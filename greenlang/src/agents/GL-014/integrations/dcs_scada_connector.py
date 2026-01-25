"""
DCS/SCADA Connector Module for GL-014 EXCHANGER-PRO (Heat Exchanger Optimization Agent).

Provides integration with Distributed Control Systems and SCADA systems for real-time
process data acquisition:
- Emerson DeltaV
- Honeywell Experion PKS
- Yokogawa CENTUM VP

Supports real-time tag subscription, alarm/event retrieval, control system status,
and setpoint management for heat exchanger optimization.

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
    Union,
)
import asyncio
import base64
import json
import logging
import time
import uuid
from collections import defaultdict, deque

import aiohttp
from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    CircuitState,
    ConnectionState,
    ConnectorType,
    DataQualityLevel,
    DataQualityResult,
    HealthCheckResult,
    HealthStatus,
    AuthenticationType,
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    ConnectorError,
    RateLimitError,
    TimeoutError,
    ValidationError,
    ProtocolError,
    with_retry,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class DCSProvider(str, Enum):
    """Supported DCS/SCADA providers."""

    EMERSON_DELTAV = "emerson_deltav"
    HONEYWELL_EXPERION = "honeywell_experion"
    YOKOGAWA_CENTUM = "yokogawa_centum"
    ABB_800XA = "abb_800xa"
    SIEMENS_PCS7 = "siemens_pcs7"
    SCHNEIDER_FOXBORO = "schneider_foxboro"
    GENERIC_OPC_DA = "generic_opc_da"


class TagQuality(str, Enum):
    """DCS tag quality status."""

    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    NOT_CONNECTED = "not_connected"
    CONFIGURATION_ERROR = "configuration_error"
    DEVICE_FAILURE = "device_failure"
    SENSOR_FAILURE = "sensor_failure"
    OUT_OF_RANGE = "out_of_range"
    MANUAL_OVERRIDE = "manual_override"


class AlarmPriority(str, Enum):
    """Alarm priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DIAGNOSTIC = "diagnostic"
    INFORMATION = "information"


class AlarmState(str, Enum):
    """Alarm state values."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    CLEARED = "cleared"
    SUPPRESSED = "suppressed"
    DISABLED = "disabled"
    SHELVED = "shelved"


class AlarmType(str, Enum):
    """Alarm types."""

    PROCESS = "process"
    EQUIPMENT = "equipment"
    SYSTEM = "system"
    SAFETY = "safety"
    ENVIRONMENTAL = "environmental"


class ControlMode(str, Enum):
    """Controller mode values."""

    AUTO = "auto"
    MANUAL = "manual"
    CASCADE = "cascade"
    REMOTE = "remote"
    LOCAL = "local"
    OUT_OF_SERVICE = "out_of_service"


class ModuleStatus(str, Enum):
    """DCS module status."""

    RUNNING = "running"
    STOPPED = "stopped"
    FAULTED = "faulted"
    INITIALIZING = "initializing"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class SubscriptionType(str, Enum):
    """Types of tag subscriptions."""

    POLLING = "polling"
    EVENT_DRIVEN = "event_driven"
    EXCEPTION_BASED = "exception_based"


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class DeltaVConfig(BaseModel):
    """Emerson DeltaV configuration."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(..., description="DeltaV server hostname")
    port: int = Field(default=8443, ge=1, le=65535, description="REST API port")
    use_ssl: bool = Field(default=True, description="Use SSL/TLS")

    # Authentication
    auth_type: AuthenticationType = Field(
        default=AuthenticationType.BASIC,
        description="Authentication type"
    )
    username: Optional[str] = Field(default=None, description="Username")
    password: Optional[str] = Field(default=None, description="Password")

    # DeltaV specific
    workstation: Optional[str] = Field(default=None, description="DeltaV workstation")
    area: Optional[str] = Field(default=None, description="Plant area")

    # Subscription settings
    default_scan_rate_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Default scan rate"
    )
    max_tags_per_subscription: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Max tags per subscription"
    )


class ExperionConfig(BaseModel):
    """Honeywell Experion PKS configuration."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(..., description="Experion server hostname")
    port: int = Field(default=5000, ge=1, le=65535, description="API port")
    use_ssl: bool = Field(default=True, description="Use SSL/TLS")

    # Authentication
    auth_type: AuthenticationType = Field(
        default=AuthenticationType.BASIC,
        description="Authentication type"
    )
    username: Optional[str] = Field(default=None, description="Username")
    password: Optional[str] = Field(default=None, description="Password")

    # Experion specific
    station_name: Optional[str] = Field(default=None, description="Station name")
    server_id: Optional[str] = Field(default=None, description="Server ID")

    # HMIWeb settings
    use_hmiweb: bool = Field(default=True, description="Use HMIWeb interface")
    hmiweb_port: int = Field(default=443, description="HMIWeb port")


class CentumConfig(BaseModel):
    """Yokogawa CENTUM VP configuration."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(..., description="CENTUM server hostname")
    port: int = Field(default=9998, ge=1, le=65535, description="API port")
    use_ssl: bool = Field(default=False, description="Use SSL/TLS")

    # Authentication
    auth_type: AuthenticationType = Field(
        default=AuthenticationType.BASIC,
        description="Authentication type"
    )
    username: Optional[str] = Field(default=None, description="Username")
    password: Optional[str] = Field(default=None, description="Password")

    # CENTUM specific
    domain: Optional[str] = Field(default=None, description="CENTUM domain")
    station: Optional[str] = Field(default=None, description="Station name")
    fcs: Optional[str] = Field(default=None, description="Field Control Station")


class DCSConnectorConfig(BaseConnectorConfig):
    """Configuration for DCS/SCADA connector."""

    model_config = ConfigDict(extra="forbid")

    # Provider settings
    provider: DCSProvider = Field(..., description="DCS/SCADA provider")

    # Provider-specific configurations
    deltav_config: Optional[DeltaVConfig] = Field(default=None, description="DeltaV config")
    experion_config: Optional[ExperionConfig] = Field(default=None, description="Experion config")
    centum_config: Optional[CentumConfig] = Field(default=None, description="CENTUM config")

    # Real-time data settings
    default_subscription_type: SubscriptionType = Field(
        default=SubscriptionType.EXCEPTION_BASED,
        description="Default subscription type"
    )
    default_scan_rate_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Default scan rate in ms"
    )
    default_deadband_percent: float = Field(
        default=0.5,
        ge=0.0,
        le=10.0,
        description="Default exception deadband percentage"
    )

    # Alarm settings
    enable_alarms: bool = Field(default=True, description="Enable alarm monitoring")
    alarm_buffer_size: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Alarm buffer size"
    )

    # Control settings
    enable_control_writes: bool = Field(
        default=False,
        description="Enable writing to control tags (DANGEROUS)"
    )
    write_confirmation_required: bool = Field(
        default=True,
        description="Require confirmation for writes"
    )

    # Heat exchanger specific
    heat_exchanger_tag_prefix: Optional[str] = Field(
        default=None,
        description="Tag prefix for heat exchangers"
    )

    @field_validator('connector_type', mode='before')
    @classmethod
    def set_connector_type(cls, v):
        return ConnectorType.DCS_SCADA


# =============================================================================
# Pydantic Models - Data Objects
# =============================================================================


class RealtimeTagValue(BaseModel):
    """Real-time tag value from DCS/SCADA."""

    model_config = ConfigDict(extra="allow")

    tag_name: str = Field(..., description="Tag name")
    value: Any = Field(..., description="Current value")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Value timestamp")
    quality: TagQuality = Field(default=TagQuality.GOOD, description="Value quality")
    quality_code: Optional[int] = Field(default=None, description="Raw quality code")

    # Engineering units
    engineering_unit: Optional[str] = Field(default=None, description="Engineering unit")
    raw_value: Optional[Any] = Field(default=None, description="Raw value before scaling")

    # Status
    is_forced: bool = Field(default=False, description="Value is forced/overridden")
    is_simulated: bool = Field(default=False, description="Value is simulated")

    # Limits
    high_limit: Optional[float] = Field(default=None, description="High limit")
    low_limit: Optional[float] = Field(default=None, description="Low limit")
    high_alarm_limit: Optional[float] = Field(default=None, description="High alarm limit")
    low_alarm_limit: Optional[float] = Field(default=None, description="Low alarm limit")


class TagSubscription(BaseModel):
    """Tag subscription configuration."""

    model_config = ConfigDict(extra="allow")

    subscription_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Subscription ID"
    )
    tag_name: str = Field(..., description="Tag name to subscribe")
    subscription_type: SubscriptionType = Field(
        default=SubscriptionType.EXCEPTION_BASED,
        description="Subscription type"
    )
    scan_rate_ms: int = Field(default=1000, ge=100, description="Scan rate in ms")
    deadband_percent: float = Field(default=0.5, ge=0, description="Deadband percentage")

    # State
    is_active: bool = Field(default=False, description="Subscription is active")
    last_value: Optional[RealtimeTagValue] = Field(default=None, description="Last value")
    value_count: int = Field(default=0, ge=0, description="Values received count")
    error_count: int = Field(default=0, ge=0, description="Error count")

    # Callback
    callback_id: Optional[str] = Field(default=None, description="Callback identifier")


class DCSAlarm(BaseModel):
    """DCS/SCADA alarm model."""

    model_config = ConfigDict(extra="allow")

    alarm_id: str = Field(..., description="Unique alarm ID")
    tag_name: str = Field(..., description="Source tag name")
    alarm_type: AlarmType = Field(default=AlarmType.PROCESS, description="Alarm type")
    priority: AlarmPriority = Field(default=AlarmPriority.MEDIUM, description="Priority")
    state: AlarmState = Field(default=AlarmState.ACTIVE, description="Alarm state")

    # Alarm details
    message: str = Field(..., description="Alarm message")
    description: Optional[str] = Field(default=None, description="Detailed description")
    condition: Optional[str] = Field(default=None, description="Alarm condition")

    # Values
    alarm_value: Optional[Any] = Field(default=None, description="Value at alarm")
    alarm_limit: Optional[float] = Field(default=None, description="Alarm limit")
    deviation: Optional[float] = Field(default=None, description="Deviation from limit")

    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Alarm time")
    acknowledged_time: Optional[datetime] = Field(default=None, description="Ack time")
    cleared_time: Optional[datetime] = Field(default=None, description="Clear time")

    # User
    acknowledged_by: Optional[str] = Field(default=None, description="Acknowledged by user")

    # Equipment
    equipment_id: Optional[str] = Field(default=None, description="Related equipment")
    area: Optional[str] = Field(default=None, description="Plant area")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ControllerStatus(BaseModel):
    """DCS controller/module status."""

    model_config = ConfigDict(extra="allow")

    module_name: str = Field(..., description="Module/controller name")
    module_type: str = Field(default="controller", description="Module type")
    status: ModuleStatus = Field(default=ModuleStatus.UNKNOWN, description="Module status")

    # Control status
    control_mode: ControlMode = Field(default=ControlMode.AUTO, description="Control mode")
    setpoint: Optional[float] = Field(default=None, description="Current setpoint")
    process_value: Optional[float] = Field(default=None, description="Process value")
    output: Optional[float] = Field(default=None, description="Control output")

    # PID tuning (if applicable)
    kp: Optional[float] = Field(default=None, description="Proportional gain")
    ki: Optional[float] = Field(default=None, description="Integral gain")
    kd: Optional[float] = Field(default=None, description="Derivative gain")

    # Limits
    output_high_limit: Optional[float] = Field(default=None, description="Output high limit")
    output_low_limit: Optional[float] = Field(default=None, description="Output low limit")

    # Health
    communication_status: str = Field(default="ok", description="Communication status")
    last_update: datetime = Field(default_factory=datetime.utcnow)

    metadata: Dict[str, Any] = Field(default_factory=dict)


class HeatExchangerControlTags(BaseModel):
    """Control tags for a heat exchanger."""

    model_config = ConfigDict(extra="allow")

    equipment_id: str = Field(..., description="Heat exchanger equipment ID")

    # Temperature control
    hot_inlet_temp_pv: Optional[str] = Field(default=None, description="Hot inlet temp PV tag")
    hot_outlet_temp_pv: Optional[str] = Field(default=None, description="Hot outlet temp PV tag")
    hot_outlet_temp_sp: Optional[str] = Field(default=None, description="Hot outlet temp SP tag")
    cold_inlet_temp_pv: Optional[str] = Field(default=None, description="Cold inlet temp PV tag")
    cold_outlet_temp_pv: Optional[str] = Field(default=None, description="Cold outlet temp PV tag")
    cold_outlet_temp_sp: Optional[str] = Field(default=None, description="Cold outlet temp SP tag")

    # Flow control
    hot_flow_pv: Optional[str] = Field(default=None, description="Hot flow PV tag")
    hot_flow_sp: Optional[str] = Field(default=None, description="Hot flow SP tag")
    hot_flow_cv: Optional[str] = Field(default=None, description="Hot flow CV tag")
    cold_flow_pv: Optional[str] = Field(default=None, description="Cold flow PV tag")
    cold_flow_sp: Optional[str] = Field(default=None, description="Cold flow SP tag")
    cold_flow_cv: Optional[str] = Field(default=None, description="Cold flow CV tag")

    # Pressure
    hot_inlet_pressure_pv: Optional[str] = Field(default=None, description="Hot inlet P tag")
    hot_outlet_pressure_pv: Optional[str] = Field(default=None, description="Hot outlet P tag")
    cold_inlet_pressure_pv: Optional[str] = Field(default=None, description="Cold inlet P tag")
    cold_outlet_pressure_pv: Optional[str] = Field(default=None, description="Cold outlet P tag")

    # Calculated
    heat_duty_pv: Optional[str] = Field(default=None, description="Heat duty PV tag")
    ua_coefficient_pv: Optional[str] = Field(default=None, description="UA coefficient tag")
    fouling_factor_pv: Optional[str] = Field(default=None, description="Fouling factor tag")
    effectiveness_pv: Optional[str] = Field(default=None, description="Effectiveness tag")

    # Status
    control_mode_tag: Optional[str] = Field(default=None, description="Control mode tag")
    equipment_status_tag: Optional[str] = Field(default=None, description="Equipment status tag")

    def get_all_tags(self) -> List[str]:
        """Get all non-None tag names."""
        tags = []
        for field_name, field_value in self.__dict__.items():
            if field_name != "equipment_id" and field_value and isinstance(field_value, str):
                tags.append(field_value)
        return tags


class HeatExchangerRealtimeData(BaseModel):
    """Real-time data snapshot for a heat exchanger."""

    model_config = ConfigDict(extra="allow")

    equipment_id: str = Field(..., description="Equipment ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Temperatures (in engineering units from DCS)
    hot_inlet_temp: Optional[float] = Field(default=None)
    hot_outlet_temp: Optional[float] = Field(default=None)
    hot_outlet_temp_sp: Optional[float] = Field(default=None)
    cold_inlet_temp: Optional[float] = Field(default=None)
    cold_outlet_temp: Optional[float] = Field(default=None)
    cold_outlet_temp_sp: Optional[float] = Field(default=None)

    # Flows
    hot_flow: Optional[float] = Field(default=None)
    hot_flow_sp: Optional[float] = Field(default=None)
    hot_flow_cv: Optional[float] = Field(default=None)
    cold_flow: Optional[float] = Field(default=None)
    cold_flow_sp: Optional[float] = Field(default=None)
    cold_flow_cv: Optional[float] = Field(default=None)

    # Pressures
    hot_inlet_pressure: Optional[float] = Field(default=None)
    hot_outlet_pressure: Optional[float] = Field(default=None)
    cold_inlet_pressure: Optional[float] = Field(default=None)
    cold_outlet_pressure: Optional[float] = Field(default=None)

    # Calculated
    heat_duty: Optional[float] = Field(default=None)
    ua_coefficient: Optional[float] = Field(default=None)
    fouling_factor: Optional[float] = Field(default=None)
    effectiveness: Optional[float] = Field(default=None)

    # Pressure drops
    hot_pressure_drop: Optional[float] = Field(default=None)
    cold_pressure_drop: Optional[float] = Field(default=None)

    # Status
    control_mode: Optional[ControlMode] = Field(default=None)
    equipment_status: Optional[str] = Field(default=None)

    # Quality
    data_quality_score: float = Field(default=1.0, ge=0, le=1)
    bad_quality_tags: List[str] = Field(default_factory=list)

    # Active alarms
    active_alarm_count: int = Field(default=0, ge=0)
    alarms: List[DCSAlarm] = Field(default_factory=list)


class SetpointChangeRequest(BaseModel):
    """Request to change a setpoint."""

    model_config = ConfigDict(extra="forbid")

    tag_name: str = Field(..., description="Setpoint tag name")
    new_value: float = Field(..., description="New setpoint value")
    reason: str = Field(..., min_length=1, description="Reason for change")
    requester: str = Field(..., description="User requesting change")

    # Safety
    confirm: bool = Field(default=False, description="Confirmation flag")
    ramp_rate: Optional[float] = Field(
        default=None,
        ge=0,
        description="Ramp rate per minute"
    )
    ramp_duration_minutes: Optional[float] = Field(
        default=None,
        ge=0,
        description="Ramp duration"
    )


class SetpointChangeResponse(BaseModel):
    """Response to setpoint change request."""

    model_config = ConfigDict(extra="allow")

    request: SetpointChangeRequest = Field(..., description="Original request")
    success: bool = Field(..., description="Change successful")
    old_value: Optional[float] = Field(default=None, description="Previous value")
    new_value: Optional[float] = Field(default=None, description="Confirmed new value")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = Field(default=None, description="Error if failed")


# =============================================================================
# Provider-Specific Adapters
# =============================================================================


class DCSProviderAdapter:
    """Base adapter for DCS provider-specific transformations."""

    def __init__(self, config: DCSConnectorConfig) -> None:
        self._config = config

    def get_base_url(self) -> str:
        """Get the base URL for API calls."""
        raise NotImplementedError

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        raise NotImplementedError

    async def read_tag(
        self,
        session: aiohttp.ClientSession,
        tag_name: str
    ) -> RealtimeTagValue:
        """Read a single tag value."""
        raise NotImplementedError

    async def read_tags(
        self,
        session: aiohttp.ClientSession,
        tag_names: List[str]
    ) -> Dict[str, RealtimeTagValue]:
        """Read multiple tag values."""
        raise NotImplementedError

    async def write_tag(
        self,
        session: aiohttp.ClientSession,
        tag_name: str,
        value: Any
    ) -> bool:
        """Write a tag value."""
        raise NotImplementedError


class DeltaVAdapter(DCSProviderAdapter):
    """Adapter for Emerson DeltaV."""

    def get_base_url(self) -> str:
        config = self._config.deltav_config
        protocol = "https" if config.use_ssl else "http"
        return f"{protocol}://{config.host}:{config.port}/api"

    def get_auth_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        config = self._config.deltav_config
        if config.auth_type == AuthenticationType.BASIC:
            credentials = base64.b64encode(
                f"{config.username}:{config.password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"
        return headers

    async def read_tag(
        self,
        session: aiohttp.ClientSession,
        tag_name: str
    ) -> RealtimeTagValue:
        """Read tag from DeltaV."""
        url = f"{self.get_base_url()}/v1/tags/{tag_name}/value"

        async with session.get(url, headers=self.get_auth_headers()) as response:
            response.raise_for_status()
            data = await response.json()

            return RealtimeTagValue(
                tag_name=tag_name,
                value=data.get("value"),
                timestamp=datetime.fromisoformat(data.get("timestamp", "").replace("Z", "+00:00")),
                quality=self._map_quality(data.get("quality", "good")),
                engineering_unit=data.get("unit"),
            )

    async def read_tags(
        self,
        session: aiohttp.ClientSession,
        tag_names: List[str]
    ) -> Dict[str, RealtimeTagValue]:
        """Read multiple tags from DeltaV."""
        url = f"{self.get_base_url()}/v1/tags/values"

        async with session.post(
            url,
            headers=self.get_auth_headers(),
            json={"tags": tag_names}
        ) as response:
            response.raise_for_status()
            data = await response.json()

            results = {}
            for item in data.get("values", []):
                tag_name = item.get("tag")
                results[tag_name] = RealtimeTagValue(
                    tag_name=tag_name,
                    value=item.get("value"),
                    timestamp=datetime.fromisoformat(
                        item.get("timestamp", "").replace("Z", "+00:00")
                    ),
                    quality=self._map_quality(item.get("quality", "good")),
                    engineering_unit=item.get("unit"),
                )

            return results

    async def write_tag(
        self,
        session: aiohttp.ClientSession,
        tag_name: str,
        value: Any
    ) -> bool:
        """Write tag to DeltaV."""
        url = f"{self.get_base_url()}/v1/tags/{tag_name}/value"

        async with session.put(
            url,
            headers=self.get_auth_headers(),
            json={"value": value}
        ) as response:
            return response.status == 200

    def _map_quality(self, quality: str) -> TagQuality:
        quality_map = {
            "good": TagQuality.GOOD,
            "uncertain": TagQuality.UNCERTAIN,
            "bad": TagQuality.BAD,
        }
        return quality_map.get(quality.lower(), TagQuality.UNCERTAIN)


class ExperionAdapter(DCSProviderAdapter):
    """Adapter for Honeywell Experion PKS."""

    def get_base_url(self) -> str:
        config = self._config.experion_config
        if config.use_hmiweb:
            protocol = "https" if config.use_ssl else "http"
            return f"{protocol}://{config.host}:{config.hmiweb_port}/hmiweb/api"
        protocol = "https" if config.use_ssl else "http"
        return f"{protocol}://{config.host}:{config.port}/api"

    def get_auth_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        config = self._config.experion_config
        if config.auth_type == AuthenticationType.BASIC:
            credentials = base64.b64encode(
                f"{config.username}:{config.password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"
        return headers

    async def read_tag(
        self,
        session: aiohttp.ClientSession,
        tag_name: str
    ) -> RealtimeTagValue:
        """Read tag from Experion."""
        url = f"{self.get_base_url()}/points/{tag_name}"

        async with session.get(url, headers=self.get_auth_headers()) as response:
            response.raise_for_status()
            data = await response.json()

            return RealtimeTagValue(
                tag_name=tag_name,
                value=data.get("PV", data.get("value")),
                timestamp=datetime.utcnow(),
                quality=self._map_quality(data.get("status", "good")),
                engineering_unit=data.get("EU", data.get("unit")),
            )

    async def read_tags(
        self,
        session: aiohttp.ClientSession,
        tag_names: List[str]
    ) -> Dict[str, RealtimeTagValue]:
        """Read multiple tags from Experion."""
        url = f"{self.get_base_url()}/points/batch"

        async with session.post(
            url,
            headers=self.get_auth_headers(),
            json={"pointNames": tag_names}
        ) as response:
            response.raise_for_status()
            data = await response.json()

            results = {}
            for item in data.get("points", []):
                tag_name = item.get("name")
                results[tag_name] = RealtimeTagValue(
                    tag_name=tag_name,
                    value=item.get("PV", item.get("value")),
                    timestamp=datetime.utcnow(),
                    quality=self._map_quality(item.get("status", "good")),
                    engineering_unit=item.get("EU"),
                )

            return results

    async def write_tag(
        self,
        session: aiohttp.ClientSession,
        tag_name: str,
        value: Any
    ) -> bool:
        """Write tag to Experion."""
        url = f"{self.get_base_url()}/points/{tag_name}/write"

        async with session.post(
            url,
            headers=self.get_auth_headers(),
            json={"value": value}
        ) as response:
            return response.status in [200, 204]

    def _map_quality(self, status: str) -> TagQuality:
        if status.lower() in ["good", "ok", "normal"]:
            return TagQuality.GOOD
        elif status.lower() in ["bad", "failed", "error"]:
            return TagQuality.BAD
        return TagQuality.UNCERTAIN


class CentumAdapter(DCSProviderAdapter):
    """Adapter for Yokogawa CENTUM VP."""

    def get_base_url(self) -> str:
        config = self._config.centum_config
        protocol = "https" if config.use_ssl else "http"
        return f"{protocol}://{config.host}:{config.port}/centum/api"

    def get_auth_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        config = self._config.centum_config
        if config.auth_type == AuthenticationType.BASIC:
            credentials = base64.b64encode(
                f"{config.username}:{config.password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"
        return headers

    async def read_tag(
        self,
        session: aiohttp.ClientSession,
        tag_name: str
    ) -> RealtimeTagValue:
        """Read tag from CENTUM."""
        url = f"{self.get_base_url()}/tags/{tag_name}"

        async with session.get(url, headers=self.get_auth_headers()) as response:
            response.raise_for_status()
            data = await response.json()

            return RealtimeTagValue(
                tag_name=tag_name,
                value=data.get("value"),
                timestamp=datetime.utcnow(),
                quality=self._map_quality(data.get("quality", 192)),
                engineering_unit=data.get("unit"),
            )

    async def read_tags(
        self,
        session: aiohttp.ClientSession,
        tag_names: List[str]
    ) -> Dict[str, RealtimeTagValue]:
        """Read multiple tags from CENTUM."""
        url = f"{self.get_base_url()}/tags/read"

        async with session.post(
            url,
            headers=self.get_auth_headers(),
            json={"tagNames": tag_names}
        ) as response:
            response.raise_for_status()
            data = await response.json()

            results = {}
            for item in data.get("tags", []):
                tag_name = item.get("tagName")
                results[tag_name] = RealtimeTagValue(
                    tag_name=tag_name,
                    value=item.get("value"),
                    timestamp=datetime.utcnow(),
                    quality=self._map_quality(item.get("quality", 192)),
                    engineering_unit=item.get("unit"),
                )

            return results

    async def write_tag(
        self,
        session: aiohttp.ClientSession,
        tag_name: str,
        value: Any
    ) -> bool:
        """Write tag to CENTUM."""
        url = f"{self.get_base_url()}/tags/{tag_name}/write"

        async with session.post(
            url,
            headers=self.get_auth_headers(),
            json={"value": value}
        ) as response:
            return response.status in [200, 204]

    def _map_quality(self, quality_code: int) -> TagQuality:
        # CENTUM uses OPC quality codes
        if quality_code >= 192:
            return TagQuality.GOOD
        elif quality_code >= 64:
            return TagQuality.UNCERTAIN
        return TagQuality.BAD


# =============================================================================
# DCS/SCADA Connector Implementation
# =============================================================================


class DCSConnector(BaseConnector):
    """
    DCS/SCADA Connector for GL-014 EXCHANGER-PRO.

    Provides integration with distributed control systems for real-time
    process data acquisition and control:
    - Emerson DeltaV
    - Honeywell Experion PKS
    - Yokogawa CENTUM VP

    Features:
    - Real-time tag value reading
    - Tag subscription for continuous data
    - Alarm monitoring and retrieval
    - Control system status monitoring
    - Setpoint changes (with safety controls)
    - Heat exchanger specific data aggregation
    """

    def __init__(self, config: DCSConnectorConfig) -> None:
        """Initialize DCS/SCADA connector."""
        super().__init__(config)
        self._dcs_config = config

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

        # Provider adapter
        self._adapter = self._create_adapter()

        # Subscriptions
        self._subscriptions: Dict[str, TagSubscription] = {}
        self._subscription_task: Optional[asyncio.Task] = None

        # Alarm buffer
        self._alarm_buffer: deque = deque(maxlen=config.alarm_buffer_size)

        # Heat exchanger tag sets
        self._heat_exchanger_tags: Dict[str, HeatExchangerControlTags] = {}

        # Value callbacks
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

    def _create_adapter(self) -> DCSProviderAdapter:
        """Create provider-specific adapter."""
        if self._dcs_config.provider == DCSProvider.EMERSON_DELTAV:
            return DeltaVAdapter(self._dcs_config)
        elif self._dcs_config.provider == DCSProvider.HONEYWELL_EXPERION:
            return ExperionAdapter(self._dcs_config)
        elif self._dcs_config.provider == DCSProvider.YOKOGAWA_CENTUM:
            return CentumAdapter(self._dcs_config)
        else:
            raise ConfigurationError(f"Unsupported DCS provider: {self._dcs_config.provider}")

    async def connect(self) -> None:
        """Establish connection to DCS/SCADA."""
        self._logger.info(
            f"Connecting to {self._dcs_config.provider.value} DCS..."
        )

        # Create HTTP session
        timeout = aiohttp.ClientTimeout(
            total=self._config.connection_timeout_seconds,
            connect=30,
            sock_read=self._config.read_timeout_seconds
        )

        # SSL context - in production, configure proper certificates
        ssl_context = False  # Disable SSL verification for development

        connector = aiohttp.TCPConnector(
            limit=self._config.pool_max_size,
            limit_per_host=self._config.pool_max_size,
            ssl=ssl_context,
        )

        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
        )

        # Test connection
        try:
            await self._test_connection()
            self._state = ConnectionState.CONNECTED
            self._logger.info(
                f"Successfully connected to {self._dcs_config.provider.value}"
            )

            # Start subscription task
            if self._subscriptions:
                self._subscription_task = asyncio.create_task(
                    self._subscription_loop()
                )

        except Exception as e:
            await self.disconnect()
            raise ConnectionError(f"Failed to connect to DCS: {str(e)}")

    async def disconnect(self) -> None:
        """Disconnect from DCS/SCADA."""
        # Stop subscription task
        if self._subscription_task:
            self._subscription_task.cancel()
            try:
                await self._subscription_task
            except asyncio.CancelledError:
                pass
            self._subscription_task = None

        # Close session
        if self._session:
            await self._session.close()
            self._session = None

        self._state = ConnectionState.DISCONNECTED
        self._logger.info("Disconnected from DCS")

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on DCS connection."""
        start_time = time.time()

        try:
            if not self._session:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Session not initialized",
                    latency_ms=0.0
                )

            await self._test_connection()
            latency_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="DCS connection healthy",
                latency_ms=latency_ms,
                details={
                    "provider": self._dcs_config.provider.value,
                    "active_subscriptions": len(self._subscriptions),
                }
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000
            )

    async def validate_configuration(self) -> bool:
        """Validate DCS connector configuration."""
        provider = self._dcs_config.provider

        if provider == DCSProvider.EMERSON_DELTAV:
            if not self._dcs_config.deltav_config:
                raise ConfigurationError("DeltaV configuration required")
            if not self._dcs_config.deltav_config.host:
                raise ConfigurationError("DeltaV host required")

        elif provider == DCSProvider.HONEYWELL_EXPERION:
            if not self._dcs_config.experion_config:
                raise ConfigurationError("Experion configuration required")
            if not self._dcs_config.experion_config.host:
                raise ConfigurationError("Experion host required")

        elif provider == DCSProvider.YOKOGAWA_CENTUM:
            if not self._dcs_config.centum_config:
                raise ConfigurationError("CENTUM configuration required")
            if not self._dcs_config.centum_config.host:
                raise ConfigurationError("CENTUM host required")

        return True

    async def _test_connection(self) -> None:
        """Test connection with a simple read."""
        # Try to read a system tag or any available tag
        try:
            base_url = self._adapter.get_base_url()
            headers = self._adapter.get_auth_headers()

            # Provider-specific health endpoint
            if self._dcs_config.provider == DCSProvider.EMERSON_DELTAV:
                url = f"{base_url}/v1/health"
            elif self._dcs_config.provider == DCSProvider.HONEYWELL_EXPERION:
                url = f"{base_url}/status"
            else:
                url = f"{base_url}/health"

            async with self._session.get(url, headers=headers) as response:
                if response.status >= 400:
                    raise ConnectionError(f"Health check failed: {response.status}")

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection test failed: {str(e)}")

    # =========================================================================
    # Tag Operations
    # =========================================================================

    async def read_tag(self, tag_name: str) -> RealtimeTagValue:
        """
        Read a single tag value.

        Args:
            tag_name: Tag name to read

        Returns:
            Real-time tag value
        """
        async def _do_read() -> RealtimeTagValue:
            return await self._adapter.read_tag(self._session, tag_name)

        return await self.execute_with_protection(
            _do_read,
            operation_name="read_tag",
            use_cache=True,
            cache_key=self._generate_cache_key("read_tag", tag_name),
        )

    async def read_tags(self, tag_names: List[str]) -> Dict[str, RealtimeTagValue]:
        """
        Read multiple tag values.

        Args:
            tag_names: List of tag names to read

        Returns:
            Dictionary of tag values keyed by tag name
        """
        async def _do_read() -> Dict[str, RealtimeTagValue]:
            return await self._adapter.read_tags(self._session, tag_names)

        return await self.execute_with_protection(
            _do_read,
            operation_name="read_tags",
            use_cache=True,
            cache_key=self._generate_cache_key("read_tags", *sorted(tag_names)),
        )

    async def write_tag(
        self,
        request: SetpointChangeRequest
    ) -> SetpointChangeResponse:
        """
        Write a tag value (setpoint change).

        Args:
            request: Setpoint change request

        Returns:
            Setpoint change response
        """
        if not self._dcs_config.enable_control_writes:
            return SetpointChangeResponse(
                request=request,
                success=False,
                error_message="Control writes are disabled"
            )

        if self._dcs_config.write_confirmation_required and not request.confirm:
            return SetpointChangeResponse(
                request=request,
                success=False,
                error_message="Confirmation required for setpoint changes"
            )

        # Get current value
        current = await self.read_tag(request.tag_name)
        old_value = current.value if isinstance(current.value, (int, float)) else None

        # Perform write
        try:
            success = await self._adapter.write_tag(
                self._session,
                request.tag_name,
                request.new_value
            )

            if success:
                self._logger.info(
                    f"Setpoint changed: {request.tag_name} "
                    f"{old_value} -> {request.new_value} "
                    f"by {request.requester}: {request.reason}"
                )

            return SetpointChangeResponse(
                request=request,
                success=success,
                old_value=old_value,
                new_value=request.new_value if success else None,
            )

        except Exception as e:
            return SetpointChangeResponse(
                request=request,
                success=False,
                old_value=old_value,
                error_message=str(e)
            )

    # =========================================================================
    # Subscription Operations
    # =========================================================================

    async def subscribe(
        self,
        tag_name: str,
        callback: Optional[Callable[[RealtimeTagValue], None]] = None,
        scan_rate_ms: Optional[int] = None,
        deadband_percent: Optional[float] = None,
    ) -> TagSubscription:
        """
        Subscribe to tag value changes.

        Args:
            tag_name: Tag name to subscribe
            callback: Callback function for value changes
            scan_rate_ms: Scan rate in milliseconds
            deadband_percent: Exception deadband percentage

        Returns:
            Tag subscription
        """
        subscription = TagSubscription(
            tag_name=tag_name,
            subscription_type=self._dcs_config.default_subscription_type,
            scan_rate_ms=scan_rate_ms or self._dcs_config.default_scan_rate_ms,
            deadband_percent=deadband_percent or self._dcs_config.default_deadband_percent,
            is_active=True,
        )

        self._subscriptions[tag_name] = subscription

        if callback:
            callback_id = str(uuid.uuid4())
            subscription.callback_id = callback_id
            self._callbacks[tag_name].append(callback)

        # Start subscription task if not running
        if self._subscription_task is None and self._state == ConnectionState.CONNECTED:
            self._subscription_task = asyncio.create_task(
                self._subscription_loop()
            )

        self._logger.info(f"Subscribed to tag: {tag_name}")
        return subscription

    async def unsubscribe(self, tag_name: str) -> bool:
        """Unsubscribe from a tag."""
        if tag_name in self._subscriptions:
            del self._subscriptions[tag_name]
            self._callbacks.pop(tag_name, None)
            self._logger.info(f"Unsubscribed from tag: {tag_name}")
            return True
        return False

    async def _subscription_loop(self) -> None:
        """Background task for processing subscriptions."""
        while True:
            try:
                if not self._subscriptions:
                    await asyncio.sleep(1)
                    continue

                # Group subscriptions by scan rate
                scan_groups: Dict[int, List[str]] = defaultdict(list)
                for tag_name, sub in self._subscriptions.items():
                    if sub.is_active:
                        scan_groups[sub.scan_rate_ms].append(tag_name)

                # Process each scan rate group
                for scan_rate_ms, tag_names in scan_groups.items():
                    if not tag_names:
                        continue

                    try:
                        values = await self.read_tags(tag_names)

                        for tag_name, value in values.items():
                            subscription = self._subscriptions.get(tag_name)
                            if subscription:
                                # Check deadband
                                should_notify = True
                                if subscription.last_value and subscription.deadband_percent > 0:
                                    if isinstance(value.value, (int, float)) and \
                                       isinstance(subscription.last_value.value, (int, float)):
                                        change_percent = abs(
                                            (value.value - subscription.last_value.value) /
                                            subscription.last_value.value * 100
                                        ) if subscription.last_value.value != 0 else 100
                                        should_notify = change_percent >= subscription.deadband_percent

                                if should_notify:
                                    subscription.last_value = value
                                    subscription.value_count += 1

                                    # Call callbacks
                                    for callback in self._callbacks.get(tag_name, []):
                                        try:
                                            callback(value)
                                        except Exception as e:
                                            self._logger.error(
                                                f"Callback error for {tag_name}: {e}"
                                            )

                    except Exception as e:
                        self._logger.error(f"Subscription read error: {e}")
                        for tag_name in tag_names:
                            if tag_name in self._subscriptions:
                                self._subscriptions[tag_name].error_count += 1

                # Sleep for minimum scan rate
                min_scan_rate = min(scan_groups.keys()) if scan_groups else 1000
                await asyncio.sleep(min_scan_rate / 1000)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Subscription loop error: {e}")
                await asyncio.sleep(1)

    # =========================================================================
    # Alarm Operations
    # =========================================================================

    async def get_active_alarms(
        self,
        equipment_id: Optional[str] = None,
        priority: Optional[AlarmPriority] = None,
        alarm_type: Optional[AlarmType] = None,
    ) -> List[DCSAlarm]:
        """
        Get active alarms.

        Args:
            equipment_id: Filter by equipment
            priority: Filter by priority
            alarm_type: Filter by type

        Returns:
            List of active alarms
        """
        # Provider-specific alarm retrieval
        base_url = self._adapter.get_base_url()
        headers = self._adapter.get_auth_headers()

        params = {"state": "active"}
        if equipment_id:
            params["equipment"] = equipment_id
        if priority:
            params["priority"] = priority.value

        async with self._session.get(
            f"{base_url}/alarms",
            headers=headers,
            params=params
        ) as response:
            if response.status == 404:
                return []  # No alarm endpoint
            response.raise_for_status()
            data = await response.json()

            alarms = []
            for item in data.get("alarms", []):
                alarm = DCSAlarm(
                    alarm_id=item.get("id", str(uuid.uuid4())),
                    tag_name=item.get("tag", item.get("source", "")),
                    alarm_type=AlarmType(item.get("type", "process")),
                    priority=AlarmPriority(item.get("priority", "medium")),
                    state=AlarmState(item.get("state", "active")),
                    message=item.get("message", ""),
                    timestamp=datetime.utcnow(),
                )
                alarms.append(alarm)

            return alarms

    async def acknowledge_alarm(
        self,
        alarm_id: str,
        user: str,
        comment: Optional[str] = None
    ) -> bool:
        """Acknowledge an alarm."""
        base_url = self._adapter.get_base_url()
        headers = self._adapter.get_auth_headers()

        async with self._session.post(
            f"{base_url}/alarms/{alarm_id}/acknowledge",
            headers=headers,
            json={"user": user, "comment": comment}
        ) as response:
            return response.status in [200, 204]

    # =========================================================================
    # Heat Exchanger Operations
    # =========================================================================

    async def register_heat_exchanger_tags(
        self,
        tags: HeatExchangerControlTags
    ) -> None:
        """Register control tags for a heat exchanger."""
        self._heat_exchanger_tags[tags.equipment_id] = tags
        self._logger.info(
            f"Registered {len(tags.get_all_tags())} control tags for {tags.equipment_id}"
        )

    async def get_heat_exchanger_realtime_data(
        self,
        equipment_id: str
    ) -> HeatExchangerRealtimeData:
        """
        Get real-time data for a heat exchanger.

        Args:
            equipment_id: Heat exchanger equipment ID

        Returns:
            Real-time data snapshot
        """
        tags = self._heat_exchanger_tags.get(equipment_id)
        if not tags:
            raise ValidationError(f"No tags registered for equipment {equipment_id}")

        # Read all tags
        all_tag_names = tags.get_all_tags()
        if not all_tag_names:
            raise ValidationError(f"No tags configured for equipment {equipment_id}")

        values = await self.read_tags(all_tag_names)

        # Build data object
        data = HeatExchangerRealtimeData(
            equipment_id=equipment_id,
            timestamp=datetime.utcnow()
        )

        bad_quality_tags = []

        # Map tag values to data fields
        tag_field_map = {
            tags.hot_inlet_temp_pv: 'hot_inlet_temp',
            tags.hot_outlet_temp_pv: 'hot_outlet_temp',
            tags.hot_outlet_temp_sp: 'hot_outlet_temp_sp',
            tags.cold_inlet_temp_pv: 'cold_inlet_temp',
            tags.cold_outlet_temp_pv: 'cold_outlet_temp',
            tags.cold_outlet_temp_sp: 'cold_outlet_temp_sp',
            tags.hot_flow_pv: 'hot_flow',
            tags.hot_flow_sp: 'hot_flow_sp',
            tags.hot_flow_cv: 'hot_flow_cv',
            tags.cold_flow_pv: 'cold_flow',
            tags.cold_flow_sp: 'cold_flow_sp',
            tags.cold_flow_cv: 'cold_flow_cv',
            tags.hot_inlet_pressure_pv: 'hot_inlet_pressure',
            tags.hot_outlet_pressure_pv: 'hot_outlet_pressure',
            tags.cold_inlet_pressure_pv: 'cold_inlet_pressure',
            tags.cold_outlet_pressure_pv: 'cold_outlet_pressure',
            tags.heat_duty_pv: 'heat_duty',
            tags.ua_coefficient_pv: 'ua_coefficient',
            tags.fouling_factor_pv: 'fouling_factor',
            tags.effectiveness_pv: 'effectiveness',
        }

        for tag_name, field_name in tag_field_map.items():
            if not tag_name:
                continue

            if tag_name in values:
                tag_value = values[tag_name]
                if tag_value.quality == TagQuality.GOOD:
                    setattr(data, field_name, tag_value.value)
                else:
                    bad_quality_tags.append(tag_name)

        # Calculate pressure drops
        if data.hot_inlet_pressure and data.hot_outlet_pressure:
            data.hot_pressure_drop = data.hot_inlet_pressure - data.hot_outlet_pressure
        if data.cold_inlet_pressure and data.cold_outlet_pressure:
            data.cold_pressure_drop = data.cold_inlet_pressure - data.cold_outlet_pressure

        # Data quality score
        total_tags = len([t for t in tag_field_map.keys() if t])
        good_tags = total_tags - len(bad_quality_tags)
        data.data_quality_score = good_tags / total_tags if total_tags > 0 else 0
        data.bad_quality_tags = bad_quality_tags

        # Get active alarms
        if self._dcs_config.enable_alarms:
            try:
                alarms = await self.get_active_alarms(equipment_id=equipment_id)
                data.alarms = alarms
                data.active_alarm_count = len(alarms)
            except Exception as e:
                self._logger.warning(f"Failed to get alarms for {equipment_id}: {e}")

        return data

    async def subscribe_heat_exchanger(
        self,
        equipment_id: str,
        callback: Optional[Callable[[HeatExchangerRealtimeData], None]] = None
    ) -> List[TagSubscription]:
        """
        Subscribe to all tags for a heat exchanger.

        Args:
            equipment_id: Heat exchanger equipment ID
            callback: Optional callback for aggregated data

        Returns:
            List of subscriptions created
        """
        tags = self._heat_exchanger_tags.get(equipment_id)
        if not tags:
            raise ValidationError(f"No tags registered for equipment {equipment_id}")

        subscriptions = []
        for tag_name in tags.get_all_tags():
            sub = await self.subscribe(tag_name)
            subscriptions.append(sub)

        # If callback provided, set up aggregation
        if callback:
            async def _aggregated_callback():
                while True:
                    try:
                        data = await self.get_heat_exchanger_realtime_data(equipment_id)
                        callback(data)
                    except Exception as e:
                        self._logger.error(f"Aggregation callback error: {e}")
                    await asyncio.sleep(self._dcs_config.default_scan_rate_ms / 1000)

            asyncio.create_task(_aggregated_callback())

        return subscriptions

    # =========================================================================
    # Controller Status Operations
    # =========================================================================

    async def get_controller_status(
        self,
        module_name: str
    ) -> ControllerStatus:
        """
        Get controller/module status.

        Args:
            module_name: Controller/module name

        Returns:
            Controller status
        """
        base_url = self._adapter.get_base_url()
        headers = self._adapter.get_auth_headers()

        async with self._session.get(
            f"{base_url}/modules/{module_name}/status",
            headers=headers
        ) as response:
            if response.status == 404:
                return ControllerStatus(
                    module_name=module_name,
                    status=ModuleStatus.UNKNOWN
                )
            response.raise_for_status()
            data = await response.json()

            return ControllerStatus(
                module_name=module_name,
                module_type=data.get("type", "controller"),
                status=ModuleStatus(data.get("status", "unknown")),
                control_mode=ControlMode(data.get("mode", "auto")),
                setpoint=data.get("SP"),
                process_value=data.get("PV"),
                output=data.get("OUT"),
            )


# =============================================================================
# Factory Function
# =============================================================================


def create_dcs_connector(
    provider: DCSProvider,
    connector_name: str,
    deltav_config: Optional[DeltaVConfig] = None,
    experion_config: Optional[ExperionConfig] = None,
    centum_config: Optional[CentumConfig] = None,
    **kwargs
) -> DCSConnector:
    """
    Factory function to create DCS/SCADA connector.

    Args:
        provider: DCS provider
        connector_name: Connector name
        deltav_config: DeltaV configuration
        experion_config: Experion configuration
        centum_config: CENTUM configuration
        **kwargs: Additional configuration options

    Returns:
        Configured DCS connector
    """
    config = DCSConnectorConfig(
        connector_name=connector_name,
        connector_type=ConnectorType.DCS_SCADA,
        provider=provider,
        deltav_config=deltav_config,
        experion_config=experion_config,
        centum_config=centum_config,
        **kwargs
    )

    return DCSConnector(config)
