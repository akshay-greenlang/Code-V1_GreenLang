"""
Process Historian Connector Module for GL-014 EXCHANGER-PRO.

Provides integration with major process historians for heat exchanger time-series data:
- OSIsoft PI Web API
- Honeywell PHD (Process History Database)
- AspenTech IP.21 (InfoPlus.21)
- Generic OPC-UA connector

Supports time-series data retrieval, tag browsing, discovery, interpolated/raw data
modes, and bulk data retrieval optimization.

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
from urllib.parse import urlencode, urljoin

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


class HistorianProvider(str, Enum):
    """Supported process historian providers."""

    OSISOFT_PI = "osisoft_pi"  # OSIsoft PI System
    HONEYWELL_PHD = "honeywell_phd"  # Honeywell PHD
    ASPENTECH_IP21 = "aspentech_ip21"  # AspenTech InfoPlus.21
    OPC_UA = "opc_ua"  # Generic OPC-UA
    AVEVA_HISTORIAN = "aveva_historian"  # AVEVA/Wonderware Historian
    GE_PROFICY = "ge_proficy"  # GE Proficy Historian
    IGNITION = "ignition"  # Inductive Automation Ignition


class DataRetrievalMode(str, Enum):
    """Data retrieval modes for time-series queries."""

    RAW = "raw"  # Raw recorded values
    INTERPOLATED = "interpolated"  # Linear interpolation
    PLOT = "plot"  # Plot-optimized (min/max/avg per interval)
    AVERAGE = "average"  # Time-weighted average
    MINIMUM = "minimum"  # Minimum value per interval
    MAXIMUM = "maximum"  # Maximum value per interval
    TOTAL = "total"  # Totalized value per interval
    COUNT = "count"  # Count of values per interval
    RANGE = "range"  # Max - Min per interval
    STDEV = "stdev"  # Standard deviation per interval
    SNAPSHOT = "snapshot"  # Current/latest value


class TagQuality(str, Enum):
    """Tag data quality status."""

    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    SUBSTITUTED = "substituted"
    QUESTIONABLE = "questionable"
    NOT_AVAILABLE = "not_available"
    STALE = "stale"


class TagDataType(str, Enum):
    """Tag data types."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    BOOLEAN = "boolean"
    STRING = "string"
    DIGITAL = "digital"
    TIMESTAMP = "timestamp"
    BLOB = "blob"


class TagType(str, Enum):
    """Types of historian tags."""

    ANALOG = "analog"
    DIGITAL = "digital"
    STRING = "string"
    ARRAY = "array"
    CALCULATED = "calculated"
    TOTALIZER = "totalizer"


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class PIWebAPIConfig(BaseModel):
    """OSIsoft PI Web API configuration."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(..., description="PI Web API base URL")
    data_archive: str = Field(..., description="PI Data Archive server name")
    asset_server: Optional[str] = Field(default=None, description="AF Server name")
    asset_database: Optional[str] = Field(default=None, description="AF Database name")

    # Authentication
    auth_type: AuthenticationType = Field(
        default=AuthenticationType.BASIC,
        description="Authentication type"
    )
    username: Optional[str] = Field(default=None, description="Username")
    password: Optional[str] = Field(default=None, description="Password")
    kerberos_spn: Optional[str] = Field(default=None, description="Kerberos SPN")

    # API settings
    api_version: str = Field(default="1.0", description="PI Web API version")
    max_points_per_request: int = Field(
        default=250000,
        ge=1000,
        le=1000000,
        description="Maximum points per request"
    )
    bulk_parallelism: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Parallel requests for bulk operations"
    )


class PHDConfig(BaseModel):
    """Honeywell PHD configuration."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(..., description="PHD server hostname")
    port: int = Field(default=5465, ge=1, le=65535, description="PHD server port")
    database: str = Field(..., description="PHD database name")

    # Authentication
    auth_type: AuthenticationType = Field(
        default=AuthenticationType.BASIC,
        description="Authentication type"
    )
    username: Optional[str] = Field(default=None, description="Username")
    password: Optional[str] = Field(default=None, description="Password")

    # API settings
    api_endpoint: str = Field(default="/phd/api", description="API endpoint")
    max_tags_per_request: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum tags per request"
    )


class IP21Config(BaseModel):
    """AspenTech IP.21 configuration."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(..., description="IP.21 server hostname")
    port: int = Field(default=10014, ge=1, le=65535, description="IP.21 SQLplus port")
    http_port: int = Field(default=443, ge=1, le=65535, description="IP.21 REST API port")
    database: str = Field(default="IP_AnalogDef", description="Definition table")

    # Authentication
    auth_type: AuthenticationType = Field(
        default=AuthenticationType.BASIC,
        description="Authentication type"
    )
    username: Optional[str] = Field(default=None, description="Username")
    password: Optional[str] = Field(default=None, description="Password")

    # API settings
    use_rest_api: bool = Field(default=True, description="Use REST API vs SQLplus")
    max_rows_per_query: int = Field(
        default=100000,
        ge=1000,
        le=1000000,
        description="Maximum rows per query"
    )


class OPCUAConfig(BaseModel):
    """OPC-UA configuration."""

    model_config = ConfigDict(extra="forbid")

    endpoint_url: str = Field(..., description="OPC-UA endpoint URL")
    namespace_uri: Optional[str] = Field(default=None, description="Namespace URI")
    namespace_index: int = Field(default=2, ge=0, description="Namespace index")

    # Authentication
    auth_type: AuthenticationType = Field(
        default=AuthenticationType.NONE,
        description="Authentication type"
    )
    username: Optional[str] = Field(default=None, description="Username")
    password: Optional[str] = Field(default=None, description="Password")
    certificate_path: Optional[str] = Field(default=None, description="Certificate path")
    private_key_path: Optional[str] = Field(default=None, description="Private key path")

    # Security
    security_policy: str = Field(
        default="None",
        description="Security policy (None, Basic128Rsa15, Basic256, Basic256Sha256)"
    )
    security_mode: str = Field(
        default="None",
        description="Security mode (None, Sign, SignAndEncrypt)"
    )

    # Session settings
    session_timeout_ms: int = Field(default=60000, ge=1000, description="Session timeout")
    subscription_publish_interval_ms: int = Field(
        default=1000,
        ge=100,
        description="Subscription publish interval"
    )


class ProcessHistorianConnectorConfig(BaseConnectorConfig):
    """Configuration for process historian connector."""

    model_config = ConfigDict(extra="forbid")

    # Provider settings
    provider: HistorianProvider = Field(..., description="Historian provider")

    # Provider-specific configurations
    pi_config: Optional[PIWebAPIConfig] = Field(default=None, description="PI Web API config")
    phd_config: Optional[PHDConfig] = Field(default=None, description="PHD config")
    ip21_config: Optional[IP21Config] = Field(default=None, description="IP.21 config")
    opcua_config: Optional[OPCUAConfig] = Field(default=None, description="OPC-UA config")

    # Default query settings
    default_retrieval_mode: DataRetrievalMode = Field(
        default=DataRetrievalMode.INTERPOLATED,
        description="Default data retrieval mode"
    )
    default_interval_seconds: int = Field(
        default=60,
        ge=1,
        le=86400,
        description="Default interpolation interval"
    )
    max_query_duration_days: int = Field(
        default=365,
        ge=1,
        le=3650,
        description="Maximum query duration in days"
    )

    # Heat exchanger specific tags
    enable_tag_discovery: bool = Field(
        default=True,
        description="Enable automatic tag discovery"
    )
    tag_filter_patterns: List[str] = Field(
        default_factory=lambda: [
            "*HX*",  # Heat exchanger
            "*TEMP*",  # Temperature
            "*FLOW*",  # Flow rate
            "*PRESS*",  # Pressure
            "*DUTY*",  # Heat duty
            "*UA*",  # Overall heat transfer coefficient
            "*FOUL*",  # Fouling
            "*DT*",  # Delta temperature
            "*LMTD*",  # Log mean temperature difference
        ],
        description="Tag filter patterns for discovery"
    )

    @field_validator('connector_type', mode='before')
    @classmethod
    def set_connector_type(cls, v):
        return ConnectorType.PROCESS_HISTORIAN


# =============================================================================
# Pydantic Models - Data Objects
# =============================================================================


class TagDefinition(BaseModel):
    """Tag definition/metadata model."""

    model_config = ConfigDict(extra="allow")

    tag_name: str = Field(..., description="Tag name/identifier")
    tag_id: Optional[str] = Field(default=None, description="Unique tag ID")
    description: Optional[str] = Field(default=None, description="Tag description")

    # Classification
    tag_type: TagType = Field(default=TagType.ANALOG, description="Tag type")
    data_type: TagDataType = Field(default=TagDataType.FLOAT64, description="Data type")

    # Engineering units
    engineering_unit: Optional[str] = Field(default=None, description="Engineering unit")
    zero_scale: Optional[float] = Field(default=None, description="Zero scale value")
    span_scale: Optional[float] = Field(default=None, description="Span scale value")

    # Location
    point_source: Optional[str] = Field(default=None, description="Point source")
    server: Optional[str] = Field(default=None, description="Server name")
    database: Optional[str] = Field(default=None, description="Database name")

    # Archiving
    compression_enabled: bool = Field(default=True, description="Compression enabled")
    exception_dev: Optional[float] = Field(default=None, description="Exception deviation")
    compression_dev: Optional[float] = Field(default=None, description="Compression deviation")

    # Heat exchanger specific
    equipment_id: Optional[str] = Field(default=None, description="Related equipment ID")
    measurement_type: Optional[str] = Field(
        default=None,
        description="Measurement type (inlet_temp, outlet_temp, flow, etc.)"
    )
    stream_side: Optional[str] = Field(
        default=None,
        description="Stream side (hot, cold, tube, shell)"
    )

    # Metadata
    created_date: Optional[datetime] = Field(default=None, description="Creation date")
    modified_date: Optional[datetime] = Field(default=None, description="Last modified date")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TagValue(BaseModel):
    """Single tag value with timestamp and quality."""

    model_config = ConfigDict(extra="allow")

    tag_name: str = Field(..., description="Tag name")
    timestamp: datetime = Field(..., description="Value timestamp")
    value: Any = Field(..., description="Tag value")
    quality: TagQuality = Field(default=TagQuality.GOOD, description="Value quality")
    quality_code: Optional[int] = Field(default=None, description="Raw quality code")

    # Optional context
    engineering_unit: Optional[str] = Field(default=None, description="Engineering unit")
    is_annotated: bool = Field(default=False, description="Has annotation")
    annotation: Optional[str] = Field(default=None, description="Annotation text")


class TimeSeriesData(BaseModel):
    """Time series data for a tag."""

    model_config = ConfigDict(extra="allow")

    tag_name: str = Field(..., description="Tag name")
    tag_definition: Optional[TagDefinition] = Field(default=None, description="Tag definition")

    # Query parameters
    start_time: datetime = Field(..., description="Query start time")
    end_time: datetime = Field(..., description="Query end time")
    retrieval_mode: DataRetrievalMode = Field(..., description="Retrieval mode used")
    interval_seconds: Optional[int] = Field(default=None, description="Interval for interpolated")

    # Data
    values: List[TagValue] = Field(default_factory=list, description="Time series values")
    point_count: int = Field(default=0, ge=0, description="Number of points")

    # Statistics
    min_value: Optional[float] = Field(default=None, description="Minimum value")
    max_value: Optional[float] = Field(default=None, description="Maximum value")
    avg_value: Optional[float] = Field(default=None, description="Average value")
    stdev_value: Optional[float] = Field(default=None, description="Standard deviation")

    # Quality summary
    good_quality_percent: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Percentage of good quality values"
    )
    bad_quality_count: int = Field(default=0, ge=0, description="Count of bad quality values")

    # Metadata
    query_duration_ms: Optional[float] = Field(default=None, description="Query duration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BulkTimeSeriesRequest(BaseModel):
    """Request for bulk time series data retrieval."""

    model_config = ConfigDict(extra="forbid")

    tag_names: List[str] = Field(..., min_length=1, description="Tag names to retrieve")
    start_time: datetime = Field(..., description="Start time")
    end_time: datetime = Field(..., description="End time")
    retrieval_mode: DataRetrievalMode = Field(
        default=DataRetrievalMode.INTERPOLATED,
        description="Retrieval mode"
    )
    interval_seconds: Optional[int] = Field(
        default=60,
        ge=1,
        description="Interval for interpolated mode"
    )
    include_tag_definitions: bool = Field(
        default=False,
        description="Include tag definitions in response"
    )
    max_points_per_tag: Optional[int] = Field(
        default=None,
        description="Maximum points per tag"
    )


class BulkTimeSeriesResponse(BaseModel):
    """Response for bulk time series data retrieval."""

    model_config = ConfigDict(extra="allow")

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request ID"
    )
    request: BulkTimeSeriesRequest = Field(..., description="Original request")

    # Results
    results: Dict[str, TimeSeriesData] = Field(
        default_factory=dict,
        description="Results by tag name"
    )
    successful_tags: List[str] = Field(default_factory=list, description="Successful tags")
    failed_tags: List[str] = Field(default_factory=list, description="Failed tags")
    errors: Dict[str, str] = Field(default_factory=dict, description="Errors by tag")

    # Summary
    total_tags: int = Field(default=0, ge=0, description="Total tags requested")
    total_points: int = Field(default=0, ge=0, description="Total points retrieved")
    query_duration_ms: float = Field(default=0.0, ge=0, description="Total query duration")

    # Quality
    overall_quality_score: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Overall data quality score"
    )


class TagSearchRequest(BaseModel):
    """Request for tag search/discovery."""

    model_config = ConfigDict(extra="forbid")

    search_pattern: str = Field(..., description="Search pattern (supports wildcards)")
    tag_type: Optional[TagType] = Field(default=None, description="Filter by tag type")
    point_source: Optional[str] = Field(default=None, description="Filter by point source")
    description_pattern: Optional[str] = Field(
        default=None,
        description="Description search pattern"
    )
    max_results: int = Field(default=1000, ge=1, le=10000, description="Maximum results")
    include_metadata: bool = Field(default=True, description="Include full metadata")


class TagSearchResponse(BaseModel):
    """Response for tag search/discovery."""

    model_config = ConfigDict(extra="allow")

    request: TagSearchRequest = Field(..., description="Original request")
    tags: List[TagDefinition] = Field(default_factory=list, description="Found tags")
    total_count: int = Field(default=0, ge=0, description="Total matching tags")
    returned_count: int = Field(default=0, ge=0, description="Returned count")
    truncated: bool = Field(default=False, description="Results truncated")
    query_duration_ms: float = Field(default=0.0, ge=0, description="Query duration")


# =============================================================================
# Heat Exchanger Specific Data Models
# =============================================================================


class HeatExchangerTagSet(BaseModel):
    """Complete tag set for a heat exchanger."""

    model_config = ConfigDict(extra="allow")

    equipment_id: str = Field(..., description="Heat exchanger equipment ID")
    equipment_name: Optional[str] = Field(default=None, description="Equipment name")

    # Hot side tags
    hot_inlet_temp_tag: Optional[str] = Field(default=None, description="Hot inlet temp tag")
    hot_outlet_temp_tag: Optional[str] = Field(default=None, description="Hot outlet temp tag")
    hot_flow_tag: Optional[str] = Field(default=None, description="Hot side flow tag")
    hot_inlet_pressure_tag: Optional[str] = Field(default=None, description="Hot inlet pressure")
    hot_outlet_pressure_tag: Optional[str] = Field(default=None, description="Hot outlet pressure")

    # Cold side tags
    cold_inlet_temp_tag: Optional[str] = Field(default=None, description="Cold inlet temp tag")
    cold_outlet_temp_tag: Optional[str] = Field(default=None, description="Cold outlet temp tag")
    cold_flow_tag: Optional[str] = Field(default=None, description="Cold side flow tag")
    cold_inlet_pressure_tag: Optional[str] = Field(default=None, description="Cold inlet pressure")
    cold_outlet_pressure_tag: Optional[str] = Field(default=None, description="Cold outlet pressure")

    # Calculated/derived tags
    duty_tag: Optional[str] = Field(default=None, description="Heat duty tag")
    ua_tag: Optional[str] = Field(default=None, description="UA coefficient tag")
    lmtd_tag: Optional[str] = Field(default=None, description="LMTD tag")
    effectiveness_tag: Optional[str] = Field(default=None, description="Effectiveness tag")
    fouling_factor_tag: Optional[str] = Field(default=None, description="Fouling factor tag")
    pressure_drop_hot_tag: Optional[str] = Field(default=None, description="Hot side dP tag")
    pressure_drop_cold_tag: Optional[str] = Field(default=None, description="Cold side dP tag")

    # All tag names for convenience
    all_tags: List[str] = Field(default_factory=list, description="All tag names")

    def get_all_tags(self) -> List[str]:
        """Get all non-None tag names."""
        tags = []
        for field_name in [
            'hot_inlet_temp_tag', 'hot_outlet_temp_tag', 'hot_flow_tag',
            'hot_inlet_pressure_tag', 'hot_outlet_pressure_tag',
            'cold_inlet_temp_tag', 'cold_outlet_temp_tag', 'cold_flow_tag',
            'cold_inlet_pressure_tag', 'cold_outlet_pressure_tag',
            'duty_tag', 'ua_tag', 'lmtd_tag', 'effectiveness_tag',
            'fouling_factor_tag', 'pressure_drop_hot_tag', 'pressure_drop_cold_tag'
        ]:
            value = getattr(self, field_name, None)
            if value:
                tags.append(value)
        return tags


class HeatExchangerSnapshot(BaseModel):
    """Current/snapshot data for a heat exchanger."""

    model_config = ConfigDict(extra="allow")

    equipment_id: str = Field(..., description="Heat exchanger equipment ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Snapshot time")

    # Hot side values
    hot_inlet_temp: Optional[float] = Field(default=None, description="Hot inlet temp")
    hot_outlet_temp: Optional[float] = Field(default=None, description="Hot outlet temp")
    hot_flow: Optional[float] = Field(default=None, description="Hot side flow")
    hot_inlet_pressure: Optional[float] = Field(default=None, description="Hot inlet pressure")
    hot_outlet_pressure: Optional[float] = Field(default=None, description="Hot outlet pressure")

    # Cold side values
    cold_inlet_temp: Optional[float] = Field(default=None, description="Cold inlet temp")
    cold_outlet_temp: Optional[float] = Field(default=None, description="Cold outlet temp")
    cold_flow: Optional[float] = Field(default=None, description="Cold side flow")
    cold_inlet_pressure: Optional[float] = Field(default=None, description="Cold inlet pressure")
    cold_outlet_pressure: Optional[float] = Field(default=None, description="Cold outlet pressure")

    # Calculated values
    duty: Optional[float] = Field(default=None, description="Heat duty")
    ua_coefficient: Optional[float] = Field(default=None, description="UA coefficient")
    lmtd: Optional[float] = Field(default=None, description="LMTD")
    effectiveness: Optional[float] = Field(default=None, description="Effectiveness")
    fouling_factor: Optional[float] = Field(default=None, description="Fouling factor")
    hot_pressure_drop: Optional[float] = Field(default=None, description="Hot side dP")
    cold_pressure_drop: Optional[float] = Field(default=None, description="Cold side dP")

    # Quality
    data_quality_score: float = Field(default=1.0, ge=0, le=1, description="Data quality")
    missing_tags: List[str] = Field(default_factory=list, description="Missing tags")
    bad_quality_tags: List[str] = Field(default_factory=list, description="Bad quality tags")

    # Units
    temp_unit: str = Field(default="C", description="Temperature unit")
    flow_unit: str = Field(default="kg/s", description="Flow unit")
    pressure_unit: str = Field(default="kPa", description="Pressure unit")
    duty_unit: str = Field(default="kW", description="Duty unit")


# =============================================================================
# Provider-Specific Adapters
# =============================================================================


class HistorianProviderAdapter:
    """Base adapter for historian provider-specific API transformations."""

    def __init__(self, config: ProcessHistorianConnectorConfig) -> None:
        self._config = config

    def get_base_url(self) -> str:
        """Get the base URL for API calls."""
        raise NotImplementedError

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        raise NotImplementedError

    async def build_time_series_request(
        self,
        tag_name: str,
        start_time: datetime,
        end_time: datetime,
        mode: DataRetrievalMode,
        interval_seconds: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Build time series request (url, params, body)."""
        raise NotImplementedError

    def parse_time_series_response(
        self,
        response_data: Dict[str, Any],
        tag_name: str
    ) -> List[TagValue]:
        """Parse time series response into TagValue list."""
        raise NotImplementedError


class PIWebAPIAdapter(HistorianProviderAdapter):
    """Adapter for OSIsoft PI Web API."""

    def get_base_url(self) -> str:
        return self._config.pi_config.base_url.rstrip("/")

    def get_auth_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self._config.pi_config.auth_type == AuthenticationType.BASIC:
            credentials = base64.b64encode(
                f"{self._config.pi_config.username}:{self._config.pi_config.password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"

        return headers

    async def build_time_series_request(
        self,
        tag_name: str,
        start_time: datetime,
        end_time: datetime,
        mode: DataRetrievalMode,
        interval_seconds: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Build PI Web API request for time series data."""
        base_url = self.get_base_url()

        # URL encode the tag path
        web_id = await self._get_tag_web_id(tag_name)

        # Build endpoint based on retrieval mode
        if mode == DataRetrievalMode.RAW:
            endpoint = f"{base_url}/streams/{web_id}/recorded"
        elif mode == DataRetrievalMode.INTERPOLATED:
            endpoint = f"{base_url}/streams/{web_id}/interpolated"
        elif mode == DataRetrievalMode.PLOT:
            endpoint = f"{base_url}/streams/{web_id}/plot"
        elif mode in [
            DataRetrievalMode.AVERAGE, DataRetrievalMode.MINIMUM,
            DataRetrievalMode.MAXIMUM, DataRetrievalMode.TOTAL
        ]:
            endpoint = f"{base_url}/streams/{web_id}/summary"
        elif mode == DataRetrievalMode.SNAPSHOT:
            endpoint = f"{base_url}/streams/{web_id}/value"
        else:
            endpoint = f"{base_url}/streams/{web_id}/interpolated"

        # Build query parameters
        params = {
            "startTime": start_time.isoformat(),
            "endTime": end_time.isoformat(),
        }

        if mode == DataRetrievalMode.INTERPOLATED and interval_seconds:
            params["interval"] = f"{interval_seconds}s"

        if mode in [
            DataRetrievalMode.AVERAGE, DataRetrievalMode.MINIMUM,
            DataRetrievalMode.MAXIMUM, DataRetrievalMode.TOTAL
        ]:
            params["summaryType"] = mode.value.capitalize()
            if interval_seconds:
                params["summaryDuration"] = f"{interval_seconds}s"

        return endpoint, params, None

    async def _get_tag_web_id(self, tag_name: str) -> str:
        """Get PI Web ID for a tag (placeholder - would query API)."""
        # In real implementation, this would query the PI Web API
        # to get the WebID for the tag
        return base64.urlsafe_b64encode(tag_name.encode()).decode()

    def parse_time_series_response(
        self,
        response_data: Dict[str, Any],
        tag_name: str
    ) -> List[TagValue]:
        """Parse PI Web API response."""
        values = []
        items = response_data.get("Items", [])

        for item in items:
            timestamp_str = item.get("Timestamp", "")
            value = item.get("Value", None)
            good = item.get("Good", True)

            try:
                timestamp = datetime.fromisoformat(
                    timestamp_str.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                continue

            # Handle digital states
            if isinstance(value, dict):
                value = value.get("Value", value.get("Name", str(value)))

            quality = TagQuality.GOOD if good else TagQuality.BAD

            values.append(TagValue(
                tag_name=tag_name,
                timestamp=timestamp,
                value=value,
                quality=quality
            ))

        return values


class PHDAdapter(HistorianProviderAdapter):
    """Adapter for Honeywell PHD."""

    def get_base_url(self) -> str:
        config = self._config.phd_config
        return f"https://{config.host}:{config.port}{config.api_endpoint}"

    def get_auth_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self._config.phd_config.auth_type == AuthenticationType.BASIC:
            credentials = base64.b64encode(
                f"{self._config.phd_config.username}:{self._config.phd_config.password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"

        return headers

    async def build_time_series_request(
        self,
        tag_name: str,
        start_time: datetime,
        end_time: datetime,
        mode: DataRetrievalMode,
        interval_seconds: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Build PHD API request."""
        base_url = self.get_base_url()
        endpoint = f"{base_url}/data/history"

        # PHD uses different sample type codes
        sample_type_map = {
            DataRetrievalMode.RAW: "raw",
            DataRetrievalMode.INTERPOLATED: "interpolated",
            DataRetrievalMode.AVERAGE: "average",
            DataRetrievalMode.MINIMUM: "minimum",
            DataRetrievalMode.MAXIMUM: "maximum",
            DataRetrievalMode.SNAPSHOT: "snapshot",
        }

        body = {
            "tagnames": [tag_name],
            "starttime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "endtime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "sampletype": sample_type_map.get(mode, "interpolated"),
        }

        if interval_seconds and mode == DataRetrievalMode.INTERPOLATED:
            body["samplefrequency"] = interval_seconds

        return endpoint, {}, body

    def parse_time_series_response(
        self,
        response_data: Dict[str, Any],
        tag_name: str
    ) -> List[TagValue]:
        """Parse PHD API response."""
        values = []
        data = response_data.get("data", {}).get(tag_name, [])

        for item in data:
            timestamp_str = item.get("timestamp", "")
            value = item.get("value")
            quality_code = item.get("confidence", 100)

            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except (ValueError, TypeError):
                continue

            quality = TagQuality.GOOD if quality_code >= 50 else TagQuality.BAD

            values.append(TagValue(
                tag_name=tag_name,
                timestamp=timestamp,
                value=value,
                quality=quality,
                quality_code=quality_code
            ))

        return values


class IP21Adapter(HistorianProviderAdapter):
    """Adapter for AspenTech IP.21."""

    def get_base_url(self) -> str:
        config = self._config.ip21_config
        return f"https://{config.host}:{config.http_port}/processdata/api/v1"

    def get_auth_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self._config.ip21_config.auth_type == AuthenticationType.BASIC:
            credentials = base64.b64encode(
                f"{self._config.ip21_config.username}:{self._config.ip21_config.password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"

        return headers

    async def build_time_series_request(
        self,
        tag_name: str,
        start_time: datetime,
        end_time: datetime,
        mode: DataRetrievalMode,
        interval_seconds: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Build IP.21 REST API request."""
        base_url = self.get_base_url()
        endpoint = f"{base_url}/history"

        # IP.21 aggregation types
        aggregation_map = {
            DataRetrievalMode.RAW: "actual",
            DataRetrievalMode.INTERPOLATED: "interpolated",
            DataRetrievalMode.AVERAGE: "average",
            DataRetrievalMode.MINIMUM: "min",
            DataRetrievalMode.MAXIMUM: "max",
            DataRetrievalMode.TOTAL: "total",
            DataRetrievalMode.SNAPSHOT: "snapshot",
        }

        body = {
            "tags": [tag_name],
            "startTime": start_time.isoformat(),
            "endTime": end_time.isoformat(),
            "aggregation": aggregation_map.get(mode, "interpolated"),
        }

        if interval_seconds:
            body["period"] = f"PT{interval_seconds}S"

        return endpoint, {}, body

    def parse_time_series_response(
        self,
        response_data: Dict[str, Any],
        tag_name: str
    ) -> List[TagValue]:
        """Parse IP.21 REST API response."""
        values = []
        tag_data = response_data.get("tags", {}).get(tag_name, {})
        samples = tag_data.get("samples", [])

        for sample in samples:
            timestamp_str = sample.get("t", "")
            value = sample.get("v")
            quality_str = sample.get("q", "Good")

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue

            quality = TagQuality.GOOD if quality_str == "Good" else TagQuality.BAD

            values.append(TagValue(
                tag_name=tag_name,
                timestamp=timestamp,
                value=value,
                quality=quality
            ))

        return values


class OPCUAAdapter(HistorianProviderAdapter):
    """Adapter for OPC-UA Historical Access."""

    def get_base_url(self) -> str:
        return self._config.opcua_config.endpoint_url

    def get_auth_headers(self) -> Dict[str, str]:
        # OPC-UA uses session-based auth, not HTTP headers
        return {}

    async def build_time_series_request(
        self,
        tag_name: str,
        start_time: datetime,
        end_time: datetime,
        mode: DataRetrievalMode,
        interval_seconds: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Build OPC-UA HA request parameters."""
        # OPC-UA uses different API - this returns request structure
        # Real implementation would use asyncua library

        params = {
            "node_id": tag_name,
            "start_time": start_time,
            "end_time": end_time,
            "aggregate_type": mode.value,
        }

        if interval_seconds:
            params["processing_interval_ms"] = interval_seconds * 1000

        return "", params, None

    def parse_time_series_response(
        self,
        response_data: Dict[str, Any],
        tag_name: str
    ) -> List[TagValue]:
        """Parse OPC-UA response."""
        values = []
        data_values = response_data.get("DataValues", [])

        for dv in data_values:
            timestamp = dv.get("SourceTimestamp")
            value = dv.get("Value", {}).get("Value")
            status_code = dv.get("StatusCode", {}).get("value", 0)

            if timestamp is None:
                continue

            # OPC-UA status code interpretation
            is_good = (status_code & 0xC0000000) == 0

            values.append(TagValue(
                tag_name=tag_name,
                timestamp=timestamp,
                value=value,
                quality=TagQuality.GOOD if is_good else TagQuality.BAD,
                quality_code=status_code
            ))

        return values


# =============================================================================
# Process Historian Connector Implementation
# =============================================================================


class ProcessHistorianConnector(BaseConnector):
    """
    Process Historian Connector for GL-014 EXCHANGER-PRO.

    Provides integration with enterprise process historians including:
    - OSIsoft PI Web API
    - Honeywell PHD
    - AspenTech IP.21
    - Generic OPC-UA

    Features:
    - Time-series data retrieval (raw, interpolated, aggregated)
    - Tag browsing and discovery
    - Bulk data retrieval optimization
    - Heat exchanger specific tag management
    - Data quality assessment
    """

    def __init__(self, config: ProcessHistorianConnectorConfig) -> None:
        """Initialize process historian connector."""
        super().__init__(config)
        self._historian_config = config

        # HTTP session for REST APIs
        self._session: Optional[aiohttp.ClientSession] = None

        # Provider adapter
        self._adapter = self._create_adapter()

        # Tag cache
        self._tag_cache: Dict[str, TagDefinition] = {}
        self._heat_exchanger_tag_sets: Dict[str, HeatExchangerTagSet] = {}

    def _create_adapter(self) -> HistorianProviderAdapter:
        """Create provider-specific adapter."""
        if self._historian_config.provider == HistorianProvider.OSISOFT_PI:
            return PIWebAPIAdapter(self._historian_config)
        elif self._historian_config.provider == HistorianProvider.HONEYWELL_PHD:
            return PHDAdapter(self._historian_config)
        elif self._historian_config.provider == HistorianProvider.ASPENTECH_IP21:
            return IP21Adapter(self._historian_config)
        elif self._historian_config.provider == HistorianProvider.OPC_UA:
            return OPCUAAdapter(self._historian_config)
        else:
            raise ConfigurationError(f"Unsupported provider: {self._historian_config.provider}")

    async def connect(self) -> None:
        """Establish connection to process historian."""
        self._logger.info(
            f"Connecting to {self._historian_config.provider.value} historian..."
        )

        # Create aiohttp session
        timeout = aiohttp.ClientTimeout(
            total=self._config.connection_timeout_seconds,
            connect=30,
            sock_read=self._config.read_timeout_seconds
        )

        connector = aiohttp.TCPConnector(
            limit=self._config.pool_max_size,
            limit_per_host=self._config.pool_max_size,
            keepalive_timeout=self._config.pool_keepalive_timeout_seconds,
            ssl=False  # Configure based on provider
        )

        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=self._adapter.get_auth_headers()
        )

        # Test connection
        try:
            await self._test_connection()
            self._state = ConnectionState.CONNECTED
            self._logger.info(
                f"Successfully connected to {self._historian_config.provider.value}"
            )
        except Exception as e:
            await self.disconnect()
            raise ConnectionError(f"Failed to connect to historian: {str(e)}")

    async def disconnect(self) -> None:
        """Disconnect from process historian."""
        if self._session:
            await self._session.close()
            self._session = None
        self._state = ConnectionState.DISCONNECTED
        self._logger.info("Disconnected from process historian")

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on historian connection."""
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
                message="Historian connection healthy",
                latency_ms=latency_ms,
                details={
                    "provider": self._historian_config.provider.value,
                    "base_url": self._adapter.get_base_url(),
                }
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000
            )

    async def validate_configuration(self) -> bool:
        """Validate historian connector configuration."""
        provider = self._historian_config.provider

        if provider == HistorianProvider.OSISOFT_PI:
            if not self._historian_config.pi_config:
                raise ConfigurationError("PI Web API configuration required")
            if not self._historian_config.pi_config.base_url:
                raise ConfigurationError("PI Web API base URL required")
            if not self._historian_config.pi_config.data_archive:
                raise ConfigurationError("PI Data Archive name required")

        elif provider == HistorianProvider.HONEYWELL_PHD:
            if not self._historian_config.phd_config:
                raise ConfigurationError("PHD configuration required")
            if not self._historian_config.phd_config.host:
                raise ConfigurationError("PHD host required")

        elif provider == HistorianProvider.ASPENTECH_IP21:
            if not self._historian_config.ip21_config:
                raise ConfigurationError("IP.21 configuration required")
            if not self._historian_config.ip21_config.host:
                raise ConfigurationError("IP.21 host required")

        elif provider == HistorianProvider.OPC_UA:
            if not self._historian_config.opcua_config:
                raise ConfigurationError("OPC-UA configuration required")
            if not self._historian_config.opcua_config.endpoint_url:
                raise ConfigurationError("OPC-UA endpoint URL required")

        return True

    async def _test_connection(self) -> None:
        """Test connection with a simple API call."""
        if self._historian_config.provider == HistorianProvider.OPC_UA:
            # OPC-UA uses different connection test
            return

        # HTTP-based historians
        base_url = self._adapter.get_base_url()
        headers = self._adapter.get_auth_headers()

        async with self._session.get(base_url, headers=headers) as response:
            if response.status >= 400:
                raise ConnectionError(
                    f"Connection test failed with status {response.status}"
                )

    # =========================================================================
    # Tag Operations
    # =========================================================================

    async def search_tags(self, request: TagSearchRequest) -> TagSearchResponse:
        """
        Search for tags in the historian.

        Args:
            request: Tag search request parameters

        Returns:
            Tag search response with matching tags
        """
        start_time = time.time()

        async def _do_search() -> TagSearchResponse:
            return await self._search_tags_impl(request)

        response = await self.execute_with_protection(
            _do_search,
            operation_name="search_tags",
            use_cache=True,
            cache_key=self._generate_cache_key(
                "search_tags",
                request.model_dump()
            )
        )

        response.query_duration_ms = (time.time() - start_time) * 1000
        return response

    async def _search_tags_impl(self, request: TagSearchRequest) -> TagSearchResponse:
        """Implementation of tag search."""
        # Provider-specific tag search
        base_url = self._adapter.get_base_url()

        if self._historian_config.provider == HistorianProvider.OSISOFT_PI:
            # PI Web API search
            search_url = f"{base_url}/points/search"
            params = {
                "query": request.search_pattern,
                "maxCount": request.max_results,
            }
            if request.point_source:
                params["pointSource"] = request.point_source

            async with self._session.get(search_url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

            tags = []
            for item in data.get("Items", []):
                tag = TagDefinition(
                    tag_name=item.get("Name", ""),
                    tag_id=item.get("WebId"),
                    description=item.get("Descriptor"),
                    point_source=item.get("PointSource"),
                    engineering_unit=item.get("EngineeringUnits"),
                    data_type=self._map_pi_data_type(item.get("PointType", "")),
                )
                tags.append(tag)

        else:
            # Generic implementation for other providers
            tags = []

        return TagSearchResponse(
            request=request,
            tags=tags,
            total_count=len(tags),
            returned_count=len(tags),
            truncated=len(tags) >= request.max_results,
        )

    def _map_pi_data_type(self, pi_type: str) -> TagDataType:
        """Map PI point type to TagDataType."""
        type_map = {
            "Float32": TagDataType.FLOAT32,
            "Float64": TagDataType.FLOAT64,
            "Int16": TagDataType.INT16,
            "Int32": TagDataType.INT32,
            "Digital": TagDataType.DIGITAL,
            "String": TagDataType.STRING,
        }
        return type_map.get(pi_type, TagDataType.FLOAT64)

    async def get_tag_definition(self, tag_name: str) -> Optional[TagDefinition]:
        """Get tag definition/metadata."""
        # Check cache first
        if tag_name in self._tag_cache:
            return self._tag_cache[tag_name]

        # Search for tag
        search_result = await self.search_tags(TagSearchRequest(
            search_pattern=tag_name,
            max_results=1
        ))

        if search_result.tags:
            tag = search_result.tags[0]
            self._tag_cache[tag_name] = tag
            return tag

        return None

    async def browse_tags(
        self,
        path: str = "*",
        max_results: int = 1000
    ) -> List[TagDefinition]:
        """Browse tags in a hierarchy."""
        search_result = await self.search_tags(TagSearchRequest(
            search_pattern=path,
            max_results=max_results
        ))
        return search_result.tags

    # =========================================================================
    # Time Series Data Operations
    # =========================================================================

    async def get_time_series(
        self,
        tag_name: str,
        start_time: datetime,
        end_time: datetime,
        mode: Optional[DataRetrievalMode] = None,
        interval_seconds: Optional[int] = None
    ) -> TimeSeriesData:
        """
        Get time series data for a single tag.

        Args:
            tag_name: Tag name to retrieve
            start_time: Start time
            end_time: End time
            mode: Retrieval mode (defaults to config default)
            interval_seconds: Interval for interpolated mode

        Returns:
            Time series data
        """
        mode = mode or self._historian_config.default_retrieval_mode
        interval_seconds = interval_seconds or self._historian_config.default_interval_seconds

        async def _do_query() -> TimeSeriesData:
            return await self._get_time_series_impl(
                tag_name, start_time, end_time, mode, interval_seconds
            )

        cache_key = self._generate_cache_key(
            "time_series",
            tag_name,
            start_time.isoformat(),
            end_time.isoformat(),
            mode.value,
            interval_seconds
        )

        return await self.execute_with_protection(
            _do_query,
            operation_name="get_time_series",
            use_cache=True,
            cache_key=cache_key
        )

    async def _get_time_series_impl(
        self,
        tag_name: str,
        start_time: datetime,
        end_time: datetime,
        mode: DataRetrievalMode,
        interval_seconds: int
    ) -> TimeSeriesData:
        """Implementation of time series retrieval."""
        query_start = time.time()

        # Build request
        endpoint, params, body = await self._adapter.build_time_series_request(
            tag_name, start_time, end_time, mode, interval_seconds
        )

        # Execute request
        if body:
            async with self._session.post(endpoint, params=params, json=body) as response:
                response.raise_for_status()
                response_data = await response.json()
        else:
            async with self._session.get(endpoint, params=params) as response:
                response.raise_for_status()
                response_data = await response.json()

        # Parse response
        values = self._adapter.parse_time_series_response(response_data, tag_name)

        # Calculate statistics
        numeric_values = [
            v.value for v in values
            if isinstance(v.value, (int, float)) and v.quality == TagQuality.GOOD
        ]

        min_val = min(numeric_values) if numeric_values else None
        max_val = max(numeric_values) if numeric_values else None
        avg_val = sum(numeric_values) / len(numeric_values) if numeric_values else None

        # Calculate quality metrics
        good_count = sum(1 for v in values if v.quality == TagQuality.GOOD)
        good_percent = (good_count / len(values) * 100) if values else 100.0
        bad_count = len(values) - good_count

        return TimeSeriesData(
            tag_name=tag_name,
            start_time=start_time,
            end_time=end_time,
            retrieval_mode=mode,
            interval_seconds=interval_seconds if mode == DataRetrievalMode.INTERPOLATED else None,
            values=values,
            point_count=len(values),
            min_value=min_val,
            max_value=max_val,
            avg_value=avg_val,
            good_quality_percent=good_percent,
            bad_quality_count=bad_count,
            query_duration_ms=(time.time() - query_start) * 1000
        )

    async def get_bulk_time_series(
        self,
        request: BulkTimeSeriesRequest
    ) -> BulkTimeSeriesResponse:
        """
        Get time series data for multiple tags efficiently.

        Args:
            request: Bulk request with tag names and time range

        Returns:
            Bulk response with data for all tags
        """
        query_start = time.time()

        results: Dict[str, TimeSeriesData] = {}
        successful_tags: List[str] = []
        failed_tags: List[str] = []
        errors: Dict[str, str] = {}

        # Process tags in parallel batches
        batch_size = self._get_bulk_batch_size()

        for i in range(0, len(request.tag_names), batch_size):
            batch = request.tag_names[i:i + batch_size]
            tasks = [
                self.get_time_series(
                    tag_name=tag,
                    start_time=request.start_time,
                    end_time=request.end_time,
                    mode=request.retrieval_mode,
                    interval_seconds=request.interval_seconds
                )
                for tag in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for tag, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    failed_tags.append(tag)
                    errors[tag] = str(result)
                else:
                    results[tag] = result
                    successful_tags.append(tag)

        # Calculate totals
        total_points = sum(r.point_count for r in results.values())

        # Overall quality
        quality_scores = [
            r.good_quality_percent / 100 for r in results.values()
        ]
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 1.0

        return BulkTimeSeriesResponse(
            request=request,
            results=results,
            successful_tags=successful_tags,
            failed_tags=failed_tags,
            errors=errors,
            total_tags=len(request.tag_names),
            total_points=total_points,
            query_duration_ms=(time.time() - query_start) * 1000,
            overall_quality_score=overall_quality
        )

    def _get_bulk_batch_size(self) -> int:
        """Get optimal batch size for bulk operations."""
        if self._historian_config.provider == HistorianProvider.OSISOFT_PI:
            return self._historian_config.pi_config.bulk_parallelism if self._historian_config.pi_config else 4
        elif self._historian_config.provider == HistorianProvider.HONEYWELL_PHD:
            return self._historian_config.phd_config.max_tags_per_request if self._historian_config.phd_config else 100
        return 10

    async def get_snapshot(self, tag_name: str) -> Optional[TagValue]:
        """Get current/snapshot value for a tag."""
        result = await self.get_time_series(
            tag_name=tag_name,
            start_time=datetime.utcnow() - timedelta(hours=1),
            end_time=datetime.utcnow(),
            mode=DataRetrievalMode.SNAPSHOT
        )

        if result.values:
            return result.values[-1]
        return None

    async def get_multiple_snapshots(
        self,
        tag_names: List[str]
    ) -> Dict[str, TagValue]:
        """Get current values for multiple tags."""
        result = await self.get_bulk_time_series(BulkTimeSeriesRequest(
            tag_names=tag_names,
            start_time=datetime.utcnow() - timedelta(hours=1),
            end_time=datetime.utcnow(),
            retrieval_mode=DataRetrievalMode.SNAPSHOT
        ))

        snapshots = {}
        for tag_name, data in result.results.items():
            if data.values:
                snapshots[tag_name] = data.values[-1]

        return snapshots

    # =========================================================================
    # Heat Exchanger Specific Operations
    # =========================================================================

    async def register_heat_exchanger_tags(
        self,
        tag_set: HeatExchangerTagSet
    ) -> None:
        """Register tag set for a heat exchanger."""
        self._heat_exchanger_tag_sets[tag_set.equipment_id] = tag_set
        tag_set.all_tags = tag_set.get_all_tags()
        self._logger.info(
            f"Registered {len(tag_set.all_tags)} tags for {tag_set.equipment_id}"
        )

    async def discover_heat_exchanger_tags(
        self,
        equipment_id: str
    ) -> Optional[HeatExchangerTagSet]:
        """Attempt to discover tags for a heat exchanger."""
        tag_set = HeatExchangerTagSet(equipment_id=equipment_id)

        # Search for tags containing equipment ID
        search_result = await self.search_tags(TagSearchRequest(
            search_pattern=f"*{equipment_id}*",
            max_results=100
        ))

        for tag_def in search_result.tags:
            tag_name_lower = tag_def.tag_name.lower()
            desc_lower = (tag_def.description or "").lower()

            # Hot inlet temperature
            if any(x in tag_name_lower for x in ["hot", "shell"]) and "inlet" in tag_name_lower and "temp" in tag_name_lower:
                tag_set.hot_inlet_temp_tag = tag_def.tag_name
            elif any(x in tag_name_lower for x in ["hot", "shell"]) and "outlet" in tag_name_lower and "temp" in tag_name_lower:
                tag_set.hot_outlet_temp_tag = tag_def.tag_name
            elif any(x in tag_name_lower for x in ["cold", "tube"]) and "inlet" in tag_name_lower and "temp" in tag_name_lower:
                tag_set.cold_inlet_temp_tag = tag_def.tag_name
            elif any(x in tag_name_lower for x in ["cold", "tube"]) and "outlet" in tag_name_lower and "temp" in tag_name_lower:
                tag_set.cold_outlet_temp_tag = tag_def.tag_name
            elif any(x in tag_name_lower for x in ["hot", "shell"]) and "flow" in tag_name_lower:
                tag_set.hot_flow_tag = tag_def.tag_name
            elif any(x in tag_name_lower for x in ["cold", "tube"]) and "flow" in tag_name_lower:
                tag_set.cold_flow_tag = tag_def.tag_name
            elif "duty" in tag_name_lower or "heat" in tag_name_lower:
                tag_set.duty_tag = tag_def.tag_name
            elif "ua" in tag_name_lower or "coeff" in tag_name_lower:
                tag_set.ua_tag = tag_def.tag_name
            elif "lmtd" in tag_name_lower:
                tag_set.lmtd_tag = tag_def.tag_name
            elif "foul" in tag_name_lower:
                tag_set.fouling_factor_tag = tag_def.tag_name

        tag_set.all_tags = tag_set.get_all_tags()

        if tag_set.all_tags:
            self._heat_exchanger_tag_sets[equipment_id] = tag_set
            return tag_set

        return None

    async def get_heat_exchanger_snapshot(
        self,
        equipment_id: str
    ) -> HeatExchangerSnapshot:
        """Get current snapshot data for a heat exchanger."""
        tag_set = self._heat_exchanger_tag_sets.get(equipment_id)

        if not tag_set:
            tag_set = await self.discover_heat_exchanger_tags(equipment_id)
            if not tag_set:
                raise ValidationError(f"No tags found for equipment {equipment_id}")

        # Get snapshots for all tags
        snapshots = await self.get_multiple_snapshots(tag_set.get_all_tags())

        # Build snapshot object
        snapshot = HeatExchangerSnapshot(
            equipment_id=equipment_id,
            timestamp=datetime.utcnow()
        )

        missing_tags = []
        bad_quality_tags = []

        # Map tag values to snapshot fields
        tag_field_map = {
            tag_set.hot_inlet_temp_tag: 'hot_inlet_temp',
            tag_set.hot_outlet_temp_tag: 'hot_outlet_temp',
            tag_set.hot_flow_tag: 'hot_flow',
            tag_set.cold_inlet_temp_tag: 'cold_inlet_temp',
            tag_set.cold_outlet_temp_tag: 'cold_outlet_temp',
            tag_set.cold_flow_tag: 'cold_flow',
            tag_set.duty_tag: 'duty',
            tag_set.ua_tag: 'ua_coefficient',
            tag_set.lmtd_tag: 'lmtd',
            tag_set.fouling_factor_tag: 'fouling_factor',
        }

        for tag_name, field_name in tag_field_map.items():
            if not tag_name:
                continue

            if tag_name in snapshots:
                tag_value = snapshots[tag_name]
                if tag_value.quality == TagQuality.GOOD:
                    setattr(snapshot, field_name, tag_value.value)
                else:
                    bad_quality_tags.append(tag_name)
            else:
                missing_tags.append(tag_name)

        # Calculate data quality score
        total_tags = len([t for t in tag_field_map.keys() if t])
        good_tags = total_tags - len(missing_tags) - len(bad_quality_tags)
        snapshot.data_quality_score = good_tags / total_tags if total_tags > 0 else 0

        snapshot.missing_tags = missing_tags
        snapshot.bad_quality_tags = bad_quality_tags

        return snapshot

    async def get_heat_exchanger_history(
        self,
        equipment_id: str,
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int = 60
    ) -> Dict[str, TimeSeriesData]:
        """Get historical data for a heat exchanger."""
        tag_set = self._heat_exchanger_tag_sets.get(equipment_id)

        if not tag_set:
            tag_set = await self.discover_heat_exchanger_tags(equipment_id)
            if not tag_set:
                raise ValidationError(f"No tags found for equipment {equipment_id}")

        result = await self.get_bulk_time_series(BulkTimeSeriesRequest(
            tag_names=tag_set.get_all_tags(),
            start_time=start_time,
            end_time=end_time,
            retrieval_mode=DataRetrievalMode.INTERPOLATED,
            interval_seconds=interval_seconds
        ))

        return result.results


# =============================================================================
# Factory Function
# =============================================================================


def create_process_historian_connector(
    provider: HistorianProvider,
    connector_name: str,
    pi_config: Optional[PIWebAPIConfig] = None,
    phd_config: Optional[PHDConfig] = None,
    ip21_config: Optional[IP21Config] = None,
    opcua_config: Optional[OPCUAConfig] = None,
    **kwargs
) -> ProcessHistorianConnector:
    """
    Factory function to create process historian connector.

    Args:
        provider: Historian provider
        connector_name: Connector name
        pi_config: PI Web API configuration
        phd_config: PHD configuration
        ip21_config: IP.21 configuration
        opcua_config: OPC-UA configuration
        **kwargs: Additional configuration options

    Returns:
        Configured ProcessHistorianConnector
    """
    config = ProcessHistorianConnectorConfig(
        connector_name=connector_name,
        connector_type=ConnectorType.PROCESS_HISTORIAN,
        provider=provider,
        pi_config=pi_config,
        phd_config=phd_config,
        ip21_config=ip21_config,
        opcua_config=opcua_config,
        **kwargs
    )

    return ProcessHistorianConnector(config)
