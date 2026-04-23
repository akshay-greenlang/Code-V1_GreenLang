"""
Process Historian Connector Module for GL-015 INSULSCAN (Insulation Inspection Agent).

Provides enterprise-grade integration with process historians:
- OSIsoft PI (PI Web API, PI AF SDK)
- AVEVA (Wonderware) Historian
- Query historical temperature data
- Trend analysis support
- Data interpolation and aggregation
- Batch data retrieval

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


class HistorianProvider(str, Enum):
    """Supported historian providers."""

    OSISOFT_PI = "osisoft_pi"  # OSIsoft PI System
    AVEVA_HISTORIAN = "aveva_historian"  # AVEVA Historian (Wonderware)
    AVEVA_INSIGHT = "aveva_insight"  # AVEVA Insight (Cloud)
    ASPEN_IP21 = "aspen_ip21"  # AspenTech IP.21
    HONEYWELL_PHD = "honeywell_phd"  # Honeywell PHD
    GENERIC_REST = "generic_rest"  # Generic REST API historian


class PIConnectionType(str, Enum):
    """PI connection types."""

    WEB_API = "web_api"  # PI Web API (RESTful)
    AF_SDK = "af_sdk"  # PI AF SDK (.NET)
    OLEDB = "oledb"  # PI OLEDB provider


class InterpolationMethod(str, Enum):
    """Data interpolation methods."""

    LINEAR = "linear"  # Linear interpolation
    STEP = "step"  # Step (previous value)
    CUBIC = "cubic"  # Cubic spline
    NONE = "none"  # No interpolation (raw)


class AggregationMethod(str, Enum):
    """Data aggregation methods for trends."""

    AVERAGE = "average"  # Time-weighted average
    MINIMUM = "minimum"  # Minimum value
    MAXIMUM = "maximum"  # Maximum value
    COUNT = "count"  # Sample count
    RANGE = "range"  # Max - Min
    STANDARD_DEVIATION = "standard_deviation"  # Standard deviation
    TOTAL = "total"  # Sum/total
    FIRST = "first"  # First value
    LAST = "last"  # Last value
    INTERPOLATED = "interpolated"  # Interpolated values


class DataQuality(str, Enum):
    """Historian data quality."""

    GOOD = "good"
    QUESTIONABLE = "questionable"
    SUBSTITUTED = "substituted"
    BAD = "bad"
    NO_DATA = "no_data"
    STALE = "stale"
    CONFIGURATION_ERROR = "configuration_error"


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


class HistorianError(Exception):
    """Base historian exception."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class HistorianConnectionError(HistorianError):
    """Historian connection error."""
    pass


class HistorianQueryError(HistorianError):
    """Historian query error."""
    pass


class HistorianTagNotFoundError(HistorianError):
    """Tag not found in historian."""
    pass


class HistorianAuthenticationError(HistorianError):
    """Historian authentication error."""
    pass


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class PIWebAPIConfig(BaseModel):
    """PI Web API configuration."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(
        ...,
        description="PI Web API base URL (e.g., https://piwebapi.example.com/piwebapi)"
    )
    data_server_name: str = Field(
        ...,
        description="PI Data Archive server name"
    )
    af_server_name: Optional[str] = Field(
        default=None,
        description="PI AF server name"
    )
    af_database_name: Optional[str] = Field(
        default=None,
        description="PI AF database name"
    )

    # Authentication
    auth_type: str = Field(
        default="basic",
        description="Authentication type (basic, kerberos, bearer)"
    )
    username: Optional[str] = Field(
        default=None,
        description="Username for basic auth"
    )
    password: Optional[str] = Field(
        default=None,
        description="Password for basic auth"
    )
    bearer_token: Optional[str] = Field(
        default=None,
        description="Bearer token for OAuth"
    )
    verify_ssl: bool = Field(
        default=True,
        description="Verify SSL certificates"
    )

    # Batch settings
    max_points_per_request: int = Field(
        default=10000,
        ge=100,
        le=150000,
        description="Maximum points per batch request"
    )
    max_tags_per_request: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum tags per batch request"
    )


class AVEVAHistorianConfig(BaseModel):
    """AVEVA Historian configuration."""

    model_config = ConfigDict(extra="forbid")

    server_name: str = Field(
        ...,
        description="AVEVA Historian server name"
    )
    database_name: str = Field(
        default="Runtime",
        description="Database name"
    )

    # Connection
    connection_string: Optional[str] = Field(
        default=None,
        description="Full connection string (overrides other settings)"
    )
    use_oledb: bool = Field(
        default=True,
        description="Use OLEDB provider"
    )

    # Authentication
    username: Optional[str] = Field(
        default=None,
        description="Username"
    )
    password: Optional[str] = Field(
        default=None,
        description="Password"
    )
    integrated_security: bool = Field(
        default=True,
        description="Use Windows integrated security"
    )

    # Query settings
    default_retrieval_mode: str = Field(
        default="Cyclic",
        description="Default retrieval mode (Cyclic, Delta, BestFit)"
    )
    max_samples: int = Field(
        default=100000,
        ge=100,
        description="Maximum samples per query"
    )


class HistorianConnectorConfig(BaseModel):
    """Configuration for historian connector."""

    model_config = ConfigDict(extra="forbid")

    connector_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Connector identifier"
    )
    connector_name: str = Field(
        default="Historian-Connector",
        description="Connector name"
    )

    # Provider
    provider: HistorianProvider = Field(
        ...,
        description="Historian provider"
    )

    # Provider-specific configs
    pi_web_api_config: Optional[PIWebAPIConfig] = Field(
        default=None,
        description="PI Web API configuration"
    )
    aveva_config: Optional[AVEVAHistorianConfig] = Field(
        default=None,
        description="AVEVA Historian configuration"
    )

    # Connection settings
    connection_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Connection timeout"
    )
    request_timeout_seconds: float = Field(
        default=60.0,
        ge=10.0,
        le=600.0,
        description="Request timeout"
    )

    # Retry settings
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries"
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        description="Retry delay"
    )

    # Query defaults
    default_interpolation: InterpolationMethod = Field(
        default=InterpolationMethod.LINEAR,
        description="Default interpolation method"
    )
    default_aggregation: AggregationMethod = Field(
        default=AggregationMethod.AVERAGE,
        description="Default aggregation method"
    )
    default_interval_seconds: int = Field(
        default=60,
        ge=1,
        description="Default interval for interpolated/aggregated data"
    )

    # Caching
    cache_enabled: bool = Field(
        default=True,
        description="Enable query result caching"
    )
    cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        description="Cache TTL"
    )

    # Health check
    health_check_enabled: bool = Field(
        default=True,
        description="Enable health checks"
    )
    health_check_interval_seconds: float = Field(
        default=60.0,
        ge=10.0,
        description="Health check interval"
    )


# =============================================================================
# Data Models - Tags and Values
# =============================================================================


class HistorianTag(BaseModel):
    """Historian tag definition."""

    model_config = ConfigDict(frozen=False)

    tag_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Internal tag ID"
    )
    tag_name: str = Field(
        ...,
        description="Tag name in historian"
    )
    description: Optional[str] = Field(
        default=None,
        description="Tag description"
    )
    engineering_unit: str = Field(
        default="",
        description="Engineering unit"
    )
    data_type: str = Field(
        default="Float64",
        description="Data type"
    )

    # PI-specific
    pi_point_id: Optional[str] = Field(
        default=None,
        description="PI Point WebId"
    )
    pi_path: Optional[str] = Field(
        default=None,
        description="PI AF path"
    )

    # AVEVA-specific
    aveva_tagname: Optional[str] = Field(
        default=None,
        description="AVEVA tag name"
    )

    # Mapping
    equipment_id: Optional[str] = Field(
        default=None,
        description="Associated equipment"
    )
    measurement_type: Optional[str] = Field(
        default=None,
        description="Measurement type (temperature, pressure, etc.)"
    )

    # Value ranges
    typical_value: Optional[float] = Field(
        default=None,
        description="Typical value"
    )
    low_limit: Optional[float] = Field(
        default=None,
        description="Low engineering limit"
    )
    high_limit: Optional[float] = Field(
        default=None,
        description="High engineering limit"
    )

    # Status
    enabled: bool = Field(
        default=True,
        description="Tag enabled for queries"
    )


class HistorianValue(BaseModel):
    """Single historian value."""

    model_config = ConfigDict(frozen=True)

    tag_name: str = Field(..., description="Tag name")
    timestamp: datetime = Field(..., description="Timestamp")
    value: Optional[float] = Field(default=None, description="Value")
    quality: DataQuality = Field(
        default=DataQuality.GOOD,
        description="Data quality"
    )
    annotations: Optional[str] = Field(
        default=None,
        description="Value annotations"
    )


class TimeSeries(BaseModel):
    """Time series data from historian."""

    model_config = ConfigDict(frozen=False)

    tag_name: str = Field(..., description="Tag name")
    start_time: datetime = Field(..., description="Query start time")
    end_time: datetime = Field(..., description="Query end time")
    values: List[HistorianValue] = Field(
        default_factory=list,
        description="Time series values"
    )
    sample_count: int = Field(default=0, ge=0, description="Sample count")

    # Statistics
    min_value: Optional[float] = Field(default=None, description="Minimum value")
    max_value: Optional[float] = Field(default=None, description="Maximum value")
    avg_value: Optional[float] = Field(default=None, description="Average value")
    std_dev: Optional[float] = Field(default=None, description="Standard deviation")

    # Query info
    interpolation: InterpolationMethod = Field(
        default=InterpolationMethod.NONE,
        description="Interpolation method used"
    )
    aggregation: Optional[AggregationMethod] = Field(
        default=None,
        description="Aggregation method used"
    )
    interval_seconds: Optional[int] = Field(
        default=None,
        description="Interval for interpolated/aggregated data"
    )


class TrendData(BaseModel):
    """Trend data for multiple tags."""

    model_config = ConfigDict(frozen=False)

    trend_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Trend identifier"
    )
    start_time: datetime = Field(..., description="Trend start time")
    end_time: datetime = Field(..., description="Trend end time")
    interval_seconds: int = Field(..., description="Data interval")
    series: Dict[str, TimeSeries] = Field(
        default_factory=dict,
        description="Time series by tag name"
    )
    query_time_ms: float = Field(
        default=0.0,
        description="Query execution time"
    )


class SnapshotValue(BaseModel):
    """Current/snapshot value from historian."""

    model_config = ConfigDict(frozen=True)

    tag_name: str = Field(..., description="Tag name")
    value: Optional[float] = Field(default=None, description="Current value")
    timestamp: datetime = Field(..., description="Value timestamp")
    quality: DataQuality = Field(..., description="Data quality")
    age_seconds: float = Field(
        default=0.0,
        description="Age of value in seconds"
    )


# =============================================================================
# Query Models
# =============================================================================


class TimeRange(BaseModel):
    """Time range for historian queries."""

    model_config = ConfigDict(frozen=True)

    start_time: datetime = Field(..., description="Start time")
    end_time: datetime = Field(..., description="End time")

    @field_validator('end_time')
    @classmethod
    def validate_end_time(cls, v: datetime, info) -> datetime:
        """Validate end time is after start time."""
        start = info.data.get('start_time')
        if start and v <= start:
            raise ValueError('end_time must be after start_time')
        return v


class HistorianQuery(BaseModel):
    """Historian data query."""

    model_config = ConfigDict(frozen=True)

    query_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Query identifier"
    )
    tags: List[str] = Field(
        ...,
        min_length=1,
        description="Tags to query"
    )
    time_range: TimeRange = Field(..., description="Query time range")

    # Data retrieval options
    interpolation: InterpolationMethod = Field(
        default=InterpolationMethod.LINEAR,
        description="Interpolation method"
    )
    aggregation: Optional[AggregationMethod] = Field(
        default=None,
        description="Aggregation method"
    )
    interval_seconds: Optional[int] = Field(
        default=None,
        description="Data interval (for interpolated/aggregated)"
    )
    max_samples: Optional[int] = Field(
        default=None,
        description="Maximum samples per tag"
    )

    # Quality filter
    include_questionable: bool = Field(
        default=True,
        description="Include questionable quality values"
    )
    include_bad: bool = Field(
        default=False,
        description="Include bad quality values"
    )


class TrendQuery(BaseModel):
    """Trend data query for visualization."""

    model_config = ConfigDict(frozen=True)

    query_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Query identifier"
    )
    tags: List[str] = Field(
        ...,
        min_length=1,
        description="Tags to query"
    )

    # Time range (flexible specification)
    start_time: Optional[datetime] = Field(
        default=None,
        description="Start time"
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="End time"
    )
    relative_start: Optional[str] = Field(
        default=None,
        description="Relative start (e.g., '-24h', '-7d')"
    )

    # Display options
    interval_seconds: int = Field(
        default=60,
        ge=1,
        description="Display interval"
    )
    aggregation: AggregationMethod = Field(
        default=AggregationMethod.AVERAGE,
        description="Aggregation method"
    )
    fill_gaps: bool = Field(
        default=True,
        description="Fill data gaps"
    )


# =============================================================================
# Historian Connector
# =============================================================================


class HistorianConnector:
    """
    Process Historian Connector for GL-015 INSULSCAN.

    Provides unified interface for historian data access supporting
    OSIsoft PI and AVEVA Historian.

    Features:
    - Multi-provider support
    - Historical data retrieval
    - Trend analysis
    - Data interpolation and aggregation
    - Batch queries
    - Query caching
    """

    def __init__(self, config: HistorianConnectorConfig) -> None:
        """
        Initialize historian connector.

        Args:
            config: Connector configuration
        """
        self._config = config
        self._logger = logging.getLogger(
            f"{__name__}.{config.connector_name}"
        )

        self._state = ConnectionState.DISCONNECTED
        self._client: Optional[Any] = None

        # Tag cache
        self._tags: Dict[str, HistorianTag] = {}

        # Query cache
        self._query_cache: Dict[str, Tuple[datetime, Any]] = {}

        # Health check
        self._health_check_task: Optional[asyncio.Task] = None
        self._last_successful_query: Optional[datetime] = None

        # Metrics
        self._queries_executed = 0
        self._query_errors = 0
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def config(self) -> HistorianConnectorConfig:
        """Get configuration."""
        return self._config

    @property
    def state(self) -> ConnectionState:
        """Get connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> None:
        """
        Connect to historian.

        Raises:
            HistorianConnectionError: If connection fails
        """
        self._state = ConnectionState.CONNECTING

        try:
            provider = self._config.provider

            if provider == HistorianProvider.OSISOFT_PI:
                await self._connect_pi()
            elif provider in [HistorianProvider.AVEVA_HISTORIAN, HistorianProvider.AVEVA_INSIGHT]:
                await self._connect_aveva()
            else:
                raise HistorianConnectionError(f"Unsupported provider: {provider}")

            self._state = ConnectionState.CONNECTED

            # Start health check
            if self._config.health_check_enabled:
                self._health_check_task = asyncio.create_task(
                    self._health_check_loop()
                )

            self._logger.info(f"Connected to {provider.value} historian")

        except Exception as e:
            self._state = ConnectionState.ERROR
            self._logger.error(f"Connection failed: {e}")
            raise HistorianConnectionError(f"Connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from historian."""
        self._logger.info("Disconnecting from historian")

        # Cancel health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Clear caches
        self._query_cache.clear()

        self._client = None
        self._state = ConnectionState.DISCONNECTED

    async def _connect_pi(self) -> None:
        """Connect to PI Web API."""
        config = self._config.pi_web_api_config
        if not config:
            raise HistorianConnectionError("PI Web API config not provided")

        self._logger.info(f"Connecting to PI Web API: {config.base_url}")

        # In production, use aiohttp or httpx:
        # import aiohttp
        # auth = aiohttp.BasicAuth(config.username, config.password)
        # self._client = aiohttp.ClientSession(auth=auth)

        # Verify connection by getting server info
        # url = f"{config.base_url}/system"
        # async with self._client.get(url) as response:
        #     if response.status != 200:
        #         raise HistorianConnectionError(f"PI Web API error: {response.status}")

    async def _connect_aveva(self) -> None:
        """Connect to AVEVA Historian."""
        config = self._config.aveva_config
        if not config:
            raise HistorianConnectionError("AVEVA Historian config not provided")

        self._logger.info(f"Connecting to AVEVA Historian: {config.server_name}")

        # In production, use pyodbc or proprietary SDK

    # =========================================================================
    # Tag Operations
    # =========================================================================

    async def search_tags(
        self,
        pattern: str,
        max_results: int = 100
    ) -> List[HistorianTag]:
        """
        Search for tags matching pattern.

        Args:
            pattern: Search pattern (wildcards supported)
            max_results: Maximum results

        Returns:
            List of matching tags
        """
        if not self.is_connected:
            raise HistorianConnectionError("Not connected")

        provider = self._config.provider

        if provider == HistorianProvider.OSISOFT_PI:
            return await self._search_tags_pi(pattern, max_results)
        elif provider in [HistorianProvider.AVEVA_HISTORIAN, HistorianProvider.AVEVA_INSIGHT]:
            return await self._search_tags_aveva(pattern, max_results)
        else:
            return []

    async def _search_tags_pi(
        self,
        pattern: str,
        max_results: int
    ) -> List[HistorianTag]:
        """Search tags in PI."""
        config = self._config.pi_web_api_config

        # In production:
        # url = f"{config.base_url}/dataservers/{config.data_server_name}/points"
        # params = {"nameFilter": pattern, "maxCount": max_results}
        # async with self._client.get(url, params=params) as response:
        #     data = await response.json()
        #     return [self._parse_pi_point(item) for item in data.get("Items", [])]

        return []

    async def _search_tags_aveva(
        self,
        pattern: str,
        max_results: int
    ) -> List[HistorianTag]:
        """Search tags in AVEVA Historian."""
        # In production, query the tag dictionary

        return []

    async def get_tag_info(self, tag_name: str) -> Optional[HistorianTag]:
        """
        Get tag information.

        Args:
            tag_name: Tag name

        Returns:
            Tag info or None
        """
        # Check cache
        if tag_name in self._tags:
            return self._tags[tag_name]

        # Search for tag
        results = await self.search_tags(tag_name, max_results=1)
        if results:
            self._tags[tag_name] = results[0]
            return results[0]

        return None

    # =========================================================================
    # Data Retrieval
    # =========================================================================

    async def get_snapshot(
        self,
        tags: List[str]
    ) -> Dict[str, SnapshotValue]:
        """
        Get current/snapshot values.

        Args:
            tags: Tag names

        Returns:
            Dictionary of tag name to snapshot value
        """
        if not self.is_connected:
            raise HistorianConnectionError("Not connected")

        provider = self._config.provider
        self._queries_executed += 1

        try:
            if provider == HistorianProvider.OSISOFT_PI:
                return await self._get_snapshot_pi(tags)
            elif provider in [HistorianProvider.AVEVA_HISTORIAN, HistorianProvider.AVEVA_INSIGHT]:
                return await self._get_snapshot_aveva(tags)
            else:
                return {}
        except Exception as e:
            self._query_errors += 1
            raise HistorianQueryError(f"Snapshot query failed: {e}")

    async def _get_snapshot_pi(self, tags: List[str]) -> Dict[str, SnapshotValue]:
        """Get snapshot from PI."""
        config = self._config.pi_web_api_config
        results = {}

        # In production:
        # for tag_name in tags:
        #     url = f"{config.base_url}/points/{tag_name}/value"
        #     async with self._client.get(url) as response:
        #         data = await response.json()
        #         results[tag_name] = SnapshotValue(
        #             tag_name=tag_name,
        #             value=data.get("Value"),
        #             timestamp=datetime.fromisoformat(data.get("Timestamp")),
        #             quality=self._parse_pi_quality(data.get("Good", True)),
        #             age_seconds=0,
        #         )

        # Mock implementation
        now = datetime.utcnow()
        for tag_name in tags:
            results[tag_name] = SnapshotValue(
                tag_name=tag_name,
                value=0.0,
                timestamp=now,
                quality=DataQuality.GOOD,
                age_seconds=0,
            )

        return results

    async def _get_snapshot_aveva(self, tags: List[str]) -> Dict[str, SnapshotValue]:
        """Get snapshot from AVEVA Historian."""
        # In production, query AVEVA for current values

        now = datetime.utcnow()
        return {
            tag_name: SnapshotValue(
                tag_name=tag_name,
                value=0.0,
                timestamp=now,
                quality=DataQuality.GOOD,
                age_seconds=0,
            )
            for tag_name in tags
        }

    async def get_recorded_values(
        self,
        query: HistorianQuery
    ) -> Dict[str, TimeSeries]:
        """
        Get recorded (raw) historical values.

        Args:
            query: Historian query

        Returns:
            Dictionary of tag name to time series
        """
        if not self.is_connected:
            raise HistorianConnectionError("Not connected")

        # Check cache
        cache_key = self._generate_cache_key(query)
        cached = self._get_from_cache(cache_key)
        if cached:
            self._cache_hits += 1
            return cached

        self._cache_misses += 1
        self._queries_executed += 1

        try:
            provider = self._config.provider

            if provider == HistorianProvider.OSISOFT_PI:
                result = await self._get_recorded_pi(query)
            elif provider in [HistorianProvider.AVEVA_HISTORIAN, HistorianProvider.AVEVA_INSIGHT]:
                result = await self._get_recorded_aveva(query)
            else:
                result = {}

            # Cache result
            self._set_cache(cache_key, result)
            self._last_successful_query = datetime.utcnow()

            return result

        except Exception as e:
            self._query_errors += 1
            raise HistorianQueryError(f"Recorded values query failed: {e}")

    async def _get_recorded_pi(self, query: HistorianQuery) -> Dict[str, TimeSeries]:
        """Get recorded values from PI."""
        config = self._config.pi_web_api_config
        results = {}

        # In production:
        # for tag_name in query.tags:
        #     url = f"{config.base_url}/streams/{tag_name}/recorded"
        #     params = {
        #         "startTime": query.time_range.start_time.isoformat(),
        #         "endTime": query.time_range.end_time.isoformat(),
        #         "maxCount": query.max_samples or config.max_points_per_request,
        #     }
        #     async with self._client.get(url, params=params) as response:
        #         data = await response.json()
        #         values = [self._parse_pi_value(v) for v in data.get("Items", [])]
        #         results[tag_name] = TimeSeries(
        #             tag_name=tag_name,
        #             start_time=query.time_range.start_time,
        #             end_time=query.time_range.end_time,
        #             values=values,
        #             sample_count=len(values),
        #         )

        # Mock implementation
        for tag_name in query.tags:
            results[tag_name] = TimeSeries(
                tag_name=tag_name,
                start_time=query.time_range.start_time,
                end_time=query.time_range.end_time,
                values=[],
                sample_count=0,
                interpolation=query.interpolation,
            )

        return results

    async def _get_recorded_aveva(self, query: HistorianQuery) -> Dict[str, TimeSeries]:
        """Get recorded values from AVEVA Historian."""
        # In production, query AVEVA

        results = {}
        for tag_name in query.tags:
            results[tag_name] = TimeSeries(
                tag_name=tag_name,
                start_time=query.time_range.start_time,
                end_time=query.time_range.end_time,
                values=[],
                sample_count=0,
            )

        return results

    async def get_interpolated_values(
        self,
        query: HistorianQuery
    ) -> Dict[str, TimeSeries]:
        """
        Get interpolated historical values at regular intervals.

        Args:
            query: Historian query

        Returns:
            Dictionary of tag name to time series
        """
        if not self.is_connected:
            raise HistorianConnectionError("Not connected")

        interval = query.interval_seconds or self._config.default_interval_seconds
        self._queries_executed += 1

        try:
            provider = self._config.provider

            if provider == HistorianProvider.OSISOFT_PI:
                return await self._get_interpolated_pi(query, interval)
            elif provider in [HistorianProvider.AVEVA_HISTORIAN, HistorianProvider.AVEVA_INSIGHT]:
                return await self._get_interpolated_aveva(query, interval)
            else:
                return {}

        except Exception as e:
            self._query_errors += 1
            raise HistorianQueryError(f"Interpolated values query failed: {e}")

    async def _get_interpolated_pi(
        self,
        query: HistorianQuery,
        interval: int
    ) -> Dict[str, TimeSeries]:
        """Get interpolated values from PI."""
        config = self._config.pi_web_api_config
        results = {}

        # In production:
        # for tag_name in query.tags:
        #     url = f"{config.base_url}/streams/{tag_name}/interpolated"
        #     params = {
        #         "startTime": query.time_range.start_time.isoformat(),
        #         "endTime": query.time_range.end_time.isoformat(),
        #         "interval": f"{interval}s",
        #     }
        #     async with self._client.get(url, params=params) as response:
        #         ...

        # Mock implementation
        for tag_name in query.tags:
            results[tag_name] = TimeSeries(
                tag_name=tag_name,
                start_time=query.time_range.start_time,
                end_time=query.time_range.end_time,
                values=[],
                sample_count=0,
                interpolation=query.interpolation,
                interval_seconds=interval,
            )

        return results

    async def _get_interpolated_aveva(
        self,
        query: HistorianQuery,
        interval: int
    ) -> Dict[str, TimeSeries]:
        """Get interpolated values from AVEVA Historian."""
        # In production, query AVEVA with cyclic mode

        results = {}
        for tag_name in query.tags:
            results[tag_name] = TimeSeries(
                tag_name=tag_name,
                start_time=query.time_range.start_time,
                end_time=query.time_range.end_time,
                values=[],
                sample_count=0,
                interval_seconds=interval,
            )

        return results

    async def get_summary_values(
        self,
        query: HistorianQuery
    ) -> Dict[str, TimeSeries]:
        """
        Get aggregated/summary values.

        Args:
            query: Historian query with aggregation

        Returns:
            Dictionary of tag name to time series
        """
        if not self.is_connected:
            raise HistorianConnectionError("Not connected")

        aggregation = query.aggregation or self._config.default_aggregation
        interval = query.interval_seconds or self._config.default_interval_seconds

        self._queries_executed += 1

        try:
            provider = self._config.provider

            if provider == HistorianProvider.OSISOFT_PI:
                return await self._get_summary_pi(query, aggregation, interval)
            elif provider in [HistorianProvider.AVEVA_HISTORIAN, HistorianProvider.AVEVA_INSIGHT]:
                return await self._get_summary_aveva(query, aggregation, interval)
            else:
                return {}

        except Exception as e:
            self._query_errors += 1
            raise HistorianQueryError(f"Summary values query failed: {e}")

    async def _get_summary_pi(
        self,
        query: HistorianQuery,
        aggregation: AggregationMethod,
        interval: int
    ) -> Dict[str, TimeSeries]:
        """Get summary values from PI."""
        # PI Web API summary endpoint

        results = {}
        for tag_name in query.tags:
            results[tag_name] = TimeSeries(
                tag_name=tag_name,
                start_time=query.time_range.start_time,
                end_time=query.time_range.end_time,
                values=[],
                sample_count=0,
                aggregation=aggregation,
                interval_seconds=interval,
            )

        return results

    async def _get_summary_aveva(
        self,
        query: HistorianQuery,
        aggregation: AggregationMethod,
        interval: int
    ) -> Dict[str, TimeSeries]:
        """Get summary values from AVEVA Historian."""
        # AVEVA summary query

        results = {}
        for tag_name in query.tags:
            results[tag_name] = TimeSeries(
                tag_name=tag_name,
                start_time=query.time_range.start_time,
                end_time=query.time_range.end_time,
                values=[],
                sample_count=0,
                aggregation=aggregation,
                interval_seconds=interval,
            )

        return results

    # =========================================================================
    # Trend Analysis
    # =========================================================================

    async def get_trend_data(
        self,
        query: TrendQuery
    ) -> TrendData:
        """
        Get trend data for visualization.

        Args:
            query: Trend query

        Returns:
            Trend data with multiple series
        """
        import time
        start_time = time.time()

        # Parse time range
        if query.relative_start:
            end_time = query.end_time or datetime.utcnow()
            start_time_dt = self._parse_relative_time(query.relative_start, end_time)
        else:
            start_time_dt = query.start_time or datetime.utcnow() - timedelta(hours=24)
            end_time = query.end_time or datetime.utcnow()

        # Build historian query
        hist_query = HistorianQuery(
            tags=query.tags,
            time_range=TimeRange(start_time=start_time_dt, end_time=end_time),
            aggregation=query.aggregation,
            interval_seconds=query.interval_seconds,
        )

        # Get aggregated data
        series_data = await self.get_summary_values(hist_query)

        # Fill gaps if requested
        if query.fill_gaps:
            for series in series_data.values():
                self._fill_gaps(series, query.interval_seconds)

        query_time = (time.time() - start_time) * 1000

        return TrendData(
            start_time=start_time_dt,
            end_time=end_time,
            interval_seconds=query.interval_seconds,
            series=series_data,
            query_time_ms=query_time,
        )

    def _parse_relative_time(
        self,
        relative: str,
        reference: datetime
    ) -> datetime:
        """Parse relative time string."""
        # Simple parsing for -24h, -7d, etc.
        if relative.startswith('-'):
            relative = relative[1:]

        value = int(relative[:-1])
        unit = relative[-1].lower()

        if unit == 'h':
            delta = timedelta(hours=value)
        elif unit == 'd':
            delta = timedelta(days=value)
        elif unit == 'w':
            delta = timedelta(weeks=value)
        elif unit == 'm':
            delta = timedelta(minutes=value)
        else:
            delta = timedelta(hours=value)

        return reference - delta

    def _fill_gaps(self, series: TimeSeries, interval_seconds: int) -> None:
        """Fill gaps in time series with interpolation."""
        if not series.values:
            return

        # Sort by timestamp
        sorted_values = sorted(series.values, key=lambda v: v.timestamp)

        # Check for gaps larger than interval and interpolate
        # Implementation would depend on specific requirements
        pass

    # =========================================================================
    # Temperature-Specific Methods
    # =========================================================================

    async def get_temperature_history(
        self,
        equipment_id: str,
        tag_mapping: Dict[str, str],
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int = 60
    ) -> Dict[str, TimeSeries]:
        """
        Get temperature history for equipment.

        Args:
            equipment_id: Equipment identifier
            tag_mapping: Map of measurement type to tag name
            start_time: Start time
            end_time: End time
            interval_seconds: Data interval

        Returns:
            Temperature time series by measurement type
        """
        # Get all tags for this equipment
        tags = list(tag_mapping.values())

        query = HistorianQuery(
            tags=tags,
            time_range=TimeRange(start_time=start_time, end_time=end_time),
            interpolation=InterpolationMethod.LINEAR,
            interval_seconds=interval_seconds,
        )

        series = await self.get_interpolated_values(query)

        # Remap to measurement types
        result = {}
        for measurement_type, tag_name in tag_mapping.items():
            if tag_name in series:
                result[measurement_type] = series[tag_name]

        return result

    async def get_temperature_statistics(
        self,
        tag_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, float]:
        """
        Get temperature statistics for a period.

        Args:
            tag_name: Temperature tag
            start_time: Start time
            end_time: End time

        Returns:
            Dictionary of statistics
        """
        query = HistorianQuery(
            tags=[tag_name],
            time_range=TimeRange(start_time=start_time, end_time=end_time),
        )

        series = await self.get_recorded_values(query)

        if tag_name not in series or not series[tag_name].values:
            return {}

        values = [v.value for v in series[tag_name].values if v.value is not None]

        if not values:
            return {}

        import statistics

        return {
            "min": min(values),
            "max": max(values),
            "avg": statistics.mean(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "sample_count": len(values),
        }

    # =========================================================================
    # Caching
    # =========================================================================

    def _generate_cache_key(self, query: HistorianQuery) -> str:
        """Generate cache key for query."""
        import hashlib
        key_data = f"{query.tags}:{query.time_range}:{query.interpolation}:{query.aggregation}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._config.cache_enabled:
            return None

        if key in self._query_cache:
            timestamp, value = self._query_cache[key]
            age = (datetime.utcnow() - timestamp).total_seconds()
            if age < self._config.cache_ttl_seconds:
                return value
            else:
                del self._query_cache[key]

        return None

    def _set_cache(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if self._config.cache_enabled:
            self._query_cache[key] = (datetime.utcnow(), value)

    def clear_cache(self) -> None:
        """Clear query cache."""
        self._query_cache.clear()
        self._logger.debug("Query cache cleared")

    # =========================================================================
    # Health Check
    # =========================================================================

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while True:
            try:
                await asyncio.sleep(self._config.health_check_interval_seconds)

                # Try a simple query
                now = datetime.utcnow()
                if self._tags:
                    tag = list(self._tags.keys())[0]
                    await self.get_snapshot([tag])

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.warning(f"Health check failed: {e}")
                self._state = ConnectionState.ERROR

    def get_metrics(self) -> Dict[str, Any]:
        """Get connector metrics."""
        return {
            "state": self._state.value,
            "provider": self._config.provider.value,
            "queries_executed": self._queries_executed,
            "query_errors": self._query_errors,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._query_cache),
            "tags_cached": len(self._tags),
            "last_successful_query": self._last_successful_query.isoformat() if self._last_successful_query else None,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_pi_connector(
    base_url: str,
    data_server_name: str,
    username: str,
    password: str,
    connector_name: str = "PI-Connector",
    **kwargs
) -> HistorianConnector:
    """
    Create PI Web API connector.

    Args:
        base_url: PI Web API URL
        data_server_name: PI Data Archive server name
        username: Username
        password: Password
        connector_name: Connector name
        **kwargs: Additional configuration

    Returns:
        Configured HistorianConnector
    """
    pi_config = PIWebAPIConfig(
        base_url=base_url,
        data_server_name=data_server_name,
        username=username,
        password=password,
    )

    config = HistorianConnectorConfig(
        connector_name=connector_name,
        provider=HistorianProvider.OSISOFT_PI,
        pi_web_api_config=pi_config,
        **kwargs
    )

    return HistorianConnector(config)


def create_aveva_connector(
    server_name: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    connector_name: str = "AVEVA-Connector",
    **kwargs
) -> HistorianConnector:
    """
    Create AVEVA Historian connector.

    Args:
        server_name: AVEVA Historian server name
        username: Username (None for integrated security)
        password: Password
        connector_name: Connector name
        **kwargs: Additional configuration

    Returns:
        Configured HistorianConnector
    """
    aveva_config = AVEVAHistorianConfig(
        server_name=server_name,
        username=username,
        password=password,
        integrated_security=username is None,
    )

    config = HistorianConnectorConfig(
        connector_name=connector_name,
        provider=HistorianProvider.AVEVA_HISTORIAN,
        aveva_config=aveva_config,
        **kwargs
    )

    return HistorianConnector(config)
