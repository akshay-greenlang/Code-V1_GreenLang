"""
GL-020 ECONOPULSE - Process Historian Integration Module

Enterprise-grade process historian connectors providing:
- OSIsoft PI historian integration
- AspenTech InfoPlus.21 historian integration
- Wonderware historian integration
- Time-series data storage and retrieval
- Compressed data handling
- Calculated tag support

Thread-safe with connection pooling and circuit breaker pattern.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class AggregateType(Enum):
    """Time-series data aggregation types."""
    RAW = "raw"
    INTERPOLATED = "interpolated"
    AVERAGE = "average"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    TOTAL = "total"
    COUNT = "count"
    RANGE = "range"
    STDEV = "stdev"
    VARIANCE = "variance"
    TIME_WEIGHTED_AVERAGE = "time_weighted_average"


class CompressionMode(Enum):
    """Data compression modes."""
    NONE = "none"
    SWINGING_DOOR = "swinging_door"
    EXCEPTION_DEVIATION = "exception_deviation"
    BOXCAR = "boxcar"


class TagDataType(Enum):
    """Tag data types."""
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    STRING = "string"
    DIGITAL = "digital"
    BLOB = "blob"


class TagQuality(Enum):
    """Tag value quality codes."""
    GOOD = 192
    UNCERTAIN = 64
    BAD = 0
    SUBSTITUTED = 24
    ANNOTATED = 200


class ConnectionStatus(Enum):
    """Connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TimeSeriesPoint:
    """Single point in a time series."""
    timestamp: datetime
    value: Any
    quality: TagQuality = TagQuality.GOOD
    annotated: bool = False
    annotation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "quality": self.quality.name,
            "annotated": self.annotated,
            "annotation": self.annotation,
        }


@dataclass
class TimeSeriesData:
    """Time series data container."""
    tag_name: str
    points: List[TimeSeriesPoint] = field(default_factory=list)
    data_type: TagDataType = TagDataType.FLOAT64
    unit: str = ""
    description: str = ""

    def __len__(self) -> int:
        return len(self.points)

    def timestamps(self) -> List[datetime]:
        """Get list of timestamps."""
        return [p.timestamp for p in self.points]

    def values(self) -> List[Any]:
        """Get list of values."""
        return [p.value for p in self.points]

    def qualities(self) -> List[TagQuality]:
        """Get list of quality codes."""
        return [p.quality for p in self.points]

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame({
            "timestamp": self.timestamps(),
            "value": self.values(),
            "quality": [q.name for q in self.qualities()],
        })

    def filter_by_quality(self, min_quality: TagQuality = TagQuality.UNCERTAIN) -> "TimeSeriesData":
        """Filter points by minimum quality."""
        filtered_points = [
            p for p in self.points
            if p.quality.value >= min_quality.value
        ]
        return TimeSeriesData(
            tag_name=self.tag_name,
            points=filtered_points,
            data_type=self.data_type,
            unit=self.unit,
            description=self.description,
        )

    def resample(
        self,
        interval_seconds: int,
        aggregate: AggregateType = AggregateType.AVERAGE,
    ) -> "TimeSeriesData":
        """Resample time series to regular intervals."""
        if not self.points:
            return TimeSeriesData(tag_name=self.tag_name)

        resampled_points = []
        start_time = self.points[0].timestamp
        end_time = self.points[-1].timestamp

        current_time = start_time
        while current_time <= end_time:
            next_time = current_time + timedelta(seconds=interval_seconds)

            # Get points in interval
            interval_points = [
                p for p in self.points
                if current_time <= p.timestamp < next_time
            ]

            if interval_points:
                values = [p.value for p in interval_points if isinstance(p.value, (int, float))]

                if values:
                    if aggregate == AggregateType.AVERAGE:
                        agg_value = sum(values) / len(values)
                    elif aggregate == AggregateType.MINIMUM:
                        agg_value = min(values)
                    elif aggregate == AggregateType.MAXIMUM:
                        agg_value = max(values)
                    elif aggregate == AggregateType.TOTAL:
                        agg_value = sum(values)
                    elif aggregate == AggregateType.COUNT:
                        agg_value = len(values)
                    else:
                        agg_value = values[0]

                    resampled_points.append(TimeSeriesPoint(
                        timestamp=current_time,
                        value=agg_value,
                        quality=TagQuality.GOOD,
                    ))

            current_time = next_time

        return TimeSeriesData(
            tag_name=self.tag_name,
            points=resampled_points,
            data_type=self.data_type,
            unit=self.unit,
            description=self.description,
        )


@dataclass
class CompressedDataConfig:
    """Data compression configuration."""
    mode: CompressionMode = CompressionMode.SWINGING_DOOR
    exception_deviation: float = 0.5  # Percent of span
    compression_deviation: float = 1.0  # Percent of span
    compression_timeout_seconds: float = 3600.0
    minimum_points_per_day: int = 100


@dataclass
class CalculatedTag:
    """Calculated/virtual tag definition."""
    tag_name: str
    expression: str  # PI or AF expression
    source_tags: List[str] = field(default_factory=list)
    unit: str = ""
    description: str = ""
    evaluation_interval_seconds: float = 60.0
    enabled: bool = True

    def validate_expression(self) -> Tuple[bool, str]:
        """Validate the calculation expression."""
        # Basic syntax validation
        if not self.expression:
            return False, "Expression is empty"

        # Check for required source tags
        for tag in self.source_tags:
            if tag not in self.expression:
                return False, f"Source tag {tag} not found in expression"

        return True, "Expression is valid"


@dataclass
class HistorianConfig:
    """Historian connection configuration."""
    name: str
    host: str
    port: int
    database: str = ""
    username: Optional[str] = None
    password: Optional[str] = None  # Retrieved from vault
    use_ssl: bool = True
    connection_timeout_seconds: float = 30.0
    read_timeout_seconds: float = 60.0
    max_connections: int = 5
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class TagSearchResult:
    """Result of tag search operation."""
    tag_name: str
    description: str
    data_type: TagDataType
    unit: str
    created_date: Optional[datetime] = None
    point_class: str = ""
    source: str = ""


@dataclass
class WriteResult:
    """Result of data write operation."""
    tag_name: str
    points_written: int
    points_failed: int
    success: bool
    error_message: Optional[str] = None


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._last_failure_time:
                    elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self._state = CircuitBreakerState.HALF_OPEN
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN

    def is_available(self) -> bool:
        return self.state != CircuitBreakerState.OPEN


# =============================================================================
# Abstract Base Class
# =============================================================================

class HistorianConnectorBase(ABC):
    """
    Abstract base class for process historian connectors.

    Provides common functionality for all historian types including:
    - Connection management with circuit breaker
    - Connection pooling
    - Retry logic
    - Thread-safe concurrent access
    """

    def __init__(
        self,
        config: HistorianConfig,
        vault_client=None,
    ):
        """
        Initialize historian connector.

        Args:
            config: Historian connection configuration
            vault_client: Optional vault client for credential retrieval
        """
        self.config = config
        self.vault_client = vault_client

        # Retrieve credentials from vault if available
        if vault_client and config.password is None:
            config.password = vault_client.get_secret(f"historian_{config.name}_password")

        self._connected = False
        self._lock = threading.RLock()
        self._status = ConnectionStatus.DISCONNECTED

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
        )

        # Connection pool
        self._connection_pool: List[Any] = []
        self._pool_lock = threading.Lock()

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=config.max_connections)

        # Tag cache
        self._tag_cache: Dict[str, TagSearchResult] = {}

        logger.info(f"Initialized historian connector: {config.name} at {config.host}")

    @property
    def is_connected(self) -> bool:
        """Check if connector is connected."""
        with self._lock:
            return self._connected

    @property
    def status(self) -> ConnectionStatus:
        """Get connection status."""
        with self._lock:
            return self._status

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to historian.

        Returns:
            True if connection successful.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from historian."""
        pass

    @abstractmethod
    async def read_raw(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        max_points: int = 10000,
    ) -> Dict[str, TimeSeriesData]:
        """
        Read raw archived data.

        Args:
            tag_names: List of tag names to read
            start_time: Start of time range
            end_time: End of time range
            max_points: Maximum points per tag

        Returns:
            Dictionary mapping tag names to TimeSeriesData
        """
        pass

    @abstractmethod
    async def read_interpolated(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int = 60,
    ) -> Dict[str, TimeSeriesData]:
        """
        Read interpolated data at regular intervals.

        Args:
            tag_names: List of tag names
            start_time: Start of time range
            end_time: End of time range
            interval_seconds: Interpolation interval

        Returns:
            Dictionary mapping tag names to TimeSeriesData
        """
        pass

    @abstractmethod
    async def read_aggregated(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int,
        aggregate_type: AggregateType,
    ) -> Dict[str, TimeSeriesData]:
        """
        Read aggregated data.

        Args:
            tag_names: List of tag names
            start_time: Start of time range
            end_time: End of time range
            interval_seconds: Aggregation interval
            aggregate_type: Type of aggregation

        Returns:
            Dictionary mapping tag names to TimeSeriesData
        """
        pass

    @abstractmethod
    async def write_values(
        self,
        tag_name: str,
        values: List[TimeSeriesPoint],
    ) -> WriteResult:
        """
        Write values to historian.

        Args:
            tag_name: Tag to write to
            values: List of points to write

        Returns:
            WriteResult with status
        """
        pass

    @abstractmethod
    async def search_tags(
        self,
        pattern: str,
        max_results: int = 1000,
    ) -> List[TagSearchResult]:
        """
        Search for tags matching pattern.

        Args:
            pattern: Search pattern (supports wildcards)
            max_results: Maximum results to return

        Returns:
            List of matching tags
        """
        pass

    async def read_snapshot(
        self,
        tag_names: List[str],
    ) -> Dict[str, TimeSeriesPoint]:
        """
        Read current snapshot values.

        Args:
            tag_names: List of tag names

        Returns:
            Dictionary mapping tag names to current values
        """
        now = datetime.now()
        data = await self.read_raw(
            tag_names=tag_names,
            start_time=now - timedelta(minutes=5),
            end_time=now,
            max_points=1,
        )

        result = {}
        for tag_name, ts_data in data.items():
            if ts_data.points:
                result[tag_name] = ts_data.points[-1]

        return result

    async def close(self) -> None:
        """Clean up resources."""
        await self.disconnect()
        self._executor.shutdown(wait=True)


# =============================================================================
# OSIsoft PI Connector
# =============================================================================

class OSIsoftPIConnector(HistorianConnectorBase):
    """
    OSIsoft PI Historian Connector.

    Provides access to OSIsoft PI Data Archive through PI Web API or PI SDK.

    Features:
    - PI Web API REST client
    - Bulk data retrieval
    - Calculated data support
    - AF (Asset Framework) integration
    """

    def __init__(
        self,
        config: HistorianConfig,
        vault_client=None,
        use_web_api: bool = True,
        af_server: Optional[str] = None,
    ):
        """
        Initialize PI connector.

        Args:
            config: Historian configuration
            vault_client: Optional vault client
            use_web_api: Use PI Web API (True) or PI SDK (False)
            af_server: Optional AF server name
        """
        super().__init__(config, vault_client)
        self.use_web_api = use_web_api
        self.af_server = af_server

        self._http_client = None
        self._pi_server = None

    async def connect(self) -> bool:
        """Connect to PI Data Archive."""
        if not self._circuit_breaker.is_available():
            logger.warning("PI connection blocked by circuit breaker")
            return False

        try:
            if self.use_web_api:
                await self._connect_web_api()
            else:
                await self._connect_sdk()

            self._connected = True
            self._status = ConnectionStatus.CONNECTED
            self._circuit_breaker.record_success()
            logger.info(f"Connected to PI Data Archive: {self.config.host}")
            return True

        except Exception as e:
            self._circuit_breaker.record_failure()
            self._status = ConnectionStatus.FAILED
            logger.error(f"Failed to connect to PI: {e}")
            return False

    async def _connect_web_api(self) -> None:
        """Connect via PI Web API."""
        import httpx

        base_url = f"https://{self.config.host}:{self.config.port}/piwebapi"

        auth = None
        if self.config.username and self.config.password:
            auth = httpx.BasicAuth(self.config.username, self.config.password)

        self._http_client = httpx.AsyncClient(
            base_url=base_url,
            auth=auth,
            verify=self.config.use_ssl,
            timeout=self.config.read_timeout_seconds,
        )

        # Test connection
        response = await self._http_client.get("/system")
        response.raise_for_status()

    async def _connect_sdk(self) -> None:
        """Connect via PI SDK (requires OSIsoft.AFSDK)."""
        # Note: This requires the OSIsoft AF SDK to be installed
        # from OSIsoft.AF import PIServer

        raise NotImplementedError(
            "PI SDK connection requires OSIsoft.AFSDK. "
            "Use use_web_api=True for REST API access."
        )

    async def disconnect(self) -> None:
        """Disconnect from PI."""
        try:
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None

            self._connected = False
            self._status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from PI Data Archive")

        except Exception as e:
            logger.error(f"Error disconnecting from PI: {e}")

    async def read_raw(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        max_points: int = 10000,
    ) -> Dict[str, TimeSeriesData]:
        """Read raw archived data from PI."""
        if not self._circuit_breaker.is_available():
            raise ConnectionError("Circuit breaker open")

        results = {}

        try:
            for tag_name in tag_names:
                # Get WebId for tag
                web_id = await self._get_tag_web_id(tag_name)

                if not web_id:
                    logger.warning(f"Tag not found: {tag_name}")
                    continue

                # Build query
                params = {
                    "startTime": start_time.isoformat(),
                    "endTime": end_time.isoformat(),
                    "maxCount": max_points,
                }

                response = await self._http_client.get(
                    f"/streams/{web_id}/recorded",
                    params=params,
                )
                response.raise_for_status()

                data = response.json()
                points = self._parse_pi_values(data.get("Items", []))

                results[tag_name] = TimeSeriesData(
                    tag_name=tag_name,
                    points=points,
                )

            self._circuit_breaker.record_success()
            return results

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Error reading from PI: {e}")
            raise

    async def read_interpolated(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int = 60,
    ) -> Dict[str, TimeSeriesData]:
        """Read interpolated data from PI."""
        if not self._circuit_breaker.is_available():
            raise ConnectionError("Circuit breaker open")

        results = {}

        try:
            for tag_name in tag_names:
                web_id = await self._get_tag_web_id(tag_name)

                if not web_id:
                    continue

                params = {
                    "startTime": start_time.isoformat(),
                    "endTime": end_time.isoformat(),
                    "interval": f"{interval_seconds}s",
                }

                response = await self._http_client.get(
                    f"/streams/{web_id}/interpolated",
                    params=params,
                )
                response.raise_for_status()

                data = response.json()
                points = self._parse_pi_values(data.get("Items", []))

                results[tag_name] = TimeSeriesData(
                    tag_name=tag_name,
                    points=points,
                )

            self._circuit_breaker.record_success()
            return results

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Error reading interpolated from PI: {e}")
            raise

    async def read_aggregated(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int,
        aggregate_type: AggregateType,
    ) -> Dict[str, TimeSeriesData]:
        """Read aggregated data from PI."""
        if not self._circuit_breaker.is_available():
            raise ConnectionError("Circuit breaker open")

        # Map aggregate type to PI summary type
        summary_map = {
            AggregateType.AVERAGE: "Average",
            AggregateType.MINIMUM: "Minimum",
            AggregateType.MAXIMUM: "Maximum",
            AggregateType.TOTAL: "Total",
            AggregateType.COUNT: "Count",
            AggregateType.STDEV: "StdDev",
            AggregateType.RANGE: "Range",
        }

        summary_type = summary_map.get(aggregate_type, "Average")
        results = {}

        try:
            for tag_name in tag_names:
                web_id = await self._get_tag_web_id(tag_name)

                if not web_id:
                    continue

                params = {
                    "startTime": start_time.isoformat(),
                    "endTime": end_time.isoformat(),
                    "summaryType": summary_type,
                    "summaryDuration": f"{interval_seconds}s",
                }

                response = await self._http_client.get(
                    f"/streams/{web_id}/summary",
                    params=params,
                )
                response.raise_for_status()

                data = response.json()
                points = self._parse_pi_summary(data.get("Items", []), summary_type)

                results[tag_name] = TimeSeriesData(
                    tag_name=tag_name,
                    points=points,
                )

            self._circuit_breaker.record_success()
            return results

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Error reading aggregated from PI: {e}")
            raise

    async def write_values(
        self,
        tag_name: str,
        values: List[TimeSeriesPoint],
    ) -> WriteResult:
        """Write values to PI archive."""
        if not self._circuit_breaker.is_available():
            return WriteResult(
                tag_name=tag_name,
                points_written=0,
                points_failed=len(values),
                success=False,
                error_message="Circuit breaker open",
            )

        try:
            web_id = await self._get_tag_web_id(tag_name)

            if not web_id:
                return WriteResult(
                    tag_name=tag_name,
                    points_written=0,
                    points_failed=len(values),
                    success=False,
                    error_message=f"Tag not found: {tag_name}",
                )

            # Format values for PI Web API
            items = []
            for point in values:
                items.append({
                    "Timestamp": point.timestamp.isoformat(),
                    "Value": point.value,
                    "Good": point.quality == TagQuality.GOOD,
                })

            response = await self._http_client.post(
                f"/streams/{web_id}/recorded",
                json=items,
            )
            response.raise_for_status()

            self._circuit_breaker.record_success()
            return WriteResult(
                tag_name=tag_name,
                points_written=len(values),
                points_failed=0,
                success=True,
            )

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Error writing to PI: {e}")
            return WriteResult(
                tag_name=tag_name,
                points_written=0,
                points_failed=len(values),
                success=False,
                error_message=str(e),
            )

    async def search_tags(
        self,
        pattern: str,
        max_results: int = 1000,
    ) -> List[TagSearchResult]:
        """Search for PI tags."""
        try:
            # Get data server WebId
            response = await self._http_client.get("/dataservers")
            response.raise_for_status()
            servers = response.json().get("Items", [])

            if not servers:
                return []

            server_web_id = servers[0]["WebId"]

            # Search for points
            params = {
                "nameFilter": pattern,
                "maxCount": max_results,
            }

            response = await self._http_client.get(
                f"/dataservers/{server_web_id}/points",
                params=params,
            )
            response.raise_for_status()

            results = []
            for item in response.json().get("Items", []):
                results.append(TagSearchResult(
                    tag_name=item.get("Name", ""),
                    description=item.get("Descriptor", ""),
                    data_type=TagDataType.FLOAT64,
                    unit=item.get("EngineeringUnits", ""),
                    point_class=item.get("PointClass", ""),
                ))

            return results

        except Exception as e:
            logger.error(f"Error searching PI tags: {e}")
            return []

    async def _get_tag_web_id(self, tag_name: str) -> Optional[str]:
        """Get WebId for a PI tag."""
        # Check cache
        if tag_name in self._tag_cache:
            return self._tag_cache[tag_name].point_class  # Store WebId in point_class

        try:
            params = {"path": f"\\\\{self.config.host}\\{tag_name}"}
            response = await self._http_client.get("/points", params=params)

            if response.status_code == 200:
                data = response.json()
                web_id = data.get("WebId")

                # Cache result
                self._tag_cache[tag_name] = TagSearchResult(
                    tag_name=tag_name,
                    description=data.get("Descriptor", ""),
                    data_type=TagDataType.FLOAT64,
                    unit=data.get("EngineeringUnits", ""),
                    point_class=web_id,
                )

                return web_id

        except Exception as e:
            logger.warning(f"Error getting WebId for {tag_name}: {e}")

        return None

    def _parse_pi_values(self, items: List[Dict]) -> List[TimeSeriesPoint]:
        """Parse PI Web API value response."""
        points = []
        for item in items:
            try:
                timestamp = datetime.fromisoformat(
                    item["Timestamp"].replace("Z", "+00:00")
                )
                value = item.get("Value")
                good = item.get("Good", True)

                points.append(TimeSeriesPoint(
                    timestamp=timestamp,
                    value=value,
                    quality=TagQuality.GOOD if good else TagQuality.BAD,
                ))

            except Exception as e:
                logger.warning(f"Error parsing PI value: {e}")

        return points

    def _parse_pi_summary(
        self, items: List[Dict], summary_type: str
    ) -> List[TimeSeriesPoint]:
        """Parse PI Web API summary response."""
        points = []
        for item in items:
            try:
                type_data = item.get("Type", {})
                value_data = item.get("Value", {})

                if type_data.get("Name") == summary_type:
                    timestamp = datetime.fromisoformat(
                        value_data.get("Timestamp", "").replace("Z", "+00:00")
                    )
                    value = value_data.get("Value")

                    points.append(TimeSeriesPoint(
                        timestamp=timestamp,
                        value=value,
                        quality=TagQuality.GOOD,
                    ))

            except Exception as e:
                logger.warning(f"Error parsing PI summary: {e}")

        return points

    async def read_af_attribute(
        self,
        element_path: str,
        attribute_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> TimeSeriesData:
        """Read AF (Asset Framework) attribute data."""
        if not self.af_server:
            raise ValueError("AF server not configured")

        try:
            # Get AF attribute WebId
            params = {
                "path": f"\\\\{self.af_server}\\{element_path}|{attribute_name}"
            }

            response = await self._http_client.get("/attributes", params=params)
            response.raise_for_status()

            web_id = response.json().get("WebId")

            if not web_id:
                raise ValueError(f"AF attribute not found: {element_path}|{attribute_name}")

            # Read data
            params = {
                "startTime": start_time.isoformat(),
                "endTime": end_time.isoformat(),
            }

            response = await self._http_client.get(
                f"/streams/{web_id}/recorded",
                params=params,
            )
            response.raise_for_status()

            points = self._parse_pi_values(response.json().get("Items", []))

            return TimeSeriesData(
                tag_name=f"{element_path}|{attribute_name}",
                points=points,
            )

        except Exception as e:
            logger.error(f"Error reading AF attribute: {e}")
            raise


# =============================================================================
# AspenTech InfoPlus.21 Connector
# =============================================================================

class AspenInfoPlusConnector(HistorianConnectorBase):
    """
    AspenTech InfoPlus.21 Historian Connector.

    Provides access to InfoPlus.21 historian via SQLplus or REST API.
    """

    def __init__(
        self,
        config: HistorianConfig,
        vault_client=None,
    ):
        super().__init__(config, vault_client)
        self._connection = None

    async def connect(self) -> bool:
        """Connect to InfoPlus.21."""
        if not self._circuit_breaker.is_available():
            logger.warning("IP.21 connection blocked by circuit breaker")
            return False

        try:
            # Connect via ODBC or native driver
            import pyodbc

            conn_str = (
                f"Driver={{AspenTech SQLplus}};Server={self.config.host};"
                f"Port={self.config.port};Database={self.config.database};"
            )

            if self.config.username:
                conn_str += f"UID={self.config.username};PWD={self.config.password};"

            self._connection = pyodbc.connect(conn_str)

            self._connected = True
            self._status = ConnectionStatus.CONNECTED
            self._circuit_breaker.record_success()
            logger.info(f"Connected to InfoPlus.21: {self.config.host}")
            return True

        except Exception as e:
            self._circuit_breaker.record_failure()
            self._status = ConnectionStatus.FAILED
            logger.error(f"Failed to connect to IP.21: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from InfoPlus.21."""
        try:
            if self._connection:
                self._connection.close()
                self._connection = None

            self._connected = False
            self._status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from InfoPlus.21")

        except Exception as e:
            logger.error(f"Error disconnecting from IP.21: {e}")

    async def read_raw(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        max_points: int = 10000,
    ) -> Dict[str, TimeSeriesData]:
        """Read raw data from InfoPlus.21."""
        if not self._circuit_breaker.is_available():
            raise ConnectionError("Circuit breaker open")

        results = {}

        try:
            cursor = self._connection.cursor()

            for tag_name in tag_names:
                # SQLplus query for raw data
                query = f"""
                SELECT TS, VALUE, STATUS
                FROM HISTORY
                WHERE NAME = '{tag_name}'
                AND TS >= '{start_time.strftime("%Y-%m-%d %H:%M:%S")}'
                AND TS <= '{end_time.strftime("%Y-%m-%d %H:%M:%S")}'
                ORDER BY TS
                """

                cursor.execute(query)
                rows = cursor.fetchmany(max_points)

                points = []
                for row in rows:
                    points.append(TimeSeriesPoint(
                        timestamp=row[0],
                        value=row[1],
                        quality=TagQuality.GOOD if row[2] == 0 else TagQuality.BAD,
                    ))

                results[tag_name] = TimeSeriesData(
                    tag_name=tag_name,
                    points=points,
                )

            cursor.close()
            self._circuit_breaker.record_success()
            return results

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Error reading from IP.21: {e}")
            raise

    async def read_interpolated(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int = 60,
    ) -> Dict[str, TimeSeriesData]:
        """Read interpolated data from InfoPlus.21."""
        if not self._circuit_breaker.is_available():
            raise ConnectionError("Circuit breaker open")

        results = {}

        try:
            cursor = self._connection.cursor()

            for tag_name in tag_names:
                # SQLplus query for interpolated data
                query = f"""
                SELECT TS, VALUE
                FROM HISTORY
                WHERE NAME = '{tag_name}'
                AND TS >= '{start_time.strftime("%Y-%m-%d %H:%M:%S")}'
                AND TS <= '{end_time.strftime("%Y-%m-%d %H:%M:%S")}'
                REQUEST = INTERP
                PERIOD = {interval_seconds}
                """

                cursor.execute(query)
                rows = cursor.fetchall()

                points = []
                for row in rows:
                    points.append(TimeSeriesPoint(
                        timestamp=row[0],
                        value=row[1],
                        quality=TagQuality.GOOD,
                    ))

                results[tag_name] = TimeSeriesData(
                    tag_name=tag_name,
                    points=points,
                )

            cursor.close()
            self._circuit_breaker.record_success()
            return results

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Error reading interpolated from IP.21: {e}")
            raise

    async def read_aggregated(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int,
        aggregate_type: AggregateType,
    ) -> Dict[str, TimeSeriesData]:
        """Read aggregated data from InfoPlus.21."""
        if not self._circuit_breaker.is_available():
            raise ConnectionError("Circuit breaker open")

        # Map aggregate type to IP.21 function
        func_map = {
            AggregateType.AVERAGE: "AVG",
            AggregateType.MINIMUM: "MIN",
            AggregateType.MAXIMUM: "MAX",
            AggregateType.TOTAL: "TOTAL",
            AggregateType.COUNT: "COUNT",
            AggregateType.STDEV: "STDEV",
        }

        func_name = func_map.get(aggregate_type, "AVG")
        results = {}

        try:
            cursor = self._connection.cursor()

            for tag_name in tag_names:
                query = f"""
                SELECT TS, {func_name}(VALUE)
                FROM HISTORY
                WHERE NAME = '{tag_name}'
                AND TS >= '{start_time.strftime("%Y-%m-%d %H:%M:%S")}'
                AND TS <= '{end_time.strftime("%Y-%m-%d %H:%M:%S")}'
                GROUP BY TS
                PERIOD = {interval_seconds}
                """

                cursor.execute(query)
                rows = cursor.fetchall()

                points = []
                for row in rows:
                    points.append(TimeSeriesPoint(
                        timestamp=row[0],
                        value=row[1],
                        quality=TagQuality.GOOD,
                    ))

                results[tag_name] = TimeSeriesData(
                    tag_name=tag_name,
                    points=points,
                )

            cursor.close()
            self._circuit_breaker.record_success()
            return results

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Error reading aggregated from IP.21: {e}")
            raise

    async def write_values(
        self,
        tag_name: str,
        values: List[TimeSeriesPoint],
    ) -> WriteResult:
        """Write values to InfoPlus.21."""
        if not self._circuit_breaker.is_available():
            return WriteResult(
                tag_name=tag_name,
                points_written=0,
                points_failed=len(values),
                success=False,
                error_message="Circuit breaker open",
            )

        try:
            cursor = self._connection.cursor()

            written = 0
            for point in values:
                query = f"""
                INSERT INTO HISTORY (NAME, TS, VALUE, STATUS)
                VALUES ('{tag_name}', '{point.timestamp.strftime("%Y-%m-%d %H:%M:%S")}',
                        {point.value}, {0 if point.quality == TagQuality.GOOD else 1})
                """
                cursor.execute(query)
                written += 1

            self._connection.commit()
            cursor.close()

            self._circuit_breaker.record_success()
            return WriteResult(
                tag_name=tag_name,
                points_written=written,
                points_failed=len(values) - written,
                success=True,
            )

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Error writing to IP.21: {e}")
            return WriteResult(
                tag_name=tag_name,
                points_written=0,
                points_failed=len(values),
                success=False,
                error_message=str(e),
            )

    async def search_tags(
        self,
        pattern: str,
        max_results: int = 1000,
    ) -> List[TagSearchResult]:
        """Search for IP.21 tags."""
        try:
            cursor = self._connection.cursor()

            query = f"""
            SELECT NAME, DESCRIPTION, IP_ENG_UNITS
            FROM ATDEFMAP
            WHERE NAME LIKE '{pattern.replace("*", "%")}'
            """

            cursor.execute(query)
            rows = cursor.fetchmany(max_results)

            results = []
            for row in rows:
                results.append(TagSearchResult(
                    tag_name=row[0],
                    description=row[1] or "",
                    data_type=TagDataType.FLOAT64,
                    unit=row[2] or "",
                ))

            cursor.close()
            return results

        except Exception as e:
            logger.error(f"Error searching IP.21 tags: {e}")
            return []


# =============================================================================
# Wonderware Historian Connector
# =============================================================================

class WonderwareHistorianConnector(HistorianConnectorBase):
    """
    Wonderware (AVEVA) Historian Connector.

    Provides access to Wonderware IndustrialSQL Server / Historian via OLEDB or REST.
    """

    def __init__(
        self,
        config: HistorianConfig,
        vault_client=None,
    ):
        super().__init__(config, vault_client)
        self._connection = None

    async def connect(self) -> bool:
        """Connect to Wonderware Historian."""
        if not self._circuit_breaker.is_available():
            logger.warning("Wonderware connection blocked by circuit breaker")
            return False

        try:
            import pyodbc

            conn_str = (
                f"Driver={{InSQL}};Server={self.config.host};"
                f"Database=Runtime;"
            )

            if self.config.username:
                conn_str += f"UID={self.config.username};PWD={self.config.password};"
            else:
                conn_str += "Trusted_Connection=yes;"

            self._connection = pyodbc.connect(conn_str)

            self._connected = True
            self._status = ConnectionStatus.CONNECTED
            self._circuit_breaker.record_success()
            logger.info(f"Connected to Wonderware Historian: {self.config.host}")
            return True

        except Exception as e:
            self._circuit_breaker.record_failure()
            self._status = ConnectionStatus.FAILED
            logger.error(f"Failed to connect to Wonderware: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Wonderware Historian."""
        try:
            if self._connection:
                self._connection.close()
                self._connection = None

            self._connected = False
            self._status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from Wonderware Historian")

        except Exception as e:
            logger.error(f"Error disconnecting from Wonderware: {e}")

    async def read_raw(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        max_points: int = 10000,
    ) -> Dict[str, TimeSeriesData]:
        """Read raw data from Wonderware Historian."""
        if not self._circuit_breaker.is_available():
            raise ConnectionError("Circuit breaker open")

        results = {}

        try:
            cursor = self._connection.cursor()

            tags_list = ",".join([f"'{t}'" for t in tag_names])

            query = f"""
            SELECT TagName, DateTime, Value, Quality
            FROM History
            WHERE TagName IN ({tags_list})
            AND DateTime >= '{start_time.strftime("%Y-%m-%d %H:%M:%S")}'
            AND DateTime <= '{end_time.strftime("%Y-%m-%d %H:%M:%S")}'
            AND wwRetrievalMode = 'Full'
            ORDER BY TagName, DateTime
            """

            cursor.execute(query)
            rows = cursor.fetchmany(max_points * len(tag_names))

            # Group by tag name
            tag_data: Dict[str, List[TimeSeriesPoint]] = {t: [] for t in tag_names}

            for row in rows:
                tag_name = row[0]
                if tag_name in tag_data:
                    quality = TagQuality.GOOD if row[3] >= 192 else TagQuality.BAD
                    tag_data[tag_name].append(TimeSeriesPoint(
                        timestamp=row[1],
                        value=row[2],
                        quality=quality,
                    ))

            for tag_name, points in tag_data.items():
                results[tag_name] = TimeSeriesData(
                    tag_name=tag_name,
                    points=points,
                )

            cursor.close()
            self._circuit_breaker.record_success()
            return results

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Error reading from Wonderware: {e}")
            raise

    async def read_interpolated(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int = 60,
    ) -> Dict[str, TimeSeriesData]:
        """Read interpolated data from Wonderware Historian."""
        if not self._circuit_breaker.is_available():
            raise ConnectionError("Circuit breaker open")

        results = {}

        try:
            cursor = self._connection.cursor()

            tags_list = ",".join([f"'{t}'" for t in tag_names])

            query = f"""
            SELECT TagName, DateTime, Value
            FROM History
            WHERE TagName IN ({tags_list})
            AND DateTime >= '{start_time.strftime("%Y-%m-%d %H:%M:%S")}'
            AND DateTime <= '{end_time.strftime("%Y-%m-%d %H:%M:%S")}'
            AND wwRetrievalMode = 'Cyclic'
            AND wwResolution = {interval_seconds * 1000}
            ORDER BY TagName, DateTime
            """

            cursor.execute(query)
            rows = cursor.fetchall()

            # Group by tag name
            tag_data: Dict[str, List[TimeSeriesPoint]] = {t: [] for t in tag_names}

            for row in rows:
                tag_name = row[0]
                if tag_name in tag_data:
                    tag_data[tag_name].append(TimeSeriesPoint(
                        timestamp=row[1],
                        value=row[2],
                        quality=TagQuality.GOOD,
                    ))

            for tag_name, points in tag_data.items():
                results[tag_name] = TimeSeriesData(
                    tag_name=tag_name,
                    points=points,
                )

            cursor.close()
            self._circuit_breaker.record_success()
            return results

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Error reading interpolated from Wonderware: {e}")
            raise

    async def read_aggregated(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int,
        aggregate_type: AggregateType,
    ) -> Dict[str, TimeSeriesData]:
        """Read aggregated data from Wonderware Historian."""
        if not self._circuit_breaker.is_available():
            raise ConnectionError("Circuit breaker open")

        # Map aggregate type to Wonderware retrieval mode
        mode_map = {
            AggregateType.AVERAGE: "Average",
            AggregateType.MINIMUM: "Minimum",
            AggregateType.MAXIMUM: "Maximum",
            AggregateType.TOTAL: "Total",
            AggregateType.COUNT: "Count",
            AggregateType.STDEV: "StdDev",
        }

        retrieval_mode = mode_map.get(aggregate_type, "Average")
        results = {}

        try:
            cursor = self._connection.cursor()

            tags_list = ",".join([f"'{t}'" for t in tag_names])

            query = f"""
            SELECT TagName, DateTime, Value
            FROM History
            WHERE TagName IN ({tags_list})
            AND DateTime >= '{start_time.strftime("%Y-%m-%d %H:%M:%S")}'
            AND DateTime <= '{end_time.strftime("%Y-%m-%d %H:%M:%S")}'
            AND wwRetrievalMode = '{retrieval_mode}'
            AND wwResolution = {interval_seconds * 1000}
            ORDER BY TagName, DateTime
            """

            cursor.execute(query)
            rows = cursor.fetchall()

            tag_data: Dict[str, List[TimeSeriesPoint]] = {t: [] for t in tag_names}

            for row in rows:
                tag_name = row[0]
                if tag_name in tag_data:
                    tag_data[tag_name].append(TimeSeriesPoint(
                        timestamp=row[1],
                        value=row[2],
                        quality=TagQuality.GOOD,
                    ))

            for tag_name, points in tag_data.items():
                results[tag_name] = TimeSeriesData(
                    tag_name=tag_name,
                    points=points,
                )

            cursor.close()
            self._circuit_breaker.record_success()
            return results

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Error reading aggregated from Wonderware: {e}")
            raise

    async def write_values(
        self,
        tag_name: str,
        values: List[TimeSeriesPoint],
    ) -> WriteResult:
        """Write values to Wonderware Historian."""
        if not self._circuit_breaker.is_available():
            return WriteResult(
                tag_name=tag_name,
                points_written=0,
                points_failed=len(values),
                success=False,
                error_message="Circuit breaker open",
            )

        try:
            cursor = self._connection.cursor()

            written = 0
            for point in values:
                query = f"""
                INSERT INTO History (TagName, DateTime, Value, Quality)
                VALUES ('{tag_name}', '{point.timestamp.strftime("%Y-%m-%d %H:%M:%S")}',
                        {point.value}, {192 if point.quality == TagQuality.GOOD else 0})
                """
                cursor.execute(query)
                written += 1

            self._connection.commit()
            cursor.close()

            self._circuit_breaker.record_success()
            return WriteResult(
                tag_name=tag_name,
                points_written=written,
                points_failed=len(values) - written,
                success=True,
            )

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Error writing to Wonderware: {e}")
            return WriteResult(
                tag_name=tag_name,
                points_written=0,
                points_failed=len(values),
                success=False,
                error_message=str(e),
            )

    async def search_tags(
        self,
        pattern: str,
        max_results: int = 1000,
    ) -> List[TagSearchResult]:
        """Search for Wonderware tags."""
        try:
            cursor = self._connection.cursor()

            query = f"""
            SELECT TOP {max_results}
                TagName, Description, EUKey, ProviderID
            FROM Tag
            WHERE TagName LIKE '{pattern.replace("*", "%")}'
            """

            cursor.execute(query)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append(TagSearchResult(
                    tag_name=row[0],
                    description=row[1] or "",
                    data_type=TagDataType.FLOAT64,
                    unit=row[2] or "",
                    source=row[3] or "",
                ))

            cursor.close()
            return results

        except Exception as e:
            logger.error(f"Error searching Wonderware tags: {e}")
            return []
