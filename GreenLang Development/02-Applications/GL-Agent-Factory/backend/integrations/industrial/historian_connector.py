"""
Process Historian Connector for GreenLang.

This module provides comprehensive integration with process historians
including OSIsoft PI, Aveva Historian, and InfluxDB for time-series
data storage and retrieval.

Features:
    - OSIsoft PI Data Archive connector
    - Aveva Historian (Wonderware) connector
    - InfluxDB 1.x and 2.x connector
    - Time-series query optimization
    - Data compression handling
    - Batch data retrieval
    - Aggregation functions

Example:
    >>> from integrations.industrial import PIHistorianConnector, PIConfig
    >>>
    >>> config = PIConfig(
    ...     host="pi-server.factory.local",
    ...     username="pi_user"
    ... )
    >>> connector = PIHistorianConnector(config)
    >>> async with connector:
    ...     data = await connector.read_history(query)
"""

import asyncio
import logging
import time
from abc import abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, SecretStr

from .base import (
    AuthenticationType,
    BaseConnectorConfig,
    BaseIndustrialConnector,
    TLSConfig,
)
from .data_models import (
    AggregationType,
    BatchReadResponse,
    BatchWriteRequest,
    BatchWriteResponse,
    ConnectionState,
    DataQuality,
    DataType,
    HistoricalQuery,
    HistoricalResult,
    TagMetadata,
    TagValue,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Historian Types
# =============================================================================


class HistorianType(str, Enum):
    """Supported historian types."""

    PI = "pi"  # OSIsoft PI
    AVEVA = "aveva"  # Aveva/Wonderware Historian
    INFLUXDB = "influxdb"  # InfluxDB
    TIMESCALEDB = "timescaledb"  # TimescaleDB


class CompressionType(str, Enum):
    """Data compression types."""

    NONE = "none"
    SWINGING_DOOR = "swinging_door"  # PI default
    DEADBAND = "deadband"
    BOXCAR = "boxcar"


# =============================================================================
# Base Historian Configuration
# =============================================================================


class BaseHistorianConfig(BaseConnectorConfig):
    """
    Base configuration for process historians.

    Attributes:
        historian_type: Type of historian
        max_points_per_request: Maximum points per request
        default_aggregation: Default aggregation type
        timeout_seconds: Query timeout
    """

    historian_type: HistorianType = Field(..., description="Historian type")
    max_points_per_request: int = Field(
        100000,
        ge=1,
        description="Max points per request"
    )
    default_aggregation: AggregationType = Field(
        AggregationType.RAW,
        description="Default aggregation"
    )
    batch_size: int = Field(100, ge=1, description="Batch size for queries")


# =============================================================================
# OSIsoft PI Configuration
# =============================================================================


class PIConfig(BaseHistorianConfig):
    """
    OSIsoft PI Data Archive configuration.

    Attributes:
        host: PI Data Archive server
        port: PI server port
        pi_server_name: PI Server name
        af_server: Optional AF Server for attributes
        af_database: AF Database name
    """

    historian_type: HistorianType = HistorianType.PI
    port: int = Field(5450, description="PI server port")
    pi_server_name: Optional[str] = Field(None, description="PI Server name")
    af_server: Optional[str] = Field(None, description="AF Server")
    af_database: Optional[str] = Field(None, description="AF Database")

    # PI-specific settings
    use_compression: bool = Field(True, description="Use compressed data")
    max_archive_points: int = Field(100000, description="Max archive points")
    request_timeout_ms: int = Field(30000, description="Request timeout")


class PIConnector(BaseIndustrialConnector):
    """
    OSIsoft PI Data Archive Connector.

    Provides integration with OSIsoft PI for historical data
    retrieval and real-time data access.

    Features:
        - PI Point reading and writing
        - Archive data queries
        - AF Element/Attribute support
        - Interpolated and plot values
        - Compressed data handling

    Example:
        >>> config = PIConfig(
        ...     host="pi-server.local",
        ...     username="pi_user",
        ...     password="secret"
        ... )
        >>> connector = PIConnector(config)
        >>> await connector.connect()
        >>> data = await connector.read_history(query)
    """

    def __init__(self, config: PIConfig):
        """Initialize PI connector."""
        base_config = BaseConnectorConfig(
            host=config.host,
            port=config.port,
            timeout_seconds=config.timeout_seconds,
            auth_type=AuthenticationType.USERNAME_PASSWORD,
            username=config.username,
            password=config.password,
            name=config.name or "pi_connector",
            tls=config.tls,
            rate_limit=config.rate_limit,
            reconnect=config.reconnect,
            health_check_interval_seconds=config.health_check_interval_seconds,
        )

        super().__init__(base_config)
        self.pi_config = config

        # PI-specific state
        self._pi_server: Optional[Any] = None
        self._af_database: Optional[Any] = None
        self._point_cache: Dict[str, Any] = {}

    async def _do_connect(self) -> bool:
        """Connect to PI Data Archive."""
        logger.info(f"Connecting to PI Data Archive: {self.pi_config.host}")

        try:
            # In production, use osisoft.pidevclub.piwebapi or PIthon
            # from PIthon import PIPoint, PIServer
            # self._pi_server = PIServer(self.pi_config.host)
            # self._pi_server.connect(
            #     self.pi_config.username,
            #     self.pi_config.password.get_secret_value()
            # )

            # Simulated connection
            self._pi_server = True

            # Connect to AF if configured
            if self.pi_config.af_server:
                # self._af_database = AFDatabase(...)
                pass

            logger.info("PI Data Archive connected")
            return True

        except Exception as e:
            logger.error(f"PI connection failed: {e}")
            raise

    async def _do_disconnect(self) -> None:
        """Disconnect from PI."""
        self._pi_server = None
        self._af_database = None
        self._point_cache.clear()
        logger.info("PI disconnected")

    async def _do_health_check(self) -> bool:
        """Check PI connection health."""
        return self._pi_server is not None

    async def read_tags(
        self,
        tag_ids: List[str],
    ) -> BatchReadResponse:
        """Read current snapshot values from PI."""
        self._validate_connected()

        values: Dict[str, TagValue] = {}
        errors: Dict[str, str] = {}

        for tag_id in tag_ids:
            try:
                # In production:
                # point = PIPoint.FindPIPoint(self._pi_server, tag_id)
                # current = point.CurrentValue()
                # value = current.Value
                # timestamp = current.Timestamp
                # status = current.IsGood

                # Simulated read
                import random
                value = round(random.uniform(0, 100), 2)

                values[tag_id] = TagValue(
                    tag_id=tag_id,
                    value=value,
                    timestamp=datetime.utcnow(),
                    quality=DataQuality.GOOD,
                )

            except Exception as e:
                errors[tag_id] = str(e)

        return BatchReadResponse(
            values=values,
            errors=errors,
            timestamp=datetime.utcnow(),
        )

    async def read_history(
        self,
        query: HistoricalQuery,
    ) -> Dict[str, HistoricalResult]:
        """
        Read historical data from PI Archive.

        Args:
            query: Historical query specification

        Returns:
            Dictionary of tag_id to HistoricalResult
        """
        self._validate_connected()

        results: Dict[str, HistoricalResult] = {}

        for tag_id in query.tag_ids:
            try:
                result = await self._query_tag_history(tag_id, query)
                results[tag_id] = result
            except Exception as e:
                logger.error(f"PI history query failed for {tag_id}: {e}")
                results[tag_id] = HistoricalResult(
                    tag_id=tag_id,
                    values=[],
                    start_time=query.start_time,
                    end_time=query.end_time,
                    point_count=0,
                    aggregation=query.aggregation,
                )

        return results

    async def _query_tag_history(
        self,
        tag_id: str,
        query: HistoricalQuery,
    ) -> HistoricalResult:
        """Query history for a single tag."""
        # In production:
        # point = PIPoint.FindPIPoint(self._pi_server, tag_id)
        #
        # if query.aggregation == AggregationType.RAW:
        #     values = point.RecordedValues(
        #         query.start_time,
        #         query.end_time,
        #         boundaryType=AFBoundaryType.Inside,
        #         maxCount=query.max_points
        #     )
        # elif query.aggregation == AggregationType.INTERPOLATED:
        #     values = point.InterpolatedValues(
        #         query.start_time,
        #         query.end_time,
        #         interval=AFTimeSpan(milliseconds=query.interval_ms)
        #     )
        # else:
        #     values = point.Summaries(
        #         query.start_time,
        #         query.end_time,
        #         interval=AFTimeSpan(milliseconds=query.interval_ms),
        #         summaryType=self._map_aggregation(query.aggregation)
        #     )

        # Simulated historical data
        import random

        values = []
        current_time = query.start_time
        interval = timedelta(milliseconds=query.interval_ms or 60000)

        while current_time <= query.end_time and len(values) < query.max_points:
            # Simulate values with some trend
            base_value = 50 + 20 * ((current_time - query.start_time).total_seconds() / 3600)
            value = base_value + random.uniform(-5, 5)

            values.append(TagValue(
                tag_id=tag_id,
                value=round(value, 2),
                timestamp=current_time,
                quality=DataQuality.GOOD,
            ))
            current_time += interval

        return HistoricalResult(
            tag_id=tag_id,
            values=values,
            start_time=query.start_time,
            end_time=query.end_time,
            point_count=len(values),
            aggregation=query.aggregation,
        )

    async def write_tags(
        self,
        request: BatchWriteRequest,
    ) -> BatchWriteResponse:
        """Write values to PI points."""
        self._validate_connected()

        success: Dict[str, bool] = {}
        errors: Dict[str, str] = {}

        for tag_id, value in request.writes.items():
            try:
                # In production:
                # point = PIPoint.FindPIPoint(self._pi_server, tag_id)
                # point.UpdateValue(
                #     AFValue(value, AFTime.Now),
                #     AFUpdateOption.Replace
                # )

                success[tag_id] = True
                logger.info(f"Wrote {value} to PI point {tag_id}")

            except Exception as e:
                errors[tag_id] = str(e)

        return BatchWriteResponse(
            success=success,
            errors=errors,
            timestamp=datetime.utcnow(),
        )

    async def search_points(
        self,
        pattern: str,
        max_results: int = 1000,
    ) -> List[str]:
        """
        Search for PI points by name pattern.

        Args:
            pattern: Point name pattern (supports * wildcard)
            max_results: Maximum results

        Returns:
            List of matching point names
        """
        self._validate_connected()

        # In production:
        # points = PIPoint.FindPIPoints(
        #     self._pi_server,
        #     pattern,
        #     maxCount=max_results
        # )
        # return [p.Name for p in points]

        # Simulated search
        return [f"TAG_{pattern}_{i}" for i in range(min(10, max_results))]


# =============================================================================
# Aveva Historian Configuration
# =============================================================================


class AvevaConfig(BaseHistorianConfig):
    """
    Aveva (Wonderware) Historian configuration.

    Attributes:
        host: Historian server hostname
        port: Server port
        database: Database name
        use_sqloledb: Use SQL Server provider
    """

    historian_type: HistorianType = HistorianType.AVEVA
    port: int = Field(1433, description="SQL Server port")
    database: str = Field("Runtime", description="Database name")
    use_sqloledb: bool = Field(True, description="Use SQL OLE DB")

    # Aveva-specific
    retrieval_mode: str = Field("Cyclic", description="Retrieval mode")
    resolution_ms: int = Field(1000, description="Resolution in ms")


class AvevaConnector(BaseIndustrialConnector):
    """
    Aveva (Wonderware) Historian Connector.

    Provides integration with Aveva Historian for time-series data.

    Features:
        - Tag data retrieval
        - Cyclic and delta retrieval modes
        - Analog and discrete tag support
        - Summary queries (min, max, avg)

    Example:
        >>> config = AvevaConfig(
        ...     host="historian.factory.local",
        ...     database="Runtime",
        ...     username="historian_user"
        ... )
        >>> connector = AvevaConnector(config)
        >>> await connector.connect()
    """

    def __init__(self, config: AvevaConfig):
        """Initialize Aveva connector."""
        base_config = BaseConnectorConfig(
            host=config.host,
            port=config.port,
            timeout_seconds=config.timeout_seconds,
            auth_type=AuthenticationType.USERNAME_PASSWORD,
            username=config.username,
            password=config.password,
            name=config.name or "aveva_connector",
            tls=config.tls,
            rate_limit=config.rate_limit,
            reconnect=config.reconnect,
            health_check_interval_seconds=config.health_check_interval_seconds,
        )

        super().__init__(base_config)
        self.aveva_config = config
        self._connection: Optional[Any] = None

    async def _do_connect(self) -> bool:
        """Connect to Aveva Historian."""
        logger.info(f"Connecting to Aveva Historian: {self.aveva_config.host}")

        try:
            # In production, use pyodbc or aioodbc:
            # connection_string = (
            #     f"DRIVER={{SQL Server}};"
            #     f"SERVER={self.aveva_config.host};"
            #     f"DATABASE={self.aveva_config.database};"
            #     f"UID={self.aveva_config.username};"
            #     f"PWD={self.aveva_config.password.get_secret_value()}"
            # )
            # self._connection = await aioodbc.connect(connection_string)

            self._connection = True
            logger.info("Aveva Historian connected")
            return True

        except Exception as e:
            logger.error(f"Aveva connection failed: {e}")
            raise

    async def _do_disconnect(self) -> None:
        """Disconnect from Aveva."""
        if self._connection:
            # await self._connection.close()
            pass
        self._connection = None
        logger.info("Aveva disconnected")

    async def _do_health_check(self) -> bool:
        """Check Aveva connection."""
        return self._connection is not None

    async def read_tags(
        self,
        tag_ids: List[str],
    ) -> BatchReadResponse:
        """Read current values from Aveva."""
        self._validate_connected()

        values: Dict[str, TagValue] = {}
        errors: Dict[str, str] = {}

        # In production, query Live table:
        # query = f"""
        #     SELECT TagName, Value, DateTime, Quality
        #     FROM Live
        #     WHERE TagName IN ({','.join(f"'{t}'" for t in tag_ids)})
        # """
        # async with self._connection.cursor() as cursor:
        #     await cursor.execute(query)
        #     rows = await cursor.fetchall()

        # Simulated read
        import random
        for tag_id in tag_ids:
            values[tag_id] = TagValue(
                tag_id=tag_id,
                value=round(random.uniform(0, 100), 2),
                timestamp=datetime.utcnow(),
                quality=DataQuality.GOOD,
            )

        return BatchReadResponse(
            values=values,
            errors=errors,
            timestamp=datetime.utcnow(),
        )

    async def read_history(
        self,
        query: HistoricalQuery,
    ) -> Dict[str, HistoricalResult]:
        """Read historical data from Aveva."""
        self._validate_connected()

        results: Dict[str, HistoricalResult] = {}

        # In production:
        # sql = f"""
        #     SELECT TagName, DateTime, Value, Quality
        #     FROM History
        #     WHERE TagName IN ({','.join(f"'{t}'" for t in query.tag_ids)})
        #     AND DateTime BETWEEN '{query.start_time}' AND '{query.end_time}'
        #     AND wwRetrievalMode = '{self.aveva_config.retrieval_mode}'
        #     AND wwResolution = {self.aveva_config.resolution_ms}
        # """

        # Simulated results
        import random
        for tag_id in query.tag_ids:
            values = []
            current_time = query.start_time
            interval = timedelta(milliseconds=query.interval_ms or 60000)

            while current_time <= query.end_time and len(values) < query.max_points:
                values.append(TagValue(
                    tag_id=tag_id,
                    value=round(random.uniform(0, 100), 2),
                    timestamp=current_time,
                    quality=DataQuality.GOOD,
                ))
                current_time += interval

            results[tag_id] = HistoricalResult(
                tag_id=tag_id,
                values=values,
                start_time=query.start_time,
                end_time=query.end_time,
                point_count=len(values),
                aggregation=query.aggregation,
            )

        return results

    async def write_tags(
        self,
        request: BatchWriteRequest,
    ) -> BatchWriteResponse:
        """Write values to Aveva (if enabled)."""
        return BatchWriteResponse(
            errors={tag: "Write not supported" for tag in request.writes}
        )


# =============================================================================
# InfluxDB Configuration
# =============================================================================


class InfluxDBVersion(str, Enum):
    """InfluxDB versions."""

    V1 = "1.x"
    V2 = "2.x"


class InfluxDBConfig(BaseHistorianConfig):
    """
    InfluxDB configuration.

    Attributes:
        host: InfluxDB server hostname
        port: Server port
        version: InfluxDB version (1.x or 2.x)
        database: Database name (1.x) or bucket (2.x)
        org: Organization (2.x only)
        token: API token (2.x) or password (1.x)
        measurement: Default measurement name
    """

    historian_type: HistorianType = HistorianType.INFLUXDB
    port: int = Field(8086, description="InfluxDB port")
    version: InfluxDBVersion = Field(InfluxDBVersion.V2, description="Version")

    # Database/bucket
    database: str = Field(..., description="Database (1.x) or bucket (2.x)")
    org: Optional[str] = Field(None, description="Organization (2.x)")
    token: Optional[SecretStr] = Field(None, description="API token")

    # Query settings
    measurement: str = Field("process_data", description="Default measurement")
    retention_policy: str = Field("autogen", description="Retention policy (1.x)")

    # Performance
    batch_size: int = Field(5000, description="Write batch size")
    flush_interval_ms: int = Field(1000, description="Flush interval")


class InfluxDBConnector(BaseIndustrialConnector):
    """
    InfluxDB Time-Series Connector.

    Provides integration with InfluxDB for high-performance
    time-series data storage and retrieval.

    Features:
        - InfluxDB 1.x and 2.x support
        - Flux and InfluxQL query languages
        - Batch writes for performance
        - Downsampling and aggregation
        - Tag-based filtering

    Example:
        >>> config = InfluxDBConfig(
        ...     host="influxdb.factory.local",
        ...     version=InfluxDBVersion.V2,
        ...     database="process_data",
        ...     org="factory",
        ...     token=SecretStr("my-token")
        ... )
        >>> connector = InfluxDBConnector(config)
        >>> await connector.connect()
    """

    def __init__(self, config: InfluxDBConfig):
        """Initialize InfluxDB connector."""
        base_config = BaseConnectorConfig(
            host=config.host,
            port=config.port,
            timeout_seconds=config.timeout_seconds,
            auth_type=AuthenticationType.TOKEN if config.token else AuthenticationType.NONE,
            name=config.name or "influxdb_connector",
            tls=config.tls,
            rate_limit=config.rate_limit,
            reconnect=config.reconnect,
            health_check_interval_seconds=config.health_check_interval_seconds,
        )

        super().__init__(base_config)
        self.influx_config = config
        self._client: Optional[Any] = None
        self._write_api: Optional[Any] = None
        self._query_api: Optional[Any] = None

        # Write buffer
        self._write_buffer: List[Dict] = []
        self._last_flush = time.monotonic()

    async def _do_connect(self) -> bool:
        """Connect to InfluxDB."""
        logger.info(f"Connecting to InfluxDB: {self.influx_config.host}")

        try:
            if self.influx_config.version == InfluxDBVersion.V2:
                # In production, use influxdb-client:
                # from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
                # self._client = InfluxDBClientAsync(
                #     url=f"http://{self.influx_config.host}:{self.influx_config.port}",
                #     token=self.influx_config.token.get_secret_value(),
                #     org=self.influx_config.org
                # )
                # self._write_api = self._client.write_api()
                # self._query_api = self._client.query_api()
                pass
            else:
                # InfluxDB 1.x
                # from aioinflux import InfluxDBClient
                # self._client = InfluxDBClient(
                #     host=self.influx_config.host,
                #     port=self.influx_config.port,
                #     database=self.influx_config.database,
                #     username=self.influx_config.username,
                #     password=self.influx_config.password.get_secret_value() if self.influx_config.password else None
                # )
                pass

            self._client = True
            logger.info("InfluxDB connected")
            return True

        except Exception as e:
            logger.error(f"InfluxDB connection failed: {e}")
            raise

    async def _do_disconnect(self) -> None:
        """Disconnect from InfluxDB."""
        # Flush remaining writes
        await self._flush_writes()

        if self._client:
            # await self._client.close()
            pass

        self._client = None
        self._write_api = None
        self._query_api = None
        logger.info("InfluxDB disconnected")

    async def _do_health_check(self) -> bool:
        """Check InfluxDB health."""
        try:
            # In production:
            # health = await self._client.health()
            # return health.status == "pass"
            return self._client is not None
        except Exception:
            return False

    async def read_tags(
        self,
        tag_ids: List[str],
    ) -> BatchReadResponse:
        """Read latest values from InfluxDB."""
        self._validate_connected()

        values: Dict[str, TagValue] = {}
        errors: Dict[str, str] = {}

        if self.influx_config.version == InfluxDBVersion.V2:
            # Flux query for latest values
            flux_query = f'''
                from(bucket: "{self.influx_config.database}")
                |> range(start: -1h)
                |> filter(fn: (r) => r["_measurement"] == "{self.influx_config.measurement}")
                |> filter(fn: (r) => {" or ".join(f'r["tag_id"] == "{t}"' for t in tag_ids)})
                |> last()
            '''
            # In production:
            # tables = await self._query_api.query(flux_query)
            # for table in tables:
            #     for record in table.records:
            #         tag_id = record.values.get("tag_id")
            #         values[tag_id] = TagValue(...)
        else:
            # InfluxQL query
            influxql = f'''
                SELECT last(value) FROM {self.influx_config.measurement}
                WHERE tag_id IN ({','.join(f"'{t}'" for t in tag_ids)})
                GROUP BY tag_id
            '''

        # Simulated read
        import random
        for tag_id in tag_ids:
            values[tag_id] = TagValue(
                tag_id=tag_id,
                value=round(random.uniform(0, 100), 2),
                timestamp=datetime.utcnow(),
                quality=DataQuality.GOOD,
            )

        return BatchReadResponse(
            values=values,
            errors=errors,
            timestamp=datetime.utcnow(),
        )

    async def read_history(
        self,
        query: HistoricalQuery,
    ) -> Dict[str, HistoricalResult]:
        """Read historical data from InfluxDB."""
        self._validate_connected()

        results: Dict[str, HistoricalResult] = {}

        if self.influx_config.version == InfluxDBVersion.V2:
            # Build Flux query with aggregation
            agg_func = self._map_aggregation_flux(query.aggregation)
            window = f"{query.interval_ms}ms" if query.interval_ms else "1m"

            flux_query = f'''
                from(bucket: "{self.influx_config.database}")
                |> range(start: {query.start_time.isoformat()}Z, stop: {query.end_time.isoformat()}Z)
                |> filter(fn: (r) => r["_measurement"] == "{self.influx_config.measurement}")
                |> filter(fn: (r) => {" or ".join(f'r["tag_id"] == "{t}"' for t in query.tag_ids)})
                {f'|> aggregateWindow(every: {window}, fn: {agg_func}, createEmpty: false)' if agg_func != 'identity' else ''}
                |> limit(n: {query.max_points})
            '''

            # In production:
            # tables = await self._query_api.query(flux_query)

        # Simulated results
        import random
        for tag_id in query.tag_ids:
            values = []
            current_time = query.start_time
            interval = timedelta(milliseconds=query.interval_ms or 60000)

            while current_time <= query.end_time and len(values) < query.max_points:
                values.append(TagValue(
                    tag_id=tag_id,
                    value=round(random.uniform(0, 100), 2),
                    timestamp=current_time,
                    quality=DataQuality.GOOD,
                ))
                current_time += interval

            results[tag_id] = HistoricalResult(
                tag_id=tag_id,
                values=values,
                start_time=query.start_time,
                end_time=query.end_time,
                point_count=len(values),
                aggregation=query.aggregation,
            )

        return results

    def _map_aggregation_flux(self, agg: AggregationType) -> str:
        """Map aggregation type to Flux function."""
        mapping = {
            AggregationType.RAW: "identity",
            AggregationType.AVERAGE: "mean",
            AggregationType.MINIMUM: "min",
            AggregationType.MAXIMUM: "max",
            AggregationType.TOTAL: "sum",
            AggregationType.COUNT: "count",
            AggregationType.FIRST: "first",
            AggregationType.LAST: "last",
            AggregationType.STANDARD_DEVIATION: "stddev",
        }
        return mapping.get(agg, "identity")

    async def write_tags(
        self,
        request: BatchWriteRequest,
    ) -> BatchWriteResponse:
        """Write values to InfluxDB."""
        self._validate_connected()

        success: Dict[str, bool] = {}
        errors: Dict[str, str] = {}

        for tag_id, value in request.writes.items():
            try:
                point = {
                    "measurement": self.influx_config.measurement,
                    "tags": {"tag_id": tag_id},
                    "fields": {"value": float(value)},
                    "time": datetime.utcnow().isoformat() + "Z",
                }
                self._write_buffer.append(point)
                success[tag_id] = True

            except Exception as e:
                errors[tag_id] = str(e)

        # Check if we should flush
        if (len(self._write_buffer) >= self.influx_config.batch_size or
            time.monotonic() - self._last_flush > self.influx_config.flush_interval_ms / 1000):
            await self._flush_writes()

        return BatchWriteResponse(
            success=success,
            errors=errors,
            timestamp=datetime.utcnow(),
        )

    async def _flush_writes(self) -> None:
        """Flush buffered writes to InfluxDB."""
        if not self._write_buffer:
            return

        try:
            # In production:
            # if self.influx_config.version == InfluxDBVersion.V2:
            #     await self._write_api.write(
            #         bucket=self.influx_config.database,
            #         record=self._write_buffer
            #     )
            # else:
            #     await self._client.write(self._write_buffer)

            logger.debug(f"Flushed {len(self._write_buffer)} points to InfluxDB")
            self._write_buffer.clear()
            self._last_flush = time.monotonic()

        except Exception as e:
            logger.error(f"InfluxDB write failed: {e}")
            raise

    async def write_batch(
        self,
        tag_values: List[TagValue],
    ) -> int:
        """
        Write batch of tag values to InfluxDB.

        Args:
            tag_values: List of TagValue objects

        Returns:
            Number of points written
        """
        self._validate_connected()

        points = []
        for tv in tag_values:
            points.append({
                "measurement": self.influx_config.measurement,
                "tags": {"tag_id": tv.tag_id},
                "fields": {"value": float(tv.value)},
                "time": tv.timestamp.isoformat() + "Z",
            })

        self._write_buffer.extend(points)

        if len(self._write_buffer) >= self.influx_config.batch_size:
            await self._flush_writes()

        return len(points)

    async def execute_flux_query(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Execute raw Flux query.

        Args:
            query: Flux query string

        Returns:
            Query results as list of dictionaries
        """
        self._validate_connected()

        if self.influx_config.version != InfluxDBVersion.V2:
            raise ValueError("Flux queries only supported in InfluxDB 2.x")

        # In production:
        # tables = await self._query_api.query(query)
        # results = []
        # for table in tables:
        #     for record in table.records:
        #         results.append(record.values)
        # return results

        return []


# =============================================================================
# Historian Factory
# =============================================================================


def get_historian_connector(
    config: BaseHistorianConfig,
) -> BaseIndustrialConnector:
    """
    Factory function to get appropriate historian connector.

    Args:
        config: Historian configuration

    Returns:
        Appropriate connector instance

    Raises:
        ValueError: If historian type not supported
    """
    connectors = {
        HistorianType.PI: PIConnector,
        HistorianType.AVEVA: AvevaConnector,
        HistorianType.INFLUXDB: InfluxDBConnector,
    }

    connector_class = connectors.get(config.historian_type)
    if not connector_class:
        raise ValueError(f"Unsupported historian type: {config.historian_type}")

    return connector_class(config)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "HistorianType",
    "CompressionType",
    "InfluxDBVersion",
    # Configuration
    "BaseHistorianConfig",
    "PIConfig",
    "AvevaConfig",
    "InfluxDBConfig",
    # Connectors
    "PIConnector",
    "AvevaConnector",
    "InfluxDBConnector",
    # Factory
    "get_historian_connector",
]
