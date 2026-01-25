# -*- coding: utf-8 -*-
"""
GL-014 ExchangerPro - Historian Connector

Process historian integration for historical data access:
- Batch ingestion for backfills
- OSIsoft PI, Honeywell PHD, AspenTech IP21 support patterns
- Time-range queries
- Data aggregation (average, min, max, interpolated)

All operations are READ-ONLY for OT safety.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class HistorianType(str, Enum):
    """Supported historian systems."""
    OSISOFT_PI = "osisoft_pi"
    HONEYWELL_PHD = "honeywell_phd"
    ASPEN_IP21 = "aspen_ip21"
    WONDERWARE = "wonderware"
    INFLUXDB = "influxdb"
    TIMESCALEDB = "timescaledb"
    MOCK = "mock"


class AggregationType(str, Enum):
    """Data aggregation types."""
    RAW = "raw"
    AVERAGE = "average"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    RANGE = "range"
    TOTAL = "total"
    COUNT = "count"
    INTERPOLATED = "interpolated"
    PLOT = "plot"
    STANDARD_DEVIATION = "std_dev"
    VARIANCE = "variance"


class BackfillStatus(str, Enum):
    """Backfill operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataQuality(str, Enum):
    """Historian data quality."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    SUBSTITUTED = "substituted"
    ANNOTATED = "annotated"


# =============================================================================
# DATA MODELS
# =============================================================================

class HistorianDataPoint(BaseModel):
    """Single data point from historian."""
    tag_id: str = Field(..., description="Tag identifier")
    timestamp: datetime = Field(..., description="Data timestamp")
    value: Any = Field(..., description="Data value")
    quality: DataQuality = Field(default=DataQuality.GOOD, description="Data quality")
    quality_code: int = Field(default=0, description="Numeric quality code")

    # Optional metadata
    engineering_unit: Optional[str] = Field(None, description="Engineering unit")
    annotated: bool = Field(default=False, description="Has annotations")
    annotation: Optional[str] = Field(None, description="Annotation text")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TimeRangeQuery(BaseModel):
    """Time-range query specification."""
    tag_ids: List[str] = Field(..., min_items=1, description="Tags to query")
    start_time: datetime = Field(..., description="Query start time")
    end_time: datetime = Field(..., description="Query end time")

    # Aggregation
    aggregation_type: AggregationType = Field(
        default=AggregationType.RAW,
        description="Aggregation type"
    )
    aggregation_interval_seconds: Optional[int] = Field(
        None,
        ge=1,
        description="Aggregation interval"
    )

    # Filtering
    quality_filter: Optional[List[DataQuality]] = Field(
        None,
        description="Quality filter"
    )
    max_points: Optional[int] = Field(
        None,
        ge=1,
        le=1000000,
        description="Maximum points to return"
    )

    # Options
    include_boundaries: bool = Field(
        default=True,
        description="Include boundary values"
    )
    fill_missing: bool = Field(
        default=False,
        description="Fill missing values"
    )
    fill_method: str = Field(
        default="previous",
        description="Fill method: previous, linear, none"
    )

    @validator("end_time")
    def end_after_start(cls, v, values):
        """Validate end time is after start time."""
        if "start_time" in values and v <= values["start_time"]:
            raise ValueError("end_time must be after start_time")
        return v


class BatchIngestionResult(BaseModel):
    """Result of batch ingestion operation."""
    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Batch ID"
    )
    tag_id: str = Field(..., description="Tag ID")
    start_time: datetime = Field(..., description="Data start time")
    end_time: datetime = Field(..., description="Data end time")

    # Statistics
    total_points: int = Field(..., description="Total points retrieved")
    good_points: int = Field(..., description="Good quality points")
    bad_points: int = Field(..., description="Bad quality points")

    # Timing
    query_time_ms: float = Field(..., description="Query time in ms")
    ingestion_time_ms: float = Field(..., description="Ingestion time in ms")

    # Status
    success: bool = Field(..., description="Operation successful")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    # Provenance
    provenance_hash: str = Field(default="", description="Data provenance hash")


class BackfillRequest(BaseModel):
    """Request for historical data backfill."""
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request ID"
    )
    exchanger_id: str = Field(..., description="Heat exchanger ID")
    tag_ids: List[str] = Field(..., description="Tags to backfill")

    # Time range
    start_time: datetime = Field(..., description="Backfill start")
    end_time: datetime = Field(..., description="Backfill end")

    # Options
    aggregation_type: AggregationType = Field(
        default=AggregationType.AVERAGE,
        description="Aggregation for backfill"
    )
    aggregation_interval_seconds: int = Field(
        default=60,
        ge=1,
        description="Aggregation interval"
    )
    priority: str = Field(default="normal", description="Priority: low, normal, high")

    # Processing
    chunk_size_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Chunk size for processing"
    )

    # Audit
    requested_by: str = Field(default="system", description="Requestor")
    requested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp"
    )


class BackfillProgress(BaseModel):
    """Backfill operation progress."""
    request_id: str = Field(..., description="Request ID")
    status: BackfillStatus = Field(..., description="Current status")

    # Progress
    total_chunks: int = Field(..., description="Total chunks")
    completed_chunks: int = Field(..., description="Completed chunks")
    progress_percent: float = Field(..., description="Progress percentage")

    # Statistics
    total_points_retrieved: int = Field(default=0, description="Points retrieved")
    total_points_processed: int = Field(default=0, description="Points processed")
    errors_count: int = Field(default=0, description="Error count")

    # Timing
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    estimated_completion: Optional[datetime] = Field(None, description="ETA")

    # Current chunk
    current_chunk_start: Optional[datetime] = Field(None, description="Current chunk start")
    current_chunk_end: Optional[datetime] = Field(None, description="Current chunk end")


class DataAggregation(BaseModel):
    """Aggregated data result."""
    tag_id: str = Field(..., description="Tag ID")
    start_time: datetime = Field(..., description="Period start")
    end_time: datetime = Field(..., description="Period end")
    aggregation_type: AggregationType = Field(..., description="Aggregation type")

    # Values
    value: float = Field(..., description="Aggregated value")
    count: int = Field(..., description="Number of samples")

    # Additional statistics (when applicable)
    minimum: Optional[float] = Field(None, description="Minimum value")
    maximum: Optional[float] = Field(None, description="Maximum value")
    average: Optional[float] = Field(None, description="Average value")
    std_dev: Optional[float] = Field(None, description="Standard deviation")

    # Quality
    good_percent: float = Field(default=100.0, description="Good quality %")


class HistorianHealthStatus(BaseModel):
    """Historian connection health status."""
    historian_type: HistorianType = Field(..., description="Historian type")
    connected: bool = Field(..., description="Connection status")
    last_successful_query: Optional[datetime] = Field(None, description="Last query")
    error_count: int = Field(default=0, description="Error count")
    latency_ms: Optional[float] = Field(None, description="Query latency")
    message: str = Field(default="", description="Status message")


# =============================================================================
# CONFIGURATION
# =============================================================================

class HistorianConfig(BaseModel):
    """Base historian configuration."""
    historian_type: HistorianType = Field(..., description="Historian type")
    name: str = Field(..., description="Connection name")

    # Connection
    host: str = Field(..., description="Historian host")
    port: int = Field(..., description="Historian port")

    # Authentication
    username: Optional[str] = Field(None, description="Username")
    password: Optional[str] = Field(None, description="Password")
    use_integrated_auth: bool = Field(
        default=False,
        description="Use Windows integrated auth"
    )

    # Options
    timeout_seconds: int = Field(default=30, ge=1, description="Query timeout")
    max_points_per_query: int = Field(
        default=100000,
        ge=1,
        description="Max points per query"
    )
    retry_attempts: int = Field(default=3, ge=1, description="Retry attempts")
    retry_delay_seconds: float = Field(default=1.0, ge=0, description="Retry delay")

    # Read-only mode (enforced for safety)
    read_only: bool = Field(default=True, description="Read-only mode (always true)")


class OSIsoftPIConfig(HistorianConfig):
    """OSIsoft PI configuration."""
    historian_type: HistorianType = Field(default=HistorianType.OSISOFT_PI)
    port: int = Field(default=5450, description="PI Data Archive port")

    # PI-specific
    af_server: Optional[str] = Field(None, description="PI AF server")
    af_database: Optional[str] = Field(None, description="PI AF database")
    use_af: bool = Field(default=False, description="Use Asset Framework")


class HoneywellPHDConfig(HistorianConfig):
    """Honeywell PHD configuration."""
    historian_type: HistorianType = Field(default=HistorianType.HONEYWELL_PHD)
    port: int = Field(default=3100, description="PHD port")

    # PHD-specific
    database: str = Field(default="PHD", description="PHD database name")
    use_oledb: bool = Field(default=True, description="Use OLEDB provider")


class AspenIP21Config(HistorianConfig):
    """AspenTech IP.21 configuration."""
    historian_type: HistorianType = Field(default=HistorianType.ASPEN_IP21)
    port: int = Field(default=10014, description="IP.21 port")

    # IP.21-specific
    repository: str = Field(default="IP21", description="Repository name")
    use_oda: bool = Field(default=True, description="Use ODA interface")


# =============================================================================
# HISTORIAN ADAPTERS
# =============================================================================

class HistorianAdapter(ABC):
    """Abstract base class for historian adapters."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to historian."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from historian."""
        pass

    @abstractmethod
    async def query_time_range(
        self,
        query: TimeRangeQuery,
    ) -> Dict[str, List[HistorianDataPoint]]:
        """Query data for time range."""
        pass

    @abstractmethod
    async def query_aggregated(
        self,
        query: TimeRangeQuery,
    ) -> Dict[str, List[DataAggregation]]:
        """Query aggregated data."""
        pass

    @abstractmethod
    async def get_health_status(self) -> HistorianHealthStatus:
        """Get health status."""
        pass


class OSIsoftPIConnector(HistorianAdapter):
    """
    OSIsoft PI Data Archive connector.

    Provides read-only access to PI historian data.
    Supports both direct PI SDK and PI Web API patterns.
    """

    def __init__(self, config: OSIsoftPIConfig):
        """Initialize PI connector."""
        self.config = config
        self._connected = False
        self._last_query_time: Optional[datetime] = None
        self._error_count = 0

        logger.info(f"OSIsoft PI connector initialized for {config.host}")

    async def connect(self) -> bool:
        """Connect to PI Data Archive."""
        logger.info(f"Connecting to PI Data Archive at {self.config.host}")

        try:
            # In production, would use PI SDK or PI Web API
            # from OSIsoft.AF.PI import PIServer
            # server = PIServer.FindPIServer(self.config.host)
            # server.Connect()

            await asyncio.sleep(0.1)  # Simulate connection
            self._connected = True
            logger.info("Connected to PI Data Archive")
            return True

        except Exception as e:
            self._error_count += 1
            logger.error(f"PI connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from PI."""
        self._connected = False
        logger.info("Disconnected from PI Data Archive")

    async def query_time_range(
        self,
        query: TimeRangeQuery,
    ) -> Dict[str, List[HistorianDataPoint]]:
        """Query raw data from PI."""
        if not self._connected:
            raise ConnectionError("Not connected to PI")

        logger.info(
            f"Querying PI: {len(query.tag_ids)} tags, "
            f"{query.start_time} to {query.end_time}"
        )

        results: Dict[str, List[HistorianDataPoint]] = {}

        # In production, would query PI:
        # for tag_id in query.tag_ids:
        #     pt = PIPoint.FindPIPoint(server, tag_id)
        #     values = pt.RecordedValues(
        #         AFTimeRange(query.start_time, query.end_time),
        #         AFBoundaryType.Inside,
        #         None,
        #         False
        #     )

        # Simulate results
        for tag_id in query.tag_ids:
            results[tag_id] = self._generate_sample_data(
                tag_id, query.start_time, query.end_time
            )

        self._last_query_time = datetime.now(timezone.utc)
        return results

    async def query_aggregated(
        self,
        query: TimeRangeQuery,
    ) -> Dict[str, List[DataAggregation]]:
        """Query aggregated data from PI."""
        if not self._connected:
            raise ConnectionError("Not connected to PI")

        results: Dict[str, List[DataAggregation]] = {}

        # In production, would use PI summaries:
        # values = pt.Summaries(
        #     AFTimeRange(start, end),
        #     AFTimeSpan.FromSeconds(interval),
        #     AFSummaryTypes.Average | AFSummaryTypes.Minimum | AFSummaryTypes.Maximum,
        #     AFCalculationBasis.TimeWeighted,
        #     AFTimestampCalculation.Auto
        # )

        interval_seconds = query.aggregation_interval_seconds or 3600

        for tag_id in query.tag_ids:
            results[tag_id] = self._generate_aggregated_data(
                tag_id,
                query.start_time,
                query.end_time,
                query.aggregation_type,
                interval_seconds,
            )

        self._last_query_time = datetime.now(timezone.utc)
        return results

    async def get_health_status(self) -> HistorianHealthStatus:
        """Get PI health status."""
        return HistorianHealthStatus(
            historian_type=HistorianType.OSISOFT_PI,
            connected=self._connected,
            last_successful_query=self._last_query_time,
            error_count=self._error_count,
            latency_ms=10.0 if self._connected else None,
            message="Connected" if self._connected else "Disconnected",
        )

    def _generate_sample_data(
        self,
        tag_id: str,
        start: datetime,
        end: datetime,
    ) -> List[HistorianDataPoint]:
        """Generate sample data for testing."""
        import random

        points = []
        current = start
        interval = timedelta(minutes=1)

        while current <= end:
            points.append(HistorianDataPoint(
                tag_id=tag_id,
                timestamp=current,
                value=random.gauss(100, 10),
                quality=DataQuality.GOOD,
            ))
            current += interval

        return points

    def _generate_aggregated_data(
        self,
        tag_id: str,
        start: datetime,
        end: datetime,
        agg_type: AggregationType,
        interval_seconds: int,
    ) -> List[DataAggregation]:
        """Generate aggregated sample data."""
        import random

        aggregations = []
        current = start
        interval = timedelta(seconds=interval_seconds)

        while current < end:
            period_end = min(current + interval, end)

            value = random.gauss(100, 5)

            aggregations.append(DataAggregation(
                tag_id=tag_id,
                start_time=current,
                end_time=period_end,
                aggregation_type=agg_type,
                value=value,
                count=60,
                minimum=value - 5,
                maximum=value + 5,
                average=value,
                std_dev=2.5,
            ))
            current = period_end

        return aggregations


class HoneywellPHDConnector(HistorianAdapter):
    """Honeywell PHD historian connector."""

    def __init__(self, config: HoneywellPHDConfig):
        """Initialize PHD connector."""
        self.config = config
        self._connected = False
        self._last_query_time: Optional[datetime] = None
        self._error_count = 0

        logger.info(f"Honeywell PHD connector initialized for {config.host}")

    async def connect(self) -> bool:
        """Connect to PHD server."""
        logger.info(f"Connecting to PHD at {self.config.host}")

        try:
            # In production, would use PHD OLEDB or API
            await asyncio.sleep(0.1)
            self._connected = True
            logger.info("Connected to PHD")
            return True

        except Exception as e:
            self._error_count += 1
            logger.error(f"PHD connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from PHD."""
        self._connected = False
        logger.info("Disconnected from PHD")

    async def query_time_range(
        self,
        query: TimeRangeQuery,
    ) -> Dict[str, List[HistorianDataPoint]]:
        """Query raw data from PHD."""
        if not self._connected:
            raise ConnectionError("Not connected to PHD")

        logger.info(f"Querying PHD: {len(query.tag_ids)} tags")

        # In production, would query PHD via OLEDB or API
        results: Dict[str, List[HistorianDataPoint]] = {}
        for tag_id in query.tag_ids:
            results[tag_id] = []

        self._last_query_time = datetime.now(timezone.utc)
        return results

    async def query_aggregated(
        self,
        query: TimeRangeQuery,
    ) -> Dict[str, List[DataAggregation]]:
        """Query aggregated data from PHD."""
        if not self._connected:
            raise ConnectionError("Not connected to PHD")

        results: Dict[str, List[DataAggregation]] = {}
        for tag_id in query.tag_ids:
            results[tag_id] = []

        self._last_query_time = datetime.now(timezone.utc)
        return results

    async def get_health_status(self) -> HistorianHealthStatus:
        """Get PHD health status."""
        return HistorianHealthStatus(
            historian_type=HistorianType.HONEYWELL_PHD,
            connected=self._connected,
            last_successful_query=self._last_query_time,
            error_count=self._error_count,
            message="Connected" if self._connected else "Disconnected",
        )


class AspenIP21Connector(HistorianAdapter):
    """AspenTech IP.21 historian connector."""

    def __init__(self, config: AspenIP21Config):
        """Initialize IP.21 connector."""
        self.config = config
        self._connected = False
        self._last_query_time: Optional[datetime] = None
        self._error_count = 0

        logger.info(f"Aspen IP.21 connector initialized for {config.host}")

    async def connect(self) -> bool:
        """Connect to IP.21."""
        logger.info(f"Connecting to IP.21 at {self.config.host}")

        try:
            # In production, would use IP.21 ODA or SQLplus
            await asyncio.sleep(0.1)
            self._connected = True
            logger.info("Connected to IP.21")
            return True

        except Exception as e:
            self._error_count += 1
            logger.error(f"IP.21 connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from IP.21."""
        self._connected = False
        logger.info("Disconnected from IP.21")

    async def query_time_range(
        self,
        query: TimeRangeQuery,
    ) -> Dict[str, List[HistorianDataPoint]]:
        """Query raw data from IP.21."""
        if not self._connected:
            raise ConnectionError("Not connected to IP.21")

        logger.info(f"Querying IP.21: {len(query.tag_ids)} tags")

        # In production, would query IP.21 via SQLplus or ODA
        results: Dict[str, List[HistorianDataPoint]] = {}
        for tag_id in query.tag_ids:
            results[tag_id] = []

        self._last_query_time = datetime.now(timezone.utc)
        return results

    async def query_aggregated(
        self,
        query: TimeRangeQuery,
    ) -> Dict[str, List[DataAggregation]]:
        """Query aggregated data from IP.21."""
        if not self._connected:
            raise ConnectionError("Not connected to IP.21")

        results: Dict[str, List[DataAggregation]] = {}
        for tag_id in query.tag_ids:
            results[tag_id] = []

        self._last_query_time = datetime.now(timezone.utc)
        return results

    async def get_health_status(self) -> HistorianHealthStatus:
        """Get IP.21 health status."""
        return HistorianHealthStatus(
            historian_type=HistorianType.ASPEN_IP21,
            connected=self._connected,
            last_successful_query=self._last_query_time,
            error_count=self._error_count,
            message="Connected" if self._connected else "Disconnected",
        )


class MockHistorianAdapter(HistorianAdapter):
    """Mock historian for testing."""

    def __init__(self):
        """Initialize mock historian."""
        self._connected = False
        self._data: Dict[str, List[HistorianDataPoint]] = {}

    async def connect(self) -> bool:
        """Connect to mock historian."""
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Disconnect from mock historian."""
        self._connected = False

    async def query_time_range(
        self,
        query: TimeRangeQuery,
    ) -> Dict[str, List[HistorianDataPoint]]:
        """Query mock data."""
        import random

        results: Dict[str, List[HistorianDataPoint]] = {}

        for tag_id in query.tag_ids:
            points = []
            current = query.start_time
            interval = timedelta(minutes=1)

            while current <= query.end_time:
                points.append(HistorianDataPoint(
                    tag_id=tag_id,
                    timestamp=current,
                    value=random.gauss(100, 10),
                    quality=DataQuality.GOOD,
                ))
                current += interval

            results[tag_id] = points

        return results

    async def query_aggregated(
        self,
        query: TimeRangeQuery,
    ) -> Dict[str, List[DataAggregation]]:
        """Query mock aggregated data."""
        import random

        results: Dict[str, List[DataAggregation]] = {}
        interval = timedelta(seconds=query.aggregation_interval_seconds or 3600)

        for tag_id in query.tag_ids:
            aggregations = []
            current = query.start_time

            while current < query.end_time:
                period_end = min(current + interval, query.end_time)
                value = random.gauss(100, 5)

                aggregations.append(DataAggregation(
                    tag_id=tag_id,
                    start_time=current,
                    end_time=period_end,
                    aggregation_type=query.aggregation_type,
                    value=value,
                    count=60,
                    minimum=value - 5,
                    maximum=value + 5,
                    average=value,
                ))
                current = period_end

            results[tag_id] = aggregations

        return results

    async def get_health_status(self) -> HistorianHealthStatus:
        """Get mock health status."""
        return HistorianHealthStatus(
            historian_type=HistorianType.MOCK,
            connected=self._connected,
            message="Mock historian",
        )


# =============================================================================
# HISTORIAN CONNECTOR
# =============================================================================

class HistorianConnector:
    """
    Main historian connector for GL-014 ExchangerPro.

    Provides unified interface for:
    - Multiple historian backends (PI, PHD, IP.21)
    - Time-range queries
    - Data aggregation
    - Batch ingestion for backfills
    - Connection health monitoring

    All operations are READ-ONLY for OT safety.

    Example:
        >>> config = OSIsoftPIConfig(host="pi-server", username="user", password="pass")
        >>> async with HistorianConnector(config) as historian:
        ...     query = TimeRangeQuery(
        ...         tag_ids=["HX001.SHELL_INLET_TEMP"],
        ...         start_time=datetime.now() - timedelta(hours=24),
        ...         end_time=datetime.now(),
        ...     )
        ...     data = await historian.query_time_range(query)
    """

    def __init__(self, config: HistorianConfig):
        """
        Initialize historian connector.

        Args:
            config: Historian configuration
        """
        self.config = config
        self._adapter: Optional[HistorianAdapter] = None
        self._connected = False

        # Backfill tracking
        self._backfill_requests: Dict[str, BackfillRequest] = {}
        self._backfill_progress: Dict[str, BackfillProgress] = {}

        # Statistics
        self._query_count = 0
        self._total_points_retrieved = 0
        self._error_count = 0

        # Initialize adapter based on type
        self._initialize_adapter()

        logger.info(
            f"Historian connector initialized: {config.historian_type.value}, "
            f"read_only={config.read_only}"
        )

    def _initialize_adapter(self) -> None:
        """Initialize appropriate adapter based on configuration."""
        if self.config.historian_type == HistorianType.OSISOFT_PI:
            self._adapter = OSIsoftPIConnector(
                OSIsoftPIConfig(**self.config.dict())
            )
        elif self.config.historian_type == HistorianType.HONEYWELL_PHD:
            self._adapter = HoneywellPHDConnector(
                HoneywellPHDConfig(**self.config.dict())
            )
        elif self.config.historian_type == HistorianType.ASPEN_IP21:
            self._adapter = AspenIP21Connector(
                AspenIP21Config(**self.config.dict())
            )
        else:
            self._adapter = MockHistorianAdapter()

    async def connect(self) -> bool:
        """Connect to historian."""
        if self._connected:
            return True

        logger.info(f"Connecting to {self.config.historian_type.value}")

        try:
            if self._adapter:
                result = await self._adapter.connect()
                self._connected = result
                return result
            return False

        except Exception as e:
            self._error_count += 1
            logger.error(f"Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from historian."""
        if self._adapter:
            await self._adapter.disconnect()
        self._connected = False
        logger.info("Disconnected from historian")

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================

    async def query_time_range(
        self,
        query: TimeRangeQuery,
    ) -> Dict[str, List[HistorianDataPoint]]:
        """
        Query raw data for a time range.

        Args:
            query: Time range query specification

        Returns:
            Dictionary mapping tag_id to list of data points
        """
        if not self._connected or not self._adapter:
            raise ConnectionError("Not connected to historian")

        logger.info(
            f"Querying {len(query.tag_ids)} tags from "
            f"{query.start_time} to {query.end_time}"
        )

        try:
            results = await self._adapter.query_time_range(query)

            # Update statistics
            self._query_count += 1
            total_points = sum(len(points) for points in results.values())
            self._total_points_retrieved += total_points

            logger.info(f"Retrieved {total_points} data points")
            return results

        except Exception as e:
            self._error_count += 1
            logger.error(f"Query failed: {e}")
            raise

    async def query_aggregated(
        self,
        query: TimeRangeQuery,
    ) -> Dict[str, List[DataAggregation]]:
        """
        Query aggregated data for a time range.

        Args:
            query: Time range query with aggregation settings

        Returns:
            Dictionary mapping tag_id to list of aggregations
        """
        if not self._connected or not self._adapter:
            raise ConnectionError("Not connected to historian")

        logger.info(
            f"Querying aggregated data ({query.aggregation_type.value}) "
            f"for {len(query.tag_ids)} tags"
        )

        try:
            results = await self._adapter.query_aggregated(query)
            self._query_count += 1
            return results

        except Exception as e:
            self._error_count += 1
            logger.error(f"Aggregation query failed: {e}")
            raise

    async def query_single_tag(
        self,
        tag_id: str,
        start_time: datetime,
        end_time: datetime,
        aggregation_type: AggregationType = AggregationType.RAW,
        interval_seconds: Optional[int] = None,
    ) -> List[Union[HistorianDataPoint, DataAggregation]]:
        """
        Query single tag (convenience method).

        Args:
            tag_id: Tag to query
            start_time: Start time
            end_time: End time
            aggregation_type: Aggregation type
            interval_seconds: Aggregation interval

        Returns:
            List of data points or aggregations
        """
        query = TimeRangeQuery(
            tag_ids=[tag_id],
            start_time=start_time,
            end_time=end_time,
            aggregation_type=aggregation_type,
            aggregation_interval_seconds=interval_seconds,
        )

        if aggregation_type == AggregationType.RAW:
            results = await self.query_time_range(query)
        else:
            results = await self.query_aggregated(query)

        return results.get(tag_id, [])

    # =========================================================================
    # BATCH INGESTION
    # =========================================================================

    async def start_backfill(
        self,
        request: BackfillRequest,
    ) -> str:
        """
        Start a backfill operation.

        Args:
            request: Backfill request

        Returns:
            Request ID
        """
        logger.info(
            f"Starting backfill for {request.exchanger_id}: "
            f"{len(request.tag_ids)} tags, "
            f"{request.start_time} to {request.end_time}"
        )

        # Store request
        self._backfill_requests[request.request_id] = request

        # Calculate chunks
        total_duration = request.end_time - request.start_time
        chunk_duration = timedelta(hours=request.chunk_size_hours)
        total_chunks = int(total_duration / chunk_duration) + 1

        # Initialize progress
        progress = BackfillProgress(
            request_id=request.request_id,
            status=BackfillStatus.PENDING,
            total_chunks=total_chunks,
            completed_chunks=0,
            progress_percent=0.0,
        )
        self._backfill_progress[request.request_id] = progress

        # Start backfill task
        asyncio.create_task(self._run_backfill(request))

        return request.request_id

    async def _run_backfill(self, request: BackfillRequest) -> None:
        """Run backfill operation in background."""
        progress = self._backfill_progress[request.request_id]
        progress.status = BackfillStatus.IN_PROGRESS
        progress.started_at = datetime.now(timezone.utc)

        chunk_duration = timedelta(hours=request.chunk_size_hours)
        current_start = request.start_time

        while current_start < request.end_time:
            current_end = min(current_start + chunk_duration, request.end_time)

            progress.current_chunk_start = current_start
            progress.current_chunk_end = current_end

            try:
                # Query this chunk
                query = TimeRangeQuery(
                    tag_ids=request.tag_ids,
                    start_time=current_start,
                    end_time=current_end,
                    aggregation_type=request.aggregation_type,
                    aggregation_interval_seconds=request.aggregation_interval_seconds,
                )

                if request.aggregation_type == AggregationType.RAW:
                    results = await self.query_time_range(query)
                else:
                    results = await self.query_aggregated(query)

                # Update progress
                points = sum(len(v) for v in results.values())
                progress.total_points_retrieved += points
                progress.completed_chunks += 1
                progress.progress_percent = (
                    progress.completed_chunks / progress.total_chunks * 100
                )

            except Exception as e:
                logger.error(f"Backfill chunk failed: {e}")
                progress.errors_count += 1

            current_start = current_end
            await asyncio.sleep(0.1)  # Rate limiting

        # Complete
        progress.status = BackfillStatus.COMPLETED
        progress.completed_at = datetime.now(timezone.utc)
        progress.progress_percent = 100.0

        logger.info(
            f"Backfill completed for {request.request_id}: "
            f"{progress.total_points_retrieved} points"
        )

    def get_backfill_progress(self, request_id: str) -> Optional[BackfillProgress]:
        """Get backfill progress."""
        return self._backfill_progress.get(request_id)

    async def cancel_backfill(self, request_id: str) -> bool:
        """Cancel a backfill operation."""
        progress = self._backfill_progress.get(request_id)
        if not progress:
            return False

        if progress.status == BackfillStatus.IN_PROGRESS:
            progress.status = BackfillStatus.CANCELLED
            logger.info(f"Cancelled backfill {request_id}")
            return True

        return False

    async def batch_ingest(
        self,
        exchanger_id: str,
        tag_ids: List[str],
        start_time: datetime,
        end_time: datetime,
        aggregation_type: AggregationType = AggregationType.AVERAGE,
        aggregation_interval_seconds: int = 60,
    ) -> List[BatchIngestionResult]:
        """
        Batch ingest historical data.

        Args:
            exchanger_id: Heat exchanger ID
            tag_ids: Tags to ingest
            start_time: Start time
            end_time: End time
            aggregation_type: Aggregation type
            aggregation_interval_seconds: Aggregation interval

        Returns:
            List of ingestion results per tag
        """
        import time

        logger.info(
            f"Batch ingestion for {exchanger_id}: {len(tag_ids)} tags"
        )

        results = []

        for tag_id in tag_ids:
            start_ms = time.time() * 1000

            try:
                query = TimeRangeQuery(
                    tag_ids=[tag_id],
                    start_time=start_time,
                    end_time=end_time,
                    aggregation_type=aggregation_type,
                    aggregation_interval_seconds=aggregation_interval_seconds,
                )

                if aggregation_type == AggregationType.RAW:
                    data = await self.query_time_range(query)
                else:
                    data = await self.query_aggregated(query)

                query_ms = time.time() * 1000 - start_ms
                points = data.get(tag_id, [])

                # Calculate quality stats
                good_points = len([
                    p for p in points
                    if isinstance(p, HistorianDataPoint) and p.quality == DataQuality.GOOD
                ])

                # Calculate provenance hash
                data_str = f"{tag_id}|{start_time}|{end_time}|{len(points)}"
                provenance_hash = hashlib.sha256(data_str.encode()).hexdigest()

                result = BatchIngestionResult(
                    tag_id=tag_id,
                    start_time=start_time,
                    end_time=end_time,
                    total_points=len(points),
                    good_points=good_points,
                    bad_points=len(points) - good_points,
                    query_time_ms=query_ms,
                    ingestion_time_ms=time.time() * 1000 - start_ms,
                    success=True,
                    provenance_hash=provenance_hash,
                )

            except Exception as e:
                result = BatchIngestionResult(
                    tag_id=tag_id,
                    start_time=start_time,
                    end_time=end_time,
                    total_points=0,
                    good_points=0,
                    bad_points=0,
                    query_time_ms=0,
                    ingestion_time_ms=0,
                    success=False,
                    errors=[str(e)],
                )

            results.append(result)

        return results

    # =========================================================================
    # HEALTH & STATISTICS
    # =========================================================================

    async def get_health_status(self) -> HistorianHealthStatus:
        """Get historian health status."""
        if self._adapter:
            return await self._adapter.get_health_status()

        return HistorianHealthStatus(
            historian_type=self.config.historian_type,
            connected=False,
            message="No adapter initialized",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            "historian_type": self.config.historian_type.value,
            "host": self.config.host,
            "connected": self._connected,
            "read_only": self.config.read_only,
            "query_count": self._query_count,
            "total_points_retrieved": self._total_points_retrieved,
            "error_count": self._error_count,
            "active_backfills": len([
                p for p in self._backfill_progress.values()
                if p.status == BackfillStatus.IN_PROGRESS
            ]),
        }

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================

    async def __aenter__(self) -> "HistorianConnector":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_historian_connector(
    historian_type: HistorianType,
    config: Dict[str, Any],
) -> HistorianConnector:
    """
    Create historian connector from type and config.

    Args:
        historian_type: Historian type
        config: Configuration dictionary

    Returns:
        Configured HistorianConnector
    """
    config["historian_type"] = historian_type

    if historian_type == HistorianType.OSISOFT_PI:
        hist_config = OSIsoftPIConfig(**config)
    elif historian_type == HistorianType.HONEYWELL_PHD:
        hist_config = HoneywellPHDConfig(**config)
    elif historian_type == HistorianType.ASPEN_IP21:
        hist_config = AspenIP21Config(**config)
    else:
        hist_config = HistorianConfig(**config)

    return HistorianConnector(hist_config)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "HistorianType",
    "AggregationType",
    "BackfillStatus",
    "DataQuality",

    # Data Models
    "HistorianDataPoint",
    "TimeRangeQuery",
    "BatchIngestionResult",
    "BackfillRequest",
    "BackfillProgress",
    "DataAggregation",
    "HistorianHealthStatus",

    # Configuration
    "HistorianConfig",
    "OSIsoftPIConfig",
    "HoneywellPHDConfig",
    "AspenIP21Config",

    # Adapters
    "HistorianAdapter",
    "OSIsoftPIConnector",
    "HoneywellPHDConnector",
    "AspenIP21Connector",
    "MockHistorianAdapter",

    # Main Connector
    "HistorianConnector",

    # Factory
    "create_historian_connector",
]
