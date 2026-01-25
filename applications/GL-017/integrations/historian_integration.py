"""
GL-017 CONDENSYNC Historian Integration Module

PI System / OSIsoft integration for historical data retrieval,
trend analysis, performance baseline comparison, and batch export.

Features:
- PI Web API integration
- Historical data retrieval with interpolation
- Trend analysis queries
- Performance baseline comparison
- Batch data export to CSV/JSON

Author: GreenLang AI Platform
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import json
import csv
import io

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class HistorianError(Exception):
    """Base exception for historian integration."""
    pass


class HistorianConnectionError(HistorianError):
    """Raised when historian connection fails."""
    pass


class HistorianQueryError(HistorianError):
    """Raised when query fails."""
    pass


class HistorianDataError(HistorianError):
    """Raised when data processing fails."""
    pass


# =============================================================================
# Enums and Constants
# =============================================================================

class InterpolationType(Enum):
    """Data interpolation types."""
    NONE = "none"
    LINEAR = "linear"
    STEP = "step"
    STEP_BEFORE = "step_before"
    STEP_AFTER = "step_after"


class AggregationType(Enum):
    """Data aggregation types."""
    AVERAGE = "average"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    TOTAL = "total"
    COUNT = "count"
    RANGE = "range"
    STD_DEV = "std_dev"
    FIRST = "first"
    LAST = "last"


class ExportFormat(Enum):
    """Export file formats."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"


class DataQuality(Enum):
    """PI data quality indicators."""
    GOOD = "good"
    BAD = "bad"
    QUESTIONABLE = "questionable"
    SUBSTITUTED = "substituted"
    NO_DATA = "no_data"


# =============================================================================
# Data Models
# =============================================================================

class HistorianConfig(BaseModel):
    """Configuration for historian integration."""

    base_url: str = Field(
        ...,
        description="PI Web API base URL"
    )
    api_version: str = Field(
        default="1.14.0",
        description="PI Web API version"
    )
    username: Optional[str] = Field(
        default=None,
        description="Username for authentication"
    )
    password: Optional[str] = Field(
        default=None,
        description="Password for authentication (from vault)"
    )
    verify_ssl: bool = Field(
        default=True,
        description="Verify SSL certificates"
    )
    connection_timeout: float = Field(
        default=10.0,
        description="Connection timeout in seconds"
    )
    request_timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds"
    )
    max_points_per_request: int = Field(
        default=10000,
        description="Maximum data points per request"
    )
    default_interpolation: InterpolationType = Field(
        default=InterpolationType.LINEAR,
        description="Default interpolation type"
    )
    pi_data_archive: str = Field(
        default="PI-Server",
        description="PI Data Archive name"
    )
    af_database: str = Field(
        default="Condenser",
        description="AF Database name"
    )

    @validator("base_url")
    def validate_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("Base URL must start with http:// or https://")
        return v.rstrip("/")


@dataclass
class TrendQuery:
    """Query parameters for trend data retrieval."""

    tag_names: List[str]
    start_time: datetime
    end_time: datetime
    interval: Optional[timedelta] = None  # For interpolated data
    aggregation: Optional[AggregationType] = None
    interpolation: InterpolationType = InterpolationType.LINEAR
    include_quality: bool = True
    max_points: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tag_names": self.tag_names,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "interval": str(self.interval) if self.interval else None,
            "aggregation": self.aggregation.value if self.aggregation else None,
            "interpolation": self.interpolation.value,
            "include_quality": self.include_quality,
            "max_points": self.max_points
        }


@dataclass
class HistoricalDataPoint:
    """Single historical data point."""

    tag_name: str
    timestamp: datetime
    value: Any
    quality: DataQuality
    units: Optional[str] = None
    annotations: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tag_name": self.tag_name,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "quality": self.quality.value,
            "units": self.units,
            "annotations": self.annotations
        }


@dataclass
class PerformanceBaseline:
    """Performance baseline data for comparison."""

    baseline_id: str
    name: str
    description: str
    created_date: datetime
    valid_from: datetime
    valid_to: datetime

    # Baseline metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Operating conditions
    operating_conditions: Dict[str, Any] = field(default_factory=dict)

    # Statistical bounds
    upper_bounds: Dict[str, float] = field(default_factory=dict)
    lower_bounds: Dict[str, float] = field(default_factory=dict)

    def is_valid(self, timestamp: datetime) -> bool:
        """Check if baseline is valid for given timestamp."""
        return self.valid_from <= timestamp <= self.valid_to

    def compare(
        self,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """Compare current metrics against baseline."""
        comparison = {}

        for metric_name, baseline_value in self.metrics.items():
            if metric_name not in current_metrics:
                continue

            current_value = current_metrics[metric_name]
            deviation = current_value - baseline_value
            deviation_percent = (
                (deviation / baseline_value * 100) if baseline_value != 0 else 0
            )

            upper = self.upper_bounds.get(metric_name, baseline_value * 1.1)
            lower = self.lower_bounds.get(metric_name, baseline_value * 0.9)

            comparison[metric_name] = {
                "baseline": baseline_value,
                "current": current_value,
                "deviation": deviation,
                "deviation_percent": deviation_percent,
                "within_bounds": lower <= current_value <= upper,
                "upper_bound": upper,
                "lower_bound": lower
            }

        return comparison

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "baseline_id": self.baseline_id,
            "name": self.name,
            "description": self.description,
            "created_date": self.created_date.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat(),
            "metrics": self.metrics,
            "operating_conditions": self.operating_conditions,
            "upper_bounds": self.upper_bounds,
            "lower_bounds": self.lower_bounds
        }


@dataclass
class BatchExportConfig:
    """Configuration for batch data export."""

    tag_names: List[str]
    start_time: datetime
    end_time: datetime
    format: ExportFormat = ExportFormat.CSV
    interval: Optional[timedelta] = None
    include_quality: bool = True
    include_units: bool = True
    output_path: Optional[str] = None
    compress: bool = False


# =============================================================================
# Historian Integration Class
# =============================================================================

class HistorianIntegration:
    """
    PI System / OSIsoft integration for historical data.

    Provides:
    - Historical data retrieval with interpolation
    - Trend analysis queries
    - Performance baseline management
    - Batch data export
    """

    def __init__(self, config: HistorianConfig):
        """
        Initialize historian integration.

        Args:
            config: Historian configuration
        """
        self.config = config

        self._client = None
        self._connected = False
        self._web_ids: Dict[str, str] = {}  # Tag name -> WebID mapping

        # Cached baselines
        self._baselines: Dict[str, PerformanceBaseline] = {}

        # Statistics
        self._stats = {
            "queries_total": 0,
            "queries_success": 0,
            "queries_failed": 0,
            "points_retrieved": 0,
            "exports_total": 0,
            "last_query": None
        }

        logger.info(f"Historian Integration initialized for {config.base_url}")

    @property
    def is_connected(self) -> bool:
        """Check if connected to historian."""
        return self._connected

    async def connect(self) -> None:
        """
        Establish connection to PI Web API.

        Raises:
            HistorianConnectionError: If connection fails
        """
        logger.info(f"Connecting to PI Web API at {self.config.base_url}")

        try:
            await self._create_client()
            await self._verify_connection()
            await self._load_tag_web_ids()

            self._connected = True
            logger.info("Successfully connected to PI Web API")

        except Exception as e:
            logger.error(f"Failed to connect to historian: {e}")
            raise HistorianConnectionError(f"Connection failed: {e}")

    async def _create_client(self) -> None:
        """Create HTTP client for PI Web API."""
        # Simulated client
        self._client = {
            "base_url": self.config.base_url,
            "auth": (self.config.username, self.config.password),
            "verify_ssl": self.config.verify_ssl,
            "connected": False
        }

        await asyncio.sleep(0.1)  # Simulate connection
        self._client["connected"] = True

    async def _verify_connection(self) -> None:
        """Verify connection to PI Web API."""
        # In production: GET /piwebapi/system/versions
        await asyncio.sleep(0.05)
        logger.debug("PI Web API connection verified")

    async def _load_tag_web_ids(self) -> None:
        """Load WebIDs for condenser tags."""
        condenser_tags = [
            "Condenser.CoolingWater.InletTemp",
            "Condenser.CoolingWater.OutletTemp",
            "Condenser.CoolingWater.FlowRate",
            "Condenser.Vacuum.Pressure",
            "Condenser.Hotwell.Level",
            "Condenser.Condensate.FlowRate",
            "Condenser.Performance.TTD",
            "Condenser.Performance.CleanlinessFactor",
            "Condenser.Performance.Duty",
            "Condenser.AirRemoval.LeakageRate"
        ]

        for tag in condenser_tags:
            # Simulated WebID (in production: lookup via PI Web API)
            self._web_ids[tag] = f"P21WebID_{tag.replace('.', '_')}"

    async def disconnect(self) -> None:
        """Disconnect from historian."""
        logger.info("Disconnecting from PI Web API")

        if self._client:
            self._client["connected"] = False
            self._client = None

        self._connected = False
        logger.info("Disconnected from PI Web API")

    async def get_historical_data(
        self,
        query: TrendQuery
    ) -> Dict[str, List[HistoricalDataPoint]]:
        """
        Retrieve historical data for specified tags.

        Args:
            query: Trend query parameters

        Returns:
            Dictionary mapping tag names to lists of data points

        Raises:
            HistorianQueryError: If query fails
        """
        if not self._connected:
            raise HistorianQueryError("Not connected to historian")

        self._stats["queries_total"] += 1

        try:
            results = {}

            for tag_name in query.tag_names:
                data_points = await self._query_tag_data(tag_name, query)
                results[tag_name] = data_points
                self._stats["points_retrieved"] += len(data_points)

            self._stats["queries_success"] += 1
            self._stats["last_query"] = datetime.utcnow()

            logger.info(
                f"Retrieved {sum(len(v) for v in results.values())} points "
                f"for {len(query.tag_names)} tags"
            )

            return results

        except Exception as e:
            self._stats["queries_failed"] += 1
            logger.error(f"Historical data query failed: {e}")
            raise HistorianQueryError(f"Query failed: {e}")

    async def _query_tag_data(
        self,
        tag_name: str,
        query: TrendQuery
    ) -> List[HistoricalDataPoint]:
        """Query data for a single tag."""
        import random

        # Simulated data generation
        data_points = []
        current_time = query.start_time
        interval = query.interval or timedelta(minutes=1)

        # Base values for simulation
        base_values = {
            "Condenser.CoolingWater.InletTemp": 25.0,
            "Condenser.CoolingWater.OutletTemp": 35.0,
            "Condenser.CoolingWater.FlowRate": 50000.0,
            "Condenser.Vacuum.Pressure": 5.0,
            "Condenser.Hotwell.Level": 50.0,
            "Condenser.Condensate.FlowRate": 1000.0,
            "Condenser.Performance.TTD": 3.5,
            "Condenser.Performance.CleanlinessFactor": 0.85,
            "Condenser.Performance.Duty": 400000.0,
            "Condenser.AirRemoval.LeakageRate": 2.0
        }

        base_value = base_values.get(tag_name, 50.0)

        while current_time <= query.end_time:
            # Simulate value with some variation
            value = base_value * (1 + random.uniform(-0.1, 0.1))

            data_points.append(HistoricalDataPoint(
                tag_name=tag_name,
                timestamp=current_time,
                value=round(value, 4),
                quality=DataQuality.GOOD,
                units=self._get_tag_units(tag_name)
            ))

            current_time += interval

            # Limit points
            if query.max_points and len(data_points) >= query.max_points:
                break

        return data_points

    def _get_tag_units(self, tag_name: str) -> str:
        """Get engineering units for a tag."""
        units_map = {
            "InletTemp": "degC",
            "OutletTemp": "degC",
            "FlowRate": "m3/h",
            "Pressure": "kPa",
            "Level": "%",
            "TTD": "degC",
            "CleanlinessFactor": "",
            "Duty": "kW",
            "LeakageRate": "kg/h"
        }

        for key, unit in units_map.items():
            if key in tag_name:
                return unit
        return ""

    async def get_aggregated_data(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        aggregation: AggregationType,
        interval: timedelta
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve aggregated historical data.

        Args:
            tag_names: List of tag names
            start_time: Start of time range
            end_time: End of time range
            aggregation: Aggregation type
            interval: Aggregation interval

        Returns:
            Dictionary mapping tag names to aggregated data
        """
        query = TrendQuery(
            tag_names=tag_names,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            aggregation=aggregation
        )

        raw_data = await self.get_historical_data(query)

        # Aggregate data
        aggregated = {}
        for tag_name, points in raw_data.items():
            aggregated[tag_name] = self._aggregate_points(points, interval, aggregation)

        return aggregated

    def _aggregate_points(
        self,
        points: List[HistoricalDataPoint],
        interval: timedelta,
        aggregation: AggregationType
    ) -> List[Dict[str, Any]]:
        """Aggregate data points by interval."""
        if not points:
            return []

        results = []
        bucket_start = points[0].timestamp
        bucket_values = []

        for point in points:
            if point.timestamp >= bucket_start + interval:
                # Complete current bucket
                if bucket_values:
                    agg_value = self._calculate_aggregate(bucket_values, aggregation)
                    results.append({
                        "timestamp": bucket_start.isoformat(),
                        "value": agg_value,
                        "count": len(bucket_values)
                    })

                # Start new bucket
                bucket_start = point.timestamp
                bucket_values = []

            if point.value is not None:
                bucket_values.append(point.value)

        # Final bucket
        if bucket_values:
            agg_value = self._calculate_aggregate(bucket_values, aggregation)
            results.append({
                "timestamp": bucket_start.isoformat(),
                "value": agg_value,
                "count": len(bucket_values)
            })

        return results

    def _calculate_aggregate(
        self,
        values: List[float],
        aggregation: AggregationType
    ) -> float:
        """Calculate aggregate value."""
        if not values:
            return 0.0

        if aggregation == AggregationType.AVERAGE:
            return sum(values) / len(values)
        elif aggregation == AggregationType.MINIMUM:
            return min(values)
        elif aggregation == AggregationType.MAXIMUM:
            return max(values)
        elif aggregation == AggregationType.TOTAL:
            return sum(values)
        elif aggregation == AggregationType.COUNT:
            return len(values)
        elif aggregation == AggregationType.RANGE:
            return max(values) - min(values)
        elif aggregation == AggregationType.STD_DEV:
            avg = sum(values) / len(values)
            variance = sum((v - avg) ** 2 for v in values) / len(values)
            return variance ** 0.5
        elif aggregation == AggregationType.FIRST:
            return values[0]
        elif aggregation == AggregationType.LAST:
            return values[-1]
        else:
            return sum(values) / len(values)

    async def create_baseline(
        self,
        name: str,
        description: str,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        operating_conditions: Optional[Dict[str, Any]] = None
    ) -> PerformanceBaseline:
        """
        Create a performance baseline from historical data.

        Args:
            name: Baseline name
            description: Baseline description
            tag_names: Tags to include in baseline
            start_time: Start of baseline period
            end_time: End of baseline period
            operating_conditions: Operating conditions during baseline

        Returns:
            Created PerformanceBaseline
        """
        # Get historical data for baseline period
        query = TrendQuery(
            tag_names=tag_names,
            start_time=start_time,
            end_time=end_time,
            interval=timedelta(minutes=5)
        )

        historical_data = await self.get_historical_data(query)

        # Calculate baseline metrics
        metrics = {}
        upper_bounds = {}
        lower_bounds = {}

        for tag_name, points in historical_data.items():
            if not points:
                continue

            values = [p.value for p in points if p.value is not None]
            if not values:
                continue

            avg = sum(values) / len(values)
            std_dev = (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5

            metrics[tag_name] = avg
            upper_bounds[tag_name] = avg + 2 * std_dev
            lower_bounds[tag_name] = avg - 2 * std_dev

        baseline = PerformanceBaseline(
            baseline_id=f"BL-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            name=name,
            description=description,
            created_date=datetime.utcnow(),
            valid_from=start_time,
            valid_to=end_time + timedelta(days=365),  # Valid for 1 year
            metrics=metrics,
            operating_conditions=operating_conditions or {},
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds
        )

        self._baselines[baseline.baseline_id] = baseline

        logger.info(f"Created performance baseline: {baseline.baseline_id}")
        return baseline

    async def compare_to_baseline(
        self,
        baseline_id: str,
        current_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Compare current performance to baseline.

        Args:
            baseline_id: Baseline identifier
            current_data: Current metric values

        Returns:
            Comparison results
        """
        baseline = self._baselines.get(baseline_id)
        if not baseline:
            raise HistorianDataError(f"Baseline not found: {baseline_id}")

        comparison = baseline.compare(current_data)

        # Calculate overall performance score
        within_bounds_count = sum(
            1 for v in comparison.values() if v.get("within_bounds", False)
        )
        total_metrics = len(comparison)

        performance_score = (
            (within_bounds_count / total_metrics * 100) if total_metrics > 0 else 0
        )

        return {
            "baseline_id": baseline_id,
            "baseline_name": baseline.name,
            "timestamp": datetime.utcnow().isoformat(),
            "performance_score": round(performance_score, 2),
            "metrics_comparison": comparison,
            "metrics_within_bounds": within_bounds_count,
            "metrics_total": total_metrics
        }

    async def export_data(self, config: BatchExportConfig) -> Union[str, bytes]:
        """
        Export historical data to file.

        Args:
            config: Export configuration

        Returns:
            Exported data as string (CSV/JSON) or bytes (Parquet)
        """
        self._stats["exports_total"] += 1

        # Get data
        query = TrendQuery(
            tag_names=config.tag_names,
            start_time=config.start_time,
            end_time=config.end_time,
            interval=config.interval or timedelta(minutes=1)
        )

        historical_data = await self.get_historical_data(query)

        # Export based on format
        if config.format == ExportFormat.CSV:
            return self._export_csv(historical_data, config)
        elif config.format == ExportFormat.JSON:
            return self._export_json(historical_data, config)
        else:
            raise HistorianDataError(f"Unsupported format: {config.format}")

    def _export_csv(
        self,
        data: Dict[str, List[HistoricalDataPoint]],
        config: BatchExportConfig
    ) -> str:
        """Export data to CSV format."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        header = ["timestamp"]
        for tag_name in config.tag_names:
            header.append(tag_name)
            if config.include_quality:
                header.append(f"{tag_name}_quality")
        writer.writerow(header)

        # Get all unique timestamps
        all_timestamps = set()
        for points in data.values():
            for point in points:
                all_timestamps.add(point.timestamp)

        # Write rows
        for timestamp in sorted(all_timestamps):
            row = [timestamp.isoformat()]
            for tag_name in config.tag_names:
                points = data.get(tag_name, [])
                point = next(
                    (p for p in points if p.timestamp == timestamp),
                    None
                )
                if point:
                    row.append(point.value)
                    if config.include_quality:
                        row.append(point.quality.value)
                else:
                    row.append("")
                    if config.include_quality:
                        row.append("")
            writer.writerow(row)

        return output.getvalue()

    def _export_json(
        self,
        data: Dict[str, List[HistoricalDataPoint]],
        config: BatchExportConfig
    ) -> str:
        """Export data to JSON format."""
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "start_time": config.start_time.isoformat(),
            "end_time": config.end_time.isoformat(),
            "tags": {}
        }

        for tag_name, points in data.items():
            export_data["tags"][tag_name] = [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "value": p.value,
                    **({"quality": p.quality.value} if config.include_quality else {}),
                    **({"units": p.units} if config.include_units and p.units else {})
                }
                for p in points
            ]

        return json.dumps(export_data, indent=2)

    def get_baseline(self, baseline_id: str) -> Optional[PerformanceBaseline]:
        """Get a baseline by ID."""
        return self._baselines.get(baseline_id)

    def get_all_baselines(self) -> Dict[str, PerformanceBaseline]:
        """Get all baselines."""
        return dict(self._baselines)

    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            **self._stats,
            "connected": self._connected,
            "cached_web_ids": len(self._web_ids),
            "baselines_count": len(self._baselines)
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "connected": self._connected,
            "timestamp": datetime.utcnow().isoformat()
        }

        if not self._connected:
            health["status"] = "unhealthy"
            health["reason"] = "Not connected to historian"
            return health

        # Try a test query
        try:
            query = TrendQuery(
                tag_names=["Condenser.CoolingWater.InletTemp"],
                start_time=datetime.utcnow() - timedelta(minutes=5),
                end_time=datetime.utcnow(),
                max_points=10
            )
            await self.get_historical_data(query)
            health["test_query_success"] = True
        except Exception as e:
            health["status"] = "degraded"
            health["test_query_success"] = False
            health["reason"] = str(e)

        return health
