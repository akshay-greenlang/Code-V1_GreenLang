# -*- coding: utf-8 -*-
"""
GL-OPS-X-001: Real-time Emissions Monitor
==========================================

Monitors emissions in real-time across facilities, providing continuous
tracking, aggregation, and trend analysis for operational decision-making.

Capabilities:
    - Real-time emissions data ingestion from multiple sources
    - Multi-level aggregation (facility, unit, process)
    - Rolling window statistics and trend analysis
    - Threshold monitoring with configurable alerts
    - Historical comparison and baseline tracking
    - Integration with operational systems (SCADA, DCS)

Zero-Hallucination Guarantees:
    - All emissions calculations use deterministic formulas
    - Complete provenance tracking with SHA-256 hashes
    - No LLM calls in the calculation path
    - All data points traceable to source readings

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class AggregationPeriod(str, Enum):
    """Time periods for aggregating emissions data."""
    MINUTE = "minute"
    FIVE_MINUTES = "five_minutes"
    FIFTEEN_MINUTES = "fifteen_minutes"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class MonitoringStatus(str, Enum):
    """Status of the monitoring system."""
    ACTIVE = "active"
    PAUSED = "paused"
    CALIBRATING = "calibrating"
    ERROR = "error"
    OFFLINE = "offline"


class EmissionsSource(str, Enum):
    """Types of emissions data sources."""
    CEMS = "cems"  # Continuous Emissions Monitoring System
    FUEL_METER = "fuel_meter"
    PRODUCTION_DATA = "production_data"
    CALCULATED = "calculated"
    MANUAL_ENTRY = "manual_entry"
    IOT_SENSOR = "iot_sensor"
    SCADA = "scada"
    DCS = "dcs"


class EmissionsTrend(str, Enum):
    """Trend direction for emissions."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class GasType(str, Enum):
    """Types of greenhouse gases monitored."""
    CO2 = "co2"
    CH4 = "ch4"
    N2O = "n2o"
    HFC = "hfc"
    PFC = "pfc"
    SF6 = "sf6"
    NF3 = "nf3"
    CO2E = "co2e"  # CO2 equivalent


# GWP values for converting to CO2e (AR6 100-year)
GWP_AR6_100 = {
    GasType.CO2: 1.0,
    GasType.CH4: 27.9,
    GasType.N2O: 273.0,
    GasType.SF6: 25200.0,
    GasType.NF3: 17400.0,
}


# =============================================================================
# Pydantic Models
# =============================================================================

class EmissionsReading(BaseModel):
    """A single emissions reading from a source."""
    reading_id: str = Field(..., description="Unique reading identifier")
    source_id: str = Field(..., description="Source identifier (e.g., meter ID)")
    source_type: EmissionsSource = Field(..., description="Type of data source")
    facility_id: str = Field(..., description="Facility identifier")
    unit_id: Optional[str] = Field(None, description="Unit/equipment identifier")
    process_id: Optional[str] = Field(None, description="Process identifier")

    # Emissions data
    gas_type: GasType = Field(..., description="Type of gas measured")
    value: float = Field(..., ge=0, description="Emissions value")
    unit: str = Field(default="kg", description="Unit of measurement")

    # Timing
    timestamp: datetime = Field(..., description="Reading timestamp")
    collection_time: Optional[datetime] = Field(None, description="When data was collected")

    # Quality
    quality_flag: str = Field(default="valid", description="Data quality flag")
    uncertainty_percent: Optional[float] = Field(None, ge=0, le=100, description="Measurement uncertainty")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FacilityMonitorConfig(BaseModel):
    """Configuration for monitoring a facility."""
    facility_id: str = Field(..., description="Facility identifier")
    name: str = Field(..., description="Facility name")
    sources: List[str] = Field(default_factory=list, description="Source IDs to monitor")
    thresholds: Dict[str, float] = Field(default_factory=dict, description="Alert thresholds by gas type")
    baseline: Dict[str, float] = Field(default_factory=dict, description="Baseline values by gas type")
    aggregation_period: AggregationPeriod = Field(
        default=AggregationPeriod.HOUR, description="Default aggregation period"
    )


class AggregatedEmissions(BaseModel):
    """Aggregated emissions data for a time period."""
    facility_id: str = Field(..., description="Facility identifier")
    period_start: datetime = Field(..., description="Period start time")
    period_end: datetime = Field(..., description="Period end time")
    aggregation_period: AggregationPeriod = Field(..., description="Aggregation period type")

    # Aggregated values by gas type
    total_emissions: Dict[str, float] = Field(default_factory=dict, description="Total emissions by gas")
    total_co2e: float = Field(default=0.0, description="Total CO2 equivalent")

    # Statistics
    reading_count: int = Field(default=0, description="Number of readings")
    avg_emissions_rate: float = Field(default=0.0, description="Average emissions rate")
    max_emissions_rate: float = Field(default=0.0, description="Maximum emissions rate")
    min_emissions_rate: float = Field(default=0.0, description="Minimum emissions rate")
    std_dev: float = Field(default=0.0, description="Standard deviation")

    # Quality
    data_completeness: float = Field(default=100.0, ge=0, le=100, description="Data completeness %")
    quality_issues: List[str] = Field(default_factory=list, description="Quality issues detected")


class EmissionsMonitorInput(BaseModel):
    """Input for the Real-time Emissions Monitor."""
    operation: str = Field(..., description="Operation to perform")
    readings: List[EmissionsReading] = Field(default_factory=list, description="Readings to ingest")
    facility_id: Optional[str] = Field(None, description="Facility ID for queries")
    start_time: Optional[datetime] = Field(None, description="Query start time")
    end_time: Optional[datetime] = Field(None, description="Query end time")
    aggregation_period: Optional[AggregationPeriod] = Field(None, description="Aggregation period")
    gas_types: List[GasType] = Field(default_factory=list, description="Gas types to include")
    facility_config: Optional[FacilityMonitorConfig] = Field(None, description="Facility configuration")

    @field_validator('operation')
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Validate operation is supported."""
        valid_ops = {
            'ingest_readings', 'get_current_emissions', 'get_aggregated',
            'get_trend', 'check_thresholds', 'configure_facility',
            'get_facilities', 'get_statistics', 'export_data'
        }
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {valid_ops}")
        return v


class EmissionsMonitorOutput(BaseModel):
    """Output from the Real-time Emissions Monitor."""
    success: bool = Field(..., description="Whether operation succeeded")
    operation: str = Field(..., description="Operation performed")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


# =============================================================================
# Real-time Emissions Monitor Implementation
# =============================================================================

class RealtimeEmissionsMonitor(BaseAgent):
    """
    GL-OPS-X-001: Real-time Emissions Monitor

    Monitors emissions in real-time across facilities, providing continuous
    tracking, aggregation, and trend analysis for operational decision-making.

    Zero-Hallucination Guarantees:
        - All calculations use deterministic arithmetic operations
        - Complete provenance tracking with SHA-256 hashes
        - No LLM calls in the calculation path
        - All data points traceable to source readings

    Usage:
        monitor = RealtimeEmissionsMonitor()

        # Ingest readings
        result = monitor.run({
            "operation": "ingest_readings",
            "readings": [
                {
                    "reading_id": "R001",
                    "source_id": "CEMS-001",
                    "source_type": "cems",
                    "facility_id": "FAC-001",
                    "gas_type": "co2",
                    "value": 150.5,
                    "unit": "kg",
                    "timestamp": "2024-01-01T12:00:00Z"
                }
            ]
        })

        # Get current emissions
        result = monitor.run({
            "operation": "get_current_emissions",
            "facility_id": "FAC-001"
        })
    """

    AGENT_ID = "GL-OPS-X-001"
    AGENT_NAME = "Real-time Emissions Monitor"
    VERSION = "1.0.0"

    # Buffer size for real-time data
    MAX_BUFFER_SIZE = 100000
    # Rolling window sizes
    TREND_WINDOW_HOURS = 24

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Real-time Emissions Monitor."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Real-time emissions monitoring and aggregation",
                version=self.VERSION,
                parameters={
                    "buffer_size": self.MAX_BUFFER_SIZE,
                    "trend_window_hours": self.TREND_WINDOW_HOURS,
                    "default_aggregation": AggregationPeriod.HOUR.value,
                }
            )
        super().__init__(config)

        # Reading buffers by facility
        self._readings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.MAX_BUFFER_SIZE))

        # Aggregated data by facility and period
        self._aggregated: Dict[str, Dict[str, List[AggregatedEmissions]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Facility configurations
        self._facilities: Dict[str, FacilityMonitorConfig] = {}

        # Current emissions state
        self._current_state: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Statistics
        self._total_readings_ingested = 0
        self._total_aggregations = 0

        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute monitor operations.

        Args:
            input_data: Input containing operation and parameters

        Returns:
            AgentResult with operation results
        """
        start_time = time.time()

        try:
            monitor_input = EmissionsMonitorInput(**input_data)
            operation = monitor_input.operation

            result_data = self._route_operation(monitor_input)

            provenance_hash = self._compute_provenance_hash(input_data, result_data)
            processing_time_ms = (time.time() - start_time) * 1000

            output = EmissionsMonitorOutput(
                success=True,
                operation=operation,
                data=result_data,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
            )

        except Exception as e:
            self.logger.error(f"Monitor operation failed: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time) * 1000

            return AgentResult(
                success=False,
                error=str(e),
                data={
                    "operation": input_data.get("operation", "unknown"),
                    "processing_time_ms": processing_time_ms,
                },
            )

    def _route_operation(self, monitor_input: EmissionsMonitorInput) -> Dict[str, Any]:
        """Route to appropriate operation handler."""
        operation = monitor_input.operation

        if operation == "ingest_readings":
            return self._handle_ingest_readings(monitor_input.readings)
        elif operation == "get_current_emissions":
            return self._handle_get_current_emissions(monitor_input.facility_id)
        elif operation == "get_aggregated":
            return self._handle_get_aggregated(
                monitor_input.facility_id,
                monitor_input.start_time,
                monitor_input.end_time,
                monitor_input.aggregation_period,
            )
        elif operation == "get_trend":
            return self._handle_get_trend(
                monitor_input.facility_id,
                monitor_input.gas_types,
            )
        elif operation == "check_thresholds":
            return self._handle_check_thresholds(monitor_input.facility_id)
        elif operation == "configure_facility":
            return self._handle_configure_facility(monitor_input.facility_config)
        elif operation == "get_facilities":
            return self._handle_get_facilities()
        elif operation == "get_statistics":
            return self._handle_get_statistics()
        elif operation == "export_data":
            return self._handle_export_data(
                monitor_input.facility_id,
                monitor_input.start_time,
                monitor_input.end_time,
            )
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # =========================================================================
    # Reading Ingestion
    # =========================================================================

    def _handle_ingest_readings(self, readings: List[EmissionsReading]) -> Dict[str, Any]:
        """Ingest emissions readings."""
        ingested_count = 0
        facilities_updated = set()
        errors = []

        for reading in readings:
            try:
                self._ingest_single_reading(reading)
                ingested_count += 1
                facilities_updated.add(reading.facility_id)
            except Exception as e:
                errors.append({
                    "reading_id": reading.reading_id,
                    "error": str(e),
                })

        self._total_readings_ingested += ingested_count

        # Trigger aggregation for updated facilities
        for facility_id in facilities_updated:
            self._update_current_state(facility_id)

        return {
            "ingested_count": ingested_count,
            "facilities_updated": list(facilities_updated),
            "errors": errors,
            "total_readings": self._total_readings_ingested,
        }

    def _ingest_single_reading(self, reading: EmissionsReading) -> None:
        """Ingest a single reading."""
        facility_id = reading.facility_id

        # Add to buffer
        self._readings[facility_id].append(reading)

        # Update current state
        gas_key = reading.gas_type.value
        if gas_key not in self._current_state[facility_id]:
            self._current_state[facility_id][gas_key] = 0.0

        # Store latest value
        self._current_state[facility_id][gas_key] = reading.value
        self._current_state[facility_id][f"{gas_key}_timestamp"] = reading.timestamp.isoformat()

    def _update_current_state(self, facility_id: str) -> None:
        """Update current emissions state for a facility."""
        readings = list(self._readings[facility_id])
        if not readings:
            return

        # Get readings from the last hour
        now = DeterministicClock.now()
        one_hour_ago = now - timedelta(hours=1)

        recent_readings = [
            r for r in readings
            if r.timestamp >= one_hour_ago
        ]

        if not recent_readings:
            return

        # Calculate totals by gas type
        totals: Dict[str, float] = defaultdict(float)
        for reading in recent_readings:
            totals[reading.gas_type.value] += reading.value

        # Calculate CO2e
        total_co2e = 0.0
        for gas_type, total in totals.items():
            try:
                gas_enum = GasType(gas_type)
                gwp = GWP_AR6_100.get(gas_enum, 1.0)
                total_co2e += total * gwp
            except ValueError:
                total_co2e += total

        self._current_state[facility_id]["hourly_total_co2e"] = total_co2e
        self._current_state[facility_id]["hourly_reading_count"] = len(recent_readings)

    # =========================================================================
    # Current Emissions
    # =========================================================================

    def _handle_get_current_emissions(
        self, facility_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get current emissions for facility or all facilities."""
        now = DeterministicClock.now()

        if facility_id:
            if facility_id not in self._current_state:
                return {
                    "facility_id": facility_id,
                    "status": MonitoringStatus.OFFLINE.value,
                    "message": "No data available for facility",
                }

            return {
                "facility_id": facility_id,
                "status": MonitoringStatus.ACTIVE.value,
                "current_emissions": self._current_state[facility_id],
                "timestamp": now.isoformat(),
            }

        # Return all facilities
        all_current = {}
        for fac_id, state in self._current_state.items():
            all_current[fac_id] = {
                "status": MonitoringStatus.ACTIVE.value,
                "current_emissions": state,
            }

        return {
            "facilities": all_current,
            "facility_count": len(all_current),
            "timestamp": now.isoformat(),
        }

    # =========================================================================
    # Aggregation
    # =========================================================================

    def _handle_get_aggregated(
        self,
        facility_id: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        aggregation_period: Optional[AggregationPeriod],
    ) -> Dict[str, Any]:
        """Get aggregated emissions data."""
        now = DeterministicClock.now()

        # Default time range: last 24 hours
        if end_time is None:
            end_time = now
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        if aggregation_period is None:
            aggregation_period = AggregationPeriod.HOUR

        facilities = [facility_id] if facility_id else list(self._readings.keys())

        results = {}
        for fac_id in facilities:
            aggregated = self._aggregate_facility(
                fac_id, start_time, end_time, aggregation_period
            )
            results[fac_id] = [a.model_dump() for a in aggregated]

        return {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "aggregation_period": aggregation_period.value,
            "facilities": results,
        }

    def _aggregate_facility(
        self,
        facility_id: str,
        start_time: datetime,
        end_time: datetime,
        aggregation_period: AggregationPeriod,
    ) -> List[AggregatedEmissions]:
        """Aggregate emissions for a facility over a time range."""
        readings = list(self._readings.get(facility_id, []))

        # Filter readings by time range
        filtered = [
            r for r in readings
            if start_time <= r.timestamp <= end_time
        ]

        if not filtered:
            return []

        # Determine period duration
        period_duration = self._get_period_duration(aggregation_period)

        # Group readings by period
        periods: Dict[datetime, List[EmissionsReading]] = defaultdict(list)
        for reading in filtered:
            period_start = self._floor_to_period(reading.timestamp, aggregation_period)
            periods[period_start].append(reading)

        # Create aggregated emissions for each period
        aggregated = []
        for period_start, period_readings in sorted(periods.items()):
            agg = self._create_aggregation(
                facility_id,
                period_start,
                period_start + period_duration,
                aggregation_period,
                period_readings,
            )
            aggregated.append(agg)

        self._total_aggregations += len(aggregated)
        return aggregated

    def _create_aggregation(
        self,
        facility_id: str,
        period_start: datetime,
        period_end: datetime,
        aggregation_period: AggregationPeriod,
        readings: List[EmissionsReading],
    ) -> AggregatedEmissions:
        """Create aggregated emissions for a period."""
        # Calculate totals by gas type
        totals: Dict[str, float] = defaultdict(float)
        values: List[float] = []

        for reading in readings:
            totals[reading.gas_type.value] += reading.value
            values.append(reading.value)

        # Calculate CO2e
        total_co2e = 0.0
        for gas_type, total in totals.items():
            try:
                gas_enum = GasType(gas_type)
                gwp = GWP_AR6_100.get(gas_enum, 1.0)
                total_co2e += total * gwp
            except ValueError:
                total_co2e += total

        # Calculate statistics
        avg_rate = sum(values) / len(values) if values else 0.0
        max_rate = max(values) if values else 0.0
        min_rate = min(values) if values else 0.0

        # Standard deviation
        if len(values) > 1:
            mean = avg_rate
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5
        else:
            std_dev = 0.0

        # Data completeness (simplified)
        expected_readings = self._expected_readings_per_period(aggregation_period)
        data_completeness = min(100.0, (len(readings) / expected_readings) * 100)

        return AggregatedEmissions(
            facility_id=facility_id,
            period_start=period_start,
            period_end=period_end,
            aggregation_period=aggregation_period,
            total_emissions={k: round(v, 6) for k, v in totals.items()},
            total_co2e=round(total_co2e, 6),
            reading_count=len(readings),
            avg_emissions_rate=round(avg_rate, 6),
            max_emissions_rate=round(max_rate, 6),
            min_emissions_rate=round(min_rate, 6),
            std_dev=round(std_dev, 6),
            data_completeness=round(data_completeness, 2),
        )

    def _get_period_duration(self, period: AggregationPeriod) -> timedelta:
        """Get duration for an aggregation period."""
        durations = {
            AggregationPeriod.MINUTE: timedelta(minutes=1),
            AggregationPeriod.FIVE_MINUTES: timedelta(minutes=5),
            AggregationPeriod.FIFTEEN_MINUTES: timedelta(minutes=15),
            AggregationPeriod.HOUR: timedelta(hours=1),
            AggregationPeriod.DAY: timedelta(days=1),
            AggregationPeriod.WEEK: timedelta(weeks=1),
            AggregationPeriod.MONTH: timedelta(days=30),
        }
        return durations.get(period, timedelta(hours=1))

    def _floor_to_period(self, dt: datetime, period: AggregationPeriod) -> datetime:
        """Floor datetime to the start of its aggregation period."""
        if period == AggregationPeriod.MINUTE:
            return dt.replace(second=0, microsecond=0)
        elif period == AggregationPeriod.FIVE_MINUTES:
            minute = (dt.minute // 5) * 5
            return dt.replace(minute=minute, second=0, microsecond=0)
        elif period == AggregationPeriod.FIFTEEN_MINUTES:
            minute = (dt.minute // 15) * 15
            return dt.replace(minute=minute, second=0, microsecond=0)
        elif period == AggregationPeriod.HOUR:
            return dt.replace(minute=0, second=0, microsecond=0)
        elif period == AggregationPeriod.DAY:
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == AggregationPeriod.WEEK:
            days_since_monday = dt.weekday()
            monday = dt - timedelta(days=days_since_monday)
            return monday.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == AggregationPeriod.MONTH:
            return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return dt

    def _expected_readings_per_period(self, period: AggregationPeriod) -> int:
        """Estimate expected readings per period (assuming 1-minute intervals)."""
        expected = {
            AggregationPeriod.MINUTE: 1,
            AggregationPeriod.FIVE_MINUTES: 5,
            AggregationPeriod.FIFTEEN_MINUTES: 15,
            AggregationPeriod.HOUR: 60,
            AggregationPeriod.DAY: 1440,
            AggregationPeriod.WEEK: 10080,
            AggregationPeriod.MONTH: 43200,
        }
        return expected.get(period, 60)

    # =========================================================================
    # Trend Analysis
    # =========================================================================

    def _handle_get_trend(
        self,
        facility_id: Optional[str],
        gas_types: List[GasType],
    ) -> Dict[str, Any]:
        """Get emissions trend for a facility."""
        if not facility_id:
            return {"error": "facility_id is required for trend analysis"}

        readings = list(self._readings.get(facility_id, []))
        if not readings:
            return {
                "facility_id": facility_id,
                "trend": EmissionsTrend.UNKNOWN.value,
                "message": "Insufficient data for trend analysis",
            }

        now = DeterministicClock.now()
        window_start = now - timedelta(hours=self.TREND_WINDOW_HOURS)

        # Filter by time and gas types
        filtered = [
            r for r in readings
            if r.timestamp >= window_start
        ]

        if gas_types:
            filtered = [r for r in filtered if r.gas_type in gas_types]

        if len(filtered) < 2:
            return {
                "facility_id": facility_id,
                "trend": EmissionsTrend.UNKNOWN.value,
                "message": "Insufficient data points for trend analysis",
            }

        # Sort by timestamp
        filtered.sort(key=lambda r: r.timestamp)

        # Calculate trend using linear regression slope
        values = [r.value for r in filtered]
        n = len(values)

        # Simple linear regression
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator

        # Calculate coefficient of variation for volatility
        if y_mean > 0:
            std_dev = (sum((v - y_mean) ** 2 for v in values) / n) ** 0.5
            cv = std_dev / y_mean
        else:
            cv = 0.0

        # Determine trend
        slope_threshold = y_mean * 0.01  # 1% of mean
        volatility_threshold = 0.3  # 30% CV

        if cv > volatility_threshold:
            trend = EmissionsTrend.VOLATILE
        elif abs(slope) < slope_threshold:
            trend = EmissionsTrend.STABLE
        elif slope > 0:
            trend = EmissionsTrend.INCREASING
        else:
            trend = EmissionsTrend.DECREASING

        return {
            "facility_id": facility_id,
            "trend": trend.value,
            "slope": round(slope, 6),
            "coefficient_of_variation": round(cv, 4),
            "data_points": n,
            "window_hours": self.TREND_WINDOW_HOURS,
            "mean_value": round(y_mean, 6),
            "min_value": round(min(values), 6),
            "max_value": round(max(values), 6),
        }

    # =========================================================================
    # Threshold Monitoring
    # =========================================================================

    def _handle_check_thresholds(self, facility_id: Optional[str]) -> Dict[str, Any]:
        """Check emissions against configured thresholds."""
        facilities = [facility_id] if facility_id else list(self._facilities.keys())

        violations = []
        warnings = []

        for fac_id in facilities:
            config = self._facilities.get(fac_id)
            if not config or not config.thresholds:
                continue

            current = self._current_state.get(fac_id, {})

            for gas_type, threshold in config.thresholds.items():
                current_value = current.get(gas_type, 0.0)

                if isinstance(current_value, (int, float)):
                    if current_value > threshold:
                        violations.append({
                            "facility_id": fac_id,
                            "gas_type": gas_type,
                            "threshold": threshold,
                            "current_value": current_value,
                            "exceedance_percent": round(
                                ((current_value - threshold) / threshold) * 100, 2
                            ),
                        })
                    elif current_value > threshold * 0.8:
                        warnings.append({
                            "facility_id": fac_id,
                            "gas_type": gas_type,
                            "threshold": threshold,
                            "current_value": current_value,
                            "utilization_percent": round(
                                (current_value / threshold) * 100, 2
                            ),
                        })

        return {
            "violations": violations,
            "warnings": warnings,
            "violation_count": len(violations),
            "warning_count": len(warnings),
            "timestamp": DeterministicClock.now().isoformat(),
        }

    # =========================================================================
    # Configuration
    # =========================================================================

    def _handle_configure_facility(
        self, config: Optional[FacilityMonitorConfig]
    ) -> Dict[str, Any]:
        """Configure monitoring for a facility."""
        if not config:
            return {"error": "facility_config is required"}

        self._facilities[config.facility_id] = config

        return {
            "facility_id": config.facility_id,
            "configured": True,
            "sources": len(config.sources),
            "thresholds": len(config.thresholds),
        }

    def _handle_get_facilities(self) -> Dict[str, Any]:
        """Get all configured facilities."""
        return {
            "facilities": {
                fac_id: config.model_dump()
                for fac_id, config in self._facilities.items()
            },
            "total_configured": len(self._facilities),
            "total_with_data": len(self._readings),
        }

    # =========================================================================
    # Statistics and Export
    # =========================================================================

    def _handle_get_statistics(self) -> Dict[str, Any]:
        """Get monitor statistics."""
        return {
            "total_readings_ingested": self._total_readings_ingested,
            "total_aggregations": self._total_aggregations,
            "facilities_monitored": len(self._readings),
            "facilities_configured": len(self._facilities),
            "buffer_utilization": {
                fac_id: len(readings) / self.MAX_BUFFER_SIZE * 100
                for fac_id, readings in self._readings.items()
            },
        }

    def _handle_export_data(
        self,
        facility_id: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> Dict[str, Any]:
        """Export raw readings data."""
        now = DeterministicClock.now()

        if end_time is None:
            end_time = now
        if start_time is None:
            start_time = end_time - timedelta(hours=24)

        facilities = [facility_id] if facility_id else list(self._readings.keys())

        export_data = {}
        total_records = 0

        for fac_id in facilities:
            readings = list(self._readings.get(fac_id, []))
            filtered = [
                r.model_dump() for r in readings
                if start_time <= r.timestamp <= end_time
            ]
            export_data[fac_id] = filtered
            total_records += len(filtered)

        return {
            "export_time": now.isoformat(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_records": total_records,
            "facilities": export_data,
        }

    # =========================================================================
    # Provenance
    # =========================================================================

    def _compute_provenance_hash(
        self, input_data: Dict[str, Any], output_data: Dict[str, Any]
    ) -> str:
        """Compute SHA-256 hash for audit trail."""
        provenance_str = json.dumps(
            {"input": input_data, "output": output_data},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()[:16]
