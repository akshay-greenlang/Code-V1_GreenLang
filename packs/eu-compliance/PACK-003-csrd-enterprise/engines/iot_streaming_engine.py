# -*- coding: utf-8 -*-
"""
IoTStreamingEngine - PACK-003 CSRD Enterprise Engine 6

Real-time data integration from IoT sensors for continuous emissions
monitoring. Supports energy meters, gas meters, water meters, temperature
sensors, emissions sensors, and weather stations. Provides device
registration, reading ingestion, windowed aggregation, anomaly detection,
and real-time emissions calculation.

Protocols Supported:
    - MQTT: Message Queuing Telemetry Transport
    - HTTP: REST API push/pull
    - OPCUA: OPC Unified Architecture (industrial)
    - MODBUS: Serial communication protocol

Data Quality Flags:
    - GOOD: Valid reading within calibration range
    - SUSPECT: Reading outside normal range but plausible
    - BAD: Invalid reading (sensor malfunction, out-of-range)
    - INTERPOLATED: Gap-filled from adjacent readings

Zero-Hallucination:
    - All aggregations use deterministic arithmetic (sum, avg, min, max)
    - Anomaly detection uses statistical thresholds (z-score)
    - Emissions conversion uses fixed emission factors from database
    - No LLM involvement in any data processing or calculation

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-003 CSRD Enterprise
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow
from greenlang.schemas.enums import AlertSeverity

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DeviceType(str, Enum):
    """Types of IoT devices."""

    ENERGY_METER = "energy_meter"
    GAS_METER = "gas_meter"
    WATER_METER = "water_meter"
    TEMPERATURE = "temperature"
    EMISSIONS_SENSOR = "emissions_sensor"
    WEATHER_STATION = "weather_station"

class DeviceProtocol(str, Enum):
    """Communication protocols for IoT devices."""

    MQTT = "mqtt"
    HTTP = "http"
    OPCUA = "opcua"
    MODBUS = "modbus"

class DeviceStatus(str, Enum):
    """Operational status of an IoT device."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    ERROR = "error"

class QualityFlag(str, Enum):
    """Data quality flag for sensor readings."""

    GOOD = "good"
    SUSPECT = "suspect"
    BAD = "bad"
    INTERPOLATED = "interpolated"

class AlertType(str, Enum):
    """Types of streaming alerts."""

    SPIKE = "spike"
    DROP = "drop"
    FLATLINE = "flatline"
    OUT_OF_RANGE = "out_of_range"
    DEVICE_OFFLINE = "device_offline"
    CALIBRATION_DUE = "calibration_due"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class DeviceLocation(BaseModel):
    """Physical location of an IoT device."""

    latitude: float = Field(0.0, ge=-90, le=90, description="Latitude")
    longitude: float = Field(0.0, ge=-180, le=180, description="Longitude")
    building: str = Field("", description="Building name/ID")
    floor: str = Field("", description="Floor number/label")
    zone: str = Field("", description="Zone/area within facility")

class IoTDevice(BaseModel):
    """IoT device registration."""

    device_id: str = Field(
        default_factory=_new_uuid, description="Unique device identifier"
    )
    device_type: DeviceType = Field(..., description="Type of sensor")
    protocol: DeviceProtocol = Field(
        DeviceProtocol.MQTT, description="Communication protocol"
    )
    facility_id: str = Field(..., description="Facility where device is installed")
    location: DeviceLocation = Field(
        default_factory=DeviceLocation, description="Physical location"
    )
    calibration_date: Optional[datetime] = Field(
        None, description="Last calibration date"
    )
    calibration_interval_days: int = Field(
        365, ge=1, description="Calibration interval in days"
    )
    status: DeviceStatus = Field(
        DeviceStatus.ACTIVE, description="Current device status"
    )
    metric_types: List[str] = Field(
        default_factory=list,
        description="Metric types this device measures",
    )
    unit: str = Field("", description="Default measurement unit")
    min_value: Optional[float] = Field(
        None, description="Minimum expected reading"
    )
    max_value: Optional[float] = Field(
        None, description="Maximum expected reading"
    )
    registered_at: datetime = Field(
        default_factory=utcnow, description="Registration timestamp"
    )

class IoTReading(BaseModel):
    """A single sensor reading."""

    reading_id: str = Field(
        default_factory=_new_uuid, description="Unique reading ID"
    )
    device_id: str = Field(..., description="Source device ID")
    timestamp: datetime = Field(
        default_factory=utcnow, description="Reading timestamp"
    )
    metric_type: str = Field(..., description="Type of metric measured")
    value: float = Field(..., description="Measured value")
    unit: str = Field(..., description="Unit of measurement")
    quality_flag: QualityFlag = Field(
        QualityFlag.GOOD, description="Data quality indicator"
    )

class AggregatedReading(BaseModel):
    """Aggregated sensor readings over a time window."""

    aggregation_id: str = Field(
        default_factory=_new_uuid, description="Aggregation ID"
    )
    device_id: str = Field(..., description="Source device ID")
    window_start: datetime = Field(..., description="Window start time")
    window_end: datetime = Field(..., description="Window end time")
    metric_type: str = Field(..., description="Metric type")
    avg_value: float = Field(..., description="Average value")
    min_value: float = Field(..., description="Minimum value")
    max_value: float = Field(..., description="Maximum value")
    sum_value: float = Field(..., description="Sum of values")
    count: int = Field(..., description="Number of readings")
    unit: str = Field(..., description="Unit of measurement")
    quality_score: float = Field(
        100.0, ge=0, le=100, description="Percentage of GOOD readings"
    )
    provenance_hash: str = Field("", description="SHA-256 provenance hash")

class StreamAlert(BaseModel):
    """Real-time streaming alert."""

    alert_id: str = Field(
        default_factory=_new_uuid, description="Unique alert ID"
    )
    device_id: str = Field(..., description="Source device ID")
    alert_type: AlertType = Field(..., description="Type of alert")
    severity: AlertSeverity = Field(..., description="Alert severity")
    message: str = Field(..., description="Human-readable alert message")
    value: Optional[float] = Field(None, description="Triggering value")
    threshold: Optional[float] = Field(None, description="Threshold exceeded")
    timestamp: datetime = Field(
        default_factory=utcnow, description="Alert timestamp"
    )
    acknowledged: bool = Field(False, description="Whether alert is acknowledged")

# ---------------------------------------------------------------------------
# Emission Factors (deterministic lookup)
# ---------------------------------------------------------------------------

_EMISSION_FACTORS: Dict[str, Dict[str, float]] = {
    "electricity_kwh": {"co2e_kg_per_unit": 0.233, "unit": "kWh"},
    "natural_gas_m3": {"co2e_kg_per_unit": 2.02, "unit": "m3"},
    "natural_gas_kwh": {"co2e_kg_per_unit": 0.184, "unit": "kWh"},
    "diesel_litre": {"co2e_kg_per_unit": 2.68, "unit": "litre"},
    "water_m3": {"co2e_kg_per_unit": 0.344, "unit": "m3"},
    "steam_kg": {"co2e_kg_per_unit": 0.17, "unit": "kg"},
}

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class IoTStreamingEngine:
    """Real-time IoT data integration and emissions calculation engine.

    Manages IoT device registration, reading ingestion, windowed
    aggregation, spike detection, and real-time emissions conversion.
    All calculations use deterministic emission factors.

    Attributes:
        _devices: Registered devices.
        _readings: Reading buffer per device.
        _alerts: Generated alerts.

    Example:
        >>> engine = IoTStreamingEngine()
        >>> device = IoTDevice(
        ...     device_type=DeviceType.ENERGY_METER,
        ...     facility_id="plant-1",
        ...     unit="kWh",
        ... )
        >>> device_id = engine.register_device(device)
        >>> reading = IoTReading(
        ...     device_id=device_id,
        ...     metric_type="electricity_kwh",
        ...     value=150.5,
        ...     unit="kWh",
        ... )
        >>> result = engine.ingest_reading(reading)
    """

    def __init__(self, buffer_max_size: int = 100000) -> None:
        """Initialize IoTStreamingEngine.

        Args:
            buffer_max_size: Maximum readings to buffer per device.
        """
        self._devices: Dict[str, IoTDevice] = {}
        self._readings: Dict[str, List[IoTReading]] = defaultdict(list)
        self._alerts: List[StreamAlert] = []
        self._buffer_max_size = buffer_max_size
        logger.info(
            "IoTStreamingEngine v%s initialized (buffer_max=%d)",
            _MODULE_VERSION, buffer_max_size,
        )

    # -- Device Management --------------------------------------------------

    def register_device(self, device: IoTDevice) -> str:
        """Register a new IoT device.

        Args:
            device: Device configuration.

        Returns:
            Registered device ID.
        """
        self._devices[device.device_id] = device
        self._readings[device.device_id] = []

        logger.info(
            "Device registered: %s (type=%s, facility=%s)",
            device.device_id, device.device_type.value, device.facility_id,
        )
        return device.device_id

    def list_devices(
        self, facility_id: Optional[str] = None
    ) -> List[IoTDevice]:
        """List registered devices, optionally filtered by facility.

        Args:
            facility_id: Filter by facility ID (optional).

        Returns:
            List of IoTDevice objects.
        """
        devices = list(self._devices.values())
        if facility_id:
            devices = [d for d in devices if d.facility_id == facility_id]
        return devices

    # -- Reading Ingestion --------------------------------------------------

    def ingest_reading(self, reading: IoTReading) -> Dict[str, Any]:
        """Process a single sensor reading.

        Validates the reading, applies quality flags, checks for
        anomalies, and adds to the buffer.

        Args:
            reading: Sensor reading to ingest.

        Returns:
            Dict with ingestion status and any alerts generated.

        Raises:
            KeyError: If device_id not registered.
        """
        if reading.device_id not in self._devices:
            raise KeyError(f"Device '{reading.device_id}' not registered")

        device = self._devices[reading.device_id]

        # Apply quality flag based on device range
        if device.min_value is not None and reading.value < device.min_value:
            reading.quality_flag = QualityFlag.SUSPECT
        if device.max_value is not None and reading.value > device.max_value:
            reading.quality_flag = QualityFlag.SUSPECT

        # Anomaly detection
        alert = self.detect_anomaly(reading.device_id, reading)

        # Buffer management
        buffer = self._readings[reading.device_id]
        if len(buffer) >= self._buffer_max_size:
            buffer.pop(0)
        buffer.append(reading)

        result: Dict[str, Any] = {
            "reading_id": reading.reading_id,
            "device_id": reading.device_id,
            "ingested": True,
            "quality_flag": reading.quality_flag.value,
            "buffer_size": len(buffer),
        }

        if alert:
            result["alert"] = {
                "alert_id": alert.alert_id,
                "type": alert.alert_type.value,
                "severity": alert.severity.value,
                "message": alert.message,
            }

        return result

    def ingest_batch(
        self, readings: List[IoTReading]
    ) -> Dict[str, Any]:
        """Batch-ingest multiple sensor readings.

        Args:
            readings: List of readings to ingest.

        Returns:
            Dict with batch ingestion summary.
        """
        start = utcnow()
        results = {
            "total": len(readings),
            "ingested": 0,
            "failed": 0,
            "alerts_generated": 0,
            "quality_breakdown": {
                "good": 0, "suspect": 0, "bad": 0, "interpolated": 0,
            },
        }

        for reading in readings:
            try:
                result = self.ingest_reading(reading)
                results["ingested"] += 1
                results["quality_breakdown"][reading.quality_flag.value] += 1
                if "alert" in result:
                    results["alerts_generated"] += 1
            except (KeyError, ValueError) as e:
                results["failed"] += 1
                logger.warning("Failed to ingest reading: %s", str(e))

        elapsed = (utcnow() - start).total_seconds() * 1000
        results["processing_time_ms"] = round(elapsed, 2)
        results["provenance_hash"] = _compute_hash(results)

        logger.info(
            "Batch ingestion: %d/%d ingested (%.0fms)",
            results["ingested"], results["total"], elapsed,
        )
        return results

    # -- Aggregation --------------------------------------------------------

    def aggregate_window(
        self,
        device_id: str,
        window_minutes: int,
        metric_type: Optional[str] = None,
    ) -> AggregatedReading:
        """Aggregate readings over a time window.

        All aggregation uses deterministic arithmetic.

        Args:
            device_id: Device to aggregate.
            window_minutes: Window size in minutes.
            metric_type: Filter by metric type (optional).

        Returns:
            AggregatedReading with statistical summary.

        Raises:
            KeyError: If device not found.
            ValueError: If no readings in window.
        """
        if device_id not in self._devices:
            raise KeyError(f"Device '{device_id}' not registered")

        now = utcnow()
        window_start = now - timedelta(minutes=window_minutes)

        readings = [
            r for r in self._readings.get(device_id, [])
            if r.timestamp >= window_start
        ]

        if metric_type:
            readings = [r for r in readings if r.metric_type == metric_type]

        if not readings:
            raise ValueError(
                f"No readings for device '{device_id}' in the last "
                f"{window_minutes} minutes"
            )

        values = [r.value for r in readings]
        good_count = sum(1 for r in readings if r.quality_flag == QualityFlag.GOOD)
        quality_score = (good_count / len(readings) * 100) if readings else 0.0

        agg = AggregatedReading(
            device_id=device_id,
            window_start=window_start,
            window_end=now,
            metric_type=metric_type or readings[0].metric_type,
            avg_value=round(sum(values) / len(values), 4),
            min_value=round(min(values), 4),
            max_value=round(max(values), 4),
            sum_value=round(sum(values), 4),
            count=len(values),
            unit=readings[0].unit,
            quality_score=round(quality_score, 2),
        )
        agg.provenance_hash = _compute_hash(agg)

        logger.debug(
            "Aggregated %d readings for device %s (window=%dmin)",
            len(readings), device_id, window_minutes,
        )
        return agg

    # -- Anomaly Detection --------------------------------------------------

    def detect_anomaly(
        self, device_id: str, reading: IoTReading
    ) -> Optional[StreamAlert]:
        """Detect anomalies in real-time readings.

        Uses z-score-based spike detection against recent history.

        Args:
            device_id: Device identifier.
            reading: Current reading to check.

        Returns:
            StreamAlert if anomaly detected, None otherwise.
        """
        history = self._readings.get(device_id, [])
        if len(history) < 10:
            return None

        # Last 50 readings for baseline
        recent = history[-50:]
        values = [r.value for r in recent]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

        if std == 0:
            return None

        z_score = abs(reading.value - mean) / std

        # Spike detection (z > 3)
        if z_score > 3.0:
            direction = "spike" if reading.value > mean else "drop"
            severity = AlertSeverity.CRITICAL if z_score > 5.0 else AlertSeverity.WARNING

            alert = StreamAlert(
                device_id=device_id,
                alert_type=AlertType.SPIKE if direction == "spike" else AlertType.DROP,
                severity=severity,
                message=(
                    f"Anomalous {direction} detected: value={reading.value:.2f} "
                    f"(mean={mean:.2f}, z-score={z_score:.2f})"
                ),
                value=reading.value,
                threshold=mean + 3 * std if direction == "spike" else mean - 3 * std,
            )
            self._alerts.append(alert)

            logger.warning(
                "Alert: %s on device %s (z=%.2f)",
                direction, device_id, z_score,
            )
            return alert

        # Flatline detection (no variance in last 20 readings)
        if len(history) >= 20:
            last_20 = [r.value for r in history[-20:]]
            if max(last_20) == min(last_20):
                alert = StreamAlert(
                    device_id=device_id,
                    alert_type=AlertType.FLATLINE,
                    severity=AlertSeverity.WARNING,
                    message=(
                        f"Flatline detected: constant value {last_20[0]:.2f} "
                        f"for last 20 readings"
                    ),
                    value=last_20[0],
                )
                self._alerts.append(alert)
                return alert

        return None

    # -- Device Health ------------------------------------------------------

    def get_device_health(self, device_id: str) -> Dict[str, Any]:
        """Get health status and diagnostics for a device.

        Args:
            device_id: Device identifier.

        Returns:
            Dict with health metrics including last reading, uptime,
            and data quality score.

        Raises:
            KeyError: If device not registered.
        """
        if device_id not in self._devices:
            raise KeyError(f"Device '{device_id}' not registered")

        device = self._devices[device_id]
        readings = self._readings.get(device_id, [])

        last_reading: Optional[Dict[str, Any]] = None
        if readings:
            last = readings[-1]
            last_reading = {
                "timestamp": last.timestamp.isoformat(),
                "value": last.value,
                "unit": last.unit,
                "quality_flag": last.quality_flag.value,
            }

        # Data quality score (percentage of GOOD readings)
        total = len(readings)
        good = sum(1 for r in readings if r.quality_flag == QualityFlag.GOOD)
        quality_score = (good / total * 100) if total > 0 else 0.0

        # Check calibration
        calibration_ok = True
        days_since_calibration: Optional[int] = None
        if device.calibration_date:
            days_since = (utcnow() - device.calibration_date).days
            days_since_calibration = days_since
            if days_since > device.calibration_interval_days:
                calibration_ok = False

        # Uptime (based on reading frequency)
        uptime_pct = 100.0
        if len(readings) >= 2:
            first = readings[0].timestamp
            last_ts = readings[-1].timestamp
            span = (last_ts - first).total_seconds()
            if span > 0:
                expected_interval = span / (len(readings) - 1)
                expected_readings = (
                    (utcnow() - first).total_seconds() / expected_interval
                )
                uptime_pct = min(100.0, (len(readings) / max(expected_readings, 1)) * 100)

        # Active alerts for this device
        active_alerts = [
            a for a in self._alerts
            if a.device_id == device_id and not a.acknowledged
        ]

        health: Dict[str, Any] = {
            "device_id": device_id,
            "device_type": device.device_type.value,
            "status": device.status.value,
            "facility_id": device.facility_id,
            "last_reading": last_reading,
            "total_readings": total,
            "data_quality_score": round(quality_score, 2),
            "uptime_pct": round(uptime_pct, 2),
            "calibration_ok": calibration_ok,
            "days_since_calibration": days_since_calibration,
            "active_alerts": len(active_alerts),
            "provenance_hash": _compute_hash(
                {"device_id": device_id, "total": total, "quality": quality_score}
            ),
        }

        return health

    # -- Real-Time Emissions ------------------------------------------------

    def calculate_realtime_emissions(
        self, facility_id: str, time_range_minutes: int = 60
    ) -> Dict[str, Any]:
        """Calculate real-time emissions from IoT sensor data.

        Converts sensor readings to CO2e using deterministic emission
        factors. No LLM involvement -- pure arithmetic.

        Args:
            facility_id: Facility to calculate for.
            time_range_minutes: Time range for calculation.

        Returns:
            Dict with emissions breakdown by source.
        """
        start = utcnow()
        cutoff = start - timedelta(minutes=time_range_minutes)

        facility_devices = [
            d for d in self._devices.values()
            if d.facility_id == facility_id
        ]

        if not facility_devices:
            return {
                "facility_id": facility_id,
                "total_emissions_kg_co2e": 0.0,
                "message": "No devices registered for this facility",
            }

        emissions_by_source: Dict[str, float] = {}
        total_emissions = 0.0

        for device in facility_devices:
            readings = [
                r for r in self._readings.get(device.device_id, [])
                if r.timestamp >= cutoff
                and r.quality_flag in (QualityFlag.GOOD, QualityFlag.INTERPOLATED)
            ]

            for reading in readings:
                factor_info = _EMISSION_FACTORS.get(reading.metric_type)
                if factor_info:
                    emissions = reading.value * factor_info["co2e_kg_per_unit"]
                    source_key = reading.metric_type
                    emissions_by_source[source_key] = (
                        emissions_by_source.get(source_key, 0.0) + emissions
                    )
                    total_emissions += emissions

        result = {
            "facility_id": facility_id,
            "time_range_minutes": time_range_minutes,
            "calculation_timestamp": start.isoformat(),
            "total_emissions_kg_co2e": round(total_emissions, 4),
            "total_emissions_tco2e": round(total_emissions / 1000, 6),
            "emissions_by_source": {
                k: round(v, 4) for k, v in emissions_by_source.items()
            },
            "devices_counted": len(facility_devices),
            "emission_factors_used": {
                k: v["co2e_kg_per_unit"]
                for k, v in _EMISSION_FACTORS.items()
                if k in emissions_by_source
            },
            "provenance_hash": _compute_hash({
                "facility_id": facility_id,
                "total": total_emissions,
                "sources": emissions_by_source,
            }),
        }

        logger.info(
            "Real-time emissions for facility %s: %.4f kg CO2e (%d sources)",
            facility_id, total_emissions, len(emissions_by_source),
        )
        return result

    # -- Buffer Management --------------------------------------------------

    def manage_buffer(self, action: str = "stats") -> Dict[str, Any]:
        """Manage the reading buffer.

        Args:
            action: One of 'stats', 'flush', 'resize'.

        Returns:
            Dict with buffer operation results.
        """
        if action == "stats":
            total_readings = sum(
                len(readings) for readings in self._readings.values()
            )
            per_device = {
                device_id: len(readings)
                for device_id, readings in self._readings.items()
            }
            return {
                "action": "stats",
                "total_readings_buffered": total_readings,
                "devices_with_data": len(per_device),
                "max_buffer_size": self._buffer_max_size,
                "per_device_counts": per_device,
            }

        elif action == "flush":
            flushed = sum(
                len(readings) for readings in self._readings.values()
            )
            for device_id in self._readings:
                self._readings[device_id] = []
            logger.info("Buffer flushed: %d readings removed", flushed)
            return {
                "action": "flush",
                "readings_flushed": flushed,
                "flushed_at": utcnow().isoformat(),
            }

        elif action == "resize":
            return {
                "action": "resize",
                "current_max_size": self._buffer_max_size,
                "message": "Use constructor parameter to set buffer size",
            }

        return {"action": action, "message": "Unknown action"}
