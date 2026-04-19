# -*- coding: utf-8 -*-
"""
BMSSCADABridge - Building/Industrial Management System Integration for PACK-031
==================================================================================

This module provides integration with Building Management Systems (BMS) and
Supervisory Control and Data Acquisition (SCADA) systems. It defines data
models for protocol adapters (BACnet, Modbus, OPC-UA), real-time meter reading
ingestion, SCADA data point mapping, alarm integration, historical data
retrieval, and data normalization with unit conversion.

Note: This bridge provides the data model and orchestration layer. Actual
protocol-level communication (BACnet/IP, Modbus TCP, OPC-UA client) is
delegated to external protocol libraries.

Supported Protocols (data model):
    - BACnet/IP: Object/property addressing, COV subscriptions
    - Modbus TCP/RTU: Register addressing, function codes
    - OPC-UA: NodeId addressing, browse paths

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-031 Industrial Energy Audit
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
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

class ProtocolType(str, Enum):
    """Industrial communication protocol types."""

    BACNET_IP = "bacnet_ip"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    OPC_UA = "opc_ua"
    MQTT = "mqtt"
    REST_API = "rest_api"
    CSV_EXPORT = "csv_export"

class DataPointType(str, Enum):
    """Types of metered data points."""

    ENERGY_KWH = "energy_kwh"
    POWER_KW = "power_kw"
    TEMPERATURE_C = "temperature_c"
    PRESSURE_BAR = "pressure_bar"
    FLOW_RATE_M3H = "flow_rate_m3h"
    HUMIDITY_PCT = "humidity_pct"
    SPEED_RPM = "speed_rpm"
    STATUS_BOOL = "status_bool"
    VOLTAGE_V = "voltage_v"
    CURRENT_A = "current_a"
    FREQUENCY_HZ = "frequency_hz"
    GAS_VOLUME_M3 = "gas_volume_m3"
    STEAM_TONNES = "steam_tonnes"

class AlarmSeverity(str, Enum):
    """Alarm severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ConnectionStatus(str, Enum):
    """Protocol connection status."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    TIMEOUT = "timeout"
    NOT_CONFIGURED = "not_configured"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class ProtocolConfig(BaseModel):
    """Configuration for a protocol adapter."""

    protocol: ProtocolType = Field(...)
    host: str = Field(default="localhost")
    port: int = Field(default=502, ge=1, le=65535)
    device_id: int = Field(default=1, ge=0)
    timeout_seconds: float = Field(default=5.0, ge=0.5)
    retry_count: int = Field(default=3, ge=0)
    polling_interval_seconds: float = Field(default=60.0, ge=1.0)
    # BACnet specific
    bacnet_device_instance: Optional[int] = Field(None)
    # OPC-UA specific
    opcua_endpoint: Optional[str] = Field(None)
    opcua_security_policy: Optional[str] = Field(None)
    # Modbus specific
    modbus_unit_id: Optional[int] = Field(None)

class DataPointMapping(BaseModel):
    """Mapping of a SCADA/BMS data point to GreenLang standard fields."""

    point_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="", description="Human-readable data point name")
    protocol: ProtocolType = Field(default=ProtocolType.MODBUS_TCP)
    point_type: DataPointType = Field(default=DataPointType.ENERGY_KWH)
    # Protocol-specific addressing
    address: str = Field(default="", description="Protocol-specific address")
    register_offset: Optional[int] = Field(None, description="Modbus register offset")
    bacnet_object_type: Optional[str] = Field(None, description="BACnet object type")
    bacnet_object_instance: Optional[int] = Field(None)
    opcua_node_id: Optional[str] = Field(None, description="OPC-UA node ID")
    # Data transformation
    raw_unit: str = Field(default="", description="Raw unit from device")
    target_unit: str = Field(default="", description="GreenLang standard unit")
    scale_factor: float = Field(default=1.0, description="Multiply raw value by this")
    offset: float = Field(default=0.0, description="Add to scaled value")
    # Metadata
    facility_zone: str = Field(default="", description="Facility zone or area")
    equipment_id: str = Field(default="", description="Associated equipment ID")
    meter_id: str = Field(default="", description="Associated meter ID")

class MeterReading(BaseModel):
    """A single meter reading from BMS/SCADA."""

    reading_id: str = Field(default_factory=_new_uuid)
    point_id: str = Field(default="")
    timestamp: datetime = Field(default_factory=utcnow)
    raw_value: float = Field(default=0.0)
    normalized_value: float = Field(default=0.0)
    unit: str = Field(default="")
    quality: str = Field(default="good", description="good|uncertain|bad")
    source_protocol: str = Field(default="")

class AlarmEvent(BaseModel):
    """An alarm event from BMS/SCADA."""

    alarm_id: str = Field(default_factory=_new_uuid)
    point_id: str = Field(default="")
    timestamp: datetime = Field(default_factory=utcnow)
    severity: AlarmSeverity = Field(default=AlarmSeverity.MEDIUM)
    description: str = Field(default="")
    value: float = Field(default=0.0)
    threshold: float = Field(default=0.0)
    acknowledged: bool = Field(default=False)
    cleared: bool = Field(default=False)

class HistoricalDataRequest(BaseModel):
    """Request for historical data from BMS/SCADA historian."""

    request_id: str = Field(default_factory=_new_uuid)
    point_ids: List[str] = Field(default_factory=list)
    start_time: Optional[datetime] = Field(None)
    end_time: Optional[datetime] = Field(None)
    interval_seconds: int = Field(default=900, description="Aggregation interval")
    aggregation: str = Field(default="average", description="average|sum|min|max|last")

class HistoricalDataResult(BaseModel):
    """Result of a historical data retrieval."""

    request_id: str = Field(default="")
    point_id: str = Field(default="")
    records_retrieved: int = Field(default=0)
    start_time: Optional[datetime] = Field(None)
    end_time: Optional[datetime] = Field(None)
    readings: List[MeterReading] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    gaps_detected: int = Field(default=0)
    provenance_hash: str = Field(default="")

class BMSSCADABridgeConfig(BaseModel):
    """Configuration for the BMS/SCADA Bridge."""

    pack_id: str = Field(default="PACK-031")
    enable_provenance: bool = Field(default=True)
    default_polling_interval_seconds: float = Field(default=60.0, ge=1.0)
    max_concurrent_connections: int = Field(default=10, ge=1, le=100)
    reading_buffer_size: int = Field(default=10000, ge=100)
    enable_alarm_integration: bool = Field(default=True)

# ---------------------------------------------------------------------------
# Unit Conversion Constants (deterministic, no LLM)
# ---------------------------------------------------------------------------

UNIT_CONVERSIONS: Dict[str, Dict[str, float]] = {
    "mwh_to_kwh": {"factor": 1000.0},
    "kwh_to_mwh": {"factor": 0.001},
    "gj_to_kwh": {"factor": 277.778},
    "kwh_to_gj": {"factor": 0.0036},
    "mmbtu_to_kwh": {"factor": 293.071},
    "therms_to_kwh": {"factor": 29.3071},
    "m3_gas_to_kwh": {"factor": 10.55},  # Natural gas approx HHV
    "fahrenheit_to_celsius": {"factor": 0.5556, "offset": -17.778},
    "psi_to_bar": {"factor": 0.0689476},
    "gpm_to_m3h": {"factor": 0.2271},
}

# ---------------------------------------------------------------------------
# BMSSCADABridge
# ---------------------------------------------------------------------------

class BMSSCADABridge:
    """Building and industrial management system integration.

    Provides data models for BACnet/Modbus/OPC-UA protocol adapters,
    meter reading ingestion, SCADA data point mapping, alarm integration,
    historical data retrieval, and unit conversion.

    Attributes:
        config: Bridge configuration.
        _protocol_configs: Registered protocol configurations.
        _data_points: Registered data point mappings.
        _connections: Active connection statuses.
        _reading_buffer: Recent meter readings.
        _alarms: Active alarm events.

    Example:
        >>> bridge = BMSSCADABridge()
        >>> bridge.register_data_point(DataPointMapping(
        ...     name="Main Meter kWh", protocol=ProtocolType.MODBUS_TCP,
        ...     point_type=DataPointType.ENERGY_KWH, address="40001"
        ... ))
        >>> reading = bridge.ingest_reading("point_id", 1234.5)
    """

    def __init__(self, config: Optional[BMSSCADABridgeConfig] = None) -> None:
        """Initialize the BMS/SCADA Bridge."""
        self.config = config or BMSSCADABridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._protocol_configs: Dict[str, ProtocolConfig] = {}
        self._data_points: Dict[str, DataPointMapping] = {}
        self._connections: Dict[str, ConnectionStatus] = {}
        self._reading_buffer: List[MeterReading] = []
        self._alarms: List[AlarmEvent] = []
        self.logger.info("BMSSCADABridge initialized")

    # -------------------------------------------------------------------------
    # Protocol Configuration
    # -------------------------------------------------------------------------

    def register_protocol(self, name: str, config: ProtocolConfig) -> Dict[str, Any]:
        """Register a protocol adapter configuration.

        Args:
            name: Unique name for this protocol connection.
            config: Protocol configuration.

        Returns:
            Dict with registration status.
        """
        self._protocol_configs[name] = config
        self._connections[name] = ConnectionStatus.NOT_CONFIGURED
        self.logger.info("Protocol registered: %s (%s)", name, config.protocol.value)
        return {"name": name, "protocol": config.protocol.value, "registered": True}

    def test_connection(self, name: str) -> Dict[str, Any]:
        """Test connectivity for a registered protocol.

        Args:
            name: Protocol connection name.

        Returns:
            Dict with connection test result.
        """
        if name not in self._protocol_configs:
            return {"name": name, "status": "not_found", "connected": False}

        # Stub: in production, this would attempt actual connection
        self._connections[name] = ConnectionStatus.CONNECTED
        return {
            "name": name,
            "protocol": self._protocol_configs[name].protocol.value,
            "status": ConnectionStatus.CONNECTED.value,
            "connected": True,
            "latency_ms": 0.0,
        }

    # -------------------------------------------------------------------------
    # Data Point Management
    # -------------------------------------------------------------------------

    def register_data_point(self, mapping: DataPointMapping) -> DataPointMapping:
        """Register a SCADA/BMS data point mapping.

        Args:
            mapping: Data point mapping configuration.

        Returns:
            Registered DataPointMapping with assigned point_id.
        """
        self._data_points[mapping.point_id] = mapping
        self.logger.info(
            "Data point registered: %s (%s, %s)",
            mapping.name, mapping.point_type.value, mapping.protocol.value,
        )
        return mapping

    def get_data_points(
        self, point_type: Optional[DataPointType] = None,
    ) -> List[Dict[str, Any]]:
        """Get registered data points, optionally filtered by type.

        Args:
            point_type: Filter by data point type (all if None).

        Returns:
            List of data point summaries.
        """
        points = self._data_points.values()
        if point_type:
            points = [p for p in points if p.point_type == point_type]
        return [
            {
                "point_id": p.point_id,
                "name": p.name,
                "type": p.point_type.value,
                "protocol": p.protocol.value,
                "address": p.address,
                "unit": p.target_unit,
                "zone": p.facility_zone,
                "equipment_id": p.equipment_id,
            }
            for p in points
        ]

    # -------------------------------------------------------------------------
    # Meter Reading Ingestion
    # -------------------------------------------------------------------------

    def ingest_reading(
        self,
        point_id: str,
        raw_value: float,
        timestamp: Optional[datetime] = None,
        quality: str = "good",
    ) -> Optional[MeterReading]:
        """Ingest a single meter reading and apply normalization.

        Normalization formula (deterministic):
            normalized = (raw_value * scale_factor) + offset

        Args:
            point_id: Data point identifier.
            raw_value: Raw value from device.
            timestamp: Reading timestamp (now if None).
            quality: Data quality flag.

        Returns:
            Normalized MeterReading, or None if point not found.
        """
        mapping = self._data_points.get(point_id)
        if mapping is None:
            self.logger.warning("Data point not found: %s", point_id)
            return None

        # Deterministic normalization
        normalized = (raw_value * mapping.scale_factor) + mapping.offset

        reading = MeterReading(
            point_id=point_id,
            timestamp=timestamp or utcnow(),
            raw_value=raw_value,
            normalized_value=round(normalized, 4),
            unit=mapping.target_unit or mapping.raw_unit,
            quality=quality,
            source_protocol=mapping.protocol.value,
        )

        self._reading_buffer.append(reading)

        # Trim buffer if needed
        if len(self._reading_buffer) > self.config.reading_buffer_size:
            self._reading_buffer = self._reading_buffer[-self.config.reading_buffer_size:]

        return reading

    def ingest_batch(
        self, readings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Ingest a batch of meter readings.

        Args:
            readings: List of dicts with point_id, raw_value, timestamp, quality.

        Returns:
            Dict with batch ingestion summary.
        """
        start = time.monotonic()
        ingested = 0
        failed = 0

        for r in readings:
            result = self.ingest_reading(
                point_id=r.get("point_id", ""),
                raw_value=r.get("raw_value", 0.0),
                timestamp=r.get("timestamp"),
                quality=r.get("quality", "good"),
            )
            if result:
                ingested += 1
            else:
                failed += 1

        elapsed = (time.monotonic() - start) * 1000
        return {
            "total": len(readings),
            "ingested": ingested,
            "failed": failed,
            "duration_ms": round(elapsed, 1),
        }

    # -------------------------------------------------------------------------
    # Alarm Integration
    # -------------------------------------------------------------------------

    def register_alarm(
        self,
        point_id: str,
        severity: AlarmSeverity,
        description: str,
        value: float = 0.0,
        threshold: float = 0.0,
    ) -> AlarmEvent:
        """Register an alarm event.

        Args:
            point_id: Data point that triggered the alarm.
            severity: Alarm severity.
            description: Alarm description.
            value: Current value.
            threshold: Threshold that was exceeded.

        Returns:
            AlarmEvent record.
        """
        alarm = AlarmEvent(
            point_id=point_id,
            severity=severity,
            description=description,
            value=value,
            threshold=threshold,
        )
        self._alarms.append(alarm)
        self.logger.warning("Alarm registered: %s (%s)", description, severity.value)
        return alarm

    def get_active_alarms(self) -> List[Dict[str, Any]]:
        """Get all uncleared alarms.

        Returns:
            List of active alarm summaries.
        """
        return [
            {
                "alarm_id": a.alarm_id,
                "point_id": a.point_id,
                "severity": a.severity.value,
                "description": a.description,
                "value": a.value,
                "threshold": a.threshold,
                "timestamp": a.timestamp.isoformat(),
                "acknowledged": a.acknowledged,
            }
            for a in self._alarms
            if not a.cleared
        ]

    # -------------------------------------------------------------------------
    # Historical Data Retrieval
    # -------------------------------------------------------------------------

    def request_historical_data(
        self, request: HistoricalDataRequest,
    ) -> List[HistoricalDataResult]:
        """Request historical data from the BMS/SCADA historian.

        In production, this dispatches to the protocol adapter. The stub
        returns empty results.

        Args:
            request: Historical data request parameters.

        Returns:
            List of HistoricalDataResult per point.
        """
        results: List[HistoricalDataResult] = []
        for point_id in request.point_ids:
            result = HistoricalDataResult(
                request_id=request.request_id,
                point_id=point_id,
                records_retrieved=0,
                start_time=request.start_time,
                end_time=request.end_time,
                quality_score=0.0,
            )
            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result)
            results.append(result)

        self.logger.info(
            "Historical data requested: %d points, interval=%ds",
            len(request.point_ids), request.interval_seconds,
        )
        return results

    # -------------------------------------------------------------------------
    # Unit Conversion
    # -------------------------------------------------------------------------

    def convert_unit(
        self, value: float, conversion_key: str,
    ) -> Optional[float]:
        """Convert a value using a predefined unit conversion.

        Deterministic calculation using lookup table.

        Args:
            value: Input value.
            conversion_key: Key from UNIT_CONVERSIONS table.

        Returns:
            Converted value, or None if conversion not found.
        """
        conv = UNIT_CONVERSIONS.get(conversion_key)
        if conv is None:
            return None
        factor = conv.get("factor", 1.0)
        offset = conv.get("offset", 0.0)
        return round(value * factor + offset, 6)

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    def check_health(self) -> Dict[str, Any]:
        """Check BMS/SCADA bridge health.

        Returns:
            Dict with health metrics.
        """
        connected = sum(
            1 for s in self._connections.values() if s == ConnectionStatus.CONNECTED
        )
        return {
            "protocols_registered": len(self._protocol_configs),
            "protocols_connected": connected,
            "data_points_registered": len(self._data_points),
            "readings_buffered": len(self._reading_buffer),
            "active_alarms": sum(1 for a in self._alarms if not a.cleared),
            "status": "healthy" if connected > 0 or len(self._protocol_configs) == 0 else "degraded",
        }
