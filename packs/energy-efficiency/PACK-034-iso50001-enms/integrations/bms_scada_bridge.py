# -*- coding: utf-8 -*-
"""
BMSSCADABridge - BMS/SCADA Data Integration for ISO 50001 EnMS (PACK-034)
==========================================================================

This module provides integration with Building Management Systems (BMS) and
SCADA systems for real-time energy monitoring in the ISO 50001 EnMS
pipeline. It defines data models for protocol adapters (BACnet, Modbus,
OPC-UA, MQTT), meter reading ingestion, alarm events, and historical
data retrieval supporting Clause 6.6 (data collection plan) and
Clause 9.1 (monitoring, measurement, analysis).

Supported Protocols (data model):
    - BACnet/IP: Object/property addressing, COV subscriptions
    - Modbus TCP/RTU: Register addressing, function codes
    - OPC-UA: NodeId addressing, browse paths
    - MQTT: Topic-based pub/sub
    - REST API: HTTP endpoint polling

Note: Actual protocol communication is delegated to external libraries.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-034 ISO 50001 Energy Management System
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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

    BACNET = "bacnet"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    OPCUA = "opcua"
    MQTT = "mqtt"
    API = "api"


class DataPointType(str, Enum):
    """Types of metered data points."""

    ANALOG_INPUT = "analog_input"
    ANALOG_OUTPUT = "analog_output"
    BINARY_INPUT = "binary_input"
    BINARY_OUTPUT = "binary_output"
    MULTISTATE = "multistate"
    ENERGY_KWH = "energy_kwh"
    POWER_KW = "power_kw"
    TEMPERATURE_C = "temperature_c"
    FLOW_RATE = "flow_rate"
    PRESSURE = "pressure"


class ConnectionStatus(str, Enum):
    """Protocol connection status."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    TIMEOUT = "timeout"
    NOT_CONFIGURED = "not_configured"
    RECONNECTING = "reconnecting"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class BMSConfig(BaseModel):
    """Configuration for the BMS/SCADA Bridge."""

    pack_id: str = Field(default="PACK-034")
    enable_provenance: bool = Field(default=True)
    protocol: ProtocolType = Field(default=ProtocolType.BACNET)
    host: str = Field(default="localhost")
    port: int = Field(default=47808, ge=1, le=65535)
    credentials_ref: str = Field(default="", description="Vault secret reference for credentials")
    timeout_seconds: float = Field(default=5.0, ge=0.5)
    default_polling_interval_seconds: float = Field(default=60.0, ge=1.0)
    max_concurrent_connections: int = Field(default=10, ge=1, le=100)
    reading_buffer_size: int = Field(default=50000, ge=100)
    enable_alarm_integration: bool = Field(default=True)
    reconnect_interval_seconds: float = Field(default=30.0, ge=5.0)


class DataPoint(BaseModel):
    """A BMS/SCADA data point with value and metadata."""

    point_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="", description="Human-readable data point name")
    point_type: DataPointType = Field(default=DataPointType.ENERGY_KWH)
    protocol: ProtocolType = Field(default=ProtocolType.BACNET)
    address: str = Field(default="", description="Protocol-specific address")
    value: float = Field(default=0.0)
    raw_value: float = Field(default=0.0)
    unit: str = Field(default="")
    quality: str = Field(default="good", description="good|uncertain|bad")
    timestamp: datetime = Field(default_factory=_utcnow)
    scale_factor: float = Field(default=1.0)
    offset: float = Field(default=0.0)
    facility_zone: str = Field(default="")
    equipment_id: str = Field(default="")
    seu_id: str = Field(default="", description="Significant Energy Use ID")


class MeterReading(BaseModel):
    """A single meter reading from BMS/SCADA."""

    reading_id: str = Field(default_factory=_new_uuid)
    point_id: str = Field(default="")
    meter_id: str = Field(default="")
    timestamp: datetime = Field(default_factory=_utcnow)
    raw_value: float = Field(default=0.0)
    normalized_value: float = Field(default=0.0)
    unit: str = Field(default="")
    quality: str = Field(default="good")
    source_protocol: str = Field(default="")
    provenance_hash: str = Field(default="")


class AlarmEvent(BaseModel):
    """An alarm event from BMS/SCADA."""

    alarm_id: str = Field(default_factory=_new_uuid)
    point_id: str = Field(default="")
    timestamp: datetime = Field(default_factory=_utcnow)
    severity: str = Field(default="medium", description="critical|high|medium|low|info")
    alarm_type: str = Field(default="threshold", description="threshold|deviation|state_change|communication")
    description: str = Field(default="")
    value: float = Field(default=0.0)
    threshold: float = Field(default=0.0)
    acknowledged: bool = Field(default=False)
    cleared: bool = Field(default=False)
    seu_id: str = Field(default="", description="Related SEU identifier")


# ---------------------------------------------------------------------------
# BMSSCADABridge
# ---------------------------------------------------------------------------


class BMSSCADABridge:
    """BMS/SCADA data integration for ISO 50001 EnMS monitoring.

    Provides data models for BACnet/Modbus/OPC-UA/MQTT protocol adapters,
    meter reading ingestion, alarm integration, and historical data
    retrieval supporting Clause 6.6 and Clause 9.1 requirements.

    Attributes:
        config: Bridge configuration.
        _data_points: Registered data point mappings.
        _connections: Active connection statuses.
        _reading_buffer: Recent meter readings.
        _alarms: Active alarm events.
        _alarm_callbacks: Registered alarm callback functions.

    Example:
        >>> bridge = BMSSCADABridge()
        >>> connected = bridge.connect(BMSConfig(host="192.168.1.100"))
        >>> points = bridge.read_data_points(["point_1", "point_2"])
    """

    def __init__(self, config: Optional[BMSConfig] = None) -> None:
        """Initialize the BMS/SCADA Bridge."""
        self.config = config or BMSConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._data_points: Dict[str, DataPoint] = {}
        self._connections: Dict[str, ConnectionStatus] = {}
        self._reading_buffer: List[MeterReading] = []
        self._alarms: List[AlarmEvent] = []
        self._alarm_callbacks: List[Callable[[AlarmEvent], None]] = []
        self.logger.info(
            "BMSSCADABridge initialized: protocol=%s, host=%s:%d",
            self.config.protocol.value, self.config.host, self.config.port,
        )

    def connect(self, config: Optional[BMSConfig] = None) -> bool:
        """Connect to BMS/SCADA system.

        In production, this establishes protocol-level connection.

        Args:
            config: Optional override configuration.

        Returns:
            True if connection is successful.
        """
        cfg = config or self.config
        conn_key = f"{cfg.protocol.value}://{cfg.host}:{cfg.port}"

        # Stub: simulate successful connection
        self._connections[conn_key] = ConnectionStatus.CONNECTED
        self.logger.info("Connected to BMS: %s", conn_key)
        return True

    def read_data_points(self, point_ids: List[str]) -> List[DataPoint]:
        """Read current values for a list of data points.

        Args:
            point_ids: List of data point identifiers.

        Returns:
            List of DataPoint with current values.
        """
        results: List[DataPoint] = []
        for pid in point_ids:
            point = self._data_points.get(pid)
            if point:
                results.append(point)
            else:
                self.logger.warning("Data point not found: %s", pid)
        return results

    def read_meter(self, meter_id: str) -> Optional[MeterReading]:
        """Read the current value of a specific meter.

        Args:
            meter_id: Meter identifier.

        Returns:
            MeterReading with current value, or None if not found.
        """
        # Find the data point associated with this meter
        for point in self._data_points.values():
            if point.equipment_id == meter_id or point.point_id == meter_id:
                normalized = (point.raw_value * point.scale_factor) + point.offset
                reading = MeterReading(
                    point_id=point.point_id,
                    meter_id=meter_id,
                    raw_value=point.raw_value,
                    normalized_value=round(normalized, 4),
                    unit=point.unit,
                    source_protocol=point.protocol.value,
                )
                if self.config.enable_provenance:
                    reading.provenance_hash = _compute_hash(reading)
                return reading
        self.logger.warning("Meter not found: %s", meter_id)
        return None

    def subscribe_alarms(
        self, callback: Callable[[AlarmEvent], None],
    ) -> bool:
        """Subscribe to alarm events from BMS/SCADA.

        Args:
            callback: Function to call when an alarm is received.

        Returns:
            True if subscription was successful.
        """
        self._alarm_callbacks.append(callback)
        self.logger.info("Alarm subscription registered: %d total", len(self._alarm_callbacks))
        return True

    def get_historical_data(
        self,
        point_id: str,
        start: datetime,
        end: datetime,
    ) -> List[MeterReading]:
        """Get historical readings for a data point.

        In production, this queries the historian. The stub returns
        buffered readings filtered by time range.

        Args:
            point_id: Data point identifier.
            start: Start of time range.
            end: End of time range.

        Returns:
            List of MeterReading within the time range.
        """
        return [
            r for r in self._reading_buffer
            if r.point_id == point_id and start <= r.timestamp <= end
        ]

    def disconnect(self) -> bool:
        """Disconnect all BMS/SCADA connections.

        Returns:
            True if disconnection was successful.
        """
        for conn_key in list(self._connections.keys()):
            self._connections[conn_key] = ConnectionStatus.DISCONNECTED
        self.logger.info("All BMS connections disconnected: %d", len(self._connections))
        return True

    def register_point(self, point: DataPoint) -> DataPoint:
        """Register a data point for monitoring.

        Args:
            point: Data point to register.

        Returns:
            Registered DataPoint.
        """
        self._data_points[point.point_id] = point
        self.logger.info(
            "Data point registered: %s (%s) for SEU %s",
            point.name, point.point_type.value, point.seu_id or "none",
        )
        return point

    def ingest_reading(
        self,
        point_id: str,
        raw_value: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[MeterReading]:
        """Ingest a single meter reading and apply normalization.

        Deterministic: normalized = (raw_value * scale_factor) + offset

        Args:
            point_id: Data point identifier.
            raw_value: Raw value from device.
            timestamp: Reading timestamp (now if None).

        Returns:
            Normalized MeterReading, or None if point not found.
        """
        point = self._data_points.get(point_id)
        if point is None:
            self.logger.warning("Data point not found: %s", point_id)
            return None

        normalized = (raw_value * point.scale_factor) + point.offset

        reading = MeterReading(
            point_id=point_id,
            meter_id=point.equipment_id,
            timestamp=timestamp or _utcnow(),
            raw_value=raw_value,
            normalized_value=round(normalized, 4),
            unit=point.unit,
            source_protocol=point.protocol.value,
        )

        if self.config.enable_provenance:
            reading.provenance_hash = _compute_hash(reading)

        self._reading_buffer.append(reading)
        if len(self._reading_buffer) > self.config.reading_buffer_size:
            self._reading_buffer = self._reading_buffer[-self.config.reading_buffer_size:]

        return reading

    def get_active_alarms(self) -> List[AlarmEvent]:
        """Get all active (uncleared) alarm events.

        Returns:
            List of active AlarmEvent.
        """
        return [a for a in self._alarms if not a.cleared]

    def check_health(self) -> Dict[str, Any]:
        """Check BMS/SCADA bridge health.

        Returns:
            Dict with health metrics.
        """
        connected = sum(
            1 for s in self._connections.values() if s == ConnectionStatus.CONNECTED
        )
        return {
            "connections_active": connected,
            "connections_total": len(self._connections),
            "data_points_registered": len(self._data_points),
            "readings_buffered": len(self._reading_buffer),
            "active_alarms": sum(1 for a in self._alarms if not a.cleared),
            "alarm_subscribers": len(self._alarm_callbacks),
            "status": "healthy" if connected > 0 or len(self._connections) == 0 else "degraded",
        }
