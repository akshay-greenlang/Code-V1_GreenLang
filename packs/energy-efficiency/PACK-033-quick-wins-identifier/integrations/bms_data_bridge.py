# -*- coding: utf-8 -*-
"""
BMSDataBridge - BMS/SCADA Data Integration for Real-Time Monitoring (PACK-033)
===============================================================================

This module provides integration with Building Management Systems (BMS) and
SCADA systems for real-time energy monitoring in the Quick Wins Identifier
pipeline. It defines data models for protocol adapters (BACnet, Modbus, OPC-UA),
meter reading ingestion, alarm events, and historical data retrieval.

Supported Protocols (data model):
    - BACnet/IP: Object/property addressing, COV subscriptions
    - Modbus TCP/RTU: Register addressing, function codes
    - OPC-UA: NodeId addressing, browse paths

Note: Actual protocol communication is delegated to external libraries.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-033 Quick Wins Identifier
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
    HUMIDITY_PCT = "humidity_pct"
    CO2_PPM = "co2_ppm"
    OCCUPANCY = "occupancy"
    LIGHTING_LUX = "lighting_lux"
    FLOW_RATE_M3H = "flow_rate_m3h"
    PRESSURE_BAR = "pressure_bar"
    STATUS_BOOL = "status_bool"


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


class BMSConfig(BaseModel):
    """Configuration for the BMS Data Bridge."""

    pack_id: str = Field(default="PACK-033")
    enable_provenance: bool = Field(default=True)
    default_polling_interval_seconds: float = Field(default=60.0, ge=1.0)
    max_concurrent_connections: int = Field(default=10, ge=1, le=100)
    reading_buffer_size: int = Field(default=10000, ge=100)
    enable_alarm_integration: bool = Field(default=True)
    protocol: ProtocolType = Field(default=ProtocolType.BACNET_IP)
    host: str = Field(default="localhost")
    port: int = Field(default=47808, ge=1, le=65535)
    timeout_seconds: float = Field(default=5.0, ge=0.5)


class DataPoint(BaseModel):
    """A BMS/SCADA data point with value and metadata."""

    point_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="", description="Human-readable data point name")
    point_type: DataPointType = Field(default=DataPointType.ENERGY_KWH)
    protocol: ProtocolType = Field(default=ProtocolType.BACNET_IP)
    address: str = Field(default="", description="Protocol-specific address")
    raw_value: float = Field(default=0.0)
    normalized_value: float = Field(default=0.0)
    unit: str = Field(default="")
    quality: str = Field(default="good", description="good|uncertain|bad")
    timestamp: datetime = Field(default_factory=_utcnow)
    scale_factor: float = Field(default=1.0)
    offset: float = Field(default=0.0)
    facility_zone: str = Field(default="")
    equipment_id: str = Field(default="")


class MeterReading(BaseModel):
    """A single meter reading from BMS/SCADA."""

    reading_id: str = Field(default_factory=_new_uuid)
    point_id: str = Field(default="")
    timestamp: datetime = Field(default_factory=_utcnow)
    raw_value: float = Field(default=0.0)
    normalized_value: float = Field(default=0.0)
    unit: str = Field(default="")
    quality: str = Field(default="good")
    source_protocol: str = Field(default="")


class AlarmEvent(BaseModel):
    """An alarm event from BMS/SCADA."""

    alarm_id: str = Field(default_factory=_new_uuid)
    point_id: str = Field(default="")
    timestamp: datetime = Field(default_factory=_utcnow)
    severity: str = Field(default="medium", description="critical|high|medium|low|info")
    description: str = Field(default="")
    value: float = Field(default=0.0)
    threshold: float = Field(default=0.0)
    acknowledged: bool = Field(default=False)
    cleared: bool = Field(default=False)


# ---------------------------------------------------------------------------
# BMSDataBridge
# ---------------------------------------------------------------------------


class BMSDataBridge:
    """BMS/SCADA data integration for real-time monitoring.

    Provides data models for BACnet/Modbus/OPC-UA protocol adapters,
    meter reading ingestion, alarm integration, and historical data
    retrieval for quick win verification.

    Attributes:
        config: Bridge configuration.
        _data_points: Registered data point mappings.
        _connections: Active connection statuses.
        _reading_buffer: Recent meter readings.
        _alarms: Active alarm events.

    Example:
        >>> bridge = BMSDataBridge()
        >>> connected = bridge.connect(BMSConfig(host="192.168.1.100"))
        >>> points = bridge.read_points(["point_1", "point_2"])
    """

    def __init__(self, config: Optional[BMSConfig] = None) -> None:
        """Initialize the BMS Data Bridge."""
        self.config = config or BMSConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._data_points: Dict[str, DataPoint] = {}
        self._connections: Dict[str, ConnectionStatus] = {}
        self._reading_buffer: List[MeterReading] = []
        self._alarms: List[AlarmEvent] = []
        self.logger.info(
            "BMSDataBridge initialized: protocol=%s, host=%s",
            self.config.protocol.value, self.config.host,
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

    def read_points(self, point_ids: List[str]) -> List[DataPoint]:
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

    def register_point(self, point: DataPoint) -> DataPoint:
        """Register a data point for monitoring.

        Args:
            point: Data point to register.

        Returns:
            Registered DataPoint.
        """
        self._data_points[point.point_id] = point
        self.logger.info(
            "Data point registered: %s (%s)", point.name, point.point_type.value
        )
        return point

    def get_historical(
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

    def get_alarms(self) -> List[AlarmEvent]:
        """Get all active (uncleared) alarm events.

        Returns:
            List of active AlarmEvent.
        """
        return [a for a in self._alarms if not a.cleared]

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
            timestamp=timestamp or _utcnow(),
            raw_value=raw_value,
            normalized_value=round(normalized, 4),
            unit=point.unit,
            source_protocol=point.protocol.value,
        )

        self._reading_buffer.append(reading)
        if len(self._reading_buffer) > self.config.reading_buffer_size:
            self._reading_buffer = self._reading_buffer[-self.config.reading_buffer_size:]

        return reading

    def check_health(self) -> Dict[str, Any]:
        """Check BMS bridge health.

        Returns:
            Dict with health metrics.
        """
        connected = sum(
            1 for s in self._connections.values() if s == ConnectionStatus.CONNECTED
        )
        return {
            "connections_active": connected,
            "data_points_registered": len(self._data_points),
            "readings_buffered": len(self._reading_buffer),
            "active_alarms": sum(1 for a in self._alarms if not a.cleared),
            "status": "healthy" if connected > 0 or len(self._connections) == 0 else "degraded",
        }
