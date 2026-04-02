# -*- coding: utf-8 -*-
"""
Shared IoT/Sensor Module for Data Layer Agents
===============================================

Provides common enums, base models, and utilities shared across the
IoT/sensor data connector agents:

    - GL-DATA-X-002: SCADA/Historians Connector Agent
    - GL-DATA-X-003: BMS Connector Agent
    - GL-DATA-X-013: IoT Meter Management Agent

All three agents handle real-time sensor/meter data and share overlapping
patterns for data quality indicators, aggregation types, connection
configuration, and data point models. This module extracts those patterns
into a single source of truth.

Usage:
    >>> from greenlang.agents.data._iot_shared import (
    ...     SensorDataQuality,
    ...     SensorAggregation,
    ...     SensorConnectionBase,
    ...     SensorDataPointBase,
    ... )

Author: GreenLang Team
Version: 1.0.0
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Optional, Union

from pydantic import Field

from greenlang.schemas import GreenLangBase

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED ENUMS
# =============================================================================

class SensorDataQuality(str, Enum):
    """
    Unified data quality indicators for sensor/meter readings.

    Union of quality indicators from:
        - SCADA: GOOD, BAD, UNCERTAIN, STALE, SUBSTITUTED
        - BMS: GOOD, UNCERTAIN, BAD, MISSING
        - IoT Meter: quality_flag string values (good, bad, uncertain, estimated)

    Values are lowercase to align with existing conventions across all three
    agent files.
    """
    GOOD = "good"
    BAD = "bad"
    UNCERTAIN = "uncertain"
    STALE = "stale"
    SUBSTITUTED = "substituted"
    MISSING = "missing"
    ESTIMATED = "estimated"


class SensorAggregation(str, Enum):
    """
    Unified aggregation types for time-series sensor data.

    Union of aggregation types from:
        - SCADA: RAW, AVERAGE, MINIMUM, MAXIMUM, SUM, COUNT, RANGE,
                 DELTA, TIME_WEIGHTED, INTERPOLATED
        - BMS: (interval-based, implicit average/sum)
        - IoT Meter: (automatic, manual, estimated reading types)

    This enum covers all aggregation modes needed across the three agents.
    """
    RAW = "raw"
    AVERAGE = "average"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    SUM = "sum"
    COUNT = "count"
    RANGE = "range"
    DELTA = "delta"
    TIME_WEIGHTED = "time_weighted"
    INTERPOLATED = "interpolated"
    LAST = "last"
    FIRST = "first"
    MEDIAN = "median"
    STDDEV = "stddev"


# =============================================================================
# SHARED BASE MODELS
# =============================================================================

class SensorConnectionBase(GreenLangBase):
    """
    Base connection configuration for sensor/meter data sources.

    Contains fields common to SCADA ConnectionConfig, BMS BMSConnectionConfig,
    and IoT meter communication setups. Agent-specific subclasses add their
    own protocol enums and domain-specific fields.

    Attributes:
        connection_id: Unique identifier for this connection.
        host: Server hostname or IP address.
        port: Server port number.
        username: Optional authentication username.
        password: Optional authentication password.
        timeout_seconds: Connection timeout in seconds (1-300).
        retry_count: Number of retry attempts on failure (0-10).
        ssl_enabled: Whether to use SSL/TLS for the connection.
        certificate_path: Optional path to SSL certificate file.
    """
    connection_id: str = Field(..., description="Unique connection identifier")
    host: str = Field(..., description="Server hostname or IP address")
    port: int = Field(..., description="Server port number")
    username: Optional[str] = Field(None, description="Authentication username")
    password: Optional[str] = Field(None, description="Authentication password")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Connection timeout")
    retry_count: int = Field(default=3, ge=0, le=10, description="Retry attempts on failure")
    ssl_enabled: bool = Field(default=True, description="Use SSL/TLS")
    certificate_path: Optional[str] = Field(None, description="Path to SSL certificate")


class SensorDataPointBase(GreenLangBase):
    """
    Base data point model for sensor/meter readings.

    Contains fields common to SCADA DataPoint, BMS BMSDataPoint, and IoT
    MeterReading. Agent-specific subclasses add their own identifier fields
    (tag_id, point_id, meter_id) and domain-specific metadata.

    Attributes:
        timestamp: When the reading was taken.
        value: The measured value (numeric or boolean/string for status).
        quality: Data quality indicator using the shared SensorDataQuality enum.
        unit: Optional engineering unit of the measurement.
    """
    timestamp: datetime = Field(..., description="Reading timestamp")
    value: Union[float, int, bool, str] = Field(..., description="Measured value")
    quality: SensorDataQuality = Field(
        default=SensorDataQuality.GOOD,
        description="Data quality indicator"
    )
    unit: Optional[str] = Field(None, description="Engineering unit")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "SensorDataQuality",
    "SensorAggregation",
    "SensorConnectionBase",
    "SensorDataPointBase",
]
