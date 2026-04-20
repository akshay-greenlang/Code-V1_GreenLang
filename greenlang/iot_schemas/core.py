# -*- coding: utf-8 -*-
"""
Canonical IoT event schemas — FY28 placeholder.

The models here are deliberately thin.  Full semantic validation (unit
coercion, provenance, equipment taxonomy) is a FY28 deliverable.  The
goal of this FY27 placeholder is to **reserve the class names and
import paths** so downstream work in PlantOS, BuildingOS, and DataCenter
CarbonOps doesn't have to paint itself into a corner.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import Field

from greenlang.schemas import GreenLangBase


class ProtocolType(str, Enum):
    """Supported IoT protocols — extend in FY28 PlantOS."""

    OPC_UA = "opc_ua"
    MQTT = "mqtt"
    MODBUS = "modbus"
    BACNET = "bacnet"       # FY28 BuildingOS
    ENERGY_STAR_PORTFOLIO = "energy_star_portfolio"  # batch fallback


class QualityFlag(str, Enum):
    """Data-quality markers forwarded from the source device."""

    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    STALE = "stale"


class CanonicalMeterReading(GreenLangBase):
    """Single meter/sensor observation in canonical form.

    The shape is intentionally close to what the existing agents already
    emit (``greenlang/agents/data/iot_meter_management_agent.py``,
    ``scada_connector_agent.py``) so that FY28 migration becomes a
    mechanical rename rather than a semantic rewrite.
    """

    site_id: str = Field(..., description="Site / facility identifier")
    meter_id: str = Field(..., description="Stable meter / sensor identifier")
    quantity: float = Field(..., description="Numeric value of the reading")
    unit: str = Field(..., description="Canonical unit (e.g. 'kWh', 'Nm3', 'm3/h')")
    ts: datetime = Field(..., description="Observation timestamp (UTC)")
    quality: QualityFlag = Field(default=QualityFlag.GOOD)
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Free-form tags (scope hint, equipment type, etc.)",
    )


class CanonicalEventEnvelope(GreenLangBase):
    """Protocol-tagged envelope wrapping a canonical payload.

    FY28 PlantOS will define richer payload types (heat-map frames,
    state-change events, alarm envelopes).  For the FY27 placeholder we
    only support :class:`CanonicalMeterReading` as the inner payload.
    """

    protocol: ProtocolType = Field(..., description="Source IoT protocol")
    source_uri: Optional[str] = Field(
        default=None,
        description="Originating endpoint (e.g. 'opc.tcp://plant-a:4840')",
    )
    received_at: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Timestamp at which the event was received (UTC)",
    )
    payload: CanonicalMeterReading
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Implementation-specific metadata (correlation IDs, "
        "pipeline hints, transformer stamps, ...)",
    )
