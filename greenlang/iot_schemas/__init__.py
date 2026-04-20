# -*- coding: utf-8 -*-
"""
GreenLang IoT Schemas — v3 L1 Data Foundation (FY28 placeholder).

**Status (2026-04-20):** scheduled for FY28 PlantOS. This module reserves
the API surface so that FY28 IoT work has a well-known home. No FY27
pilot profile (CBAM importer, CSRD filer, SB 253 Scope 1+2 reporter,
Scope 3 spend) ingests real-time IoT streams — all FY27 intake is
batch-oriented (CSV/XLSX/PDF/ERP exports).

See ``docs/fy27-scope.md`` §2.3 for the deferral rationale and FY28
acceleration triggers.

Canonical event families (stubs only — full Pydantic models land in FY28):

- :mod:`greenlang.iot_schemas.opc_ua` — OPC-UA-flavoured events
- :mod:`greenlang.iot_schemas.mqtt`   — MQTT-flavoured events
- :mod:`greenlang.iot_schemas.modbus` — Modbus register reads

Quick-start (FY28)::

    from greenlang.iot_schemas import CanonicalMeterReading, CanonicalEventEnvelope

    reading = CanonicalMeterReading(
        site_id="berlin-site-a",
        meter_id="m-001",
        quantity=1234.5,
        unit="kWh",
        ts="2028-06-01T00:00:00+00:00",
    )
    envelope = CanonicalEventEnvelope(protocol="opc_ua", payload=reading)
"""
from __future__ import annotations

from greenlang.iot_schemas.core import (
    CanonicalEventEnvelope,
    CanonicalMeterReading,
    ProtocolType,
    QualityFlag,
)

__version__ = "0.0.1-fy28-placeholder"
__fy27_scope__ = "deferred"
__fy28_owner__ = "PlantOS"

__all__ = [
    "CanonicalEventEnvelope",
    "CanonicalMeterReading",
    "ProtocolType",
    "QualityFlag",
]
