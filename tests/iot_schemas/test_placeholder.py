# -*- coding: utf-8 -*-
"""Phase 6.3 — IoT schemas placeholder smoke tests.

These tests just lock in the API surface so the FY27 placeholder stays
importable while FY28 PlantOS work lands inside it.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from greenlang.iot_schemas import (
    CanonicalEventEnvelope,
    CanonicalMeterReading,
    ProtocolType,
    QualityFlag,
    __fy27_scope__,
    __version__,
)


def test_placeholder_metadata_is_marked():
    assert __fy27_scope__ == "deferred"
    assert "placeholder" in __version__


def test_protocol_enum_complete():
    expected = {
        "opc_ua", "mqtt", "modbus", "bacnet", "energy_star_portfolio",
    }
    actual = {p.value for p in ProtocolType}
    assert expected.issubset(actual)


def test_quality_flag_enum_complete():
    actual = {q.value for q in QualityFlag}
    assert actual == {"good", "uncertain", "bad", "stale"}


def test_canonical_meter_reading_shape():
    reading = CanonicalMeterReading(
        site_id="berlin-site-a",
        meter_id="m-001",
        quantity=1234.5,
        unit="kWh",
        ts=datetime(2028, 6, 1, tzinfo=timezone.utc),
    )
    assert reading.quality == QualityFlag.GOOD
    assert reading.tags == {}
    dumped = reading.model_dump()
    assert dumped["site_id"] == "berlin-site-a"


def test_canonical_event_envelope_wraps_reading():
    payload = CanonicalMeterReading(
        site_id="s1",
        meter_id="m1",
        quantity=1.0,
        unit="kWh",
        ts=datetime.now(timezone.utc),
    )
    envelope = CanonicalEventEnvelope(
        protocol=ProtocolType.OPC_UA,
        source_uri="opc.tcp://plant-a:4840",
        payload=payload,
        metadata={"pipeline": "plantos-test"},
    )
    assert envelope.protocol == ProtocolType.OPC_UA
    assert envelope.payload.meter_id == "m1"


def test_envelope_rejects_non_canonical_payload():
    """Envelope must take only CanonicalMeterReading payloads in FY27."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        CanonicalEventEnvelope(
            protocol=ProtocolType.OPC_UA,
            payload={"site_id": "s1"},  # wrong shape
        )
