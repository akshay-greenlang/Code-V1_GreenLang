# -*- coding: utf-8 -*-
"""
Test Grid Mock Connector
=========================

Tests for deterministic grid intensity connector.

Verifies:
- Deterministic series generation
- Record/replay modes
- Snapshot integrity
- Provenance tracking
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from decimal import Decimal

from greenlang.connectors.grid.mock import GridIntensityMockConnector
from greenlang.connectors.models import GridIntensityQuery, TimeWindow
from greenlang.connectors.context import ConnectorContext, CacheMode
from greenlang.connectors.errors import ConnectorReplayRequired


class TestGridIntensityMockConnector:
    """Test suite for GridIntensityMockConnector"""

    @pytest.fixture
    def connector(self):
        """Create connector instance"""
        return GridIntensityMockConnector()

    @pytest.fixture
    def sample_query(self):
        """Create sample query for CA-ON region, 24 hours"""
        return GridIntensityQuery(
            region="CA-ON",
            window=TimeWindow(
                start=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
                end=datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc),
                resolution="hour"
            )
        )

    def test_connector_capabilities(self, connector):
        """Test connector declares correct capabilities"""
        caps = connector.capabilities

        assert caps.supports_time_series is True
        assert caps.min_resolution == "hour"
        assert caps.requires_auth is False
        assert caps.supports_streaming is False

    def test_generate_deterministic_series(self, connector, sample_query):
        """Test that same query produces same series"""
        ctx = ConnectorContext.for_record("grid/intensity/mock")

        # Run twice
        payload1, prov1 = asyncio.run(connector.fetch(sample_query, ctx))
        payload2, prov2 = asyncio.run(connector.fetch(sample_query, ctx))

        # Should be identical
        assert payload1.series == payload2.series
        assert prov1.seed == prov2.seed
        assert prov1.query_hash == prov2.query_hash

    def test_series_length_matches_hours(self, connector, sample_query):
        """Test that series length equals requested hours"""
        ctx = ConnectorContext.for_record("grid/intensity/mock")
        payload, _ = asyncio.run(connector.fetch(sample_query, ctx))

        # 24 hours requested
        assert len(payload.series) == 24

    def test_series_timestamps_ordered(self, connector, sample_query):
        """Test that series timestamps are strictly ascending"""
        ctx = ConnectorContext.for_record("grid/intensity/mock")
        payload, _ = asyncio.run(connector.fetch(sample_query, ctx))

        timestamps = [point.ts for point in payload.series]
        assert timestamps == sorted(timestamps)

    def test_series_timestamps_hourly(self, connector, sample_query):
        """Test that series has hourly timestamps"""
        ctx = ConnectorContext.for_record("grid/intensity/mock")
        payload, _ = asyncio.run(connector.fetch(sample_query, ctx))

        # Check first two points are 1 hour apart
        if len(payload.series) >= 2:
            delta = payload.series[1].ts - payload.series[0].ts
            assert delta == timedelta(hours=1)

    def test_values_use_decimal(self, connector, sample_query):
        """Test that values use Decimal type"""
        ctx = ConnectorContext.for_record("grid/intensity/mock")
        payload, _ = asyncio.run(connector.fetch(sample_query, ctx))

        for point in payload.series:
            assert isinstance(point.value, Decimal)

    def test_values_within_range(self, connector, sample_query):
        """Test that all values are within [50, 900] gCO2/kWh"""
        ctx = ConnectorContext.for_record("grid/intensity/mock")
        payload, _ = asyncio.run(connector.fetch(sample_query, ctx))

        for point in payload.series:
            assert 50 <= point.value <= 900

    def test_quality_is_simulated(self, connector, sample_query):
        """Test that all points have 'simulated' quality"""
        ctx = ConnectorContext.for_record("grid/intensity/mock")
        payload, _ = asyncio.run(connector.fetch(sample_query, ctx))

        for point in payload.series:
            assert point.quality == "simulated"

    def test_replay_mode_requires_snapshot(self, connector, sample_query):
        """Test that replay mode raises error without snapshot"""
        ctx = ConnectorContext.for_replay("grid/intensity/mock")

        with pytest.raises(ConnectorReplayRequired):
            asyncio.run(connector.fetch(sample_query, ctx))

    def test_snapshot_round_trip(self, connector, sample_query, tmp_path):
        """Test snapshot write and restore"""
        ctx = ConnectorContext.for_record(
            "grid/intensity/mock",
            cache_dir=tmp_path
        )

        # Record mode: fetch and snapshot
        payload1, prov1 = asyncio.run(connector.fetch(sample_query, ctx))
        snapshot_bytes = connector.snapshot(payload1, prov1)

        # Restore from snapshot
        payload2, prov2 = connector.restore(snapshot_bytes)

        # Should be identical
        assert payload1.series == payload2.series
        assert payload1.region == payload2.region
        assert prov1.seed == prov2.seed

    def test_provenance_complete(self, connector, sample_query):
        """Test that provenance has all required fields"""
        ctx = ConnectorContext.for_record("grid/intensity/mock")
        _, prov = asyncio.run(connector.fetch(sample_query, ctx))

        assert prov.connector_id == "grid/intensity/mock"
        assert prov.connector_version == "0.1.0"
        assert prov.mode == "record"
        assert prov.query_hash is not None
        assert prov.schema_hash is not None
        assert prov.seed is not None

    def test_different_regions_different_data(self, connector):
        """Test that different regions produce different data"""
        ctx = ConnectorContext.for_record("grid/intensity/mock")

        query_ca = GridIntensityQuery(
            region="CA-ON",
            window=TimeWindow(
                start=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
                end=datetime(2025, 1, 1, 2, 0, tzinfo=timezone.utc),
                resolution="hour"
            )
        )

        query_us = GridIntensityQuery(
            region="US-CAISO",
            window=TimeWindow(
                start=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
                end=datetime(2025, 1, 1, 2, 0, tzinfo=timezone.utc),
                resolution="hour"
            )
        )

        payload_ca, _ = asyncio.run(connector.fetch(query_ca, ctx))
        payload_us, _ = asyncio.run(connector.fetch(query_us, ctx))

        # Different regions should have different values
        assert payload_ca.series[0].value != payload_us.series[0].value
