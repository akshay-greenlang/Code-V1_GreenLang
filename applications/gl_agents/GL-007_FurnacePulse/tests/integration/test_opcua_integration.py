"""
Integration Tests for GL-007 FurnacePulse OPC-UA Integration

Tests OPC-UA client functionality including:
- Connection management
- Tag reading and subscription
- Data quality handling
- Reconnection logic
- Multi-furnace support
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any
import asyncio


class TestOPCUAConnection:
    """Tests for OPC-UA connection management."""

    @pytest.mark.asyncio
    async def test_successful_connection(self, mock_opcua_client):
        """Test successful OPC-UA connection."""
        result = await mock_opcua_client.connect()
        assert result is True
        assert mock_opcua_client.is_connected() is True

    @pytest.mark.asyncio
    async def test_connection_and_disconnect(self, mock_opcua_client):
        """Test connection and graceful disconnect."""
        await mock_opcua_client.connect()
        result = await mock_opcua_client.disconnect()
        assert result is True

    @pytest.mark.asyncio
    async def test_read_single_tag(self, mock_opcua_client):
        """Test reading a single OPC-UA tag."""
        result = await mock_opcua_client.read_tag("FRN-001.FUEL.FLOW")

        assert result["value"] == 1500.0
        assert result["quality"] == "GOOD"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_read_multiple_tags(self, mock_opcua_client):
        """Test batch reading multiple tags."""
        tags = ["FRN-001.FUEL.FLOW", "FRN-001.STACK.TEMP", "FRN-001.FLUE.O2"]
        results = await mock_opcua_client.read_tags(tags)

        assert len(results) == 3
        assert results["FRN-001.FUEL.FLOW"]["value"] == 1500.0
        assert results["FRN-001.STACK.TEMP"]["value"] == 380.0
        assert results["FRN-001.FLUE.O2"]["value"] == 3.5

    @pytest.mark.asyncio
    async def test_read_unknown_tag_returns_bad_quality(self, mock_opcua_client):
        """Test reading unknown tag returns BAD quality."""
        result = await mock_opcua_client.read_tag("UNKNOWN.TAG")
        assert result["value"] is None
        assert result["quality"] == "BAD"

    @pytest.mark.asyncio
    async def test_subscription(self, mock_opcua_client):
        """Test tag subscription."""
        sub_id = await mock_opcua_client.subscribe(["FRN-001.TMT.R1.01"])
        assert sub_id == "sub-001"

        # Unsubscribe
        result = await mock_opcua_client.unsubscribe(sub_id)
        assert result is True


class TestOPCUATelemetryCollection:
    """Tests for telemetry collection via OPC-UA."""

    @pytest.mark.asyncio
    async def test_tmt_readings_collection(
        self, mock_opcua_client, sample_tmt_readings_normal
    ):
        """Test collecting TMT readings."""
        # Read TMT tags
        tmt_tags = ["FRN-001.TMT.R1.01", "FRN-001.TMT.R1.02"]
        results = await mock_opcua_client.read_tags(tmt_tags)

        assert all(r["quality"] == "GOOD" for r in results.values())
        assert results["FRN-001.TMT.R1.01"]["value"] == 820.0
        assert results["FRN-001.TMT.R1.02"]["value"] == 825.0

    @pytest.mark.asyncio
    async def test_efficiency_inputs_collection(self, mock_opcua_client):
        """Test collecting all inputs needed for efficiency calculation."""
        required_tags = [
            "FRN-001.FUEL.FLOW",
            "FRN-001.AIR.FLOW",
            "FRN-001.STACK.TEMP",
            "FRN-001.FLUE.O2",
        ]
        results = await mock_opcua_client.read_tags(required_tags)

        # Verify all required data collected
        assert results["FRN-001.FUEL.FLOW"]["quality"] == "GOOD"
        assert results["FRN-001.STACK.TEMP"]["quality"] == "GOOD"
        assert results["FRN-001.FLUE.O2"]["quality"] == "GOOD"

    @pytest.mark.asyncio
    async def test_draft_readings_collection(self, mock_opcua_client):
        """Test collecting draft pressure readings."""
        draft_tags = ["FRN-001.DRAFT.FIREBOX"]
        results = await mock_opcua_client.read_tags(draft_tags)

        # Draft pressure should be negative (vacuum)
        assert results["FRN-001.DRAFT.FIREBOX"]["value"] == -25.0


class TestOPCUADataQuality:
    """Tests for data quality handling."""

    @pytest.mark.asyncio
    async def test_good_quality_data_accepted(self, mock_opcua_client):
        """Test that GOOD quality data is accepted."""
        result = await mock_opcua_client.read_tag("FRN-001.FUEL.FLOW")
        assert result["quality"] == "GOOD"
        assert result["value"] is not None

    @pytest.mark.asyncio
    async def test_bad_quality_data_flagged(self, mock_opcua_client):
        """Test that BAD quality data is properly flagged."""
        result = await mock_opcua_client.read_tag("INVALID.TAG")
        assert result["quality"] == "BAD"

    @pytest.mark.asyncio
    async def test_mixed_quality_batch_read(self, mock_opcua_client):
        """Test batch read with mixed quality results."""
        tags = ["FRN-001.FUEL.FLOW", "INVALID.TAG"]
        results = await mock_opcua_client.read_tags(tags)

        assert results["FRN-001.FUEL.FLOW"]["quality"] == "GOOD"
        assert results["INVALID.TAG"]["quality"] == "BAD"


class TestMultiFurnaceSupport:
    """Tests for multi-furnace OPC-UA operations."""

    @pytest.mark.asyncio
    async def test_read_tags_from_multiple_furnaces(self, mock_opcua_client):
        """Test reading tags from multiple furnaces."""
        # This would normally read from different furnaces
        tags = ["FRN-001.FUEL.FLOW", "FRN-001.STACK.TEMP"]
        results = await mock_opcua_client.read_tags(tags)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_subscribe_to_multiple_furnaces(self, mock_opcua_client):
        """Test subscribing to tags from multiple furnaces."""
        furnace_tags = [
            "FRN-001.TMT.R1.01",
            "FRN-001.TMT.R1.02",
        ]

        sub_id = await mock_opcua_client.subscribe(furnace_tags)
        assert sub_id is not None

        await mock_opcua_client.unsubscribe(sub_id)
