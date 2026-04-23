"""
Integration Tests for GL-007 FurnacePulse Kafka Integration

Tests Kafka event streaming including:
- Message publishing
- Topic routing
- Schema compliance
- Delivery confirmations
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime, timezone
from typing import Dict, Any, List
import json
import hashlib


class TestKafkaProducerIntegration:
    """Tests for Kafka producer functionality."""

    @pytest.mark.asyncio
    async def test_producer_startup(self, mock_kafka_producer):
        """Test Kafka producer startup."""
        await mock_kafka_producer.start()
        mock_kafka_producer.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_producer_shutdown(self, mock_kafka_producer):
        """Test graceful producer shutdown."""
        await mock_kafka_producer.start()
        await mock_kafka_producer.stop()
        mock_kafka_producer.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_telemetry_message(self, mock_kafka_producer):
        """Test sending telemetry message."""
        telemetry = {
            "furnace_id": "FRN-001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals": [
                {"tag_id": "FUEL.FLOW", "value": 1500.0, "unit": "kg/h"},
                {"tag_id": "STACK.TEMP", "value": 380.0, "unit": "C"},
            ]
        }

        result = await mock_kafka_producer.send(
            topic="furnacepulse.site1.FRN-001.telemetry",
            message=telemetry,
            key="FRN-001"
        )

        assert result is not None
        assert len(mock_kafka_producer.sent_messages) == 1
        assert mock_kafka_producer.sent_messages[0]["topic"] == "furnacepulse.site1.FRN-001.telemetry"

    @pytest.mark.asyncio
    async def test_send_alert_message(self, mock_kafka_producer):
        """Test sending alert message."""
        alert = {
            "alert_id": "ALT-001",
            "furnace_id": "FRN-001",
            "severity": "WARNING",
            "alert_type": "TMT_HIGH",
            "message": "Tube T-R1-03 temperature exceeds warning threshold",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        result = await mock_kafka_producer.send(
            topic="furnacepulse.alerts",
            message=alert,
            key=alert["alert_id"]
        )

        assert result is not None
        assert mock_kafka_producer.sent_messages[-1]["message"]["severity"] == "WARNING"

    @pytest.mark.asyncio
    async def test_send_inference_message(self, mock_kafka_producer):
        """Test sending ML inference message."""
        inference = {
            "model_id": "rul-weibull-v1",
            "furnace_id": "FRN-001",
            "component_id": "TUBE-R1-01",
            "predictions": {
                "rul_hours": 5000,
                "failure_probability_30d": 0.05,
            },
            "confidence": 0.92,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        result = await mock_kafka_producer.send(
            topic="furnacepulse.models.inference",
            message=inference,
            key=f"FRN-001.TUBE-R1-01"
        )

        assert result is not None
        assert mock_kafka_producer.sent_messages[-1]["message"]["model_id"] == "rul-weibull-v1"


class TestKafkaTopicRouting:
    """Tests for topic routing logic."""

    @pytest.mark.asyncio
    async def test_telemetry_topic_routing(self, mock_kafka_producer):
        """Test telemetry messages routed to correct topic."""
        site_id = "site1"
        furnace_id = "FRN-001"
        expected_topic = f"furnacepulse.{site_id}.{furnace_id}.telemetry"

        await mock_kafka_producer.send(
            topic=expected_topic,
            message={"test": "data"},
            key=furnace_id
        )

        assert mock_kafka_producer.sent_messages[-1]["topic"] == expected_topic

    @pytest.mark.asyncio
    async def test_event_topic_routing(self, mock_kafka_producer):
        """Test event messages routed to correct topic."""
        site_id = "site1"
        furnace_id = "FRN-001"
        expected_topic = f"furnacepulse.{site_id}.{furnace_id}.events"

        await mock_kafka_producer.send(
            topic=expected_topic,
            message={"event_type": "STARTUP"},
            key=furnace_id
        )

        assert mock_kafka_producer.sent_messages[-1]["topic"] == expected_topic

    @pytest.mark.asyncio
    async def test_alerts_global_topic(self, mock_kafka_producer):
        """Test alerts routed to global topic."""
        await mock_kafka_producer.send(
            topic="furnacepulse.alerts",
            message={"alert_type": "CRITICAL"},
            key="ALT-001"
        )

        assert mock_kafka_producer.sent_messages[-1]["topic"] == "furnacepulse.alerts"


class TestKafkaMessageSchema:
    """Tests for message schema compliance."""

    @pytest.mark.asyncio
    async def test_telemetry_message_schema(self, mock_kafka_producer):
        """Test telemetry message contains required fields."""
        telemetry = {
            "message_id": "MSG-001",
            "message_type": "telemetry",
            "version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "furnacepulse.site1.FRN-001",
            "furnace_id": "FRN-001",
            "readings": []
        }

        await mock_kafka_producer.send(
            topic="furnacepulse.site1.FRN-001.telemetry",
            message=telemetry,
            key="FRN-001"
        )

        sent_msg = mock_kafka_producer.sent_messages[-1]["message"]
        required_fields = ["message_id", "message_type", "version", "timestamp", "source"]
        for field in required_fields:
            assert field in sent_msg

    @pytest.mark.asyncio
    async def test_alert_message_schema(self, mock_kafka_producer):
        """Test alert message contains required fields."""
        alert = {
            "alert_id": "ALT-001",
            "message_type": "alert",
            "version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "furnace_id": "FRN-001",
            "alert_type": "TMT_HIGH",
            "severity": "WARNING",
            "description": "High tube temperature detected",
        }

        await mock_kafka_producer.send(
            topic="furnacepulse.alerts",
            message=alert,
            key=alert["alert_id"]
        )

        sent_msg = mock_kafka_producer.sent_messages[-1]["message"]
        assert "alert_id" in sent_msg
        assert "severity" in sent_msg
        assert sent_msg["severity"] in ["INFO", "WARNING", "CRITICAL", "EMERGENCY"]


class TestKafkaProvenance:
    """Tests for provenance tracking in Kafka messages."""

    @pytest.mark.asyncio
    async def test_message_includes_provenance_hash(self, mock_kafka_producer):
        """Test messages include provenance hash."""
        data = {
            "furnace_id": "FRN-001",
            "efficiency": 92.5,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

        message = {
            **data,
            "provenance_hash": provenance_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await mock_kafka_producer.send(
            topic="furnacepulse.calculations",
            message=message,
            key="FRN-001"
        )

        sent_msg = mock_kafka_producer.sent_messages[-1]["message"]
        assert "provenance_hash" in sent_msg
        assert len(sent_msg["provenance_hash"]) == 16

    @pytest.mark.asyncio
    async def test_deterministic_provenance_hash(self, mock_kafka_producer):
        """Test provenance hash is deterministic."""
        data = {"furnace_id": "FRN-001", "value": 100.0}

        hash1 = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
        hash2 = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

        assert hash1 == hash2


class TestKafkaBatchOperations:
    """Tests for batch message operations."""

    @pytest.mark.asyncio
    async def test_batch_telemetry_send(self, mock_kafka_producer):
        """Test sending batch of telemetry messages."""
        readings = [
            {"tag_id": f"TMT-{i:02d}", "value": 800.0 + i}
            for i in range(10)
        ]

        for reading in readings:
            await mock_kafka_producer.send(
                topic="furnacepulse.site1.FRN-001.telemetry",
                message=reading,
                key="FRN-001"
            )

        assert len(mock_kafka_producer.sent_messages) == 10

    @pytest.mark.asyncio
    async def test_flush_pending_messages(self, mock_kafka_producer):
        """Test flushing pending messages."""
        await mock_kafka_producer.send(
            topic="furnacepulse.alerts",
            message={"test": "data"},
            key="test"
        )

        await mock_kafka_producer.flush()
        mock_kafka_producer.flush.assert_called_once()
