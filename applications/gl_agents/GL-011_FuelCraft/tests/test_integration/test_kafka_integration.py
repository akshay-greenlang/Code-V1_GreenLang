# -*- coding: utf-8 -*-
"""
Integration Tests for Kafka Integration

Tests Kafka producer and consumer integration including:
- Message publishing
- Schema validation
- Circuit breaker behavior
- Message serialization
- Event routing

Author: GL-TestEngineer
Date: 2025-01-01
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integration.kafka_producer import (
    FuelCraftKafkaProducer,
    KafkaProducerConfig,
    MessageType,
    AlertSeverity,
    RecommendationPublishedEvent,
    AuditEventPublished,
    AlertPublishedEvent,
    MessageHeader,
    CircuitBreaker,
    CircuitBreakerState,
)


@pytest.mark.integration
@pytest.mark.asyncio
class TestKafkaProducerLifecycle:
    """Tests for Kafka producer lifecycle management."""

    async def test_producer_start_stop(self, kafka_producer_config):
        """Test producer start and stop lifecycle."""
        config = KafkaProducerConfig(**kafka_producer_config)
        producer = FuelCraftKafkaProducer(config)

        await producer.start()
        assert producer._started is True

        await producer.stop()
        assert producer._started is False

    async def test_producer_metrics_initialization(self, kafka_producer_config):
        """Test producer metrics are initialized."""
        config = KafkaProducerConfig(**kafka_producer_config)
        producer = FuelCraftKafkaProducer(config)

        await producer.start()

        metrics = producer.get_metrics()

        assert "messages_sent" in metrics
        assert "messages_failed" in metrics
        assert "bytes_sent" in metrics
        assert "circuit_breaker_state" in metrics

        await producer.stop()


@pytest.mark.integration
@pytest.mark.asyncio
class TestKafkaProducerPublishing:
    """Tests for message publishing."""

    async def test_publish_recommendation(self, kafka_producer_config):
        """Test publishing optimization recommendation."""
        config = KafkaProducerConfig(**kafka_producer_config)
        producer = FuelCraftKafkaProducer(config)

        await producer.start()

        result = await producer.publish_recommendation(
            run_id="RUN-TEST-001",
            site_id="SITE-001",
            total_cost_usd=50000.0,
            total_emissions_mtco2e=15.5,
            savings_percent=8.5,
            emission_reduction_percent=5.2,
            fuel_mix={"natural_gas": 0.7, "diesel": 0.3},
            procurement_actions=[
                {"fuel": "natural_gas", "quantity_mmbtu": 1000, "action": "procure"}
            ],
            bundle_hash="abc123def456" * 5 + "abcd",
            input_snapshot_ids={"inventory": "hash1", "prices": "hash2"},
            effective_start=datetime.now(timezone.utc),
            effective_end=datetime.now(timezone.utc) + timedelta(days=7),
            correlation_id="CORR-001",
        )

        assert result is True

        metrics = producer.get_metrics()
        assert metrics["messages_sent"] >= 1

        await producer.stop()

    async def test_publish_audit_event(self, kafka_producer_config):
        """Test publishing audit event."""
        config = KafkaProducerConfig(**kafka_producer_config)
        producer = FuelCraftKafkaProducer(config)

        await producer.start()

        result = await producer.publish_audit_event(
            event_type="optimization_completed",
            user_id="system",
            action="optimize",
            resource_type="fuel_mix",
            resource_id="RUN-TEST-001",
            run_id="RUN-TEST-001",
            before_state={"status": "pending"},
            after_state={"status": "completed"},
            change_summary="Optimization completed successfully",
            computation_hash="abc123" * 10 + "abcd",
        )

        assert result is True

        await producer.stop()

    async def test_publish_alert(self, kafka_producer_config):
        """Test publishing alert notification."""
        config = KafkaProducerConfig(**kafka_producer_config)
        producer = FuelCraftKafkaProducer(config)

        await producer.start()

        result = await producer.publish_alert(
            site_id="SITE-001",
            alert_type="low_inventory",
            alert_name="Low Natural Gas Inventory",
            severity=AlertSeverity.WARNING,
            description="Natural gas inventory below threshold",
            recommended_actions=["Increase procurement", "Check contracts"],
            related_run_id="RUN-TEST-001",
            trigger_value=15000.0,
            threshold_value=20000.0,
        )

        assert result is True

        await producer.stop()


@pytest.mark.integration
@pytest.mark.asyncio
class TestKafkaCircuitBreaker:
    """Tests for Kafka circuit breaker integration."""

    async def test_circuit_breaker_blocks_when_open(self, kafka_producer_config):
        """Test circuit breaker blocks messages when open."""
        config = KafkaProducerConfig(
            **kafka_producer_config,
            circuit_breaker_threshold=2,
            circuit_breaker_timeout_ms=1000,
        )
        producer = FuelCraftKafkaProducer(config)

        await producer.start()

        # Manually trigger circuit breaker
        producer._circuit_breaker._failure_count = 5
        producer._circuit_breaker._state = CircuitBreakerState.OPEN

        result = await producer.publish_alert(
            site_id="SITE-001",
            alert_type="test",
            alert_name="Test Alert",
            severity=AlertSeverity.INFO,
            description="Test",
        )

        # Should fail due to open circuit
        assert result is False

        await producer.stop()


@pytest.mark.integration
class TestMessageModels:
    """Tests for message model classes."""

    def test_recommendation_event_serialization(self):
        """Test recommendation event serializes to JSON."""
        event = RecommendationPublishedEvent(
            header=MessageHeader(
                message_type=MessageType.RECOMMENDATION,
                source="fuelcraft.test",
            ),
            run_id="RUN-001",
            site_id="SITE-001",
            total_cost_usd=50000.0,
            total_emissions_mtco2e=15.5,
            savings_percent=8.5,
            emission_reduction_percent=5.2,
            fuel_mix={"natural_gas": 0.7},
            bundle_hash="abc123" * 10 + "abcd",
            effective_start="2024-01-01T00:00:00Z",
            effective_end="2024-01-08T00:00:00Z",
        )

        message_bytes = event.to_kafka_message()

        assert message_bytes is not None
        assert len(message_bytes) > 0
        assert b"RUN-001" in message_bytes

    def test_audit_event_serialization(self):
        """Test audit event serializes to JSON."""
        event = AuditEventPublished(
            header=MessageHeader(
                message_type=MessageType.AUDIT,
                source="fuelcraft.audit",
            ),
            event_type="test_event",
            user_id="system",
            action="test",
            resource_type="test",
            resource_id="TEST-001",
        )

        message_bytes = event.to_kafka_message()

        assert message_bytes is not None
        assert b"test_event" in message_bytes

    def test_alert_event_serialization(self):
        """Test alert event serializes to JSON."""
        event = AlertPublishedEvent(
            header=MessageHeader(
                message_type=MessageType.ALERT,
                source="fuelcraft.alerts",
            ),
            site_id="SITE-001",
            alert_type="test_alert",
            alert_name="Test Alert",
            severity=AlertSeverity.WARNING,
            description="Test alert description",
        )

        message_bytes = event.to_kafka_message()

        assert message_bytes is not None
        assert b"test_alert" in message_bytes
        assert b"warning" in message_bytes


@pytest.mark.integration
class TestMessageHeader:
    """Tests for MessageHeader class."""

    def test_header_auto_generates_id(self):
        """Test header auto-generates message ID."""
        header = MessageHeader(
            message_type=MessageType.RECOMMENDATION,
            source="fuelcraft.test",
        )

        assert header.message_id is not None
        assert len(header.message_id) == 36  # UUID format

    def test_header_auto_generates_timestamp(self):
        """Test header auto-generates timestamp."""
        header = MessageHeader(
            message_type=MessageType.AUDIT,
            source="fuelcraft.test",
        )

        assert header.timestamp is not None


@pytest.mark.integration
class TestMessageTypeEnum:
    """Tests for MessageType enumeration."""

    def test_message_type_values(self):
        """Test MessageType enum values."""
        assert MessageType.RECOMMENDATION.value == "recommendation"
        assert MessageType.AUDIT.value == "audit"
        assert MessageType.ALERT.value == "alert"
        assert MessageType.INVENTORY_UPDATE.value == "inventory_update"
        assert MessageType.PRICE_UPDATE.value == "price_update"


@pytest.mark.integration
class TestAlertSeverityEnum:
    """Tests for AlertSeverity enumeration."""

    def test_alert_severity_values(self):
        """Test AlertSeverity enum values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.EMERGENCY.value == "emergency"


@pytest.mark.integration
@pytest.mark.asyncio
class TestKafkaProducerMetrics:
    """Tests for Kafka producer metrics."""

    async def test_metrics_track_sent_messages(self, kafka_producer_config):
        """Test metrics track sent message count."""
        config = KafkaProducerConfig(**kafka_producer_config)
        producer = FuelCraftKafkaProducer(config)

        await producer.start()

        initial_metrics = producer.get_metrics()
        initial_sent = initial_metrics["messages_sent"]

        await producer.publish_alert(
            site_id="SITE-001",
            alert_type="test",
            alert_name="Test",
            severity=AlertSeverity.INFO,
            description="Test",
        )

        final_metrics = producer.get_metrics()
        assert final_metrics["messages_sent"] == initial_sent + 1

        await producer.stop()

    async def test_metrics_track_bytes_sent(self, kafka_producer_config):
        """Test metrics track bytes sent."""
        config = KafkaProducerConfig(**kafka_producer_config)
        producer = FuelCraftKafkaProducer(config)

        await producer.start()

        await producer.publish_alert(
            site_id="SITE-001",
            alert_type="test",
            alert_name="Test Alert",
            severity=AlertSeverity.INFO,
            description="Test description",
        )

        metrics = producer.get_metrics()
        assert metrics["bytes_sent"] > 0

        await producer.stop()

    async def test_metrics_success_rate(self, kafka_producer_config):
        """Test metrics calculate success rate."""
        config = KafkaProducerConfig(**kafka_producer_config)
        producer = FuelCraftKafkaProducer(config)

        await producer.start()

        await producer.publish_alert(
            site_id="SITE-001",
            alert_type="test",
            alert_name="Test",
            severity=AlertSeverity.INFO,
            description="Test",
        )

        metrics = producer.get_metrics()
        assert metrics["success_rate"] == 1.0

        await producer.stop()
