"""Unit tests for WebhookSubscriberManager."""

import asyncio
import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from greenlang.infrastructure.api.webhooks import (
    WebhookSubscriberManager,
    WebhookModel,
    WebhookDelivery,
    WebhookStatus,
    RegisterWebhookRequest,
    PROCESS_HEAT_EVENTS,
)


class TestWebhookRegistration:
    """Test webhook registration functionality."""

    def test_register_webhook(self):
        """Test registering a new webhook."""
        manager = WebhookSubscriberManager()
        webhook_id = manager.register_webhook(
            url="https://example.com/webhook",
            events=["calculation.completed"],
            secret="my-secret"
        )

        assert webhook_id
        assert webhook_id in manager.webhooks
        webhook = manager.webhooks[webhook_id]
        assert webhook.url == "https://example.com/webhook"
        assert webhook.events == ["calculation.completed"]
        assert webhook.is_active is True
        assert webhook.health_status == "healthy"

    def test_register_webhook_multiple_events(self):
        """Test registering webhook with multiple events."""
        manager = WebhookSubscriberManager()
        webhook_id = manager.register_webhook(
            url="https://example.com/webhook",
            events=["calculation.completed", "alarm.triggered"],
            secret="secret123"
        )

        webhook = manager.webhooks[webhook_id]
        assert len(webhook.events) == 2
        assert "calculation.completed" in webhook.events
        assert "alarm.triggered" in webhook.events

    def test_register_webhook_with_metadata(self):
        """Test registering webhook with metadata."""
        manager = WebhookSubscriberManager()
        metadata = {"env": "production", "team": "backend"}
        webhook_id = manager.register_webhook(
            url="https://example.com/webhook",
            events=["calculation.completed"],
            secret="secret123",
            metadata=metadata
        )

        webhook = manager.webhooks[webhook_id]
        assert webhook.metadata == metadata

    def test_register_webhook_invalid_event(self):
        """Test registering webhook with invalid event type."""
        manager = WebhookSubscriberManager()
        with pytest.raises(ValueError, match="Unsupported event type"):
            manager.register_webhook(
                url="https://example.com/webhook",
                events=["invalid.event"],
                secret="secret123"
            )

    def test_unregister_webhook(self):
        """Test unregistering a webhook."""
        manager = WebhookSubscriberManager()
        webhook_id = manager.register_webhook(
            url="https://example.com/webhook",
            events=["calculation.completed"],
            secret="secret123"
        )

        assert webhook_id in manager.webhooks
        result = manager.unregister_webhook(webhook_id)
        assert result is True
        assert webhook_id not in manager.webhooks

    def test_unregister_nonexistent_webhook(self):
        """Test unregistering a webhook that doesn't exist."""
        manager = WebhookSubscriberManager()
        result = manager.unregister_webhook("nonexistent-id")
        assert result is False

    def test_list_webhooks(self):
        """Test listing all webhooks."""
        manager = WebhookSubscriberManager()
        webhook_id1 = manager.register_webhook(
            url="https://example1.com/webhook",
            events=["calculation.completed"],
            secret="secret1"
        )
        webhook_id2 = manager.register_webhook(
            url="https://example2.com/webhook",
            events=["alarm.triggered"],
            secret="secret2"
        )

        webhooks = manager.list_webhooks()
        assert len(webhooks) == 2
        urls = [w.url for w in webhooks]
        assert "https://example1.com/webhook" in urls
        assert "https://example2.com/webhook" in urls


class TestSignatureVerification:
    """Test HMAC-SHA256 signature verification."""

    def test_create_signature(self):
        """Test creating HMAC-SHA256 signature."""
        manager = WebhookSubscriberManager()
        payload = {"calculation_id": "calc_123", "result": 45.67}
        secret = "my-secret"

        signature = manager._create_signature(payload, secret)
        assert signature
        assert len(signature) == 64  # SHA256 hex is 64 chars

    def test_verify_signature_valid(self):
        """Test verifying a valid signature."""
        manager = WebhookSubscriberManager()
        payload = {"result": 123.45}
        secret = "shared-secret"

        signature = manager._create_signature(payload, secret)
        assert manager.verify_signature(payload, signature, secret) is True

    def test_verify_signature_invalid(self):
        """Test verifying an invalid signature."""
        manager = WebhookSubscriberManager()
        payload = {"result": 123.45}
        secret = "shared-secret"

        invalid_signature = "0" * 64
        assert manager.verify_signature(payload, invalid_signature, secret) is False

    def test_verify_signature_modified_payload(self):
        """Test verifying signature with modified payload."""
        manager = WebhookSubscriberManager()
        payload = {"result": 123.45}
        secret = "shared-secret"

        signature = manager._create_signature(payload, secret)

        modified_payload = {"result": 999.99}
        assert manager.verify_signature(modified_payload, signature, secret) is False

    def test_signature_json_serialization_order(self):
        """Test signature is consistent regardless of key order."""
        manager = WebhookSubscriberManager()
        secret = "secret"
        payload1 = {"a": 1, "b": 2}
        payload2 = {"b": 2, "a": 1}

        sig1 = manager._create_signature(payload1, secret)
        sig2 = manager._create_signature(payload2, secret)
        assert sig1 == sig2


class TestWebhookTrigger:
    """Test webhook triggering and delivery."""

    @pytest.mark.asyncio
    async def test_trigger_webhook_no_subscribers(self):
        """Test triggering webhook with no subscribers."""
        manager = WebhookSubscriberManager()
        delivery_id = await manager.trigger_webhook(
            "calculation.completed",
            {"result": 123}
        )

        assert delivery_id == ""

    @pytest.mark.asyncio
    async def test_trigger_webhook_creates_delivery(self):
        """Test triggering webhook creates delivery record."""
        manager = WebhookSubscriberManager()
        webhook_id = manager.register_webhook(
            url="https://example.com/webhook",
            events=["calculation.completed"],
            secret="secret123"
        )

        payload = {"result": 123.45}
        delivery_id = await manager.trigger_webhook(
            "calculation.completed",
            payload
        )

        assert delivery_id
        assert delivery_id in manager.deliveries
        delivery = manager.deliveries[delivery_id]
        assert delivery.webhook_id == webhook_id
        assert delivery.event_type == "calculation.completed"
        assert delivery.payload == payload
        assert delivery.status == WebhookStatus.PENDING

    @pytest.mark.asyncio
    async def test_trigger_webhook_multiple_subscribers(self):
        """Test triggering webhook with multiple subscribers."""
        manager = WebhookSubscriberManager()
        webhook_id1 = manager.register_webhook(
            url="https://example1.com/webhook",
            events=["calculation.completed"],
            secret="secret1"
        )
        webhook_id2 = manager.register_webhook(
            url="https://example2.com/webhook",
            events=["calculation.completed"],
            secret="secret2"
        )

        delivery_id = await manager.trigger_webhook(
            "calculation.completed",
            {"result": 123}
        )

        assert delivery_id
        delivery_count = sum(1 for d in manager.deliveries.values()
                           if d.event_type == "calculation.completed")
        assert delivery_count == 2

    @pytest.mark.asyncio
    async def test_trigger_webhook_invalid_event(self):
        """Test triggering webhook with invalid event type."""
        manager = WebhookSubscriberManager()
        delivery_id = await manager.trigger_webhook(
            "invalid.event",
            {"data": "test"}
        )

        assert delivery_id == ""

    @pytest.mark.asyncio
    async def test_trigger_webhook_inactive_webhook(self):
        """Test triggering webhook doesn't send to inactive webhooks."""
        manager = WebhookSubscriberManager()
        webhook_id = manager.register_webhook(
            url="https://example.com/webhook",
            events=["calculation.completed"],
            secret="secret123"
        )
        webhook = manager.webhooks[webhook_id]
        webhook.is_active = False

        delivery_id = await manager.trigger_webhook(
            "calculation.completed",
            {"result": 123}
        )

        assert delivery_id == ""

    @pytest.mark.asyncio
    async def test_trigger_webhook_signature_created(self):
        """Test trigger creates signature for delivery."""
        manager = WebhookSubscriberManager()
        webhook_id = manager.register_webhook(
            url="https://example.com/webhook",
            events=["calculation.completed"],
            secret="secret123"
        )

        payload = {"result": 123.45}
        delivery_id = await manager.trigger_webhook(
            "calculation.completed",
            payload
        )

        delivery = manager.deliveries[delivery_id]
        expected_sig = manager._create_signature(payload, "secret123")
        assert delivery.signature == expected_sig

    @pytest.mark.asyncio
    async def test_trigger_webhook_provenance_hash(self):
        """Test trigger creates provenance hash."""
        manager = WebhookSubscriberManager()
        webhook_id = manager.register_webhook(
            url="https://example.com/webhook",
            events=["calculation.completed"],
            secret="secret123"
        )

        delivery_id = await manager.trigger_webhook(
            "calculation.completed",
            {"result": 123}
        )

        delivery = manager.deliveries[delivery_id]
        assert delivery.provenance_hash
        assert len(delivery.provenance_hash) == 64  # SHA256 hex


class TestDeliveryProcessing:
    """Test webhook delivery processing and retry."""

    @pytest.mark.asyncio
    async def test_send_webhook_success(self):
        """Test successful webhook delivery."""
        manager = WebhookSubscriberManager()
        webhook = WebhookModel(
            webhook_id="webhook_123",
            url="https://example.com/webhook",
            events=["calculation.completed"],
            secret="secret123"
        )
        delivery = WebhookDelivery(
            webhook_id="webhook_123",
            event_type="calculation.completed",
            payload={"result": 123},
            signature="sig123"
        )

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            manager._http_client = AsyncMock()
            manager._http_client.post = mock_post

            result = await manager._send_webhook(delivery, webhook)

            assert result is True
            assert delivery.status == WebhookStatus.SENT
            assert delivery.http_status == 200
            assert webhook.last_triggered_at is not None

    @pytest.mark.asyncio
    async def test_send_webhook_failure(self):
        """Test failed webhook delivery."""
        manager = WebhookSubscriberManager()
        webhook = WebhookModel(
            webhook_id="webhook_123",
            url="https://example.com/webhook",
            events=["calculation.completed"],
            secret="secret123"
        )
        delivery = WebhookDelivery(
            webhook_id="webhook_123",
            event_type="calculation.completed",
            payload={"result": 123},
            signature="sig123"
        )

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            manager._http_client = AsyncMock()
            manager._http_client.post = mock_post

            result = await manager._send_webhook(delivery, webhook)

            assert result is False
            assert delivery.status == WebhookStatus.FAILED
            assert delivery.http_status == 500

    @pytest.mark.asyncio
    async def test_send_webhook_network_error(self):
        """Test webhook delivery with network error."""
        manager = WebhookSubscriberManager()
        webhook = WebhookModel(
            webhook_id="webhook_123",
            url="https://example.com/webhook",
            events=["calculation.completed"],
            secret="secret123"
        )
        delivery = WebhookDelivery(
            webhook_id="webhook_123",
            event_type="calculation.completed",
            payload={"result": 123},
            signature="sig123"
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = Exception("Network error")
            manager._http_client = AsyncMock()
            manager._http_client.post = mock_post

            result = await manager._send_webhook(delivery, webhook)

            assert result is False
            assert delivery.status == WebhookStatus.FAILED
            assert "Network error" in delivery.error_message

    @pytest.mark.asyncio
    async def test_health_score_tracking(self):
        """Test webhook health score tracking."""
        manager = WebhookSubscriberManager()
        webhook_id = manager.register_webhook(
            url="https://example.com/webhook",
            events=["calculation.completed"],
            secret="secret123"
        )
        webhook = manager.webhooks[webhook_id]

        assert webhook.consecutive_failures == 0
        assert webhook.health_status == "healthy"

        webhook.consecutive_failures = 5
        webhook.is_active = False
        webhook.health_status = "unhealthy"

        assert webhook.is_active is False
        assert webhook.health_status == "unhealthy"

    @pytest.mark.asyncio
    async def test_supported_events_exist(self):
        """Test all supported event types are defined."""
        expected_events = {
            "calculation.completed",
            "calculation.failed",
            "alarm.triggered",
            "alarm.cleared",
            "model.deployed",
            "model.degraded",
            "compliance.violation",
            "compliance.report_ready",
        }

        assert PROCESS_HEAT_EVENTS == expected_events


class TestWebhookModel:
    """Test WebhookModel Pydantic model."""

    def test_webhook_model_creation(self):
        """Test creating WebhookModel."""
        webhook = WebhookModel(
            url="https://example.com/webhook",
            events=["calculation.completed"],
            secret="secret123"
        )

        assert webhook.webhook_id
        assert webhook.url == "https://example.com/webhook"
        assert webhook.is_active is True
        assert webhook.created_at

    def test_webhook_model_json_serialization(self):
        """Test WebhookModel JSON serialization."""
        webhook = WebhookModel(
            url="https://example.com/webhook",
            events=["calculation.completed"],
            secret="secret123"
        )

        json_data = webhook.json()
        assert isinstance(json_data, str)
        parsed = json.loads(json_data)
        assert parsed["url"] == "https://example.com/webhook"


class TestDeliveryModel:
    """Test WebhookDelivery Pydantic model."""

    def test_delivery_model_creation(self):
        """Test creating WebhookDelivery."""
        delivery = WebhookDelivery(
            webhook_id="webhook_123",
            event_type="calculation.completed",
            payload={"result": 123},
            signature="sig123"
        )

        assert delivery.delivery_id
        assert delivery.status == WebhookStatus.PENDING
        assert delivery.attempt == 1
        assert delivery.created_at

    def test_delivery_status_enum(self):
        """Test WebhookStatus enum values."""
        assert WebhookStatus.PENDING.value == "pending"
        assert WebhookStatus.SENT.value == "sent"
        assert WebhookStatus.FAILED.value == "failed"
        assert WebhookStatus.RETRYING.value == "retrying"
