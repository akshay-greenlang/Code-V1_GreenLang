"""
GL-001 ThermalCommand - Webhook Manager Tests

Comprehensive tests for webhook manager including:
- Signature generation and verification
- Rate limiting
- Dead letter queue
- Idempotency tracking
- Delivery preparation

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import asyncio
from datetime import datetime, timezone, timedelta
import time
import pytest
from pydantic import SecretStr

from ..webhook_manager import (
    WebhookManager,
    DeliveryResult,
    DeliveryStatus,
    SignatureGenerator,
    RateLimiter,
    DeadLetterQueue,
    IdempotencyTracker,
)
from ..webhook_config import (
    WebhookConfig,
    WebhookEndpoint,
    EndpointRegistry,
    EndpointStatus,
    AuthenticationType,
    RateLimitConfig,
)
from ..webhook_events import (
    WebhookEventType,
    HeatPlanCreatedEvent,
)


class TestSignatureGenerator:
    """Tests for SignatureGenerator."""

    @pytest.fixture
    def generator(self):
        """Create signature generator."""
        return SignatureGenerator()

    def test_generate_signature(self, generator):
        """Test signature generation."""
        secret = "my-secret-key"
        payload = '{"event_id": "test-123"}'
        timestamp = 1700000000

        signature = generator.generate(secret, payload, timestamp)

        assert signature.startswith("v1=")
        assert len(signature) == 3 + 64  # "v1=" + 64 hex chars

    def test_signature_deterministic(self, generator):
        """Test signature is deterministic."""
        secret = "my-secret-key"
        payload = '{"event_id": "test-123"}'
        timestamp = 1700000000

        sig1 = generator.generate(secret, payload, timestamp)
        sig2 = generator.generate(secret, payload, timestamp)

        assert sig1 == sig2

    def test_signature_changes_with_payload(self, generator):
        """Test signature changes with different payloads."""
        secret = "my-secret-key"
        timestamp = 1700000000

        sig1 = generator.generate(secret, '{"id": 1}', timestamp)
        sig2 = generator.generate(secret, '{"id": 2}', timestamp)

        assert sig1 != sig2

    def test_signature_changes_with_secret(self, generator):
        """Test signature changes with different secrets."""
        payload = '{"event_id": "test-123"}'
        timestamp = 1700000000

        sig1 = generator.generate("secret1", payload, timestamp)
        sig2 = generator.generate("secret2", payload, timestamp)

        assert sig1 != sig2

    def test_signature_changes_with_timestamp(self, generator):
        """Test signature changes with different timestamps."""
        secret = "my-secret-key"
        payload = '{"event_id": "test-123"}'

        sig1 = generator.generate(secret, payload, 1700000000)
        sig2 = generator.generate(secret, payload, 1700000001)

        assert sig1 != sig2

    def test_verify_valid_signature(self, generator):
        """Test verifying valid signature."""
        secret = "my-secret-key"
        payload = '{"event_id": "test-123"}'
        timestamp = int(time.time())

        signature = generator.generate(secret, payload, timestamp)
        is_valid, error = generator.verify(secret, payload, timestamp, signature)

        assert is_valid is True
        assert error is None

    def test_verify_invalid_signature(self, generator):
        """Test verifying invalid signature."""
        secret = "my-secret-key"
        payload = '{"event_id": "test-123"}'
        timestamp = int(time.time())

        is_valid, error = generator.verify(
            secret, payload, timestamp, "v1=invalid"
        )

        assert is_valid is False
        assert "mismatch" in error.lower()

    def test_verify_expired_timestamp(self, generator):
        """Test verifying expired timestamp."""
        secret = "my-secret-key"
        payload = '{"event_id": "test-123"}'
        old_timestamp = int(time.time()) - 400  # 400 seconds ago

        signature = generator.generate(secret, payload, old_timestamp)
        is_valid, error = generator.verify(
            secret, payload, old_timestamp, signature, tolerance_seconds=300
        )

        assert is_valid is False
        assert "old" in error.lower()

    def test_verify_wrong_version(self, generator):
        """Test verifying wrong signature version."""
        is_valid, error = generator.verify(
            "secret", '{}', int(time.time()), "v2=abc123"
        )

        assert is_valid is False
        assert "version" in error.lower()


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.fixture
    def limiter(self):
        """Create rate limiter."""
        return RateLimiter()

    @pytest.fixture
    def endpoint(self):
        """Create test endpoint with rate limiting."""
        return WebhookEndpoint(
            endpoint_id="ep-001",
            name="Test Endpoint",
            url="https://example.com/webhook",
            secret=SecretStr("secret"),
            rate_limit_config=RateLimitConfig(
                requests_per_second=10.0,
                burst_size=5,
                enabled=True,
                window_seconds=60,
                max_requests_per_window=100
            )
        )

    def test_initial_requests_allowed(self, limiter, endpoint):
        """Test initial requests within burst size are allowed."""
        for _ in range(5):  # Burst size is 5
            can_proceed, wait_time = limiter.check_rate_limit(endpoint)
            assert can_proceed is True
            assert wait_time == 0.0

    def test_burst_exceeded_requires_wait(self, limiter, endpoint):
        """Test requests exceeding burst require wait."""
        # Exhaust burst
        for _ in range(5):
            limiter.check_rate_limit(endpoint)

        # Next request should require wait
        can_proceed, wait_time = limiter.check_rate_limit(endpoint)
        assert can_proceed is False
        assert wait_time > 0

    def test_rate_limiting_disabled(self, limiter):
        """Test rate limiting when disabled."""
        endpoint = WebhookEndpoint(
            endpoint_id="ep-001",
            name="Test",
            url="https://example.com/webhook",
            secret=SecretStr("secret"),
            rate_limit_config=RateLimitConfig(enabled=False)
        )

        # Should always allow when disabled
        for _ in range(100):
            can_proceed, _ = limiter.check_rate_limit(endpoint)
            assert can_proceed is True

    def test_tokens_regenerate(self, limiter, endpoint):
        """Test tokens regenerate over time."""
        # Exhaust burst
        for _ in range(5):
            limiter.check_rate_limit(endpoint)

        # Wait for token regeneration (at 10/s, wait 0.2s for 2 tokens)
        time.sleep(0.2)

        # Should be able to proceed now
        can_proceed, _ = limiter.check_rate_limit(endpoint)
        assert can_proceed is True

    def test_reset_clears_state(self, limiter, endpoint):
        """Test reset clears rate limit state."""
        # Exhaust burst
        for _ in range(5):
            limiter.check_rate_limit(endpoint)

        # Reset
        limiter.reset(endpoint.endpoint_id)

        # Should allow requests again
        can_proceed, _ = limiter.check_rate_limit(endpoint)
        assert can_proceed is True


class TestDeadLetterQueue:
    """Tests for DeadLetterQueue."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return WebhookConfig()

    @pytest.fixture
    def dlq(self, config):
        """Create DLQ."""
        return DeadLetterQueue(config)

    @pytest.fixture
    def sample_event(self):
        """Create sample event."""
        return HeatPlanCreatedEvent(
            plan_id="plan-001",
            horizon_hours=24,
            expected_cost_usd=15000.0,
            expected_emissions_kg_co2e=1200.0
        )

    @pytest.fixture
    def sample_result(self, sample_event):
        """Create sample delivery result."""
        return DeliveryResult(
            event_id=sample_event.event_id,
            endpoint_id="ep-001",
            status=DeliveryStatus.FAILED,
            error_message="Connection timeout"
        )

    def test_add_entry(self, dlq, sample_event, sample_result):
        """Test adding entry to DLQ."""
        entry_id = dlq.add(
            sample_event,
            "ep-001",
            sample_result,
            "Connection timeout"
        )

        assert entry_id is not None
        assert dlq.size() == 1

    def test_get_entry(self, dlq, sample_event, sample_result):
        """Test getting entry from DLQ."""
        entry_id = dlq.add(sample_event, "ep-001", sample_result, "Error")

        entry = dlq.get_entry(entry_id)
        assert entry is not None
        assert entry.event.event_id == sample_event.event_id
        assert entry.endpoint_id == "ep-001"

    def test_get_entries_for_endpoint(self, dlq, sample_event, sample_result):
        """Test getting entries for specific endpoint."""
        dlq.add(sample_event, "ep-001", sample_result, "Error 1")
        dlq.add(sample_event, "ep-001", sample_result, "Error 2")
        dlq.add(sample_event, "ep-002", sample_result, "Error 3")

        ep1_entries = dlq.get_entries_for_endpoint("ep-001")
        assert len(ep1_entries) == 2

        ep2_entries = dlq.get_entries_for_endpoint("ep-002")
        assert len(ep2_entries) == 1

    def test_remove_entry(self, dlq, sample_event, sample_result):
        """Test removing entry from DLQ."""
        entry_id = dlq.add(sample_event, "ep-001", sample_result, "Error")

        result = dlq.remove_entry(entry_id)
        assert result is True
        assert dlq.size() == 0

        # Remove non-existent
        result = dlq.remove_entry("nonexistent")
        assert result is False

    def test_cleanup_expired(self, dlq, sample_event, sample_result):
        """Test cleaning up expired entries."""
        # Add entry
        entry_id = dlq.add(sample_event, "ep-001", sample_result, "Error")

        # Manually set expiration to past
        entry = dlq.get_entry(entry_id)
        entry.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        # Cleanup should remove it
        removed = dlq.cleanup_expired()
        assert removed == 1
        assert dlq.size() == 0

    def test_get_statistics(self, dlq, sample_event, sample_result):
        """Test getting DLQ statistics."""
        dlq.add(sample_event, "ep-001", sample_result, "Error")
        dlq.add(sample_event, "ep-002", sample_result, "Error")

        stats = dlq.get_statistics()
        assert stats["total_entries"] == 2
        assert "ep-001" in stats["by_endpoint"]
        assert "ep-002" in stats["by_endpoint"]


class TestIdempotencyTracker:
    """Tests for IdempotencyTracker."""

    @pytest.fixture
    def tracker(self):
        """Create idempotency tracker."""
        return IdempotencyTracker(ttl_seconds=60)

    def test_not_duplicate_initially(self, tracker):
        """Test event is not duplicate initially."""
        assert tracker.is_duplicate("event-1", "ep-001") is False

    def test_is_duplicate_after_delivery(self, tracker):
        """Test event is duplicate after successful delivery."""
        result = DeliveryResult(
            event_id="event-1",
            endpoint_id="ep-001",
            status=DeliveryStatus.DELIVERED
        )
        tracker.record_delivery("event-1", "ep-001", result)

        assert tracker.is_duplicate("event-1", "ep-001") is True

    def test_not_duplicate_if_failed(self, tracker):
        """Test failed delivery is not considered duplicate."""
        result = DeliveryResult(
            event_id="event-1",
            endpoint_id="ep-001",
            status=DeliveryStatus.FAILED
        )
        tracker.record_delivery("event-1", "ep-001", result)

        assert tracker.is_duplicate("event-1", "ep-001") is False

    def test_different_endpoints_independent(self, tracker):
        """Test different endpoints are independent."""
        result = DeliveryResult(
            event_id="event-1",
            endpoint_id="ep-001",
            status=DeliveryStatus.DELIVERED
        )
        tracker.record_delivery("event-1", "ep-001", result)

        assert tracker.is_duplicate("event-1", "ep-001") is True
        assert tracker.is_duplicate("event-1", "ep-002") is False

    def test_get_cached_result(self, tracker):
        """Test getting cached result."""
        result = DeliveryResult(
            event_id="event-1",
            endpoint_id="ep-001",
            status=DeliveryStatus.DELIVERED,
            http_status_code=200
        )
        tracker.record_delivery("event-1", "ep-001", result)

        cached = tracker.get_cached_result("event-1", "ep-001")
        assert cached is not None
        assert cached.http_status_code == 200

    def test_cleanup_expired(self, tracker):
        """Test cleaning up expired records."""
        # Create tracker with short TTL
        short_ttl_tracker = IdempotencyTracker(ttl_seconds=0)

        result = DeliveryResult(
            event_id="event-1",
            endpoint_id="ep-001",
            status=DeliveryStatus.DELIVERED
        )
        short_ttl_tracker.record_delivery("event-1", "ep-001", result)

        # Record should expire immediately
        time.sleep(0.01)
        removed = short_ttl_tracker.cleanup_expired()
        assert removed == 1


class TestWebhookManager:
    """Tests for WebhookManager."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WebhookConfig(endpoints=[
            WebhookEndpoint(
                endpoint_id="ep-001",
                name="Test Endpoint",
                url="https://example.com/webhook",
                secret=SecretStr("test-secret"),
                event_types={WebhookEventType.HEAT_PLAN_CREATED}
            )
        ])

    @pytest.fixture
    def manager(self, config):
        """Create webhook manager."""
        return WebhookManager(config)

    @pytest.fixture
    def sample_event(self):
        """Create sample event."""
        return HeatPlanCreatedEvent(
            plan_id="plan-001",
            horizon_hours=24,
            expected_cost_usd=15000.0,
            expected_emissions_kg_co2e=1200.0
        )

    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert manager.registry.endpoint_count() == 1
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_start_and_shutdown(self, manager):
        """Test manager start and shutdown."""
        await manager.start()
        assert manager._running is True

        await manager.shutdown()
        assert manager._running is False

    def test_generate_signature(self, manager, config):
        """Test signature generation for endpoint."""
        endpoint = manager.registry.get_endpoint("ep-001")
        payload = '{"event": "test"}'
        timestamp = int(time.time())

        signature = manager.generate_signature(endpoint, payload, timestamp)

        assert signature is not None
        assert signature.startswith("v1=")

    def test_verify_signature(self, manager, config):
        """Test signature verification."""
        endpoint = manager.registry.get_endpoint("ep-001")
        payload = '{"event": "test"}'
        timestamp = int(time.time())

        signature = manager.generate_signature(endpoint, payload, timestamp)
        is_valid, error = manager.verify_signature(
            endpoint, payload, timestamp, signature
        )

        assert is_valid is True
        assert error is None

    def test_prepare_delivery(self, manager, sample_event):
        """Test delivery preparation."""
        endpoint = manager.registry.get_endpoint("ep-001")

        headers, payload, delivery_id = manager.prepare_delivery(
            sample_event, endpoint
        )

        assert "Content-Type" in headers
        assert "X-GL-Signature-256" in headers
        assert "X-GL-Timestamp" in headers
        assert "X-GL-Idempotency-Key" in headers
        assert delivery_id is not None
        assert len(payload) > 0

    def test_prepare_delivery_with_bearer_token(self, manager):
        """Test delivery preparation with bearer token auth."""
        endpoint = WebhookEndpoint(
            endpoint_id="ep-002",
            name="Bearer Endpoint",
            url="https://example.com/webhook",
            authentication_type=AuthenticationType.BEARER_TOKEN,
            bearer_token=SecretStr("bearer-token-123")
        )
        manager.registry.register_endpoint(endpoint)

        event = HeatPlanCreatedEvent(
            plan_id="plan-001",
            horizon_hours=24,
            expected_cost_usd=15000.0,
            expected_emissions_kg_co2e=1200.0
        )

        headers, _, _ = manager.prepare_delivery(event, endpoint)

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer bearer-token-123"

    def test_get_endpoints_for_event(self, manager):
        """Test getting endpoints for event type."""
        endpoints = manager.get_endpoints_for_event(
            WebhookEventType.HEAT_PLAN_CREATED
        )
        assert len(endpoints) == 1

        endpoints = manager.get_endpoints_for_event(
            WebhookEventType.MAINTENANCE_TRIGGER
        )
        assert len(endpoints) == 0

    def test_idempotency_tracking(self, manager, sample_event):
        """Test idempotency tracking."""
        endpoint = manager.registry.get_endpoint("ep-001")

        # Initially not duplicate
        assert manager.is_duplicate(sample_event, endpoint) is False

        # Record successful delivery
        result = DeliveryResult(
            event_id=sample_event.event_id,
            endpoint_id=endpoint.endpoint_id,
            status=DeliveryStatus.DELIVERED
        )
        manager.record_delivery_result(sample_event, endpoint, result)

        # Now is duplicate
        assert manager.is_duplicate(sample_event, endpoint) is True

    def test_rate_limit_check(self, manager):
        """Test rate limit checking."""
        endpoint = manager.registry.get_endpoint("ep-001")

        can_proceed, wait_time = manager.check_rate_limit(endpoint)
        assert can_proceed is True

    def test_add_to_dlq(self, manager, sample_event):
        """Test adding to dead letter queue."""
        endpoint = manager.registry.get_endpoint("ep-001")
        result = DeliveryResult(
            event_id=sample_event.event_id,
            endpoint_id=endpoint.endpoint_id,
            status=DeliveryStatus.FAILED,
            error_message="Connection failed"
        )

        entry_id = manager.add_to_dlq(
            sample_event, endpoint, result, "Connection failed"
        )

        assert entry_id is not None
        assert manager.dlq.size() == 1

    def test_get_statistics(self, manager, sample_event):
        """Test getting statistics."""
        endpoint = manager.registry.get_endpoint("ep-001")

        # Record some deliveries
        for status in [DeliveryStatus.DELIVERED, DeliveryStatus.DELIVERED, DeliveryStatus.FAILED]:
            result = DeliveryResult(
                event_id=f"event-{status}",
                endpoint_id=endpoint.endpoint_id,
                status=status
            )
            manager.record_delivery_result(sample_event, endpoint, result)

        stats = manager.get_statistics()

        assert stats["endpoint_count"] == 1
        assert stats["delivery_stats"]["total_delivered"] == 2
        assert stats["delivery_stats"]["total_failed"] == 1

    def test_update_endpoint_status(self, manager):
        """Test updating endpoint status."""
        result = manager.update_endpoint_status("ep-001", EndpointStatus.DEGRADED)
        assert result is True

        endpoint = manager.registry.get_endpoint("ep-001")
        assert endpoint.status == EndpointStatus.DEGRADED
