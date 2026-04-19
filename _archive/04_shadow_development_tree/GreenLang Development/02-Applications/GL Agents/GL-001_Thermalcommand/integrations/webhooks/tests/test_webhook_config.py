"""
GL-001 ThermalCommand - Webhook Configuration Tests

Comprehensive tests for webhook configuration including:
- Endpoint configuration and validation
- Retry configuration
- Rate limit configuration
- Endpoint registry operations
- Secret management

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from datetime import datetime, timezone
import pytest
from pydantic import SecretStr

from ..webhook_config import (
    RetryConfig,
    RateLimitConfig,
    WebhookEndpoint,
    WebhookConfig,
    EndpointRegistry,
    EndpointStatus,
    AuthenticationType,
    DeadLetterQueueConfig,
    SecretManager,
)
from ..webhook_events import WebhookEventType


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_values(self):
        """Test default retry configuration values."""
        config = RetryConfig()

        assert config.max_retries == 5
        assert config.initial_delay_ms == 1000
        assert config.max_delay_ms == 300000
        assert config.backoff_multiplier == 2.0
        assert 500 in config.retry_on_status_codes
        assert config.jitter_enabled is True

    def test_custom_values(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=3,
            initial_delay_ms=500,
            max_delay_ms=60000,
            backoff_multiplier=1.5,
            jitter_enabled=False
        )

        assert config.max_retries == 3
        assert config.initial_delay_ms == 500
        assert config.backoff_multiplier == 1.5

    def test_calculate_delay_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            initial_delay_ms=1000,
            backoff_multiplier=2.0,
            jitter_enabled=False
        )

        assert config.calculate_delay(0) == 1000  # First retry
        assert config.calculate_delay(1) == 2000  # Second retry
        assert config.calculate_delay(2) == 4000  # Third retry

    def test_calculate_delay_max_cap(self):
        """Test delay is capped at max_delay_ms."""
        config = RetryConfig(
            initial_delay_ms=1000,
            max_delay_ms=5000,
            backoff_multiplier=2.0,
            jitter_enabled=False
        )

        # After several retries, should cap at max
        assert config.calculate_delay(10) == 5000

    def test_calculate_delay_with_jitter(self):
        """Test jitter adds randomness to delay."""
        config = RetryConfig(
            initial_delay_ms=1000,
            backoff_multiplier=2.0,
            jitter_enabled=True,
            jitter_factor=0.2
        )

        delays = [config.calculate_delay(1) for _ in range(10)]

        # With jitter, delays should vary
        # Expected base delay is 2000, jitter +/- 20% = 1600-2400
        assert all(1600 <= d <= 2400 for d in delays)
        # Should not all be identical
        assert len(set(delays)) > 1

    def test_validation_max_retries(self):
        """Test max_retries validation."""
        with pytest.raises(ValueError):
            RetryConfig(max_retries=15)  # Exceeds max of 10

        with pytest.raises(ValueError):
            RetryConfig(max_retries=-1)  # Below min of 0


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_values(self):
        """Test default rate limit values."""
        config = RateLimitConfig()

        assert config.requests_per_second == 10.0
        assert config.burst_size == 50
        assert config.enabled is True
        assert config.window_seconds == 60
        assert config.max_requests_per_window == 600

    def test_custom_values(self):
        """Test custom rate limit configuration."""
        config = RateLimitConfig(
            requests_per_second=5.0,
            burst_size=20,
            enabled=False,
            window_seconds=120,
            max_requests_per_window=300
        )

        assert config.requests_per_second == 5.0
        assert config.burst_size == 20
        assert config.enabled is False

    def test_validation(self):
        """Test rate limit validation."""
        with pytest.raises(ValueError):
            RateLimitConfig(requests_per_second=0)  # Must be > 0

        with pytest.raises(ValueError):
            RateLimitConfig(burst_size=0)  # Must be >= 1


class TestWebhookEndpoint:
    """Tests for WebhookEndpoint."""

    def test_minimal_endpoint(self):
        """Test endpoint with minimal required fields."""
        endpoint = WebhookEndpoint(
            name="Test Endpoint",
            url="https://example.com/webhook"
        )

        assert endpoint.name == "Test Endpoint"
        assert endpoint.url == "https://example.com/webhook"
        assert endpoint.status == EndpointStatus.ACTIVE
        assert endpoint.authentication_type == AuthenticationType.HMAC_SHA256

    def test_endpoint_with_secret(self):
        """Test endpoint with HMAC secret."""
        endpoint = WebhookEndpoint(
            name="Test Endpoint",
            url="https://example.com/webhook",
            secret=SecretStr("my-secret-key")
        )

        assert endpoint.secret is not None
        assert endpoint.get_secret_value() == "my-secret-key"

    def test_endpoint_with_bearer_token(self):
        """Test endpoint with bearer token auth."""
        endpoint = WebhookEndpoint(
            name="Test Endpoint",
            url="https://example.com/webhook",
            authentication_type=AuthenticationType.BEARER_TOKEN,
            bearer_token=SecretStr("token-123")
        )

        assert endpoint.bearer_token is not None

    def test_endpoint_with_api_key(self):
        """Test endpoint with API key auth."""
        endpoint = WebhookEndpoint(
            name="Test Endpoint",
            url="https://example.com/webhook",
            authentication_type=AuthenticationType.API_KEY,
            api_key=SecretStr("api-key-456"),
            api_key_header="X-Custom-API-Key"
        )

        assert endpoint.api_key is not None
        assert endpoint.api_key_header == "X-Custom-API-Key"

    def test_endpoint_event_filtering(self):
        """Test endpoint event type filtering."""
        endpoint = WebhookEndpoint(
            name="Test Endpoint",
            url="https://example.com/webhook",
            secret=SecretStr("secret"),
            event_types={
                WebhookEventType.HEAT_PLAN_CREATED,
                WebhookEventType.SAFETY_ACTION_BLOCKED
            }
        )

        assert endpoint.accepts_event(WebhookEventType.HEAT_PLAN_CREATED)
        assert endpoint.accepts_event(WebhookEventType.SAFETY_ACTION_BLOCKED)
        assert not endpoint.accepts_event(WebhookEventType.MAINTENANCE_TRIGGER)

    def test_endpoint_is_active(self):
        """Test endpoint active status check."""
        active = WebhookEndpoint(
            name="Active",
            url="https://example.com/webhook",
            secret=SecretStr("secret"),
            status=EndpointStatus.ACTIVE
        )
        assert active.is_active()

        degraded = WebhookEndpoint(
            name="Degraded",
            url="https://example.com/webhook",
            secret=SecretStr("secret"),
            status=EndpointStatus.DEGRADED
        )
        assert degraded.is_active()

        paused = WebhookEndpoint(
            name="Paused",
            url="https://example.com/webhook",
            secret=SecretStr("secret"),
            status=EndpointStatus.PAUSED
        )
        assert not paused.is_active()

    def test_url_validation(self):
        """Test URL validation."""
        # Valid HTTPS URL
        endpoint = WebhookEndpoint(
            name="Test",
            url="https://example.com/webhook",
            secret=SecretStr("secret")
        )
        assert endpoint.url == "https://example.com/webhook"

        # HTTP localhost allowed
        endpoint = WebhookEndpoint(
            name="Test",
            url="http://localhost:8080/webhook",
            secret=SecretStr("secret")
        )
        assert "localhost" in endpoint.url

        # Invalid URL
        with pytest.raises(ValueError):
            WebhookEndpoint(
                name="Test",
                url="not-a-url",
                secret=SecretStr("secret")
            )

    def test_hmac_requires_secret(self):
        """Test that HMAC auth requires secret."""
        with pytest.raises(ValueError):
            WebhookEndpoint(
                name="Test",
                url="https://example.com/webhook",
                authentication_type=AuthenticationType.HMAC_SHA256
                # No secret provided
            )

    def test_custom_headers(self):
        """Test custom headers configuration."""
        endpoint = WebhookEndpoint(
            name="Test",
            url="https://example.com/webhook",
            secret=SecretStr("secret"),
            headers={
                "X-Custom-Header": "value",
                "X-Another-Header": "another-value"
            }
        )

        assert endpoint.headers["X-Custom-Header"] == "value"


class TestWebhookConfig:
    """Tests for WebhookConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = WebhookConfig()

        assert len(config.endpoints) == 0
        assert config.signature_header == "X-GL-Signature-256"
        assert config.timestamp_header == "X-GL-Timestamp"
        assert config.max_payload_size_bytes == 1048576
        assert config.signature_tolerance_seconds == 300

    def test_config_with_endpoints(self):
        """Test configuration with endpoints."""
        endpoints = [
            WebhookEndpoint(
                name="Endpoint 1",
                url="https://example1.com/webhook",
                secret=SecretStr("secret1")
            ),
            WebhookEndpoint(
                name="Endpoint 2",
                url="https://example2.com/webhook",
                secret=SecretStr("secret2")
            )
        ]

        config = WebhookConfig(endpoints=endpoints)

        assert len(config.endpoints) == 2

    def test_dlq_config(self):
        """Test dead letter queue configuration."""
        dlq_config = DeadLetterQueueConfig(
            enabled=True,
            max_size=5000,
            retention_hours=72,
            alert_threshold=50
        )

        config = WebhookConfig(dlq_config=dlq_config)

        assert config.dlq_config.max_size == 5000
        assert config.dlq_config.retention_hours == 72


class TestEndpointRegistry:
    """Tests for EndpointRegistry."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WebhookConfig(endpoints=[
            WebhookEndpoint(
                endpoint_id="ep-001",
                name="Endpoint 1",
                url="https://example1.com/webhook",
                secret=SecretStr("secret1"),
                event_types={
                    WebhookEventType.HEAT_PLAN_CREATED,
                    WebhookEventType.SETPOINT_RECOMMENDATION
                }
            ),
            WebhookEndpoint(
                endpoint_id="ep-002",
                name="Endpoint 2",
                url="https://example2.com/webhook",
                secret=SecretStr("secret2"),
                event_types={WebhookEventType.SAFETY_ACTION_BLOCKED}
            )
        ])

    @pytest.fixture
    def registry(self, config):
        """Create test registry."""
        return EndpointRegistry(config)

    def test_registry_initialization(self, registry):
        """Test registry initializes with config endpoints."""
        assert registry.endpoint_count() == 2

    def test_get_endpoint(self, registry):
        """Test getting endpoint by ID."""
        endpoint = registry.get_endpoint("ep-001")
        assert endpoint is not None
        assert endpoint.name == "Endpoint 1"

        # Non-existent endpoint
        assert registry.get_endpoint("nonexistent") is None

    def test_get_endpoints_for_event(self, registry):
        """Test getting endpoints by event type."""
        # Heat plan created - should get ep-001
        endpoints = registry.get_endpoints_for_event(WebhookEventType.HEAT_PLAN_CREATED)
        assert len(endpoints) == 1
        assert endpoints[0].endpoint_id == "ep-001"

        # Safety action - should get ep-002
        endpoints = registry.get_endpoints_for_event(WebhookEventType.SAFETY_ACTION_BLOCKED)
        assert len(endpoints) == 1
        assert endpoints[0].endpoint_id == "ep-002"

        # Setpoint recommendation - should get ep-001
        endpoints = registry.get_endpoints_for_event(WebhookEventType.SETPOINT_RECOMMENDATION)
        assert len(endpoints) == 1

        # Maintenance trigger - no endpoints
        endpoints = registry.get_endpoints_for_event(WebhookEventType.MAINTENANCE_TRIGGER)
        assert len(endpoints) == 0

    def test_register_endpoint(self, registry):
        """Test registering new endpoint."""
        new_endpoint = WebhookEndpoint(
            endpoint_id="ep-003",
            name="Endpoint 3",
            url="https://example3.com/webhook",
            secret=SecretStr("secret3"),
            event_types={WebhookEventType.MAINTENANCE_TRIGGER}
        )

        endpoint_id = registry.register_endpoint(new_endpoint)
        assert endpoint_id == "ep-003"
        assert registry.endpoint_count() == 3

        # Verify it's registered for maintenance events
        endpoints = registry.get_endpoints_for_event(WebhookEventType.MAINTENANCE_TRIGGER)
        assert len(endpoints) == 1

    def test_register_duplicate_fails(self, registry):
        """Test registering duplicate endpoint fails."""
        duplicate = WebhookEndpoint(
            endpoint_id="ep-001",  # Already exists
            name="Duplicate",
            url="https://example.com/webhook",
            secret=SecretStr("secret")
        )

        with pytest.raises(ValueError):
            registry.register_endpoint(duplicate)

    def test_unregister_endpoint(self, registry):
        """Test unregistering endpoint."""
        # Unregister existing
        result = registry.unregister_endpoint("ep-001")
        assert result is True
        assert registry.endpoint_count() == 1
        assert registry.get_endpoint("ep-001") is None

        # Unregister non-existent
        result = registry.unregister_endpoint("nonexistent")
        assert result is False

    def test_update_endpoint_status(self, registry):
        """Test updating endpoint status."""
        result = registry.update_endpoint_status("ep-001", EndpointStatus.PAUSED)
        assert result is True

        endpoint = registry.get_endpoint("ep-001")
        assert endpoint.status == EndpointStatus.PAUSED

        # Update non-existent
        result = registry.update_endpoint_status("nonexistent", EndpointStatus.ACTIVE)
        assert result is False

    def test_update_event_subscriptions(self, registry):
        """Test updating event subscriptions."""
        new_events = {
            WebhookEventType.MAINTENANCE_TRIGGER,
            WebhookEventType.SAFETY_ACTION_BLOCKED
        }

        result = registry.update_event_subscriptions("ep-001", new_events)
        assert result is True

        # ep-001 should now be subscribed to new events
        endpoints = registry.get_endpoints_for_event(WebhookEventType.MAINTENANCE_TRIGGER)
        assert len(endpoints) == 1
        assert endpoints[0].endpoint_id == "ep-001"

        # Should no longer be subscribed to old events
        endpoints = registry.get_endpoints_for_event(WebhookEventType.HEAT_PLAN_CREATED)
        assert len(endpoints) == 0

    def test_get_all_endpoints(self, registry):
        """Test getting all endpoints."""
        all_endpoints = registry.get_all_endpoints()
        assert len(all_endpoints) == 2

        # Test active_only filter
        registry.update_endpoint_status("ep-001", EndpointStatus.DISABLED)
        active_endpoints = registry.get_all_endpoints(active_only=True)
        assert len(active_endpoints) == 1

    def test_active_endpoint_count(self, registry):
        """Test active endpoint count."""
        assert registry.active_endpoint_count() == 2

        registry.update_endpoint_status("ep-001", EndpointStatus.PAUSED)
        assert registry.active_endpoint_count() == 1

    def test_generate_secret(self, registry):
        """Test secret generation."""
        secret = registry.generate_secret()
        assert len(secret) == 64  # 32 bytes hex = 64 chars

        secret2 = registry.generate_secret()
        assert secret != secret2  # Should be unique


class TestSecretManager:
    """Tests for SecretManager."""

    @pytest.fixture
    def manager(self):
        """Create test secret manager."""
        return SecretManager(storage_backend="memory")

    def test_generate_secret(self, manager):
        """Test secret generation."""
        secret = manager.generate_secret()
        assert len(secret) == 64  # 32 bytes hex

        secret2 = manager.generate_secret(16)
        assert len(secret2) == 32  # 16 bytes hex

    def test_store_and_get_secret(self, manager):
        """Test storing and retrieving secrets."""
        manager.store_secret("ep-001", "my-secret")

        retrieved = manager.get_secret("ep-001")
        assert retrieved == "my-secret"

        # Non-existent
        assert manager.get_secret("nonexistent") is None

    def test_delete_secret(self, manager):
        """Test deleting secrets."""
        manager.store_secret("ep-001", "my-secret")

        result = manager.delete_secret("ep-001")
        assert result is True
        assert manager.get_secret("ep-001") is None

        # Delete non-existent
        result = manager.delete_secret("nonexistent")
        assert result is False

    def test_needs_rotation(self, manager):
        """Test rotation check."""
        # Non-existent secret needs rotation
        assert manager.needs_rotation("ep-001") is True

        # Newly stored secret doesn't need rotation
        manager.store_secret("ep-001", "my-secret", rotation_days=90)
        assert manager.needs_rotation("ep-001") is False

    def test_rotate_secret(self, manager):
        """Test secret rotation."""
        manager.store_secret("ep-001", "old-secret")

        new_secret = manager.rotate_secret("ep-001")
        assert new_secret is not None
        assert new_secret != "old-secret"
        assert manager.get_secret("ep-001") == new_secret

        # Rotate non-existent
        assert manager.rotate_secret("nonexistent") is None


class TestDeadLetterQueueConfig:
    """Tests for DeadLetterQueueConfig."""

    def test_default_values(self):
        """Test default DLQ configuration."""
        config = DeadLetterQueueConfig()

        assert config.enabled is True
        assert config.max_size == 10000
        assert config.retention_hours == 168  # 7 days
        assert config.persist_to_disk is True
        assert config.alert_threshold == 100

    def test_custom_values(self):
        """Test custom DLQ configuration."""
        config = DeadLetterQueueConfig(
            enabled=False,
            max_size=5000,
            retention_hours=72,
            persist_to_disk=False,
            persistence_path="/custom/path",
            alert_threshold=50
        )

        assert config.enabled is False
        assert config.max_size == 5000
        assert config.retention_hours == 72

    def test_validation(self):
        """Test DLQ config validation."""
        with pytest.raises(ValueError):
            DeadLetterQueueConfig(max_size=50)  # Below min of 100

        with pytest.raises(ValueError):
            DeadLetterQueueConfig(retention_hours=1000)  # Exceeds max of 720
