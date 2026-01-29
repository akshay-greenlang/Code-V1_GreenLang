# -*- coding: utf-8 -*-
"""
Test Alert Webhooks Integration (FR-063)
========================================

Unit and integration tests for the alert webhooks system.

Author: GreenLang Team
Version: 1.0.0
GL-FOUND-X-001: Alert Webhooks Tests
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.orchestrator.alerting.webhooks import (
    AlertManager,
    AlertPayload,
    AlertSeverity,
    AlertType,
    WebhookConfig,
    WebhookDeliveryResult,
    WebhookDeliveryStatus,
    WebhookManager,
    compute_hmac_signature,
)
from greenlang.orchestrator.alerting.providers.slack import (
    SlackWebhookProvider,
    format_slack_payload,
)
from greenlang.orchestrator.alerting.providers.discord import (
    DiscordWebhookProvider,
    format_discord_payload,
)
from greenlang.orchestrator.alerting.providers.pagerduty import (
    PagerDutyProvider,
    format_pagerduty_payload,
    generate_dedup_key,
)
from greenlang.orchestrator.alerting.providers.custom import (
    CustomWebhookProvider,
    format_custom_payload,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_alert() -> AlertPayload:
    """Create a sample alert payload for testing."""
    return AlertPayload(
        alert_type=AlertType.RUN_FAILED,
        severity=AlertSeverity.HIGH,
        run_id="run-test-123",
        step_id="step-calculate",
        pipeline_id="pipe-emissions",
        namespace="production",
        message="Step 'calculate' failed: Division by zero",
        details={
            "error_code": "CALC_001",
            "input_file": "data.csv",
            "line_number": 42,
        },
    )


@pytest.fixture
def sample_webhook_config() -> WebhookConfig:
    """Create a sample webhook configuration for testing."""
    return WebhookConfig(
        name="test-webhook",
        provider="custom",
        url="https://webhook.example.com/alerts",
        secret="test-secret-key",
        events=["run_failed", "step_timeout"],
        severity_threshold=AlertSeverity.MEDIUM,
        retries=3,
        timeout_seconds=30,
        enabled=True,
    )


@pytest.fixture
def webhook_manager() -> WebhookManager:
    """Create a WebhookManager instance for testing."""
    return WebhookManager(
        default_timeout=10,
        default_retries=2,
        base_backoff_seconds=0.1,
        max_backoff_seconds=1.0,
    )


@pytest.fixture
def alert_manager() -> AlertManager:
    """Create an AlertManager instance for testing."""
    return AlertManager()


# =============================================================================
# ALERT PAYLOAD TESTS
# =============================================================================


class TestAlertPayload:
    """Tests for AlertPayload model."""

    def test_alert_payload_creation(self, sample_alert: AlertPayload):
        """Test creating an alert payload."""
        assert sample_alert.alert_type == AlertType.RUN_FAILED
        assert sample_alert.severity == AlertSeverity.HIGH
        assert sample_alert.run_id == "run-test-123"
        assert sample_alert.step_id == "step-calculate"
        assert sample_alert.message == "Step 'calculate' failed: Division by zero"

    def test_alert_payload_to_dict(self, sample_alert: AlertPayload):
        """Test converting alert payload to dictionary."""
        alert_dict = sample_alert.to_dict()

        assert alert_dict["alert_type"] == "run_failed"
        assert alert_dict["severity"] == "high"
        assert alert_dict["run_id"] == "run-test-123"
        assert "timestamp" in alert_dict
        assert isinstance(alert_dict["timestamp"], str)

    def test_alert_payload_to_json(self, sample_alert: AlertPayload):
        """Test serializing alert payload to JSON."""
        json_str = sample_alert.to_json()
        parsed = json.loads(json_str)

        assert parsed["alert_type"] == "run_failed"
        assert parsed["run_id"] == "run-test-123"

    def test_alert_payload_default_values(self):
        """Test alert payload default values."""
        alert = AlertPayload(
            alert_type=AlertType.SLO_BREACH,
            severity=AlertSeverity.CRITICAL,
            run_id="run-456",
            message="SLO breached",
        )

        assert alert.namespace == "default"
        assert alert.source == "greenlang-orchestrator"
        assert alert.details == {}
        assert alert.alert_id.startswith("alert-")


# =============================================================================
# WEBHOOK CONFIG TESTS
# =============================================================================


class TestWebhookConfig:
    """Tests for WebhookConfig model."""

    def test_webhook_config_creation(self, sample_webhook_config: WebhookConfig):
        """Test creating a webhook configuration."""
        assert sample_webhook_config.name == "test-webhook"
        assert sample_webhook_config.provider == "custom"
        assert sample_webhook_config.retries == 3
        assert sample_webhook_config.enabled is True

    def test_webhook_config_event_subscription(self, sample_webhook_config: WebhookConfig):
        """Test checking event subscription."""
        assert sample_webhook_config.subscribes_to(AlertType.RUN_FAILED) is True
        assert sample_webhook_config.subscribes_to(AlertType.STEP_TIMEOUT) is True
        assert sample_webhook_config.subscribes_to(AlertType.SLO_BREACH) is False

    def test_webhook_config_severity_threshold(self):
        """Test severity threshold checking."""
        config = WebhookConfig(
            name="test",
            severity_threshold=AlertSeverity.HIGH,
        )

        assert config.meets_severity_threshold(AlertSeverity.CRITICAL) is True
        assert config.meets_severity_threshold(AlertSeverity.HIGH) is True
        assert config.meets_severity_threshold(AlertSeverity.MEDIUM) is False
        assert config.meets_severity_threshold(AlertSeverity.LOW) is False

    def test_webhook_config_env_var_expansion(self):
        """Test environment variable expansion in URL."""
        os.environ["TEST_WEBHOOK_URL"] = "https://test.example.com/webhook"
        os.environ["TEST_SECRET"] = "my-secret-key"

        config = WebhookConfig(
            name="test",
            url="${TEST_WEBHOOK_URL}",
            secret="${TEST_SECRET}",
        )

        assert config.resolve_url() == "https://test.example.com/webhook"
        assert config.resolve_secret() == "my-secret-key"

        # Cleanup
        del os.environ["TEST_WEBHOOK_URL"]
        del os.environ["TEST_SECRET"]

    def test_webhook_config_empty_events_subscribes_all(self):
        """Test that empty events list subscribes to all events."""
        config = WebhookConfig(name="test", events=[])

        assert config.subscribes_to(AlertType.RUN_FAILED) is True
        assert config.subscribes_to(AlertType.SLO_BREACH) is True
        assert config.subscribes_to(AlertType.RUN_SUCCEEDED) is True


# =============================================================================
# ALERT SEVERITY TESTS
# =============================================================================


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_severity_comparison(self):
        """Test severity level comparisons."""
        assert AlertSeverity.CRITICAL >= AlertSeverity.HIGH
        assert AlertSeverity.HIGH >= AlertSeverity.MEDIUM
        assert AlertSeverity.MEDIUM >= AlertSeverity.LOW
        assert AlertSeverity.LOW >= AlertSeverity.INFO

        assert AlertSeverity.CRITICAL > AlertSeverity.HIGH
        assert not AlertSeverity.HIGH > AlertSeverity.CRITICAL

    def test_severity_from_string(self):
        """Test parsing severity from string."""
        assert AlertSeverity.from_string("critical") == AlertSeverity.CRITICAL
        assert AlertSeverity.from_string("MEDIUM") == AlertSeverity.MEDIUM
        assert AlertSeverity.from_string("Info") == AlertSeverity.INFO


# =============================================================================
# HMAC SIGNATURE TESTS
# =============================================================================


class TestHMACSignature:
    """Tests for HMAC signature computation."""

    def test_compute_hmac_signature(self):
        """Test HMAC signature computation."""
        payload = '{"message": "test"}'
        secret = "my-secret-key"

        signature = compute_hmac_signature(payload, secret)

        assert signature.startswith("sha256=")
        assert len(signature) == 71  # "sha256=" + 64 hex chars

    def test_hmac_signature_consistency(self):
        """Test that same input produces same signature."""
        payload = '{"message": "test"}'
        secret = "my-secret-key"

        sig1 = compute_hmac_signature(payload, secret)
        sig2 = compute_hmac_signature(payload, secret)

        assert sig1 == sig2

    def test_hmac_signature_different_secrets(self):
        """Test that different secrets produce different signatures."""
        payload = '{"message": "test"}'

        sig1 = compute_hmac_signature(payload, "secret1")
        sig2 = compute_hmac_signature(payload, "secret2")

        assert sig1 != sig2


# =============================================================================
# WEBHOOK MANAGER TESTS
# =============================================================================


class TestWebhookManager:
    """Tests for WebhookManager class."""

    def test_register_webhook(self, webhook_manager: WebhookManager, sample_webhook_config: WebhookConfig):
        """Test registering a webhook."""
        webhook_id = webhook_manager.register_webhook("production", sample_webhook_config)

        assert webhook_id == sample_webhook_config.webhook_id
        assert webhook_manager.get_webhook("production", webhook_id) is not None

    def test_unregister_webhook(self, webhook_manager: WebhookManager, sample_webhook_config: WebhookConfig):
        """Test unregistering a webhook."""
        webhook_id = webhook_manager.register_webhook("production", sample_webhook_config)
        result = webhook_manager.unregister_webhook("production", webhook_id)

        assert result is True
        assert webhook_manager.get_webhook("production", webhook_id) is None

    def test_unregister_nonexistent_webhook(self, webhook_manager: WebhookManager):
        """Test unregistering a non-existent webhook."""
        result = webhook_manager.unregister_webhook("production", "nonexistent-id")
        assert result is False

    def test_list_webhooks(self, webhook_manager: WebhookManager, sample_webhook_config: WebhookConfig):
        """Test listing webhooks for a namespace."""
        webhook_manager.register_webhook("production", sample_webhook_config)

        webhooks = webhook_manager.list_webhooks("production")
        assert len(webhooks) == 1
        assert webhooks[0].name == "test-webhook"

    def test_list_webhooks_empty_namespace(self, webhook_manager: WebhookManager):
        """Test listing webhooks for empty namespace."""
        webhooks = webhook_manager.list_webhooks("nonexistent")
        assert len(webhooks) == 0

    @pytest.mark.asyncio
    async def test_dispatch_alert_no_webhooks(
        self, webhook_manager: WebhookManager, sample_alert: AlertPayload
    ):
        """Test dispatching alert with no webhooks configured."""
        results = await webhook_manager.dispatch_alert("production", sample_alert)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_dispatch_alert_filters_by_event_type(
        self, webhook_manager: WebhookManager, sample_webhook_config: WebhookConfig
    ):
        """Test that dispatch filters webhooks by event type."""
        webhook_manager.register_webhook("production", sample_webhook_config)

        # Alert type that webhook doesn't subscribe to
        alert = AlertPayload(
            alert_type=AlertType.SLO_BREACH,  # Not in webhook's events
            severity=AlertSeverity.CRITICAL,
            run_id="run-123",
            message="SLO breach",
        )

        results = await webhook_manager.dispatch_alert("production", alert)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_dispatch_alert_filters_by_severity(self, webhook_manager: WebhookManager):
        """Test that dispatch filters webhooks by severity threshold."""
        config = WebhookConfig(
            name="high-priority",
            url="https://example.com/webhook",
            severity_threshold=AlertSeverity.HIGH,
            events=["run_failed"],
        )
        webhook_manager.register_webhook("production", config)

        # Alert with severity below threshold
        alert = AlertPayload(
            alert_type=AlertType.RUN_FAILED,
            severity=AlertSeverity.MEDIUM,  # Below HIGH threshold
            run_id="run-123",
            message="Low priority failure",
        )

        results = await webhook_manager.dispatch_alert("production", alert)
        assert len(results) == 0


# =============================================================================
# ALERT MANAGER TESTS
# =============================================================================


class TestAlertManager:
    """Tests for AlertManager class."""

    def test_alert_manager_initialization(self, alert_manager: AlertManager):
        """Test AlertManager initialization."""
        assert alert_manager.enabled is True
        assert alert_manager.webhook_manager is not None

    def test_alert_manager_disable(self, alert_manager: AlertManager):
        """Test disabling alerting."""
        alert_manager.enabled = False
        assert alert_manager.enabled is False

    def test_register_webhook_via_manager(self, alert_manager: AlertManager, sample_webhook_config: WebhookConfig):
        """Test registering webhook via AlertManager."""
        webhook_id = alert_manager.register_webhook("production", sample_webhook_config)
        webhooks = alert_manager.list_webhooks("production")

        assert len(webhooks) == 1
        assert webhooks[0].webhook_id == webhook_id

    @pytest.mark.asyncio
    async def test_emit_alert_when_disabled(self, alert_manager: AlertManager):
        """Test that emit_alert does nothing when disabled."""
        alert_manager.enabled = False

        results = await alert_manager.emit_alert(
            namespace="production",
            alert_type=AlertType.RUN_FAILED,
            severity=AlertSeverity.HIGH,
            run_id="run-123",
            message="Test alert",
        )

        assert len(results) == 0


# =============================================================================
# PROVIDER FORMAT TESTS
# =============================================================================


class TestSlackProvider:
    """Tests for Slack webhook provider."""

    def test_format_slack_payload(self, sample_alert: AlertPayload, sample_webhook_config: WebhookConfig):
        """Test Slack payload formatting."""
        payload = format_slack_payload(sample_alert, sample_webhook_config)

        assert "blocks" in payload
        assert "attachments" in payload
        assert "text" in payload
        assert len(payload["blocks"]) > 0

    def test_slack_payload_has_color(self, sample_alert: AlertPayload, sample_webhook_config: WebhookConfig):
        """Test that Slack payload has severity color."""
        payload = format_slack_payload(sample_alert, sample_webhook_config)

        assert payload["attachments"][0]["color"] == "#fd7e14"  # Orange for HIGH


class TestDiscordProvider:
    """Tests for Discord webhook provider."""

    def test_format_discord_payload(self, sample_alert: AlertPayload, sample_webhook_config: WebhookConfig):
        """Test Discord payload formatting."""
        payload = format_discord_payload(sample_alert, sample_webhook_config)

        assert "embeds" in payload
        assert len(payload["embeds"]) == 1
        assert "username" in payload

    def test_discord_embed_fields(self, sample_alert: AlertPayload, sample_webhook_config: WebhookConfig):
        """Test Discord embed has required fields."""
        payload = format_discord_payload(sample_alert, sample_webhook_config)
        embed = payload["embeds"][0]

        assert "title" in embed
        assert "description" in embed
        assert "color" in embed
        assert "fields" in embed
        assert "timestamp" in embed


class TestPagerDutyProvider:
    """Tests for PagerDuty webhook provider."""

    def test_generate_dedup_key(self, sample_alert: AlertPayload):
        """Test dedup key generation."""
        key1 = generate_dedup_key(sample_alert)
        key2 = generate_dedup_key(sample_alert)

        # Same alert should produce same key
        assert key1 == key2
        assert len(key1) == 32

    def test_format_pagerduty_payload(self, sample_alert: AlertPayload):
        """Test PagerDuty payload formatting."""
        config = WebhookConfig(
            name="pagerduty",
            provider="pagerduty",
            routing_key="test-routing-key",
        )
        config._routing_key = "test-routing-key"  # Direct assignment for test

        # Patch resolve_routing_key
        with patch.object(config, 'resolve_routing_key', return_value="test-routing-key"):
            payload = format_pagerduty_payload(sample_alert, config)

            assert payload["routing_key"] == "test-routing-key"
            assert payload["event_action"] == "trigger"
            assert "payload" in payload
            assert payload["payload"]["severity"] == "error"  # HIGH maps to error


class TestCustomProvider:
    """Tests for Custom webhook provider."""

    def test_format_custom_payload(self, sample_alert: AlertPayload, sample_webhook_config: WebhookConfig):
        """Test custom payload formatting."""
        payload = format_custom_payload(sample_alert, sample_webhook_config)

        assert payload["alert_id"] == sample_alert.alert_id
        assert payload["alert_type"] == "run_failed"
        assert payload["severity"] == "high"
        assert payload["run_id"] == "run-test-123"
        assert payload["message"] == sample_alert.message


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestAlertingIntegration:
    """Integration tests for the alerting system."""

    @pytest.mark.asyncio
    async def test_full_alert_lifecycle(self, alert_manager: AlertManager):
        """Test complete alert lifecycle from registration to delivery."""
        # Register a webhook
        config = WebhookConfig(
            name="integration-test",
            provider="custom",
            url="https://httpbin.org/post",  # Test endpoint
            events=["run_failed"],
            severity_threshold=AlertSeverity.LOW,
        )
        webhook_id = alert_manager.register_webhook("test", config)

        # Verify registration
        webhooks = alert_manager.list_webhooks("test")
        assert len(webhooks) == 1

        # Unregister
        result = alert_manager.unregister_webhook("test", webhook_id)
        assert result is True

        # Verify unregistration
        webhooks = alert_manager.list_webhooks("test")
        assert len(webhooks) == 0

    @pytest.mark.asyncio
    async def test_send_test_alert_no_webhooks(self, alert_manager: AlertManager):
        """Test sending test alert with no webhooks."""
        results = await alert_manager.send_test_alert(
            namespace="empty",
            webhook_id=None,
        )

        assert len(results) == 0


# =============================================================================
# CLEANUP
# =============================================================================


@pytest.fixture(autouse=True)
async def cleanup_http_clients(webhook_manager: WebhookManager):
    """Cleanup HTTP clients after tests."""
    yield
    await webhook_manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
