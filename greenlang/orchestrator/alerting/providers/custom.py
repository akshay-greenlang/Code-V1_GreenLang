# -*- coding: utf-8 -*-
"""
Custom Webhook Provider
=======================

Generic HTTP POST webhook provider for custom alert endpoints.

Supports:
- Standard JSON payload format
- Configurable headers
- Extensible payload formatting

Author: GreenLang Team
Version: 1.0.0
GL-FOUND-X-001: Custom Webhook Provider
"""

import logging
from typing import Any, Dict, Optional

from greenlang.orchestrator.alerting.webhooks import (
    AlertPayload,
    AlertSeverity,
    AlertType,
    WebhookConfig,
)

logger = logging.getLogger(__name__)


def format_custom_payload(
    alert: AlertPayload, config: WebhookConfig
) -> Dict[str, Any]:
    """
    Format alert payload for a custom webhook endpoint.

    Creates a standardized JSON payload that can be consumed by
    any HTTP endpoint that accepts POST requests.

    Args:
        alert: Alert payload to format
        config: Webhook configuration

    Returns:
        Formatted payload dictionary
    """
    payload: Dict[str, Any] = {
        "alert_id": alert.alert_id,
        "alert_type": alert.alert_type.value,
        "severity": alert.severity.value,
        "message": alert.message,
        "run_id": alert.run_id,
        "namespace": alert.namespace,
        "timestamp": alert.timestamp.isoformat(),
        "source": alert.source,
    }

    # Add optional fields if present
    if alert.pipeline_id:
        payload["pipeline_id"] = alert.pipeline_id
    if alert.step_id:
        payload["step_id"] = alert.step_id

    # Include all details
    if alert.details:
        payload["details"] = alert.details

    # Add webhook metadata if configured
    if config.metadata:
        payload["webhook_metadata"] = config.metadata

    return payload


class CustomWebhookProvider:
    """
    Custom webhook provider for generic HTTP POST endpoints.

    Provides a flexible format that can be adapted to any
    webhook endpoint that accepts JSON payloads.

    Example:
        >>> provider = CustomWebhookProvider()
        >>> payload = provider.format_alert(alert, config)
        >>> # Send payload to custom webhook URL
    """

    def __init__(self, default_content_type: str = "application/json"):
        """
        Initialize CustomWebhookProvider.

        Args:
            default_content_type: Default Content-Type header
        """
        self._default_content_type = default_content_type
        logger.info("CustomWebhookProvider initialized")

    def format_alert(
        self, alert: AlertPayload, config: WebhookConfig
    ) -> Dict[str, Any]:
        """
        Format an alert for a custom webhook.

        Args:
            alert: Alert payload to format
            config: Webhook configuration

        Returns:
            Formatted payload dictionary
        """
        return format_custom_payload(alert, config)

    def get_default_headers(self) -> Dict[str, str]:
        """Get default headers for custom webhook requests."""
        return {
            "Content-Type": self._default_content_type,
            "Accept": "application/json",
        }

    @staticmethod
    def create_envelope(
        alert: AlertPayload,
        config: WebhookConfig,
        include_signature: bool = False,
        signature: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create an envelope wrapper for the alert payload.

        Useful for webhook endpoints that expect a specific envelope format.

        Args:
            alert: Alert payload
            config: Webhook configuration
            include_signature: Whether to include signature in envelope
            signature: Pre-computed signature (if include_signature is True)

        Returns:
            Envelope dictionary containing the alert
        """
        envelope: Dict[str, Any] = {
            "version": "1.0",
            "type": "greenlang.orchestrator.alert",
            "webhook_id": config.webhook_id,
            "webhook_name": config.name,
            "data": format_custom_payload(alert, config),
        }

        if include_signature and signature:
            envelope["signature"] = signature

        return envelope


__all__ = [
    "CustomWebhookProvider",
    "format_custom_payload",
]
