# -*- coding: utf-8 -*-
"""
PagerDuty Webhook Provider
==========================

Formats alert payloads for PagerDuty Events API v2.

Supports:
- Event creation (trigger, acknowledge, resolve)
- Severity mapping to PagerDuty levels
- Dedup key for alert grouping
- Custom details and links

Reference: https://developer.pagerduty.com/docs/events-api-v2/

Author: GreenLang Team
Version: 1.0.0
GL-FOUND-X-001: PagerDuty Webhook Provider
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional

from greenlang.orchestrator.alerting.webhooks import (
    AlertPayload,
    AlertSeverity,
    AlertType,
    WebhookConfig,
)

logger = logging.getLogger(__name__)

# PagerDuty Events API v2 endpoint
PAGERDUTY_EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"

# Severity mapping to PagerDuty severity
SEVERITY_MAPPING: Dict[AlertSeverity, str] = {
    AlertSeverity.CRITICAL: "critical",
    AlertSeverity.HIGH: "error",
    AlertSeverity.MEDIUM: "warning",
    AlertSeverity.LOW: "info",
    AlertSeverity.INFO: "info",
}

# Alert type to PagerDuty event class mapping
EVENT_CLASS_MAPPING: Dict[AlertType, str] = {
    AlertType.RUN_FAILED: "pipeline_failure",
    AlertType.STEP_TIMEOUT: "step_timeout",
    AlertType.POLICY_DENIAL: "policy_violation",
    AlertType.SLO_BREACH: "slo_breach",
    AlertType.RUN_SUCCEEDED: "pipeline_success",
}


def generate_dedup_key(alert: AlertPayload) -> str:
    """
    Generate a deduplication key for PagerDuty.

    The dedup key ensures that related alerts are grouped together
    and prevents alert storms.

    Args:
        alert: Alert payload

    Returns:
        Deduplication key string
    """
    # Use run_id and alert_type for deduplication
    # This groups alerts for the same run/type together
    key_source = f"{alert.run_id}:{alert.alert_type.value}"
    if alert.step_id:
        key_source += f":{alert.step_id}"

    return hashlib.sha256(key_source.encode()).hexdigest()[:32]


def format_pagerduty_payload(
    alert: AlertPayload, config: WebhookConfig
) -> Dict[str, Any]:
    """
    Format alert payload for PagerDuty Events API v2.

    Creates a PagerDuty event with:
    - Routing key for service integration
    - Event action (trigger/acknowledge/resolve)
    - Severity level
    - Dedup key for alert grouping
    - Custom details and links

    Args:
        alert: Alert payload to format
        config: Webhook configuration

    Returns:
        Formatted PagerDuty payload dictionary
    """
    routing_key = config.resolve_routing_key()
    if not routing_key:
        logger.error("PagerDuty routing key not configured")
        return {}

    # Determine event action based on alert type
    event_action = "trigger"
    if alert.alert_type == AlertType.RUN_SUCCEEDED:
        event_action = "resolve"

    # Map severity
    pd_severity = SEVERITY_MAPPING.get(alert.severity, "warning")

    # Get event class
    event_class = EVENT_CLASS_MAPPING.get(alert.alert_type, "orchestrator_alert")

    # Generate dedup key
    dedup_key = generate_dedup_key(alert)

    # Build custom details
    custom_details: Dict[str, Any] = {
        "run_id": alert.run_id,
        "namespace": alert.namespace,
        "alert_type": alert.alert_type.value,
        "source": alert.source,
        "timestamp": alert.timestamp.isoformat(),
    }

    if alert.pipeline_id:
        custom_details["pipeline_id"] = alert.pipeline_id
    if alert.step_id:
        custom_details["step_id"] = alert.step_id

    # Add alert details
    custom_details.update(alert.details)

    # Build links
    links: List[Dict[str, str]] = []
    base_url = config.metadata.get("orchestrator_url")
    if base_url:
        links.append({
            "href": f"{base_url}/runs/{alert.run_id}",
            "text": "View Run Details",
        })

    # Build the payload
    payload: Dict[str, Any] = {
        "routing_key": routing_key,
        "event_action": event_action,
        "dedup_key": dedup_key,
        "payload": {
            "summary": alert.message[:1024],  # PagerDuty limit
            "source": f"greenlang-{alert.namespace}",
            "severity": pd_severity,
            "timestamp": alert.timestamp.isoformat(),
            "component": alert.pipeline_id or "orchestrator",
            "group": alert.namespace,
            "class": event_class,
            "custom_details": custom_details,
        },
    }

    # Add links if available
    if links:
        payload["links"] = links

    # Add images if configured
    images = config.metadata.get("images", [])
    if images:
        payload["images"] = images[:4]  # PagerDuty limit

    return payload


class PagerDutyProvider:
    """
    PagerDuty provider for alert notifications.

    Provides methods for formatting and sending alerts to PagerDuty
    using the Events API v2.

    Example:
        >>> provider = PagerDutyProvider()
        >>> payload = provider.format_alert(alert, config)
        >>> # Send payload to PagerDuty Events API
    """

    # PagerDuty Events API endpoint
    EVENTS_API_URL = PAGERDUTY_EVENTS_URL

    def __init__(self):
        """Initialize PagerDutyProvider."""
        logger.info("PagerDutyProvider initialized")

    def format_alert(
        self, alert: AlertPayload, config: WebhookConfig
    ) -> Dict[str, Any]:
        """
        Format an alert for PagerDuty.

        Args:
            alert: Alert payload to format
            config: Webhook configuration

        Returns:
            Formatted PagerDuty payload
        """
        return format_pagerduty_payload(alert, config)

    def get_severity(self, severity: AlertSeverity) -> str:
        """Get PagerDuty severity string for severity level."""
        return SEVERITY_MAPPING.get(severity, "warning")

    @staticmethod
    def create_change_event(
        routing_key: str,
        summary: str,
        source: str,
        custom_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a change event payload for PagerDuty.

        Change events are used to track deployment or configuration changes
        that may be relevant for incident correlation.

        Args:
            routing_key: PagerDuty routing key
            summary: Change event summary
            source: Source of the change
            custom_details: Additional change details

        Returns:
            Change event payload dictionary
        """
        return {
            "routing_key": routing_key,
            "payload": {
                "summary": summary,
                "source": source,
                "custom_details": custom_details or {},
            },
        }

    @staticmethod
    def get_api_url() -> str:
        """Get the PagerDuty Events API URL."""
        return PAGERDUTY_EVENTS_URL


__all__ = [
    "PagerDutyProvider",
    "format_pagerduty_payload",
    "generate_dedup_key",
    "PAGERDUTY_EVENTS_URL",
    "SEVERITY_MAPPING",
]
