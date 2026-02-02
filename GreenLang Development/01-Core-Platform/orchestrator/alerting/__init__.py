# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator Alerting System
======================================

Alert webhooks integration for operational incident response (FR-063).

This module provides:
- AlertType and AlertSeverity enums
- AlertPayload and WebhookConfig models
- WebhookManager for dispatching alerts
- Provider implementations for Slack, Discord, PagerDuty, and custom webhooks

Author: GreenLang Team
Version: 1.0.0
GL-FOUND-X-001: Alert Webhooks Integration
"""

from greenlang.orchestrator.alerting.webhooks import (
    AlertType,
    AlertSeverity,
    AlertPayload,
    WebhookConfig,
    WebhookDeliveryStatus,
    WebhookDeliveryResult,
    WebhookManager,
    AlertManager,
)

__all__ = [
    # Enums
    "AlertType",
    "AlertSeverity",
    # Models
    "AlertPayload",
    "WebhookConfig",
    "WebhookDeliveryStatus",
    "WebhookDeliveryResult",
    # Managers
    "WebhookManager",
    "AlertManager",
]
