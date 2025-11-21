# -*- coding: utf-8 -*-
"""
GreenLang Partners Module

This module provides partner ecosystem functionality including:
- Partner API and authentication
- Webhook system with delivery and retry
- Usage analytics and reporting
- White-label support
"""

from .api import (
    app as partner_api_app,
    PartnerModel,
    APIKeyModel,
    PartnerTier,
    PartnerStatus,
    APIKeyStatus,
)

from .webhooks import (
    WebhookModel,
    WebhookDeliveryModel,
    WebhookEventType,
    WebhookStatus,
    DeliveryStatus,
    WebhookEvent,
    WebhookManager,
)

from .webhook_security import (
    WebhookSignature,
    WebhookRateLimiter,
    IPWhitelist,
    WebhookValidator,
)

from .analytics import (
    AnalyticsEngine,
    MetricType,
    TimeRange,
    UsageEvent,
)

from .reporting import (
    ReportGenerator,
    ReportScheduler,
    ReportType,
    ReportFormat,
)

__all__ = [
    # API
    "partner_api_app",
    "PartnerModel",
    "APIKeyModel",
    "PartnerTier",
    "PartnerStatus",
    "APIKeyStatus",
    # Webhooks
    "WebhookModel",
    "WebhookDeliveryModel",
    "WebhookEventType",
    "WebhookStatus",
    "DeliveryStatus",
    "WebhookEvent",
    "WebhookManager",
    # Security
    "WebhookSignature",
    "WebhookRateLimiter",
    "IPWhitelist",
    "WebhookValidator",
    # Analytics
    "AnalyticsEngine",
    "MetricType",
    "TimeRange",
    "UsageEvent",
    # Reporting
    "ReportGenerator",
    "ReportScheduler",
    "ReportType",
    "ReportFormat",
]
