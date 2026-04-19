"""
GL-001 ThermalCommand - Webhook Event System

This package provides a production-grade webhook event system for notifying
external systems of ThermalCommand events. Features include:
- HMAC-SHA256 signature verification
- Idempotent delivery with exponential backoff retries
- Rate limiting per endpoint
- Dead letter queue for failed deliveries
- Circuit breaker pattern for fault tolerance
- Comprehensive delivery tracking and reporting

Event Types:
- HeatPlanCreated: New heat optimization plan created
- SetpointRecommendation: Setpoint change recommendation
- SafetyActionBlocked: Safety system blocked an action
- MaintenanceTrigger: Maintenance action triggered

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from .webhook_events import (
    WebhookEventType,
    WebhookEvent,
    HeatPlanCreatedEvent,
    SetpointRecommendationEvent,
    SafetyActionBlockedEvent,
    MaintenanceTriggerEvent,
)
from .webhook_config import (
    WebhookEndpoint,
    WebhookConfig,
    EndpointRegistry,
)
from .webhook_manager import (
    WebhookManager,
    DeliveryResult,
    DeliveryStatus,
)
from .webhook_dispatcher import (
    WebhookDispatcher,
    CircuitState,
    DispatchResult,
)

__version__ = "1.0.0"

__all__ = [
    # Event Types
    "WebhookEventType",
    "WebhookEvent",
    "HeatPlanCreatedEvent",
    "SetpointRecommendationEvent",
    "SafetyActionBlockedEvent",
    "MaintenanceTriggerEvent",
    # Configuration
    "WebhookEndpoint",
    "WebhookConfig",
    "EndpointRegistry",
    # Manager
    "WebhookManager",
    "DeliveryResult",
    "DeliveryStatus",
    # Dispatcher
    "WebhookDispatcher",
    "CircuitState",
    "DispatchResult",
]
