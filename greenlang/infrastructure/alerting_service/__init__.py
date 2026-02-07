# -*- coding: utf-8 -*-
"""
GreenLang Unified Alerting Service - OBS-004: Unified Alerting

Production-grade multi-channel alert notification service integrating
PagerDuty, Opsgenie, Slack, Email (SES/SMTP), Microsoft Teams, and
generic signed webhooks with full lifecycle management, deduplication,
escalation, on-call coordination, and operational analytics.

Key Features:
    - **6 Notification Channels**: PagerDuty, Opsgenie, Slack, Email,
      Teams, Webhook with HMAC signing
    - **Alert Lifecycle**: FIRING -> ACKNOWLEDGED -> INVESTIGATING ->
      RESOLVED with strict state-machine transitions
    - **Deduplication**: Fingerprint-based sliding-window dedup
    - **Escalation**: Policy-driven auto-escalation with on-call lookup
    - **Analytics**: MTTA, MTTR, fatigue scoring, noisy-alert ranking
    - **Alertmanager Webhook**: Native Prometheus Alertmanager receiver
    - **Template Engine**: Jinja2 templates with per-channel overrides
    - **Rate Limiting**: Global and per-channel sliding-window limiters
    - **Prometheus Metrics**: 10 metrics for monitoring the alerting
      pipeline itself

Sub-modules:
    config          - AlertingConfig with env-var overrides
    models          - Alert, NotificationResult, EscalationPolicy, OnCall
    lifecycle       - Alert lifecycle state machine
    deduplication   - Fingerprint-based dedup
    router          - Multi-rule alert routing + notification dispatch
    escalation      - Policy-driven auto-escalation
    oncall          - PagerDuty/Opsgenie on-call lookups
    analytics       - MTTA/MTTR/fatigue scoring
    metrics         - Prometheus metric definitions
    channels/       - Notification channel implementations
    templates/      - Jinja2 template engine + channel formatters
    webhook_receiver - Alertmanager webhook parser
    api/            - FastAPI REST API (17 endpoints)
    setup           - configure_alerting() + AlertingService facade

Quick Start:
    >>> from greenlang.infrastructure.alerting_service import (
    ...     configure_alerting,
    ...     get_alerting_service,
    ...     AlertingConfig,
    ... )
    >>> from fastapi import FastAPI
    >>> app = FastAPI()
    >>> configure_alerting(app)
    >>> svc = get_alerting_service(app)
    >>> alert = await svc.fire_alert({
    ...     "source": "prometheus",
    ...     "name": "HighCPU",
    ...     "severity": "critical",
    ...     "title": "CPU above 90%",
    ... })

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from greenlang.infrastructure.alerting_service.config import (
    AlertingConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
    NotificationStatus,
    NotificationChannel,
    NotificationResult,
    EscalationStep,
    EscalationPolicy,
    OnCallUser,
    OnCallSchedule,
)

# ---------------------------------------------------------------------------
# Core Components
# ---------------------------------------------------------------------------

from greenlang.infrastructure.alerting_service.lifecycle import AlertLifecycle
from greenlang.infrastructure.alerting_service.deduplication import AlertDeduplicator
from greenlang.infrastructure.alerting_service.router import AlertRouter
from greenlang.infrastructure.alerting_service.escalation import EscalationEngine
from greenlang.infrastructure.alerting_service.oncall import OnCallManager
from greenlang.infrastructure.alerting_service.analytics import AlertAnalytics

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

from greenlang.infrastructure.alerting_service.metrics import (
    PROMETHEUS_AVAILABLE,
    record_notification,
    record_notification_failure,
    record_mtta,
    record_mttr,
    record_escalation,
    record_dedup,
    update_active_alerts,
    update_fatigue_score,
    record_oncall_lookup,
)

# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

from greenlang.infrastructure.alerting_service.channels import (
    ChannelRegistry,
    create_channels,
)
from greenlang.infrastructure.alerting_service.channels.base import (
    BaseNotificationChannel,
)

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

from greenlang.infrastructure.alerting_service.templates.engine import TemplateEngine

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

from greenlang.infrastructure.alerting_service.setup import (
    AlertingService,
    configure_alerting,
    get_alerting_service,
)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    # Configuration
    "AlertingConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Models
    "Alert",
    "AlertSeverity",
    "AlertStatus",
    "NotificationStatus",
    "NotificationChannel",
    "NotificationResult",
    "EscalationStep",
    "EscalationPolicy",
    "OnCallUser",
    "OnCallSchedule",
    # Core Components
    "AlertLifecycle",
    "AlertDeduplicator",
    "AlertRouter",
    "EscalationEngine",
    "OnCallManager",
    "AlertAnalytics",
    # Metrics
    "PROMETHEUS_AVAILABLE",
    "record_notification",
    "record_notification_failure",
    "record_mtta",
    "record_mttr",
    "record_escalation",
    "record_dedup",
    "update_active_alerts",
    "update_fatigue_score",
    "record_oncall_lookup",
    # Channels
    "ChannelRegistry",
    "create_channels",
    "BaseNotificationChannel",
    # Templates
    "TemplateEngine",
    # Setup
    "AlertingService",
    "configure_alerting",
    "get_alerting_service",
]

logger.debug("Alerting Service module loaded: version=%s", __version__)
