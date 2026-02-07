# -*- coding: utf-8 -*-
"""
GreenLang Alerting Service Unit Tests (OBS-004)
================================================

Unit tests for the Unified Alerting & Notification Platform including:
- AlertingConfig configuration and environment variable parsing
- Alert, NotificationResult, EscalationPolicy, OnCallUser models
- AlertLifecycle state machine (FIRING->ACK->INVESTIGATING->RESOLVED)
- AlertDeduplicator fingerprinting and dedup window
- AlertRouter severity/team/service/time-based routing
- EscalationEngine time-based auto-escalation
- OnCallManager PD+OG schedule lookup with caching
- Channel adapters (PagerDuty, Opsgenie, Slack, Email, Teams, Webhook)
- Jinja2 template engine and channel-specific formatters
- AlertAnalytics MTTA/MTTR calculation and fatigue scoring
- Alertmanager webhook receiver parsing
- Prometheus metric recording helpers

These tests mock all external dependencies (httpx, boto3, SMTP) so they
can run without network access or cloud credentials.
"""
