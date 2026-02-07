# -*- coding: utf-8 -*-
"""
Alerting Service Configuration - OBS-004: Unified Alerting Service

Centralized configuration for the unified alerting service covering:
- PagerDuty Events API v2
- Opsgenie Alert API v2
- Slack Block Kit webhooks
- Email via AWS SES or SMTP
- Microsoft Teams Adaptive Cards
- Generic HMAC-signed webhooks
- Routing, escalation, deduplication, and rate-limiting knobs

All settings can be overridden via environment variables with the
``GL_ALERTING_`` prefix (e.g. ``GL_ALERTING_PAGERDUTY_ROUTING_KEY``).

Example:
    >>> from greenlang.infrastructure.alerting_service.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.pagerduty_enabled, cfg.slack_enabled)

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default severity routing
# ---------------------------------------------------------------------------

_DEFAULT_SEVERITY_ROUTING: Dict[str, List[str]] = {
    "critical": ["pagerduty", "opsgenie", "slack"],
    "warning": ["slack", "email"],
    "info": ["email"],
}


# ---------------------------------------------------------------------------
# AlertingConfig
# ---------------------------------------------------------------------------


@dataclass
class AlertingConfig:
    """Complete configuration for the GreenLang Unified Alerting Service.

    Attributes are grouped by concern: general, PagerDuty, Opsgenie, Slack,
    Email, Teams, Webhook, Routing, Escalation, Analytics, Dedup, and
    Rate-limiting.  Every attribute can be overridden via environment
    variable using the ``GL_ALERTING_`` prefix.
    """

    # -- General -------------------------------------------------------------
    service_name: str = "greenlang-alerting"
    environment: str = "production"
    enabled: bool = True

    # -- PagerDuty -----------------------------------------------------------
    pagerduty_enabled: bool = False
    pagerduty_routing_key: str = ""
    pagerduty_api_key: str = ""
    pagerduty_service_id: str = ""

    # -- Opsgenie ------------------------------------------------------------
    opsgenie_enabled: bool = False
    opsgenie_api_key: str = ""
    opsgenie_api_url: str = "https://api.opsgenie.com"
    opsgenie_team: str = ""

    # -- Slack ---------------------------------------------------------------
    slack_enabled: bool = False
    slack_webhook_critical: str = ""
    slack_webhook_warning: str = ""
    slack_webhook_info: str = ""

    # -- Email ---------------------------------------------------------------
    email_enabled: bool = False
    email_from: str = "alerts@greenlang.io"
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_use_ses: bool = True
    email_ses_region: str = "eu-west-1"

    # -- Microsoft Teams -----------------------------------------------------
    teams_enabled: bool = False
    teams_webhook_url: str = ""

    # -- Generic Webhook -----------------------------------------------------
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_secret: str = ""

    # -- Routing -------------------------------------------------------------
    default_severity_routing: Dict[str, List[str]] = field(
        default_factory=lambda: dict(_DEFAULT_SEVERITY_ROUTING),
    )

    # -- Escalation ----------------------------------------------------------
    escalation_enabled: bool = True
    escalation_ack_timeout_minutes: int = 15
    escalation_resolve_timeout_hours: int = 24

    # -- Analytics -----------------------------------------------------------
    analytics_enabled: bool = True
    analytics_retention_days: int = 365

    # -- Deduplication -------------------------------------------------------
    dedup_window_minutes: int = 60

    # -- Rate Limiting -------------------------------------------------------
    rate_limit_per_minute: int = 120
    rate_limit_per_channel_per_minute: int = 60

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> AlertingConfig:
        """Build an AlertingConfig from environment variables.

        Every field can be overridden via ``GL_ALERTING_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Dict values are not overridden from env (use code-level config).

        Returns:
            Populated AlertingConfig instance.
        """
        prefix = "GL_ALERTING_"

        def _env(name: str, default: Any = None) -> Optional[str]:
            return os.environ.get(f"{prefix}{name}", default)

        def _bool(name: str, default: bool) -> bool:
            val = _env(name)
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")

        def _int(name: str, default: int) -> int:
            val = _env(name)
            if val is None:
                return default
            try:
                return int(val)
            except ValueError:
                logger.warning(
                    "Invalid integer for %s%s=%s, using default %d",
                    prefix, name, val, default,
                )
                return default

        config = cls(
            service_name=_env("SERVICE_NAME", cls.service_name) or cls.service_name,
            environment=_env("ENVIRONMENT", cls.environment) or cls.environment,
            enabled=_bool("ENABLED", cls.enabled),
            # PagerDuty
            pagerduty_enabled=_bool("PAGERDUTY_ENABLED", cls.pagerduty_enabled),
            pagerduty_routing_key=_env("PAGERDUTY_ROUTING_KEY", "") or "",
            pagerduty_api_key=_env("PAGERDUTY_API_KEY", "") or "",
            pagerduty_service_id=_env("PAGERDUTY_SERVICE_ID", "") or "",
            # Opsgenie
            opsgenie_enabled=_bool("OPSGENIE_ENABLED", cls.opsgenie_enabled),
            opsgenie_api_key=_env("OPSGENIE_API_KEY", "") or "",
            opsgenie_api_url=_env("OPSGENIE_API_URL", cls.opsgenie_api_url) or cls.opsgenie_api_url,
            opsgenie_team=_env("OPSGENIE_TEAM", "") or "",
            # Slack
            slack_enabled=_bool("SLACK_ENABLED", cls.slack_enabled),
            slack_webhook_critical=_env("SLACK_WEBHOOK_CRITICAL", "") or "",
            slack_webhook_warning=_env("SLACK_WEBHOOK_WARNING", "") or "",
            slack_webhook_info=_env("SLACK_WEBHOOK_INFO", "") or "",
            # Email
            email_enabled=_bool("EMAIL_ENABLED", cls.email_enabled),
            email_from=_env("EMAIL_FROM", cls.email_from) or cls.email_from,
            email_smtp_host=_env("EMAIL_SMTP_HOST", "") or "",
            email_smtp_port=_int("EMAIL_SMTP_PORT", cls.email_smtp_port),
            email_use_ses=_bool("EMAIL_USE_SES", cls.email_use_ses),
            email_ses_region=_env("EMAIL_SES_REGION", cls.email_ses_region) or cls.email_ses_region,
            # Teams
            teams_enabled=_bool("TEAMS_ENABLED", cls.teams_enabled),
            teams_webhook_url=_env("TEAMS_WEBHOOK_URL", "") or "",
            # Webhook
            webhook_enabled=_bool("WEBHOOK_ENABLED", cls.webhook_enabled),
            webhook_url=_env("WEBHOOK_URL", "") or "",
            webhook_secret=_env("WEBHOOK_SECRET", "") or "",
            # Escalation
            escalation_enabled=_bool("ESCALATION_ENABLED", cls.escalation_enabled),
            escalation_ack_timeout_minutes=_int(
                "ESCALATION_ACK_TIMEOUT_MINUTES",
                cls.escalation_ack_timeout_minutes,
            ),
            escalation_resolve_timeout_hours=_int(
                "ESCALATION_RESOLVE_TIMEOUT_HOURS",
                cls.escalation_resolve_timeout_hours,
            ),
            # Analytics
            analytics_enabled=_bool("ANALYTICS_ENABLED", cls.analytics_enabled),
            analytics_retention_days=_int(
                "ANALYTICS_RETENTION_DAYS",
                cls.analytics_retention_days,
            ),
            # Dedup
            dedup_window_minutes=_int(
                "DEDUP_WINDOW_MINUTES", cls.dedup_window_minutes,
            ),
            # Rate limiting
            rate_limit_per_minute=_int(
                "RATE_LIMIT_PER_MINUTE", cls.rate_limit_per_minute,
            ),
            rate_limit_per_channel_per_minute=_int(
                "RATE_LIMIT_PER_CHANNEL_PER_MINUTE",
                cls.rate_limit_per_channel_per_minute,
            ),
        )

        logger.info(
            "AlertingConfig loaded: env=%s, pd=%s, og=%s, slack=%s, "
            "email=%s, teams=%s, webhook=%s",
            config.environment,
            config.pagerduty_enabled,
            config.opsgenie_enabled,
            config.slack_enabled,
            config.email_enabled,
            config.teams_enabled,
            config.webhook_enabled,
        )
        return config


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[AlertingConfig] = None
_config_lock = threading.Lock()


def get_config() -> AlertingConfig:
    """Return the singleton AlertingConfig, creating from env if needed.

    Returns:
        AlertingConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = AlertingConfig.from_env()
    return _config_instance


def set_config(config: AlertingConfig) -> None:
    """Replace the singleton AlertingConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("AlertingConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None
