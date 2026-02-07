# -*- coding: utf-8 -*-
"""
Template Engine - OBS-004: Unified Alerting Service

Jinja2-based template engine for rendering alert notification messages.
Provides per-channel template overrides and custom filters for alert-
specific formatting.

Example:
    >>> engine = TemplateEngine()
    >>> rendered = engine.render("firing", alert, "slack")

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from greenlang.infrastructure.alerting_service.models import Alert, AlertSeverity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Jinja2 import
# ---------------------------------------------------------------------------

try:
    import jinja2

    JINJA2_AVAILABLE = True
except ImportError:
    jinja2 = None  # type: ignore[assignment]
    JINJA2_AVAILABLE = False
    logger.debug("jinja2 not installed; TemplateEngine will use fallback rendering")


# ---------------------------------------------------------------------------
# Default Templates
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATES: Dict[str, str] = {
    "firing": (
        "[{{ severity_emoji(alert.severity) }} {{ alert.severity.value | upper }}] "
        "{{ alert.name }}\n\n"
        "{{ alert.title }}\n\n"
        "{% if alert.description %}{{ alert.description }}\n\n{% endif %}"
        "Service: {{ alert.service or 'N/A' }} | "
        "Team: {{ alert.team or 'N/A' }} | "
        "Env: {{ alert.environment or 'N/A' }}\n"
        "{% if alert.runbook_url %}Runbook: {{ alert.runbook_url }}\n{% endif %}"
        "{% if alert.dashboard_url %}Dashboard: {{ alert.dashboard_url }}\n{% endif %}"
    ),
    "acknowledged": (
        "Alert acknowledged by {{ alert.acknowledged_by }}.\n\n"
        "{{ alert.name }} - {{ alert.title }}\n"
        "Acknowledged at: {{ format_timestamp(alert.acknowledged_at) }}"
    ),
    "resolved": (
        "Alert resolved by {{ alert.resolved_by }}.\n\n"
        "{{ alert.name }} - {{ alert.title }}\n"
        "Resolved at: {{ format_timestamp(alert.resolved_at) }}\n"
        "{% if alert.fired_at and alert.resolved_at %}"
        "Duration: {{ format_duration(alert.fired_at, alert.resolved_at) }}"
        "{% endif %}"
    ),
    "escalated": (
        "[ESCALATION L{{ alert.escalation_level }}] {{ alert.name }}\n\n"
        "{{ alert.title }}\n\n"
        "This alert has been escalated to level {{ alert.escalation_level }} "
        "because it was not acknowledged within the configured timeout.\n\n"
        "Service: {{ alert.service or 'N/A' }} | "
        "Team: {{ alert.team or 'N/A' }}"
    ),
}

# Channel-specific overrides
_CHANNEL_TEMPLATES: Dict[str, Dict[str, str]] = {
    "slack": {
        "firing": (
            "*{{ alert.title }}*\n"
            "{{ alert.description or '' }}\n"
            "`{{ alert.severity.value | upper }}` | "
            "`{{ alert.status.value }}` | "
            "{{ alert.service or 'N/A' }}"
        ),
    },
    "email": {
        "firing": (
            "Alert: {{ alert.name }}\n"
            "Title: {{ alert.title }}\n"
            "Severity: {{ alert.severity.value | upper }}\n"
            "Status: {{ alert.status.value }}\n\n"
            "{{ alert.description or '' }}\n\n"
            "Service: {{ alert.service or 'N/A' }}\n"
            "Team: {{ alert.team or 'N/A' }}\n"
            "Environment: {{ alert.environment or 'N/A' }}\n"
        ),
    },
}


# ---------------------------------------------------------------------------
# Custom Jinja2 Filters
# ---------------------------------------------------------------------------

_EMOJI_MAP: Dict[AlertSeverity, str] = {
    AlertSeverity.CRITICAL: "🔴",
    AlertSeverity.WARNING: "🟡",
    AlertSeverity.INFO: "🔵",
}

_COLOR_MAP: Dict[AlertSeverity, str] = {
    AlertSeverity.CRITICAL: "#FF0000",
    AlertSeverity.WARNING: "#FFA500",
    AlertSeverity.INFO: "#0078D4",
}


def _severity_emoji(severity: Any) -> str:
    """Return an emoji for the severity level."""
    if isinstance(severity, AlertSeverity):
        return _EMOJI_MAP.get(severity, "⚪")
    return "⚪"


def _severity_color(severity: Any) -> str:
    """Return a color hex for the severity level."""
    if isinstance(severity, AlertSeverity):
        return _COLOR_MAP.get(severity, "#6C757D")
    return "#6C757D"


def _format_timestamp(dt: Any) -> str:
    """Format a datetime to ISO string."""
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return str(dt) if dt else "N/A"


def _format_duration(start: Any, end: Any) -> str:
    """Format the duration between two datetimes."""
    if isinstance(start, datetime) and isinstance(end, datetime):
        delta = end - start
        total_seconds = int(delta.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"
    return "N/A"


# ---------------------------------------------------------------------------
# TemplateEngine
# ---------------------------------------------------------------------------


class TemplateEngine:
    """Jinja2-based template engine for alert messages.

    Falls back to simple string formatting when Jinja2 is not available.
    """

    def __init__(self) -> None:
        self._env: Optional[Any] = None
        if JINJA2_AVAILABLE:
            self._env = jinja2.Environment(
                autoescape=False,
                undefined=jinja2.Undefined,
            )
            self._register_filters()
        logger.info(
            "TemplateEngine initialized: jinja2=%s", JINJA2_AVAILABLE,
        )

    def _register_filters(self) -> None:
        """Register custom Jinja2 filters and globals."""
        if self._env is None:
            return
        self._env.filters["severity_emoji"] = _severity_emoji
        self._env.filters["severity_color"] = _severity_color
        self._env.filters["format_timestamp"] = _format_timestamp
        self._env.globals["severity_emoji"] = _severity_emoji
        self._env.globals["severity_color"] = _severity_color
        self._env.globals["format_timestamp"] = _format_timestamp
        self._env.globals["format_duration"] = _format_duration

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        template_name: str,
        alert: Alert,
        channel: str = "default",
    ) -> str:
        """Render a named template for an alert.

        Args:
            template_name: Template name (``firing``, ``acknowledged``, etc.).
            alert: Alert providing template context.
            channel: Channel name for channel-specific overrides.

        Returns:
            Rendered message string.
        """
        template_str = self.get_template(template_name, channel)
        return self._render_string(template_str, {"alert": alert})

    def render_custom(
        self,
        template_str: str,
        context: Dict[str, Any],
    ) -> str:
        """Render an arbitrary template string.

        Args:
            template_str: Jinja2 template string.
            context: Template variables.

        Returns:
            Rendered string.
        """
        return self._render_string(template_str, context)

    def get_template(self, name: str, channel: str = "default") -> str:
        """Retrieve a template string by name and channel.

        Channel-specific templates override the default if present.

        Args:
            name: Template name.
            channel: Channel name.

        Returns:
            Template string.
        """
        # Check channel override first
        if channel != "default" and channel in _CHANNEL_TEMPLATES:
            channel_tmpl = _CHANNEL_TEMPLATES[channel].get(name)
            if channel_tmpl:
                return channel_tmpl

        return _DEFAULT_TEMPLATES.get(name, "{{ alert.title }}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _render_string(
        self,
        template_str: str,
        context: Dict[str, Any],
    ) -> str:
        """Render a template string with context.

        Args:
            template_str: Jinja2 template.
            context: Variables.

        Returns:
            Rendered string.
        """
        if JINJA2_AVAILABLE and self._env is not None:
            try:
                tmpl = self._env.from_string(template_str)
                return tmpl.render(**context)
            except Exception as exc:
                logger.warning("Template render failed: %s", exc)

        # Fallback: basic Python format
        alert = context.get("alert")
        if alert is not None:
            return f"[{alert.severity.value.upper()}] {alert.title} - {alert.description}"
        return str(context)
