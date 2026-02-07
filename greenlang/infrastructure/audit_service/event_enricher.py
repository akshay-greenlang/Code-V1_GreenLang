# -*- coding: utf-8 -*-
"""
Audit Event Enricher - SEC-005: Centralized Audit Logging Service

Enriches audit events with additional context including:
- Geo-IP location from client IP
- User agent parsing (browser, OS, device)
- Request context injection from async context vars
- Hostname and service metadata

**Design Principles:**
- Graceful degradation if enrichment services unavailable
- Non-blocking enrichment (no external network calls in hot path)
- Cache GeoIP database in memory for performance

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import socket
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .event_model import UnifiedAuditEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context Variables for Request Context
# ---------------------------------------------------------------------------

# Request context injected by middleware
_request_context: ContextVar[Dict[str, Any]] = ContextVar(
    "audit_request_context", default={}
)

# Current user context
_user_context: ContextVar[Dict[str, Any]] = ContextVar(
    "audit_user_context", default={}
)


def set_request_context(
    correlation_id: Optional[str] = None,
    request_path: Optional[str] = None,
    request_method: Optional[str] = None,
    client_ip: Optional[str] = None,
    user_agent: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Set request context for the current async context.

    Called by middleware at the start of each request.

    Args:
        correlation_id: Request correlation / trace ID.
        request_path: HTTP request path.
        request_method: HTTP method.
        client_ip: Client IP address.
        user_agent: Client User-Agent header.
        **kwargs: Additional context values.
    """
    ctx = {
        "correlation_id": correlation_id,
        "request_path": request_path,
        "request_method": request_method,
        "client_ip": client_ip,
        "user_agent": user_agent,
        **kwargs,
    }
    _request_context.set(ctx)


def set_user_context(
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    tenant_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Set user context for the current async context.

    Called by authentication middleware after user is authenticated.

    Args:
        user_id: UUID of the authenticated user.
        username: Username.
        tenant_id: UUID of the user's tenant.
        session_id: Session identifier.
        **kwargs: Additional user attributes.
    """
    ctx = {
        "user_id": user_id,
        "username": username,
        "tenant_id": tenant_id,
        "session_id": session_id,
        **kwargs,
    }
    _user_context.set(ctx)


def get_request_context() -> Dict[str, Any]:
    """Get current request context.

    Returns:
        Current request context dictionary.
    """
    return _request_context.get()


def get_user_context() -> Dict[str, Any]:
    """Get current user context.

    Returns:
        Current user context dictionary.
    """
    return _user_context.get()


def clear_context() -> None:
    """Clear both request and user context.

    Called at the end of each request.
    """
    _request_context.set({})
    _user_context.set({})


# ---------------------------------------------------------------------------
# GeoIP Stub (try import, fallback to None)
# ---------------------------------------------------------------------------


def _get_geoip_reader() -> Optional[Any]:
    """Try to get a GeoIP2 database reader.

    Returns:
        GeoIP2 reader instance or None if not available.
    """
    try:
        import geoip2.database
        import os

        # Check common GeoIP database paths
        db_paths = [
            "/usr/share/GeoIP/GeoLite2-City.mmdb",
            "/var/lib/GeoIP/GeoLite2-City.mmdb",
            os.path.expanduser("~/.geoip/GeoLite2-City.mmdb"),
        ]

        for path in db_paths:
            if os.path.exists(path):
                return geoip2.database.Reader(path)

        return None
    except ImportError:
        return None
    except Exception as e:
        logger.debug("GeoIP2 initialization failed: %s", e)
        return None


# Lazy-loaded GeoIP reader
_geoip_reader: Optional[Any] = None
_geoip_reader_initialized = False


def _lookup_geoip(ip_address: str) -> Optional[Dict[str, str]]:
    """Look up geo-location from IP address.

    Args:
        ip_address: IPv4 or IPv6 address.

    Returns:
        Dictionary with country, region, city or None if lookup fails.
    """
    global _geoip_reader, _geoip_reader_initialized

    if not _geoip_reader_initialized:
        _geoip_reader = _get_geoip_reader()
        _geoip_reader_initialized = True

    if _geoip_reader is None:
        return None

    try:
        response = _geoip_reader.city(ip_address)
        return {
            "country": response.country.iso_code or "Unknown",
            "region": (
                response.subdivisions.most_specific.name
                if response.subdivisions
                else "Unknown"
            ),
            "city": response.city.name or "Unknown",
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# User Agent Parsing Stub
# ---------------------------------------------------------------------------


def _parse_user_agent(user_agent: str) -> Optional[Dict[str, str]]:
    """Parse user agent string into components.

    Args:
        user_agent: HTTP User-Agent header value.

    Returns:
        Dictionary with browser, os, device or None if parsing fails.
    """
    try:
        from user_agents import parse

        ua = parse(user_agent)
        return {
            "browser": f"{ua.browser.family} {ua.browser.version_string}",
            "os": f"{ua.os.family} {ua.os.version_string}",
            "device": ua.device.family,
            "is_mobile": ua.is_mobile,
            "is_bot": ua.is_bot,
        }
    except ImportError:
        # Fallback: basic parsing without library
        return _parse_user_agent_basic(user_agent)
    except Exception:
        return None


def _parse_user_agent_basic(user_agent: str) -> Dict[str, str]:
    """Basic user agent parsing without external library.

    Args:
        user_agent: HTTP User-Agent header value.

    Returns:
        Dictionary with basic browser/os info.
    """
    ua_lower = user_agent.lower()

    # Detect browser
    browser = "Unknown"
    if "chrome" in ua_lower and "edg" not in ua_lower:
        browser = "Chrome"
    elif "firefox" in ua_lower:
        browser = "Firefox"
    elif "safari" in ua_lower and "chrome" not in ua_lower:
        browser = "Safari"
    elif "edg" in ua_lower:
        browser = "Edge"
    elif "msie" in ua_lower or "trident" in ua_lower:
        browser = "Internet Explorer"

    # Detect OS
    os_name = "Unknown"
    if "windows" in ua_lower:
        os_name = "Windows"
    elif "mac os" in ua_lower or "macos" in ua_lower:
        os_name = "macOS"
    elif "linux" in ua_lower:
        os_name = "Linux"
    elif "android" in ua_lower:
        os_name = "Android"
    elif "iphone" in ua_lower or "ipad" in ua_lower:
        os_name = "iOS"

    # Detect device type
    is_mobile = any(
        x in ua_lower for x in ["mobile", "android", "iphone", "ipad"]
    )
    is_bot = any(x in ua_lower for x in ["bot", "crawler", "spider"])

    return {
        "browser": browser,
        "os": os_name,
        "device": "Mobile" if is_mobile else "Desktop",
        "is_mobile": is_mobile,
        "is_bot": is_bot,
    }


# ---------------------------------------------------------------------------
# Enricher Configuration
# ---------------------------------------------------------------------------


@dataclass
class EnricherConfig:
    """Configuration for the AuditEventEnricher.

    Attributes:
        enable_geoip: Whether to perform GeoIP lookups (default True).
        enable_user_agent_parsing: Whether to parse user agents (default True).
        enable_context_injection: Whether to inject request context (default True).
        hostname: Override hostname (default: auto-detect).
        service_name: Service name for audit events (default: "greenlang").
        service_version: Service version (default: None).
    """

    enable_geoip: bool = True
    enable_user_agent_parsing: bool = True
    enable_context_injection: bool = True
    hostname: Optional[str] = None
    service_name: str = "greenlang"
    service_version: Optional[str] = None


# ---------------------------------------------------------------------------
# Audit Event Enricher
# ---------------------------------------------------------------------------


class AuditEventEnricher:
    """Enriches audit events with additional context.

    Adds geo-IP location, parsed user agent, request context,
    and system metadata to audit events.

    Example:
        >>> enricher = AuditEventEnricher()
        >>> enriched_event = enricher.enrich(event)
    """

    def __init__(self, config: Optional[EnricherConfig] = None) -> None:
        """Initialize the event enricher.

        Args:
            config: Enricher configuration.
        """
        self._config = config or EnricherConfig()
        self._hostname = self._config.hostname or self._get_hostname()

    def _get_hostname(self) -> str:
        """Get the current hostname.

        Returns:
            System hostname.
        """
        try:
            return socket.gethostname()
        except Exception:
            return "unknown"

    def enrich(self, event: UnifiedAuditEvent) -> UnifiedAuditEvent:
        """Enrich an audit event with additional context.

        Modifies the event in-place and returns it.

        Args:
            event: The audit event to enrich.

        Returns:
            The enriched event (same instance).
        """
        # Inject request context if enabled and missing
        if self._config.enable_context_injection:
            self._inject_context(event)

        # Add GeoIP if enabled and IP is present
        if (
            self._config.enable_geoip
            and event.client_ip
            and event.geo_location is None
        ):
            event.geo_location = _lookup_geoip(event.client_ip)

        # Parse user agent if enabled and present
        if (
            self._config.enable_user_agent_parsing
            and event.user_agent
            and "parsed_user_agent" not in event.metadata
        ):
            parsed = _parse_user_agent(event.user_agent)
            if parsed:
                event.metadata["parsed_user_agent"] = parsed

        # Add system metadata
        if "hostname" not in event.metadata:
            event.metadata["hostname"] = self._hostname
        if "service" not in event.metadata:
            event.metadata["service"] = self._config.service_name
        if self._config.service_version and "version" not in event.metadata:
            event.metadata["version"] = self._config.service_version

        # Ensure recorded_at is set
        if event.recorded_at is None:
            event.recorded_at = datetime.now(timezone.utc)

        return event

    def _inject_context(self, event: UnifiedAuditEvent) -> None:
        """Inject request and user context into event.

        Args:
            event: The audit event to inject context into.
        """
        request_ctx = get_request_context()
        user_ctx = get_user_context()

        # Request context
        if request_ctx:
            if event.correlation_id is None and request_ctx.get("correlation_id"):
                event.correlation_id = request_ctx["correlation_id"]
            if event.request_path is None and request_ctx.get("request_path"):
                event.request_path = request_ctx["request_path"]
            if event.request_method is None and request_ctx.get("request_method"):
                event.request_method = request_ctx["request_method"]
            if event.client_ip is None and request_ctx.get("client_ip"):
                event.client_ip = request_ctx["client_ip"]
            if event.user_agent is None and request_ctx.get("user_agent"):
                event.user_agent = request_ctx["user_agent"]

        # User context
        if user_ctx:
            if event.user_id is None and user_ctx.get("user_id"):
                event.user_id = user_ctx["user_id"]
            if event.username is None and user_ctx.get("username"):
                event.username = user_ctx["username"]
            if event.tenant_id is None and user_ctx.get("tenant_id"):
                event.tenant_id = user_ctx["tenant_id"]
            if event.session_id is None and user_ctx.get("session_id"):
                event.session_id = user_ctx["session_id"]

    def enrich_batch(
        self, events: list[UnifiedAuditEvent]
    ) -> list[UnifiedAuditEvent]:
        """Enrich a batch of audit events.

        Args:
            events: List of events to enrich.

        Returns:
            List of enriched events.
        """
        return [self.enrich(event) for event in events]


__all__ = [
    "AuditEventEnricher",
    "EnricherConfig",
    "set_request_context",
    "set_user_context",
    "get_request_context",
    "get_user_context",
    "clear_context",
]
