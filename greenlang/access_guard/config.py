# -*- coding: utf-8 -*-
"""
Access Guard Configuration - AGENT-FOUND-006: Access & Policy Guard

Centralized configuration for the Access & Policy Guard SDK covering:
- Strict mode and simulation mode
- Rate limiting defaults and burst limits
- Audit logging toggles and retention
- Tenant isolation settings
- OPA integration toggle and endpoint
- Decision and policy cache TTL
- Policy capacity limits

All settings can be overridden via environment variables with the
``GL_ACCESS_GUARD_`` prefix (e.g. ``GL_ACCESS_GUARD_STRICT_MODE``).

Example:
    >>> from greenlang.access_guard.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.strict_mode, cfg.rate_limiting_enabled)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-006 Access & Policy Guard
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_ACCESS_GUARD_"


# ---------------------------------------------------------------------------
# AccessGuardConfig
# ---------------------------------------------------------------------------


@dataclass
class AccessGuardConfig:
    """Complete configuration for the GreenLang Access & Policy Guard SDK.

    Attributes are grouped by concern: enforcement, rate limiting, audit,
    tenant isolation, OPA integration, caching, and capacity limits.

    All attributes can be overridden via environment variables using the
    ``GL_ACCESS_GUARD_`` prefix.

    Attributes:
        strict_mode: Deny by default if no matching rules.
        simulation_mode: Log decisions without enforcing.
        rate_limiting_enabled: Whether to enable request rate limiting.
        audit_enabled: Whether to enable audit event logging.
        audit_all_decisions: Log all decisions, not just denials.
        audit_retention_days: Default audit event retention in days.
        strict_tenant_isolation: Enforce strict tenant boundary checks.
        opa_enabled: Enable OPA Rego policy evaluation.
        opa_endpoint: OPA server endpoint URL.
        decision_cache_ttl_seconds: Decision cache TTL in seconds.
        policy_cache_ttl_seconds: Policy cache TTL in seconds.
        max_policies: Maximum number of policies allowed.
        max_rules_per_policy: Maximum rules per policy.
        default_rate_rpm: Default requests per minute.
        default_rate_rph: Default requests per hour.
        default_rate_rpd: Default requests per day.
        burst_limit: Default burst limit for rate limiter.
    """

    # -- Enforcement ---------------------------------------------------------
    strict_mode: bool = True
    simulation_mode: bool = False

    # -- Rate limiting -------------------------------------------------------
    rate_limiting_enabled: bool = True
    default_rate_rpm: int = 100
    default_rate_rph: int = 1000
    default_rate_rpd: int = 10000
    burst_limit: int = 20

    # -- Audit ---------------------------------------------------------------
    audit_enabled: bool = True
    audit_all_decisions: bool = True
    audit_retention_days: int = 365

    # -- Tenant isolation ----------------------------------------------------
    strict_tenant_isolation: bool = True

    # -- OPA integration -----------------------------------------------------
    opa_enabled: bool = False
    opa_endpoint: Optional[str] = None

    # -- Caching -------------------------------------------------------------
    decision_cache_ttl_seconds: int = 60
    policy_cache_ttl_seconds: int = 300

    # -- Capacity limits -----------------------------------------------------
    max_policies: int = 1000
    max_rules_per_policy: int = 100

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> AccessGuardConfig:
        """Build an AccessGuardConfig from environment variables.

        Every field can be overridden via ``GL_ACCESS_GUARD_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        String values are used as-is.

        Returns:
            Populated AccessGuardConfig instance.
        """
        prefix = _ENV_PREFIX

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

        def _str_opt(name: str, default: Optional[str]) -> Optional[str]:
            val = _env(name)
            if val is None:
                return default
            return val if val.strip() else default

        config = cls(
            strict_mode=_bool("STRICT_MODE", cls.strict_mode),
            simulation_mode=_bool("SIMULATION_MODE", cls.simulation_mode),
            rate_limiting_enabled=_bool(
                "RATE_LIMITING_ENABLED", cls.rate_limiting_enabled,
            ),
            default_rate_rpm=_int("DEFAULT_RATE_RPM", cls.default_rate_rpm),
            default_rate_rph=_int("DEFAULT_RATE_RPH", cls.default_rate_rph),
            default_rate_rpd=_int("DEFAULT_RATE_RPD", cls.default_rate_rpd),
            burst_limit=_int("BURST_LIMIT", cls.burst_limit),
            audit_enabled=_bool("AUDIT_ENABLED", cls.audit_enabled),
            audit_all_decisions=_bool(
                "AUDIT_ALL_DECISIONS", cls.audit_all_decisions,
            ),
            audit_retention_days=_int(
                "AUDIT_RETENTION_DAYS", cls.audit_retention_days,
            ),
            strict_tenant_isolation=_bool(
                "STRICT_TENANT_ISOLATION", cls.strict_tenant_isolation,
            ),
            opa_enabled=_bool("OPA_ENABLED", cls.opa_enabled),
            opa_endpoint=_str_opt("OPA_ENDPOINT", cls.opa_endpoint),
            decision_cache_ttl_seconds=_int(
                "DECISION_CACHE_TTL_SECONDS", cls.decision_cache_ttl_seconds,
            ),
            policy_cache_ttl_seconds=_int(
                "POLICY_CACHE_TTL_SECONDS", cls.policy_cache_ttl_seconds,
            ),
            max_policies=_int("MAX_POLICIES", cls.max_policies),
            max_rules_per_policy=_int(
                "MAX_RULES_PER_POLICY", cls.max_rules_per_policy,
            ),
        )

        logger.info(
            "AccessGuardConfig loaded: strict=%s, simulation=%s, "
            "rate_limit=%s, audit=%s, tenant_isolation=%s, opa=%s, "
            "max_policies=%d",
            config.strict_mode,
            config.simulation_mode,
            config.rate_limiting_enabled,
            config.audit_enabled,
            config.strict_tenant_isolation,
            config.opa_enabled,
            config.max_policies,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[AccessGuardConfig] = None
_config_lock = threading.Lock()


def get_config() -> AccessGuardConfig:
    """Return the singleton AccessGuardConfig, creating from env if needed.

    Returns:
        AccessGuardConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = AccessGuardConfig.from_env()
    return _config_instance


def set_config(config: AccessGuardConfig) -> None:
    """Replace the singleton AccessGuardConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("AccessGuardConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "AccessGuardConfig",
    "get_config",
    "set_config",
    "reset_config",
]
