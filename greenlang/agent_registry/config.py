# -*- coding: utf-8 -*-
"""
Agent Registry Configuration - AGENT-FOUND-007: Agent Registry & Service Catalog

Centralized configuration for the Agent Registry SDK covering:
- Agent capacity limits and version history depth
- Health check interval and probe timeout
- Cache settings (TTL)
- Hot-reload toggle
- Strict mode for validation enforcement
- Audit trail toggle

All settings can be overridden via environment variables with the
``GL_AGENT_REGISTRY_`` prefix (e.g. ``GL_AGENT_REGISTRY_MAX_AGENTS``).

Example:
    >>> from greenlang.agent_registry.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.max_agents, cfg.health_check_interval_seconds)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-007 Agent Registry & Service Catalog
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

_ENV_PREFIX = "GL_AGENT_REGISTRY_"


# ---------------------------------------------------------------------------
# AgentRegistryConfig
# ---------------------------------------------------------------------------


@dataclass
class AgentRegistryConfig:
    """Complete configuration for the GreenLang Agent Registry SDK.

    Attributes are grouped by concern: capacity, health checking, caching,
    hot-reload, validation, auditing, and logging.

    All attributes can be overridden via environment variables using the
    ``GL_AGENT_REGISTRY_`` prefix.

    Attributes:
        max_agents: Maximum number of distinct agents in the registry.
        max_versions_per_agent: Maximum version history per agent.
        health_check_interval_seconds: Seconds between automatic health probes.
        health_check_timeout_seconds: Timeout for a single health probe.
        cache_ttl_seconds: Time-to-live for cached query results.
        enable_hot_reload: Whether to allow runtime hot-reload of agents.
        strict_mode: Whether to enforce strict validation on all mutations.
        log_level: Python log level name for the SDK logger.
        enable_audit: Whether to record provenance audit entries.
    """

    # -- Capacity limits -----------------------------------------------------
    max_agents: int = 1000
    max_versions_per_agent: int = 50

    # -- Health checking -----------------------------------------------------
    health_check_interval_seconds: int = 60
    health_check_timeout_seconds: int = 10

    # -- Caching -------------------------------------------------------------
    cache_ttl_seconds: int = 300

    # -- Hot-reload ----------------------------------------------------------
    enable_hot_reload: bool = True

    # -- Validation ----------------------------------------------------------
    strict_mode: bool = True

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Auditing ------------------------------------------------------------
    enable_audit: bool = True

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> AgentRegistryConfig:
        """Build an AgentRegistryConfig from environment variables.

        Every field can be overridden via ``GL_AGENT_REGISTRY_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.

        Returns:
            Populated AgentRegistryConfig instance.
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

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val

        config = cls(
            max_agents=_int("MAX_AGENTS", cls.max_agents),
            max_versions_per_agent=_int(
                "MAX_VERSIONS_PER_AGENT", cls.max_versions_per_agent,
            ),
            health_check_interval_seconds=_int(
                "HEALTH_CHECK_INTERVAL_SECONDS",
                cls.health_check_interval_seconds,
            ),
            health_check_timeout_seconds=_int(
                "HEALTH_CHECK_TIMEOUT_SECONDS",
                cls.health_check_timeout_seconds,
            ),
            cache_ttl_seconds=_int("CACHE_TTL_SECONDS", cls.cache_ttl_seconds),
            enable_hot_reload=_bool("ENABLE_HOT_RELOAD", cls.enable_hot_reload),
            strict_mode=_bool("STRICT_MODE", cls.strict_mode),
            log_level=_str("LOG_LEVEL", cls.log_level),
            enable_audit=_bool("ENABLE_AUDIT", cls.enable_audit),
        )

        logger.info(
            "AgentRegistryConfig loaded: max_agents=%d, max_versions=%d, "
            "health_interval=%ds, cache_ttl=%ds, hot_reload=%s, strict=%s, "
            "audit=%s",
            config.max_agents,
            config.max_versions_per_agent,
            config.health_check_interval_seconds,
            config.cache_ttl_seconds,
            config.enable_hot_reload,
            config.strict_mode,
            config.enable_audit,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[AgentRegistryConfig] = None
_config_lock = threading.Lock()


def get_config() -> AgentRegistryConfig:
    """Return the singleton AgentRegistryConfig, creating from env if needed.

    Returns:
        AgentRegistryConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = AgentRegistryConfig.from_env()
    return _config_instance


def set_config(config: AgentRegistryConfig) -> None:
    """Replace the singleton AgentRegistryConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("AgentRegistryConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "AgentRegistryConfig",
    "get_config",
    "set_config",
    "reset_config",
]
