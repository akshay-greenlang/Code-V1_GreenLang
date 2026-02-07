# -*- coding: utf-8 -*-
"""
Assumptions Registry Configuration - AGENT-FOUND-004: Assumptions Registry

Centralized configuration for the Assumptions Registry SDK covering:
- Version limits per assumption
- Validation and change logging toggles
- Default scenario selection
- Cache settings (enable, TTL, max size)
- Registry capacity limits
- Default uncertainty percentage

All settings can be overridden via environment variables with the
``GL_ASSUMPTIONS_`` prefix (e.g. ``GL_ASSUMPTIONS_MAX_VERSIONS_PER_ASSUMPTION``).

Example:
    >>> from greenlang.assumptions.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.max_versions_per_assumption, cfg.default_scenario)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-004 Assumptions Registry
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

_ENV_PREFIX = "GL_ASSUMPTIONS_"


# ---------------------------------------------------------------------------
# AssumptionsConfig
# ---------------------------------------------------------------------------


@dataclass
class AssumptionsConfig:
    """Complete configuration for the GreenLang Assumptions Registry SDK.

    Attributes are grouped by concern: versioning, validation, scenarios,
    caching, and capacity limits.

    All attributes can be overridden via environment variables using the
    ``GL_ASSUMPTIONS_`` prefix.

    Attributes:
        max_versions_per_assumption: Maximum version history entries per assumption.
        enable_validation: Whether to validate assumption values on create/update.
        enable_change_logging: Whether to log all changes to the audit trail.
        default_scenario: Default scenario type identifier.
        cache_enabled: Whether to enable value caching.
        cache_ttl_seconds: Cache time-to-live in seconds.
        cache_max_size: Maximum number of entries in the cache.
        max_scenarios: Maximum number of scenarios allowed.
        max_assumptions: Maximum number of assumptions in the registry.
        max_validation_rules: Maximum validation rules per assumption.
        default_uncertainty_pct: Default uncertainty percentage for assumptions.
    """

    # -- Versioning ----------------------------------------------------------
    max_versions_per_assumption: int = 100

    # -- Validation ----------------------------------------------------------
    enable_validation: bool = True
    enable_change_logging: bool = True

    # -- Scenarios -----------------------------------------------------------
    default_scenario: str = "baseline"

    # -- Caching -------------------------------------------------------------
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    cache_max_size: int = 1000

    # -- Capacity limits -----------------------------------------------------
    max_scenarios: int = 50
    max_assumptions: int = 10000
    max_validation_rules: int = 20

    # -- Uncertainty ---------------------------------------------------------
    default_uncertainty_pct: float = 10.0

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> AssumptionsConfig:
        """Build an AssumptionsConfig from environment variables.

        Every field can be overridden via ``GL_ASSUMPTIONS_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated AssumptionsConfig instance.
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

        def _float(name: str, default: float) -> float:
            val = _env(name)
            if val is None:
                return default
            try:
                return float(val)
            except ValueError:
                logger.warning(
                    "Invalid float for %s%s=%s, using default %.1f",
                    prefix, name, val, default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val

        config = cls(
            max_versions_per_assumption=_int(
                "MAX_VERSIONS_PER_ASSUMPTION", cls.max_versions_per_assumption,
            ),
            enable_validation=_bool("ENABLE_VALIDATION", cls.enable_validation),
            enable_change_logging=_bool(
                "ENABLE_CHANGE_LOGGING", cls.enable_change_logging,
            ),
            default_scenario=_str("DEFAULT_SCENARIO", cls.default_scenario),
            cache_enabled=_bool("CACHE_ENABLED", cls.cache_enabled),
            cache_ttl_seconds=_int("CACHE_TTL_SECONDS", cls.cache_ttl_seconds),
            cache_max_size=_int("CACHE_MAX_SIZE", cls.cache_max_size),
            max_scenarios=_int("MAX_SCENARIOS", cls.max_scenarios),
            max_assumptions=_int("MAX_ASSUMPTIONS", cls.max_assumptions),
            max_validation_rules=_int(
                "MAX_VALIDATION_RULES", cls.max_validation_rules,
            ),
            default_uncertainty_pct=_float(
                "DEFAULT_UNCERTAINTY_PCT", cls.default_uncertainty_pct,
            ),
        )

        logger.info(
            "AssumptionsConfig loaded: max_versions=%d, validation=%s, "
            "change_log=%s, cache=%s/%ds, max_assumptions=%d",
            config.max_versions_per_assumption,
            config.enable_validation,
            config.enable_change_logging,
            config.cache_enabled,
            config.cache_ttl_seconds,
            config.max_assumptions,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[AssumptionsConfig] = None
_config_lock = threading.Lock()


def get_config() -> AssumptionsConfig:
    """Return the singleton AssumptionsConfig, creating from env if needed.

    Returns:
        AssumptionsConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = AssumptionsConfig.from_env()
    return _config_instance


def set_config(config: AssumptionsConfig) -> None:
    """Replace the singleton AssumptionsConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("AssumptionsConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "AssumptionsConfig",
    "get_config",
    "set_config",
    "reset_config",
]
