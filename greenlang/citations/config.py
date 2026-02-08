# -*- coding: utf-8 -*-
"""
Citations & Evidence Configuration - AGENT-FOUND-005: Citations & Evidence

Centralized configuration for the Citations & Evidence SDK covering:
- Citation capacity limits
- Evidence package limits
- Cache settings (enable, TTL, max size)
- Verification and change logging toggles
- Default expiration and hash validation settings

All settings can be overridden via environment variables with the
``GL_CITATIONS_`` prefix (e.g. ``GL_CITATIONS_MAX_CITATIONS``).

Example:
    >>> from greenlang.citations.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.max_citations, cfg.enable_verification)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-005 Citations & Evidence
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

_ENV_PREFIX = "GL_CITATIONS_"


# ---------------------------------------------------------------------------
# CitationsConfig
# ---------------------------------------------------------------------------


@dataclass
class CitationsConfig:
    """Complete configuration for the GreenLang Citations & Evidence SDK.

    Attributes are grouped by concern: capacity, caching, verification,
    logging, and defaults.

    All attributes can be overridden via environment variables using the
    ``GL_CITATIONS_`` prefix.

    Attributes:
        max_citations: Maximum number of citations in the registry.
        max_packages: Maximum number of evidence packages.
        cache_enabled: Whether to enable citation caching.
        cache_ttl_seconds: Cache time-to-live in seconds.
        cache_max_size: Maximum number of entries in the cache.
        enable_verification: Whether to enable citation verification on access.
        enable_change_logging: Whether to log all changes to the audit trail.
        default_expiration_years: Default expiration period for citations in years.
        enable_hash_validation: Whether to validate SHA-256 content hashes.
        max_evidence_items_per_package: Maximum evidence items per package.
    """

    # -- Capacity limits -----------------------------------------------------
    max_citations: int = 10000
    max_packages: int = 5000

    # -- Caching -------------------------------------------------------------
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    cache_max_size: int = 2000

    # -- Verification --------------------------------------------------------
    enable_verification: bool = True

    # -- Change logging ------------------------------------------------------
    enable_change_logging: bool = True

    # -- Defaults ------------------------------------------------------------
    default_expiration_years: int = 5
    enable_hash_validation: bool = True

    # -- Package limits ------------------------------------------------------
    max_evidence_items_per_package: int = 100

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> CitationsConfig:
        """Build a CitationsConfig from environment variables.

        Every field can be overridden via ``GL_CITATIONS_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.

        Returns:
            Populated CitationsConfig instance.
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

        config = cls(
            max_citations=_int("MAX_CITATIONS", cls.max_citations),
            max_packages=_int("MAX_PACKAGES", cls.max_packages),
            cache_enabled=_bool("CACHE_ENABLED", cls.cache_enabled),
            cache_ttl_seconds=_int("CACHE_TTL_SECONDS", cls.cache_ttl_seconds),
            cache_max_size=_int("CACHE_MAX_SIZE", cls.cache_max_size),
            enable_verification=_bool(
                "ENABLE_VERIFICATION", cls.enable_verification,
            ),
            enable_change_logging=_bool(
                "ENABLE_CHANGE_LOGGING", cls.enable_change_logging,
            ),
            default_expiration_years=_int(
                "DEFAULT_EXPIRATION_YEARS", cls.default_expiration_years,
            ),
            enable_hash_validation=_bool(
                "ENABLE_HASH_VALIDATION", cls.enable_hash_validation,
            ),
            max_evidence_items_per_package=_int(
                "MAX_EVIDENCE_ITEMS_PER_PACKAGE",
                cls.max_evidence_items_per_package,
            ),
        )

        logger.info(
            "CitationsConfig loaded: max_citations=%d, max_packages=%d, "
            "verification=%s, change_log=%s, cache=%s/%ds",
            config.max_citations,
            config.max_packages,
            config.enable_verification,
            config.enable_change_logging,
            config.cache_enabled,
            config.cache_ttl_seconds,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[CitationsConfig] = None
_config_lock = threading.Lock()


def get_config() -> CitationsConfig:
    """Return the singleton CitationsConfig, creating from env if needed.

    Returns:
        CitationsConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = CitationsConfig.from_env()
    return _config_instance


def set_config(config: CitationsConfig) -> None:
    """Replace the singleton CitationsConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("CitationsConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "CitationsConfig",
    "get_config",
    "set_config",
    "reset_config",
]
