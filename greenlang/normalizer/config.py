# -*- coding: utf-8 -*-
"""
Normalizer Configuration - AGENT-FOUND-003: Unit & Reference Normalizer

Centralized configuration for the Normalizer SDK covering:
- Default decimal precision for conversions
- GWP version and timeframe selection
- Conversion cache settings
- Batch processing limits
- Dimensional analysis strictness
- Canonical unit defaults

All settings can be overridden via environment variables with the
``GL_NORMALIZER_`` prefix (e.g. ``GL_NORMALIZER_DEFAULT_PRECISION``).

Example:
    >>> from greenlang.normalizer.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_precision, cfg.gwp_version)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-003 Unit & Reference Normalizer
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

_ENV_PREFIX = "GL_NORMALIZER_"


# ---------------------------------------------------------------------------
# NormalizerConfig
# ---------------------------------------------------------------------------


@dataclass
class NormalizerConfig:
    """Complete configuration for the GreenLang Normalizer SDK.

    Attributes are grouped by concern: precision, GWP, caching,
    batch processing, dimensional analysis, and canonical units.

    All attributes can be overridden via environment variables using the
    ``GL_NORMALIZER_`` prefix.

    Attributes:
        default_precision: Number of decimal places for Decimal operations.
        gwp_version: IPCC GWP version to use (AR5 or AR6).
        gwp_timeframe: GWP timeframe in years (20 or 100).
        cache_enabled: Whether to enable conversion factor caching.
        cache_ttl_seconds: Cache time-to-live in seconds.
        max_batch_size: Maximum items in a single batch operation.
        strict_dimensional_check: Reject cross-dimension conversions strictly.
        canonical_energy_unit: Default canonical energy unit.
        canonical_mass_unit: Default canonical mass unit.
        canonical_emissions_unit: Default canonical emissions unit.
    """

    # -- Decimal precision ---------------------------------------------------
    default_precision: int = 10

    # -- GWP settings -------------------------------------------------------
    gwp_version: str = "AR6"
    gwp_timeframe: int = 100

    # -- Caching ------------------------------------------------------------
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600

    # -- Batch processing ---------------------------------------------------
    max_batch_size: int = 1000

    # -- Dimensional analysis -----------------------------------------------
    strict_dimensional_check: bool = True

    # -- Canonical units ----------------------------------------------------
    canonical_energy_unit: str = "kWh"
    canonical_mass_unit: str = "kg"
    canonical_emissions_unit: str = "kgCO2e"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> NormalizerConfig:
        """Build a NormalizerConfig from environment variables.

        Every field can be overridden via ``GL_NORMALIZER_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.

        Returns:
            Populated NormalizerConfig instance.
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
            default_precision=_int("DEFAULT_PRECISION", cls.default_precision),
            gwp_version=_str("GWP_VERSION", cls.gwp_version),
            gwp_timeframe=_int("GWP_TIMEFRAME", cls.gwp_timeframe),
            cache_enabled=_bool("CACHE_ENABLED", cls.cache_enabled),
            cache_ttl_seconds=_int("CACHE_TTL_SECONDS", cls.cache_ttl_seconds),
            max_batch_size=_int("MAX_BATCH_SIZE", cls.max_batch_size),
            strict_dimensional_check=_bool(
                "STRICT_DIMENSIONAL_CHECK", cls.strict_dimensional_check,
            ),
            canonical_energy_unit=_str(
                "CANONICAL_ENERGY_UNIT", cls.canonical_energy_unit,
            ),
            canonical_mass_unit=_str(
                "CANONICAL_MASS_UNIT", cls.canonical_mass_unit,
            ),
            canonical_emissions_unit=_str(
                "CANONICAL_EMISSIONS_UNIT", cls.canonical_emissions_unit,
            ),
        )

        logger.info(
            "NormalizerConfig loaded: precision=%d, gwp=%s/%d, "
            "cache=%s, batch_max=%d, strict=%s",
            config.default_precision,
            config.gwp_version,
            config.gwp_timeframe,
            config.cache_enabled,
            config.max_batch_size,
            config.strict_dimensional_check,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[NormalizerConfig] = None
_config_lock = threading.Lock()


def get_config() -> NormalizerConfig:
    """Return the singleton NormalizerConfig, creating from env if needed.

    Returns:
        NormalizerConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = NormalizerConfig.from_env()
    return _config_instance


def set_config(config: NormalizerConfig) -> None:
    """Replace the singleton NormalizerConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("NormalizerConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "NormalizerConfig",
    "get_config",
    "set_config",
    "reset_config",
]
