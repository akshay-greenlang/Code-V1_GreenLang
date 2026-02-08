# -*- coding: utf-8 -*-
"""
Reproducibility Service Configuration - AGENT-FOUND-008: Reproducibility Agent

Centralized configuration for the Reproducibility SDK covering:
- Tolerance thresholds for numeric comparisons
- Drift detection soft/hard thresholds
- Hash algorithm and caching settings
- Environment capture, seed management, version pinning toggles
- Replay mode settings (strict mode, max duration)
- Float normalization precision
- Verification timeout
- Connection pool sizing

All settings can be overridden via environment variables with the
``GL_REPRODUCIBILITY_`` prefix (e.g. ``GL_REPRODUCIBILITY_DEFAULT_ABSOLUTE_TOLERANCE``).

Example:
    >>> from greenlang.reproducibility.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_absolute_tolerance, cfg.hash_algorithm)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-008 Reproducibility Agent
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

_ENV_PREFIX = "GL_REPRODUCIBILITY_"


# ---------------------------------------------------------------------------
# ReproducibilityConfig
# ---------------------------------------------------------------------------


@dataclass
class ReproducibilityConfig:
    """Complete configuration for the GreenLang Reproducibility SDK.

    Attributes are grouped by concern: connections, tolerances, drift,
    hashing, feature toggles, replay, normalization, verification,
    pool sizing, and logging.

    All attributes can be overridden via environment variables using the
    ``GL_REPRODUCIBILITY_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        default_absolute_tolerance: Absolute tolerance for float comparison.
        default_relative_tolerance: Relative tolerance for float comparison.
        drift_soft_threshold: Soft threshold for drift warning (fraction).
        drift_hard_threshold: Hard threshold for drift failure (fraction).
        hash_algorithm: Hash algorithm for artifact hashing.
        hash_cache_ttl_seconds: TTL for hash result caching.
        environment_capture_enabled: Whether to capture environment fingerprints.
        seed_management_enabled: Whether to manage random seeds.
        version_pinning_enabled: Whether to pin component versions.
        non_determinism_tracking_enabled: Whether to track non-determinism sources.
        replay_strict_mode: Whether replay fails on any environment mismatch.
        max_replay_duration_seconds: Maximum duration for a replay execution.
        float_normalization_decimals: Decimal places for float normalization.
        verification_timeout_seconds: Timeout for a single verification run.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        log_level: Logging level for the reproducibility service.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""

    # -- Tolerances ----------------------------------------------------------
    default_absolute_tolerance: float = 1e-9
    default_relative_tolerance: float = 1e-6

    # -- Drift ---------------------------------------------------------------
    drift_soft_threshold: float = 0.01   # 1%
    drift_hard_threshold: float = 0.05   # 5%

    # -- Hashing -------------------------------------------------------------
    hash_algorithm: str = "sha256"
    hash_cache_ttl_seconds: int = 3600

    # -- Feature toggles -----------------------------------------------------
    environment_capture_enabled: bool = True
    seed_management_enabled: bool = True
    version_pinning_enabled: bool = True
    non_determinism_tracking_enabled: bool = True

    # -- Replay --------------------------------------------------------------
    replay_strict_mode: bool = False
    max_replay_duration_seconds: int = 300

    # -- Normalization -------------------------------------------------------
    float_normalization_decimals: int = 15

    # -- Verification --------------------------------------------------------
    verification_timeout_seconds: int = 60

    # -- Pool sizing ---------------------------------------------------------
    pool_min_size: int = 2
    pool_max_size: int = 10

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> ReproducibilityConfig:
        """Build a ReproducibilityConfig from environment variables.

        Every field can be overridden via ``GL_REPRODUCIBILITY_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated ReproducibilityConfig instance.
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
                    "Invalid float for %s%s=%s, using default %s",
                    prefix, name, val, default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val

        config = cls(
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            default_absolute_tolerance=_float(
                "DEFAULT_ABSOLUTE_TOLERANCE", cls.default_absolute_tolerance,
            ),
            default_relative_tolerance=_float(
                "DEFAULT_RELATIVE_TOLERANCE", cls.default_relative_tolerance,
            ),
            drift_soft_threshold=_float(
                "DRIFT_SOFT_THRESHOLD", cls.drift_soft_threshold,
            ),
            drift_hard_threshold=_float(
                "DRIFT_HARD_THRESHOLD", cls.drift_hard_threshold,
            ),
            hash_algorithm=_str("HASH_ALGORITHM", cls.hash_algorithm),
            hash_cache_ttl_seconds=_int(
                "HASH_CACHE_TTL_SECONDS", cls.hash_cache_ttl_seconds,
            ),
            environment_capture_enabled=_bool(
                "ENVIRONMENT_CAPTURE_ENABLED", cls.environment_capture_enabled,
            ),
            seed_management_enabled=_bool(
                "SEED_MANAGEMENT_ENABLED", cls.seed_management_enabled,
            ),
            version_pinning_enabled=_bool(
                "VERSION_PINNING_ENABLED", cls.version_pinning_enabled,
            ),
            non_determinism_tracking_enabled=_bool(
                "NON_DETERMINISM_TRACKING_ENABLED",
                cls.non_determinism_tracking_enabled,
            ),
            replay_strict_mode=_bool(
                "REPLAY_STRICT_MODE", cls.replay_strict_mode,
            ),
            max_replay_duration_seconds=_int(
                "MAX_REPLAY_DURATION_SECONDS", cls.max_replay_duration_seconds,
            ),
            float_normalization_decimals=_int(
                "FLOAT_NORMALIZATION_DECIMALS", cls.float_normalization_decimals,
            ),
            verification_timeout_seconds=_int(
                "VERIFICATION_TIMEOUT_SECONDS", cls.verification_timeout_seconds,
            ),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            log_level=_str("LOG_LEVEL", cls.log_level),
        )

        logger.info(
            "ReproducibilityConfig loaded: abs_tol=%s, rel_tol=%s, "
            "drift_soft=%s, drift_hard=%s, hash=%s, cache_ttl=%ds, "
            "env_capture=%s, seeds=%s, version_pin=%s, replay_strict=%s",
            config.default_absolute_tolerance,
            config.default_relative_tolerance,
            config.drift_soft_threshold,
            config.drift_hard_threshold,
            config.hash_algorithm,
            config.hash_cache_ttl_seconds,
            config.environment_capture_enabled,
            config.seed_management_enabled,
            config.version_pinning_enabled,
            config.replay_strict_mode,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[ReproducibilityConfig] = None
_config_lock = threading.Lock()


def get_config() -> ReproducibilityConfig:
    """Return the singleton ReproducibilityConfig, creating from env if needed.

    Returns:
        ReproducibilityConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ReproducibilityConfig.from_env()
    return _config_instance


def set_config(config: ReproducibilityConfig) -> None:
    """Replace the singleton ReproducibilityConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("ReproducibilityConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "ReproducibilityConfig",
    "get_config",
    "set_config",
    "reset_config",
]
