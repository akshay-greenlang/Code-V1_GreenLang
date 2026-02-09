# -*- coding: utf-8 -*-
"""
Spend Data Categorizer Service Configuration - AGENT-DATA-009

Centralized configuration for the Spend Data Categorizer SDK covering:
- Database and cache connection defaults
- Classification settings (taxonomy, confidence thresholds)
- Emission factor source versions (EPA EEIO, EXIOBASE, DEFRA)
- Processing limits (batch size, max records, dedup threshold)
- Vendor normalization toggles
- Cache TTL settings
- Rate limiting
- Provenance tracking settings

All settings can be overridden via environment variables with the
``GL_SPEND_CAT_`` prefix (e.g. ``GL_SPEND_CAT_DEFAULT_CURRENCY``).

Example:
    >>> from greenlang.spend_categorizer.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_taxonomy, cfg.min_confidence)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-009 Spend Data Categorizer (GL-DATA-SUP-002)
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

_ENV_PREFIX = "GL_SPEND_CAT_"


# ---------------------------------------------------------------------------
# SpendCategorizerConfig
# ---------------------------------------------------------------------------


@dataclass
class SpendCategorizerConfig:
    """Complete configuration for the GreenLang Spend Data Categorizer SDK.

    Attributes are grouped by concern: connections, classification defaults,
    emission factor versions, processing limits, cache, feature toggles,
    rate limiting, and provenance tracking.

    All attributes can be overridden via environment variables using the
    ``GL_SPEND_CAT_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        s3_bucket_url: S3 bucket URL for document and report storage.
        log_level: Logging level for the spend categorizer service.
        default_currency: Default currency for spend amounts (ISO 4217).
        default_taxonomy: Default taxonomy system for classification.
        min_confidence: Minimum confidence threshold for classification.
        high_confidence_threshold: Threshold for HIGH confidence label.
        medium_confidence_threshold: Threshold for MEDIUM confidence label.
        eeio_version: EPA EEIO emission factor database version.
        exiobase_version: EXIOBASE emission factor database version.
        defra_version: DEFRA emission factor database version.
        ecoinvent_version: Ecoinvent emission factor database version.
        batch_size: Number of records per processing batch.
        max_records: Maximum total records per processing operation.
        dedup_threshold: Similarity threshold for deduplication matching.
        vendor_normalization: Whether to normalize vendor names.
        max_taxonomy_depth: Maximum depth for taxonomy code resolution.
        cache_ttl: Default cache time-to-live in seconds.
        cache_emission_factors_ttl: Cache TTL for emission factors in seconds.
        cache_taxonomy_ttl: Cache TTL for taxonomy lookups in seconds.
        enable_exiobase: Whether to enable EXIOBASE emission factors.
        enable_defra: Whether to enable DEFRA emission factors.
        enable_ecoinvent: Whether to enable Ecoinvent emission factors.
        enable_hotspot_analysis: Whether to enable hotspot analytics.
        enable_trend_analysis: Whether to enable trend analytics.
        rate_limit_rpm: Requests per minute rate limit.
        rate_limit_burst: Maximum burst requests allowed.
        enable_provenance: Whether to enable provenance tracking.
        provenance_hash_algorithm: Hash algorithm for provenance chains.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        worker_count: Number of parallel workers for batch processing.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""
    s3_bucket_url: str = ""

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Classification defaults ---------------------------------------------
    default_currency: str = "USD"
    default_taxonomy: str = "unspsc"
    min_confidence: float = 0.3
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.5

    # -- Emission factor versions --------------------------------------------
    eeio_version: str = "2024"
    exiobase_version: str = "3.8.2"
    defra_version: str = "2025"
    ecoinvent_version: str = "3.10"

    # -- Processing limits ---------------------------------------------------
    batch_size: int = 1000
    max_records: int = 100000
    dedup_threshold: float = 0.85
    vendor_normalization: bool = True
    max_taxonomy_depth: int = 6

    # -- Cache ---------------------------------------------------------------
    cache_ttl: int = 3600
    cache_emission_factors_ttl: int = 86400
    cache_taxonomy_ttl: int = 43200

    # -- Feature toggles -----------------------------------------------------
    enable_exiobase: bool = True
    enable_defra: bool = True
    enable_ecoinvent: bool = False
    enable_hotspot_analysis: bool = True
    enable_trend_analysis: bool = True

    # -- Rate limiting -------------------------------------------------------
    rate_limit_rpm: int = 120
    rate_limit_burst: int = 20

    # -- Provenance ----------------------------------------------------------
    enable_provenance: bool = True
    provenance_hash_algorithm: str = "sha256"

    # -- Pool sizing ---------------------------------------------------------
    pool_min_size: int = 2
    pool_max_size: int = 10
    worker_count: int = 4

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> SpendCategorizerConfig:
        """Build a SpendCategorizerConfig from environment variables.

        Every field can be overridden via ``GL_SPEND_CAT_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated SpendCategorizerConfig instance.
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
                    "Invalid float for %s%s=%s, using default %f",
                    prefix, name, val, default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val

        config = cls(
            # Connections
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            s3_bucket_url=_str("S3_BUCKET_URL", cls.s3_bucket_url),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
            # Classification defaults
            default_currency=_str(
                "DEFAULT_CURRENCY", cls.default_currency,
            ),
            default_taxonomy=_str(
                "DEFAULT_TAXONOMY", cls.default_taxonomy,
            ),
            min_confidence=_float(
                "MIN_CONFIDENCE", cls.min_confidence,
            ),
            high_confidence_threshold=_float(
                "HIGH_CONFIDENCE_THRESHOLD",
                cls.high_confidence_threshold,
            ),
            medium_confidence_threshold=_float(
                "MEDIUM_CONFIDENCE_THRESHOLD",
                cls.medium_confidence_threshold,
            ),
            # Emission factor versions
            eeio_version=_str(
                "EEIO_VERSION", cls.eeio_version,
            ),
            exiobase_version=_str(
                "EXIOBASE_VERSION", cls.exiobase_version,
            ),
            defra_version=_str(
                "DEFRA_VERSION", cls.defra_version,
            ),
            ecoinvent_version=_str(
                "ECOINVENT_VERSION", cls.ecoinvent_version,
            ),
            # Processing limits
            batch_size=_int(
                "BATCH_SIZE", cls.batch_size,
            ),
            max_records=_int(
                "MAX_RECORDS", cls.max_records,
            ),
            dedup_threshold=_float(
                "DEDUP_THRESHOLD", cls.dedup_threshold,
            ),
            vendor_normalization=_bool(
                "VENDOR_NORMALIZATION",
                cls.vendor_normalization,
            ),
            max_taxonomy_depth=_int(
                "MAX_TAXONOMY_DEPTH", cls.max_taxonomy_depth,
            ),
            # Cache
            cache_ttl=_int("CACHE_TTL", cls.cache_ttl),
            cache_emission_factors_ttl=_int(
                "CACHE_EMISSION_FACTORS_TTL",
                cls.cache_emission_factors_ttl,
            ),
            cache_taxonomy_ttl=_int(
                "CACHE_TAXONOMY_TTL", cls.cache_taxonomy_ttl,
            ),
            # Feature toggles
            enable_exiobase=_bool(
                "ENABLE_EXIOBASE", cls.enable_exiobase,
            ),
            enable_defra=_bool(
                "ENABLE_DEFRA", cls.enable_defra,
            ),
            enable_ecoinvent=_bool(
                "ENABLE_ECOINVENT", cls.enable_ecoinvent,
            ),
            enable_hotspot_analysis=_bool(
                "ENABLE_HOTSPOT_ANALYSIS",
                cls.enable_hotspot_analysis,
            ),
            enable_trend_analysis=_bool(
                "ENABLE_TREND_ANALYSIS",
                cls.enable_trend_analysis,
            ),
            # Rate limiting
            rate_limit_rpm=_int(
                "RATE_LIMIT_RPM", cls.rate_limit_rpm,
            ),
            rate_limit_burst=_int(
                "RATE_LIMIT_BURST", cls.rate_limit_burst,
            ),
            # Provenance
            enable_provenance=_bool(
                "ENABLE_PROVENANCE", cls.enable_provenance,
            ),
            provenance_hash_algorithm=_str(
                "PROVENANCE_HASH_ALGORITHM",
                cls.provenance_hash_algorithm,
            ),
            # Pool sizing
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            worker_count=_int("WORKER_COUNT", cls.worker_count),
        )

        logger.info(
            "SpendCategorizerConfig loaded: taxonomy=%s, currency=%s, "
            "min_confidence=%.2f, batch_size=%d, max_records=%d, "
            "dedup=%.2f, vendor_norm=%s, eeio=%s, exiobase=%s/%s, "
            "defra=%s/%s, workers=%d, provenance=%s",
            config.default_taxonomy,
            config.default_currency,
            config.min_confidence,
            config.batch_size,
            config.max_records,
            config.dedup_threshold,
            config.vendor_normalization,
            config.eeio_version,
            config.exiobase_version,
            config.enable_exiobase,
            config.defra_version,
            config.enable_defra,
            config.worker_count,
            config.enable_provenance,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[SpendCategorizerConfig] = None
_config_lock = threading.Lock()


def get_config() -> SpendCategorizerConfig:
    """Return the singleton SpendCategorizerConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path.

    Returns:
        SpendCategorizerConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = SpendCategorizerConfig.from_env()
    return _config_instance


def set_config(config: SpendCategorizerConfig) -> None:
    """Replace the singleton SpendCategorizerConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("SpendCategorizerConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "SpendCategorizerConfig",
    "get_config",
    "set_config",
    "reset_config",
]
