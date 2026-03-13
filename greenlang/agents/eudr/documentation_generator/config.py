# -*- coding: utf-8 -*-
"""
Documentation Generator Configuration - AGENT-EUDR-030

Centralized configuration for the Documentation Generator covering:
- Database and cache connection settings (PostgreSQL, Redis)
- DDS generation: reference prefix, schema version, product limits,
  provenance inclusion
- Article 9 compliance: completeness threshold, polygon requirements,
  geolocation decimal precision
- Risk documentation: criterion details, decomposition, trend data
- Mitigation documentation: evidence summaries, timelines, effectiveness
- Compliance packages: format, cross-references, table of contents,
  maximum package size
- Document versioning: max versions, retention years, amendment tracking
- Submission: timeout, retries, delay, batch size, EU Information
  System URL
- Upstream agent URLs: supply chain, risk assessment, mitigation,
  information gathering
- Report generation settings: format, appendices, regulatory refs
- Rate limiting: 5 tiers (anonymous/basic/standard/premium/admin)
- Circuit breaker: failure threshold, reset timeout, half-open calls
- Batch processing: max concurrent, timeout
- Provenance: SHA-256 hash chain settings
- Metrics: Prometheus prefix gl_eudr_dgn_

All settings overridable via environment variables with ``GL_EUDR_DGN_`` prefix.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-030 Documentation Generator (GL-EUDR-DGN-030)
Regulation: EU 2023/1115 (EUDR) Articles 4, 5, 6, 9, 10, 31, 33
Status: Production Ready
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ENV_PREFIX = "GL_EUDR_DGN_"


def _env(key: str, default: Any = None) -> Any:
    """Read environment variable with GL_EUDR_DGN_ prefix."""
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


def _env_int(key: str, default: int) -> int:
    """Read integer environment variable with GL_EUDR_DGN_ prefix."""
    val = _env(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    """Read float environment variable with GL_EUDR_DGN_ prefix."""
    val = _env(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    """Read boolean environment variable with GL_EUDR_DGN_ prefix."""
    val = _env(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_decimal(key: str, default: str) -> Decimal:
    """Read Decimal environment variable with GL_EUDR_DGN_ prefix."""
    val = _env(key)
    return Decimal(val) if val is not None else Decimal(default)


# ---------------------------------------------------------------------------
# Main Config
# ---------------------------------------------------------------------------


@dataclass
class DocumentationGeneratorConfig:
    """Centralized configuration for AGENT-EUDR-030.

    Thread-safe singleton accessed via ``get_config()``.
    All values overridable via GL_EUDR_DGN_ environment variables.
    """

    # -- 1. Database ------------------------------------------------------------
    db_host: str = field(
        default_factory=lambda: _env("DB_HOST", "localhost")
    )
    db_port: int = field(
        default_factory=lambda: _env_int("DB_PORT", 5432)
    )
    db_name: str = field(
        default_factory=lambda: _env("DB_NAME", "greenlang")
    )
    db_user: str = field(
        default_factory=lambda: _env("DB_USER", "gl")
    )
    db_password: str = field(
        default_factory=lambda: _env("DB_PASSWORD", "gl")
    )
    db_pool_min: int = field(
        default_factory=lambda: _env_int("DB_POOL_MIN", 2)
    )
    db_pool_max: int = field(
        default_factory=lambda: _env_int("DB_POOL_MAX", 10)
    )

    # -- 2. Redis ---------------------------------------------------------------
    redis_host: str = field(
        default_factory=lambda: _env("REDIS_HOST", "localhost")
    )
    redis_port: int = field(
        default_factory=lambda: _env_int("REDIS_PORT", 6379)
    )
    redis_db: int = field(
        default_factory=lambda: _env_int("REDIS_DB", 0)
    )
    redis_password: str = field(
        default_factory=lambda: _env("REDIS_PASSWORD", "")
    )
    cache_ttl: int = field(
        default_factory=lambda: _env_int("CACHE_TTL", 3600)
    )

    # -- 3. DDS Generation ------------------------------------------------------
    dds_reference_prefix: str = field(
        default_factory=lambda: _env("DDS_REFERENCE_PREFIX", "DDS")
    )
    dds_schema_version: str = field(
        default_factory=lambda: _env("DDS_SCHEMA_VERSION", "1.0")
    )
    max_products_per_dds: int = field(
        default_factory=lambda: _env_int("MAX_PRODUCTS_PER_DDS", 100)
    )
    include_provenance: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_PROVENANCE", True)
    )

    # -- 4. Article 9 Compliance ------------------------------------------------
    article9_completeness_threshold: Decimal = field(
        default_factory=lambda: _env_decimal(
            "ARTICLE9_COMPLETENESS_THRESHOLD", "0.95"
        )
    )
    require_polygon_above_4ha: bool = field(
        default_factory=lambda: _env_bool("REQUIRE_POLYGON_ABOVE_4HA", True)
    )
    geolocation_decimal_places: int = field(
        default_factory=lambda: _env_int("GEOLOCATION_DECIMAL_PLACES", 6)
    )

    # -- 5. Risk Documentation --------------------------------------------------
    include_criterion_details: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_CRITERION_DETAILS", True)
    )
    include_decomposition: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_DECOMPOSITION", True)
    )
    include_trend_data: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_TREND_DATA", True)
    )

    # -- 6. Mitigation Documentation -------------------------------------------
    include_evidence_summary: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_EVIDENCE_SUMMARY", True)
    )
    include_timeline: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_TIMELINE", True)
    )
    include_effectiveness: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_EFFECTIVENESS", True)
    )

    # -- 7. Compliance Packages -------------------------------------------------
    package_format: str = field(
        default_factory=lambda: _env("PACKAGE_FORMAT", "json")
    )
    include_cross_references: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_CROSS_REFERENCES", True)
    )
    include_table_of_contents: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_TABLE_OF_CONTENTS", True)
    )
    max_package_size_mb: int = field(
        default_factory=lambda: _env_int("MAX_PACKAGE_SIZE_MB", 500)
    )

    # -- 8. Document Versioning -------------------------------------------------
    max_versions_per_document: int = field(
        default_factory=lambda: _env_int("MAX_VERSIONS_PER_DOCUMENT", 50)
    )
    retention_years: int = field(
        default_factory=lambda: _env_int("RETENTION_YEARS", 5)
    )
    enable_amendment_tracking: bool = field(
        default_factory=lambda: _env_bool("ENABLE_AMENDMENT_TRACKING", True)
    )

    # -- 9. Submission ----------------------------------------------------------
    submission_timeout_seconds: int = field(
        default_factory=lambda: _env_int("SUBMISSION_TIMEOUT_SECONDS", 60)
    )
    max_retries: int = field(
        default_factory=lambda: _env_int("MAX_RETRIES", 3)
    )
    retry_delay_seconds: int = field(
        default_factory=lambda: _env_int("RETRY_DELAY_SECONDS", 10)
    )
    batch_size: int = field(
        default_factory=lambda: _env_int("BATCH_SIZE", 10)
    )
    eu_information_system_url: str = field(
        default_factory=lambda: _env(
            "EU_INFORMATION_SYSTEM_URL",
            "https://eudr-is.europa.eu/api/v1",
        )
    )

    # -- 10. Upstream Agent URLs ------------------------------------------------
    supply_chain_url: str = field(
        default_factory=lambda: _env(
            "SUPPLY_CHAIN_URL",
            "http://eudr-supply-chain:8001/api/v1/eudr/supply-chain",
        )
    )
    risk_assessment_url: str = field(
        default_factory=lambda: _env(
            "RISK_ASSESSMENT_URL",
            "http://eudr-risk-assessment:8028/api/v1/eudr/risk-assessment",
        )
    )
    mitigation_url: str = field(
        default_factory=lambda: _env(
            "MITIGATION_URL",
            "http://eudr-mitigation:8029/api/v1/eudr/mitigation",
        )
    )
    information_gathering_url: str = field(
        default_factory=lambda: _env(
            "INFORMATION_GATHERING_URL",
            "http://eudr-info-gathering:8027/api/v1/eudr/information-gathering",
        )
    )

    # -- 11. Report Generation --------------------------------------------------
    default_format: str = field(
        default_factory=lambda: _env("DEFAULT_FORMAT", "json")
    )
    include_appendices: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_APPENDICES", True)
    )
    include_regulatory_refs: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_REGULATORY_REFS", True)
    )

    # -- 12. Rate Limiting ------------------------------------------------------
    rate_limit_anonymous: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_ANONYMOUS", 10)
    )
    rate_limit_basic: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_BASIC", 30)
    )
    rate_limit_standard: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_STANDARD", 100)
    )
    rate_limit_premium: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_PREMIUM", 500)
    )
    rate_limit_admin: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_ADMIN", 2000)
    )

    # -- 13. Circuit Breaker ----------------------------------------------------
    circuit_breaker_failure_threshold: int = field(
        default_factory=lambda: _env_int("CB_FAILURE_THRESHOLD", 5)
    )
    circuit_breaker_reset_timeout: int = field(
        default_factory=lambda: _env_int("CB_RESET_TIMEOUT", 60)
    )
    circuit_breaker_half_open_max: int = field(
        default_factory=lambda: _env_int("CB_HALF_OPEN_MAX", 3)
    )

    # -- 14. Batch Processing ---------------------------------------------------
    max_concurrent: int = field(
        default_factory=lambda: _env_int("MAX_CONCURRENT", 10)
    )
    batch_timeout_seconds: int = field(
        default_factory=lambda: _env_int("BATCH_TIMEOUT", 300)
    )

    # -- 15. Provenance ---------------------------------------------------------
    provenance_enabled: bool = field(
        default_factory=lambda: _env_bool("PROVENANCE_ENABLED", True)
    )
    provenance_algorithm: str = "sha256"
    provenance_chain_enabled: bool = field(
        default_factory=lambda: _env_bool("PROVENANCE_CHAIN_ENABLED", True)
    )
    provenance_genesis_hash: str = (
        "0000000000000000000000000000000000000000000000000000000000000000"
    )

    # -- 16. Metrics ------------------------------------------------------------
    metrics_enabled: bool = field(
        default_factory=lambda: _env_bool("METRICS_ENABLED", True)
    )
    metrics_prefix: str = "gl_eudr_dgn_"

    # -- Logging ----------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate Article 9 completeness threshold range
        if not (Decimal("0") <= self.article9_completeness_threshold <= Decimal("1")):
            logger.warning(
                "Article 9 completeness threshold %s is outside [0, 1] range.",
                self.article9_completeness_threshold,
            )

        # Validate geolocation decimal places
        if self.geolocation_decimal_places < 1 or self.geolocation_decimal_places > 15:
            logger.warning(
                "Geolocation decimal places %d is outside [1, 15] range.",
                self.geolocation_decimal_places,
            )

        # Validate max products per DDS
        if self.max_products_per_dds < 1:
            logger.warning(
                "Max products per DDS %d must be at least 1.",
                self.max_products_per_dds,
            )

        # Validate max package size
        if self.max_package_size_mb < 1:
            logger.warning(
                "Max package size %d MB must be at least 1.",
                self.max_package_size_mb,
            )

        # Validate pool sizing
        if self.db_pool_min > self.db_pool_max:
            logger.warning(
                "DB pool min %d exceeds pool max %d.",
                self.db_pool_min,
                self.db_pool_max,
            )

        # Validate submission retries
        if self.max_retries < 0:
            logger.warning(
                "Max retries %d is negative.",
                self.max_retries,
            )

        # Validate retention years
        if self.retention_years < 5:
            logger.warning(
                "Retention years %d is below EUDR minimum of 5 years "
                "(Article 31).",
                self.retention_years,
            )

        # Validate version limits
        if self.max_versions_per_document < 1:
            logger.warning(
                "Max versions per document %d must be at least 1.",
                self.max_versions_per_document,
            )

        # Validate package format
        valid_formats = ("json", "xml", "pdf_structured")
        if self.package_format.lower() not in valid_formats:
            logger.warning(
                "Package format '%s' is not one of %s.",
                self.package_format,
                valid_formats,
            )

        logger.info(
            "DocumentationGeneratorConfig initialized: "
            "dds_prefix=%s, dds_schema=%s, max_products=%d, "
            "article9_threshold=%s, polygon_above_4ha=%s, "
            "geo_decimals=%d, "
            "package_format=%s, max_package_mb=%d, "
            "max_versions=%d, retention_years=%d, "
            "submission_timeout=%ds, max_retries=%d, "
            "eu_is_url=%s",
            self.dds_reference_prefix,
            self.dds_schema_version,
            self.max_products_per_dds,
            self.article9_completeness_threshold,
            self.require_polygon_above_4ha,
            self.geolocation_decimal_places,
            self.package_format,
            self.max_package_size_mb,
            self.max_versions_per_document,
            self.retention_years,
            self.submission_timeout_seconds,
            self.max_retries,
            self.eu_information_system_url,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[DocumentationGeneratorConfig] = None
_config_lock = threading.Lock()


def get_config() -> DocumentationGeneratorConfig:
    """Return the thread-safe singleton configuration instance.

    Returns:
        DocumentationGeneratorConfig singleton.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = DocumentationGeneratorConfig()
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing only)."""
    global _config_instance
    with _config_lock:
        _config_instance = None
