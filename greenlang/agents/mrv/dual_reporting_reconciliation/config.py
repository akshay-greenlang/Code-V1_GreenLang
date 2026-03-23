# -*- coding: utf-8 -*-
"""
Dual Reporting Reconciliation Agent Configuration - AGENT-MRV-013

Centralized configuration for the Dual Reporting Reconciliation Agent SDK
covering:
- General service settings (name, version, logging, environment, tenant)
- Database connection and pool settings (PostgreSQL)
- Reconciliation calculation defaults (GWP source, decimal precision,
  feature toggles for trends, quality, compliance)
- Materiality threshold settings (immaterial, minor, material, significant
  percentage bands for variance classification)
- Quality scoring weights (completeness, consistency, accuracy,
  transparency with sum-to-1.0 constraint)
- Trend analysis parameters (min/max periods, stability threshold)
- Regulatory compliance framework toggles (GHG Protocol, CSRD ESRS,
  CDP, SBTi, GRI, ISO 14064, RE100)
- Redis/cache settings (URL, TTL, enable flag)
- API settings (prefix, rate limit, page size)

This module implements a thread-safe singleton pattern using ``__new__``
with a class-level ``_instance``, ``_initialized`` flag, and
``threading.RLock``, ensuring exactly one configuration object exists
across the application lifecycle. All numeric settings are stored as
``Decimal`` for zero-hallucination deterministic calculations.

All settings can be overridden via environment variables with the
``GL_DRR_`` prefix.

Environment Variable Reference (GL_DRR_ prefix):
    GL_DRR_SERVICE_NAME                 - Service name for tracing
    GL_DRR_VERSION                      - Service version string
    GL_DRR_LOG_LEVEL                    - Logging level
    GL_DRR_ENVIRONMENT                  - Deployment environment
    GL_DRR_DEFAULT_TENANT               - Default tenant identifier
    GL_DRR_MAX_BATCH_SIZE               - Maximum records per batch
    GL_DRR_DB_HOST                      - PostgreSQL host
    GL_DRR_DB_PORT                      - PostgreSQL port
    GL_DRR_DB_NAME                      - PostgreSQL database name
    GL_DRR_DB_USER                      - PostgreSQL username
    GL_DRR_DB_PASSWORD                  - PostgreSQL password
    GL_DRR_DB_POOL_MIN                  - Minimum connection pool size
    GL_DRR_DB_POOL_MAX                  - Maximum connection pool size
    GL_DRR_DB_SSL_MODE                  - PostgreSQL SSL mode
    GL_DRR_TABLE_PREFIX                 - Database table name prefix
    GL_DRR_DEFAULT_GWP_SOURCE           - Default GWP source (ar4/ar5/ar6)
    GL_DRR_DECIMAL_PLACES               - Decimal places for calculations
    GL_DRR_INCLUDE_TRENDS               - Include trend analysis
    GL_DRR_INCLUDE_QUALITY              - Include quality scoring
    GL_DRR_INCLUDE_COMPLIANCE           - Include compliance checking
    GL_DRR_MAX_TRACE_STEPS              - Max provenance trace steps
    GL_DRR_IMMATERIAL_THRESHOLD         - Immaterial variance % threshold
    GL_DRR_MINOR_THRESHOLD              - Minor variance % threshold
    GL_DRR_MATERIAL_THRESHOLD           - Material variance % threshold
    GL_DRR_SIGNIFICANT_THRESHOLD        - Significant variance % threshold
    GL_DRR_COMPLETENESS_WEIGHT          - Completeness quality weight
    GL_DRR_CONSISTENCY_WEIGHT           - Consistency quality weight
    GL_DRR_ACCURACY_WEIGHT              - Accuracy quality weight
    GL_DRR_TRANSPARENCY_WEIGHT          - Transparency quality weight
    GL_DRR_ASSURANCE_THRESHOLD          - Assurance-ready score threshold
    GL_DRR_TREND_MIN_PERIODS            - Minimum periods for trend analysis
    GL_DRR_TREND_MAX_PERIODS            - Maximum periods for trend analysis
    GL_DRR_STABLE_THRESHOLD             - Stable trend variance threshold %
    GL_DRR_TREND_CONFIDENCE_LEVEL       - Trend confidence level (0-1)
    GL_DRR_TREND_SEASONALITY_ENABLED    - Enable seasonality detection
    GL_DRR_ENABLED_FRAMEWORKS           - Comma-separated frameworks
    GL_DRR_STRICT_MODE                  - Enable strict compliance mode
    GL_DRR_FAIL_ON_NON_COMPLIANT        - Fail on non-compliant results
    GL_DRR_REDIS_URL                    - Redis connection URL
    GL_DRR_CACHE_TTL                    - Cache TTL in seconds
    GL_DRR_ENABLE_CACHING               - Enable Redis caching
    GL_DRR_API_PREFIX                   - REST API route prefix
    GL_DRR_RATE_LIMIT                   - API requests per minute
    GL_DRR_PAGE_SIZE                    - Default page size for list APIs
    GL_DRR_MAX_PAGE_SIZE                - Maximum page size for list APIs
    GL_DRR_ENABLE_METRICS               - Enable Prometheus metrics export
    GL_DRR_METRICS_PREFIX               - Prometheus metrics prefix
    GL_DRR_ENABLE_TRACING               - Enable OpenTelemetry tracing
    GL_DRR_ENABLE_PROVENANCE            - Enable SHA-256 provenance tracking
    GL_DRR_GENESIS_HASH                 - Provenance chain genesis anchor
    GL_DRR_ENABLE_AUTH                  - Enable authentication middleware
    GL_DRR_WORKER_THREADS               - Worker thread pool size
    GL_DRR_ENABLE_BACKGROUND_TASKS      - Enable background task processing
    GL_DRR_HEALTH_CHECK_INTERVAL        - Health check interval (seconds)
    GL_DRR_CORS_ORIGINS                 - Comma-separated CORS origins
    GL_DRR_ENABLE_DOCS                  - Enable API documentation
    GL_DRR_ENABLED                      - Master enable/disable switch
    GL_DRR_LOCATION_BASED_LABEL         - Label for location-based method
    GL_DRR_MARKET_BASED_LABEL           - Label for market-based method
    GL_DRR_VARIANCE_PRECISION           - Decimal places for variance %
    GL_DRR_ENABLE_CROSS_SCOPE_RECON     - Enable cross-scope reconciliation
    GL_DRR_ENABLE_TEMPORAL_ALIGNMENT    - Enable temporal alignment checks
    GL_DRR_ENABLE_BOUNDARY_VALIDATION   - Enable org boundary validation
    GL_DRR_DEFAULT_REPORTING_YEAR       - Default reporting year
    GL_DRR_MAX_RECONCILIATION_RECORDS   - Max reconciliation records
    GL_DRR_ROUNDING_MODE                - Decimal rounding mode

Example:
    >>> from greenlang.agents.mrv.dual_reporting_reconciliation.config import (
    ...     DualReportingReconciliationConfig,
    ... )
    >>> cfg = DualReportingReconciliationConfig()
    >>> print(cfg.service_name, cfg.default_gwp_source)
    dual-reporting-reconciliation-service ar5

    >>> # Check singleton
    >>> cfg2 = DualReportingReconciliationConfig()
    >>> assert cfg is cfg2

    >>> # Reset for testing
    >>> DualReportingReconciliationConfig.reset()
    >>> cfg3 = DualReportingReconciliationConfig()
    >>> assert cfg is not cfg3

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-013 Dual Reporting Reconciliation (GL-MRV-X-024)
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
import os
import threading
from decimal import (
    Decimal,
    InvalidOperation,
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_FLOOR,
    ROUND_HALF_DOWN,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    ROUND_UP,
)
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX: str = "GL_DRR_"

# ---------------------------------------------------------------------------
# Valid enumeration values for configuration validation
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

_VALID_ENVIRONMENTS = frozenset({
    "development",
    "staging",
    "production",
})

_VALID_GWP_SOURCES = frozenset({"ar4", "ar5", "ar6"})

_VALID_SSL_MODES = frozenset({
    "disable",
    "allow",
    "prefer",
    "require",
    "verify-ca",
    "verify-full",
})

_VALID_ROUNDING_MODES = frozenset({
    "ROUND_HALF_UP",
    "ROUND_HALF_DOWN",
    "ROUND_HALF_EVEN",
    "ROUND_UP",
    "ROUND_DOWN",
    "ROUND_CEILING",
    "ROUND_FLOOR",
})

_ROUNDING_MODE_MAP: Dict[str, str] = {
    "ROUND_HALF_UP": ROUND_HALF_UP,
    "ROUND_HALF_DOWN": ROUND_HALF_DOWN,
    "ROUND_HALF_EVEN": ROUND_HALF_EVEN,
    "ROUND_UP": ROUND_UP,
    "ROUND_DOWN": ROUND_DOWN,
    "ROUND_CEILING": ROUND_CEILING,
    "ROUND_FLOOR": ROUND_FLOOR,
}

_VALID_FRAMEWORKS = frozenset({
    "ghg_protocol",
    "csrd_esrs",
    "cdp",
    "sbti",
    "gri",
    "iso_14064",
    "re100",
})

_VALID_MATERIALITY_CATEGORIES = frozenset({
    "immaterial",
    "minor",
    "material",
    "significant",
    "critical",
})

_VALID_VARIANCE_DIRECTIONS = frozenset({
    "location_higher",
    "market_higher",
    "equal",
})

_VALID_RECONCILIATION_STATUSES = frozenset({
    "pending",
    "in_progress",
    "completed",
    "failed",
    "cancelled",
})

_VALID_QUALITY_DIMENSIONS = frozenset({
    "completeness",
    "consistency",
    "accuracy",
    "transparency",
})

_VALID_TREND_DIRECTIONS = frozenset({
    "increasing",
    "decreasing",
    "stable",
    "volatile",
})

_VALID_SCOPE_TYPES = frozenset({
    "scope_1",
    "scope_2",
    "scope_3",
    "total",
})

_VALID_REPORTING_METHODS = frozenset({
    "location_based",
    "market_based",
})

# ---------------------------------------------------------------------------
# Default compliance frameworks
# ---------------------------------------------------------------------------

_DEFAULT_ENABLED_FRAMEWORKS: List[str] = [
    "ghg_protocol",
    "csrd_esrs",
    "cdp",
    "sbti",
    "gri",
    "iso_14064",
    "re100",
]

# ---------------------------------------------------------------------------
# Default quality dimension weights (must sum to 1.00)
# ---------------------------------------------------------------------------

_DEFAULT_COMPLETENESS_WEIGHT = Decimal("0.30")
_DEFAULT_CONSISTENCY_WEIGHT = Decimal("0.25")
_DEFAULT_ACCURACY_WEIGHT = Decimal("0.25")
_DEFAULT_TRANSPARENCY_WEIGHT = Decimal("0.20")

# ---------------------------------------------------------------------------
# Default materiality thresholds (percentage bands)
# ---------------------------------------------------------------------------

_DEFAULT_IMMATERIAL_THRESHOLD = Decimal("5")
_DEFAULT_MINOR_THRESHOLD = Decimal("15")
_DEFAULT_MATERIAL_THRESHOLD = Decimal("50")
_DEFAULT_SIGNIFICANT_THRESHOLD = Decimal("100")


# ---------------------------------------------------------------------------
# DualReportingReconciliationConfig
# ---------------------------------------------------------------------------


class DualReportingReconciliationConfig:
    """Singleton configuration for the Dual Reporting Reconciliation Agent.

    Implements a thread-safe singleton pattern via ``__new__`` with a
    class-level ``_instance``, ``_initialized`` flag, and
    ``threading.RLock``. On first instantiation, all settings are loaded
    from environment variables with the ``GL_DRR_`` prefix. Subsequent
    instantiations return the same object.

    All numeric values are stored as ``Decimal`` to ensure
    zero-hallucination deterministic arithmetic throughout the dual
    reporting reconciliation pipeline. This eliminates IEEE 754
    floating-point representation errors that could compound across
    location-based vs market-based variance calculations, quality
    scoring, and materiality threshold comparisons.

    The configuration covers nine domains:
    1. General Settings - service name, version, logging, environment,
       batch size, default tenant
    2. Database Settings - PostgreSQL connection, pool sizing, SSL mode,
       table prefix
    3. Reconciliation Defaults - GWP source, decimal places, feature
       toggles for trends/quality/compliance, trace steps, labels,
       variance precision, cross-scope/temporal/boundary flags
    4. Materiality Thresholds - immaterial, minor, material, significant
       percentage bands with ordered constraint
    5. Quality Scoring - completeness, consistency, accuracy,
       transparency weights with sum-to-1.0 constraint, assurance
       threshold
    6. Trend Analysis - min/max periods, stable threshold, confidence
       level, seasonality flag
    7. Compliance Settings - enabled regulatory frameworks, strict mode,
       fail-on-non-compliant flag
    8. Redis/Cache Settings - Redis URL, cache TTL, enable flag
    9. API Settings - route prefix, rate limit, page sizes

    The singleton can be reset for testing via :meth:`reset`. Configuration
    can be validated explicitly via :meth:`validate`, which returns a list
    of error strings (empty list means valid). Serialisation is supported
    via :meth:`to_dict` and :meth:`from_dict`.

    Attributes:
        service_name: Service name for tracing and identification.
        version: Service version string.
        log_level: Logging verbosity level.
        environment: Deployment environment (development/staging/production).
        max_batch_size: Maximum records per batch operation.
        default_tenant: Default tenant identifier for multi-tenancy.
        db_host: PostgreSQL server hostname.
        db_port: PostgreSQL server port.
        db_name: PostgreSQL database name.
        db_user: PostgreSQL username.
        db_password: PostgreSQL password (never logged or serialised).
        db_pool_min: Minimum number of connections in the pool.
        db_pool_max: Maximum number of connections in the pool.
        db_ssl_mode: PostgreSQL SSL connection mode.
        table_prefix: Database table name prefix.
        default_gwp_source: Default IPCC Assessment Report for GWP values.
        decimal_places: Number of decimal places for calculations.
        include_trends: Include trend analysis in reconciliation output.
        include_quality: Include quality scoring in reconciliation output.
        include_compliance: Include compliance checking in reconciliation.
        max_trace_steps: Maximum provenance trace steps.
        location_based_label: Display label for location-based method.
        market_based_label: Display label for market-based method.
        variance_precision: Decimal places for variance percentage display.
        enable_cross_scope_recon: Enable cross-scope reconciliation.
        enable_temporal_alignment: Enable temporal alignment checking.
        enable_boundary_validation: Enable organisational boundary checks.
        default_reporting_year: Default reporting year for reconciliation.
        max_reconciliation_records: Maximum reconciliation records per run.
        rounding_mode: Decimal rounding mode for calculations.
        immaterial_threshold: Variance % threshold for immaterial category.
        minor_threshold: Variance % threshold for minor category.
        material_threshold: Variance % threshold for material category.
        significant_threshold: Variance % threshold for significant category.
        completeness_weight: Quality weight for completeness dimension.
        consistency_weight: Quality weight for consistency dimension.
        accuracy_weight: Quality weight for accuracy dimension.
        transparency_weight: Quality weight for transparency dimension.
        assurance_threshold: Minimum score for assurance-ready status.
        trend_min_periods: Minimum reporting periods for trend analysis.
        trend_max_periods: Maximum reporting periods for trend analysis.
        stable_threshold: Variance % threshold for stable trend category.
        trend_confidence_level: Confidence level for trend projections.
        trend_seasonality_enabled: Enable seasonality detection in trends.
        enabled_frameworks: List of enabled compliance frameworks.
        strict_mode: Enable strict compliance checking.
        fail_on_non_compliant: Fail processing on non-compliant results.
        redis_url: Redis connection URL.
        cache_ttl: Cache time-to-live in seconds.
        enable_caching: Enable Redis caching.
        api_prefix: REST API URL prefix.
        rate_limit: Maximum API requests per minute.
        page_size: Default page size for list endpoints.
        max_page_size: Maximum page size for list endpoints.
        enable_metrics: Enable Prometheus metrics export.
        metrics_prefix: Prometheus metric name prefix.
        enable_tracing: Enable OpenTelemetry distributed tracing.
        enable_provenance: Enable SHA-256 provenance hash tracking.
        genesis_hash: Genesis anchor for the provenance chain.
        enable_auth: Enable authentication middleware.
        worker_threads: Thread pool size for parallel operations.
        enable_background_tasks: Enable background task processing.
        health_check_interval: Health check interval in seconds.
        cors_origins: Allowed CORS origins.
        enable_docs: Enable interactive API documentation.
        enabled: Master enable/disable switch for the agent.

    Example:
        >>> cfg = DualReportingReconciliationConfig()
        >>> cfg.default_gwp_source
        'ar5'
        >>> cfg.get_db_dsn()
        'postgresql://greenlang@localhost:5432/greenlang?sslmode=prefer'
        >>> cfg.is_framework_enabled("cdp")
        True
        >>> cfg.get_quality_weights()
        {'completeness': Decimal('0.30'), ...}
        >>> cfg.get_materiality_thresholds()
        {'immaterial': Decimal('5'), ...}
    """

    _instance: Optional[DualReportingReconciliationConfig] = None
    _initialized: bool = False
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton constructor
    # ------------------------------------------------------------------

    def __new__(cls) -> DualReportingReconciliationConfig:
        """Return the singleton instance, creating it on first call.

        Uses a threading RLock to ensure thread-safe initialisation. Only
        one instance is ever created; subsequent calls return the same
        object without acquiring the lock (double-checked locking).

        Returns:
            The singleton DualReportingReconciliationConfig instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialise configuration from environment variables.

        Guarded by the ``_initialized`` flag so that repeated calls to
        ``__init__`` (from repeated ``DualReportingReconciliationConfig()``
        calls) do not re-read environment variables or overwrite customised
        values.
        """
        if self.__class__._initialized:
            return
        with self._lock:
            if self.__class__._initialized:
                return
            self._load_from_env()
            self._validate_config()
            self.__class__._initialized = True
            logger.info(
                "DualReportingReconciliationConfig initialised from "
                "environment: service=%s, version=%s, "
                "gwp=%s, decimal_places=%d, "
                "frameworks=%s, env=%s, "
                "materiality=[%s/%s/%s/%s], "
                "quality_weights=[%s/%s/%s/%s]",
                self.service_name,
                self.version,
                self.default_gwp_source,
                self.decimal_places,
                self.enabled_frameworks,
                self.environment,
                self.immaterial_threshold,
                self.minor_threshold,
                self.material_threshold,
                self.significant_threshold,
                self.completeness_weight,
                self.consistency_weight,
                self.accuracy_weight,
                self.transparency_weight,
            )

    # ------------------------------------------------------------------
    # Environment loading
    # ------------------------------------------------------------------

    def _load_from_env(self) -> None:
        """Load all configuration from environment variables.

        Reads environment variables with the ``GL_DRR_`` prefix and
        populates all instance attributes. Each setting has a sensible
        default so the agent can start with zero environment configuration.

        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed with fallback to defaults on
        malformed input, emitting a WARNING log.
        Decimal values are parsed from string representations for
        exact precision; malformed input falls back to defaults.
        List values are parsed from comma-separated strings.
        """
        # -- 1. General Settings -----------------------------------------
        self.service_name: str = self._env_str(
            "SERVICE_NAME", "dual-reporting-reconciliation-service"
        )
        self.version: str = self._env_str(
            "VERSION", "1.0.0"
        )
        self.log_level: str = self._env_str("LOG_LEVEL", "INFO")
        self.environment: str = self._env_str(
            "ENVIRONMENT", "development"
        )
        self.max_batch_size: int = self._env_int(
            "MAX_BATCH_SIZE", 10000
        )
        self.default_tenant: str = self._env_str(
            "DEFAULT_TENANT", "default"
        )

        # -- 2. Database Settings ----------------------------------------
        self.db_host: str = self._env_str("DB_HOST", "localhost")
        self.db_port: int = self._env_int("DB_PORT", 5432)
        self.db_name: str = self._env_str("DB_NAME", "greenlang")
        self.db_user: str = self._env_str("DB_USER", "greenlang")
        self.db_password: str = self._env_str("DB_PASSWORD", "")
        self.db_pool_min: int = self._env_int("DB_POOL_MIN", 2)
        self.db_pool_max: int = self._env_int("DB_POOL_MAX", 10)
        self.db_ssl_mode: str = self._env_str("DB_SSL_MODE", "prefer")
        self.table_prefix: str = self._env_str(
            "TABLE_PREFIX", "drr_"
        )

        # -- 3. Reconciliation Defaults ----------------------------------
        self.default_gwp_source: str = self._env_str(
            "DEFAULT_GWP_SOURCE", "ar5"
        )
        self.decimal_places: int = self._env_int(
            "DECIMAL_PLACES", 8
        )
        self.include_trends: bool = self._env_bool(
            "INCLUDE_TRENDS", True
        )
        self.include_quality: bool = self._env_bool(
            "INCLUDE_QUALITY", True
        )
        self.include_compliance: bool = self._env_bool(
            "INCLUDE_COMPLIANCE", True
        )
        self.max_trace_steps: int = self._env_int(
            "MAX_TRACE_STEPS", 200
        )
        self.location_based_label: str = self._env_str(
            "LOCATION_BASED_LABEL", "Location-Based"
        )
        self.market_based_label: str = self._env_str(
            "MARKET_BASED_LABEL", "Market-Based"
        )
        self.variance_precision: int = self._env_int(
            "VARIANCE_PRECISION", 4
        )
        self.enable_cross_scope_recon: bool = self._env_bool(
            "ENABLE_CROSS_SCOPE_RECON", True
        )
        self.enable_temporal_alignment: bool = self._env_bool(
            "ENABLE_TEMPORAL_ALIGNMENT", True
        )
        self.enable_boundary_validation: bool = self._env_bool(
            "ENABLE_BOUNDARY_VALIDATION", True
        )
        self.default_reporting_year: int = self._env_int(
            "DEFAULT_REPORTING_YEAR", 2025
        )
        self.max_reconciliation_records: int = self._env_int(
            "MAX_RECONCILIATION_RECORDS", 50000
        )
        self.rounding_mode: str = self._env_str(
            "ROUNDING_MODE", "ROUND_HALF_UP"
        )

        # -- 4. Materiality Thresholds -----------------------------------
        self.immaterial_threshold: Decimal = self._env_decimal(
            "IMMATERIAL_THRESHOLD", _DEFAULT_IMMATERIAL_THRESHOLD
        )
        self.minor_threshold: Decimal = self._env_decimal(
            "MINOR_THRESHOLD", _DEFAULT_MINOR_THRESHOLD
        )
        self.material_threshold: Decimal = self._env_decimal(
            "MATERIAL_THRESHOLD", _DEFAULT_MATERIAL_THRESHOLD
        )
        self.significant_threshold: Decimal = self._env_decimal(
            "SIGNIFICANT_THRESHOLD", _DEFAULT_SIGNIFICANT_THRESHOLD
        )

        # -- 5. Quality Scoring ------------------------------------------
        self.completeness_weight: Decimal = self._env_decimal(
            "COMPLETENESS_WEIGHT", _DEFAULT_COMPLETENESS_WEIGHT
        )
        self.consistency_weight: Decimal = self._env_decimal(
            "CONSISTENCY_WEIGHT", _DEFAULT_CONSISTENCY_WEIGHT
        )
        self.accuracy_weight: Decimal = self._env_decimal(
            "ACCURACY_WEIGHT", _DEFAULT_ACCURACY_WEIGHT
        )
        self.transparency_weight: Decimal = self._env_decimal(
            "TRANSPARENCY_WEIGHT", _DEFAULT_TRANSPARENCY_WEIGHT
        )
        self.assurance_threshold: Decimal = self._env_decimal(
            "ASSURANCE_THRESHOLD", Decimal("0.90")
        )

        # -- 6. Trend Analysis -------------------------------------------
        self.trend_min_periods: int = self._env_int(
            "TREND_MIN_PERIODS", 2
        )
        self.trend_max_periods: int = self._env_int(
            "TREND_MAX_PERIODS", 10
        )
        self.stable_threshold: Decimal = self._env_decimal(
            "STABLE_THRESHOLD", Decimal("2.0")
        )
        self.trend_confidence_level: Decimal = self._env_decimal(
            "TREND_CONFIDENCE_LEVEL", Decimal("0.95")
        )
        self.trend_seasonality_enabled: bool = self._env_bool(
            "TREND_SEASONALITY_ENABLED", False
        )

        # -- 7. Compliance Settings --------------------------------------
        self.enabled_frameworks: List[str] = self._env_list(
            "ENABLED_FRAMEWORKS",
            _DEFAULT_ENABLED_FRAMEWORKS,
        )
        self.strict_mode: bool = self._env_bool(
            "STRICT_MODE", False
        )
        self.fail_on_non_compliant: bool = self._env_bool(
            "FAIL_ON_NON_COMPLIANT", False
        )

        # -- 8. Redis/Cache Settings -------------------------------------
        self.redis_url: str = self._env_str(
            "REDIS_URL", "redis://localhost:6379/0"
        )
        self.cache_ttl: int = self._env_int("CACHE_TTL", 3600)
        self.enable_caching: bool = self._env_bool(
            "ENABLE_CACHING", True
        )

        # -- 9. API Settings ---------------------------------------------
        self.api_prefix: str = self._env_str(
            "API_PREFIX", "/api/v1/dual-reporting"
        )
        self.rate_limit: int = self._env_int("RATE_LIMIT", 1000)
        self.page_size: int = self._env_int("PAGE_SIZE", 100)
        self.max_page_size: int = self._env_int("MAX_PAGE_SIZE", 10000)

        # -- Logging & Observability -------------------------------------
        self.enable_metrics: bool = self._env_bool(
            "ENABLE_METRICS", True
        )
        self.metrics_prefix: str = self._env_str(
            "METRICS_PREFIX", "gl_drr_"
        )
        self.enable_tracing: bool = self._env_bool(
            "ENABLE_TRACING", True
        )

        # -- Provenance Tracking -----------------------------------------
        self.enable_provenance: bool = self._env_bool(
            "ENABLE_PROVENANCE", True
        )
        self.genesis_hash: str = self._env_str(
            "GENESIS_HASH",
            "GL-MRV-X-024-DUAL-REPORTING-RECONCILIATION-GENESIS",
        )

        # -- Auth & Background Tasks -------------------------------------
        self.enable_auth: bool = self._env_bool("ENABLE_AUTH", True)
        self.worker_threads: int = self._env_int("WORKER_THREADS", 4)
        self.enable_background_tasks: bool = self._env_bool(
            "ENABLE_BACKGROUND_TASKS", True
        )
        self.health_check_interval: int = self._env_int(
            "HEALTH_CHECK_INTERVAL", 30
        )

        # -- CORS & Docs -------------------------------------------------
        self.cors_origins: List[str] = self._env_list(
            "CORS_ORIGINS", ["*"]
        )
        self.enable_docs: bool = self._env_bool("ENABLE_DOCS", True)

        # -- Master switch -----------------------------------------------
        self.enabled: bool = self._env_bool("ENABLED", True)

    # ------------------------------------------------------------------
    # Environment variable parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _env_str(name: str, default: str) -> str:
        """Read a string environment variable with the GL_DRR_ prefix.

        Args:
            name: Variable name suffix (after GL_DRR_).
            default: Default value if not set.

        Returns:
            The environment variable value or the default.
        """
        val = os.environ.get(f"{_ENV_PREFIX}{name}")
        if val is None:
            return default
        return val.strip()

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        """Read an integer environment variable with the GL_DRR_ prefix.

        Args:
            name: Variable name suffix (after GL_DRR_).
            default: Default value if not set or parse fails.

        Returns:
            Parsed integer value or the default.
        """
        val = os.environ.get(f"{_ENV_PREFIX}{name}")
        if val is None:
            return default
        try:
            return int(val.strip())
        except ValueError:
            logger.warning(
                "Invalid integer for %s%s=%r, using default %d",
                _ENV_PREFIX,
                name,
                val,
                default,
            )
            return default

    @staticmethod
    def _env_decimal(name: str, default: Decimal) -> Decimal:
        """Read a Decimal environment variable with the GL_DRR_ prefix.

        Uses ``Decimal(str)`` parsing for exact precision. Falls back
        to the default on ``InvalidOperation`` (malformed input) and
        emits a WARNING log.

        Args:
            name: Variable name suffix (after GL_DRR_).
            default: Default Decimal value if not set or parse fails.

        Returns:
            Parsed Decimal value or the default.
        """
        val = os.environ.get(f"{_ENV_PREFIX}{name}")
        if val is None:
            return default
        try:
            return Decimal(val.strip())
        except InvalidOperation:
            logger.warning(
                "Invalid Decimal for %s%s=%r, using default %s",
                _ENV_PREFIX,
                name,
                val,
                default,
            )
            return default

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        """Read a boolean environment variable with the GL_DRR_ prefix.

        Accepts ``true``, ``1``, ``yes`` (case-insensitive) as True.
        All other non-None values are treated as False.

        Args:
            name: Variable name suffix (after GL_DRR_).
            default: Default value if not set.

        Returns:
            Parsed boolean value or the default.
        """
        val = os.environ.get(f"{_ENV_PREFIX}{name}")
        if val is None:
            return default
        return val.strip().lower() in ("true", "1", "yes")

    @staticmethod
    def _env_list(name: str, default: List[str]) -> List[str]:
        """Read a comma-separated list environment variable.

        Args:
            name: Variable name suffix (after GL_DRR_).
            default: Default list if not set.

        Returns:
            Parsed list of stripped strings, or the default.
        """
        val = os.environ.get(f"{_ENV_PREFIX}{name}")
        if val is None:
            return list(default)
        items = [item.strip() for item in val.split(",") if item.strip()]
        if not items:
            return list(default)
        return items

    # ------------------------------------------------------------------
    # Singleton lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance for test teardown.

        After calling ``reset()``, the next instantiation of
        ``DualReportingReconciliationConfig()`` will re-read all
        environment variables and construct a fresh configuration
        object. Thread-safe.

        Example:
            >>> DualReportingReconciliationConfig.reset()
            >>> cfg = DualReportingReconciliationConfig()  # fresh instance
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
        logger.debug(
            "DualReportingReconciliationConfig singleton reset"
        )

    # ------------------------------------------------------------------
    # Internal post-load validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Run validation on load and log warnings for issues.

        Called automatically during ``__init__`` after loading from
        environment. Logs warnings but does not raise exceptions to
        allow graceful degradation. Use :meth:`validate` for strict
        validation that returns error lists.
        """
        errors = self.validate()
        if errors:
            logger.warning(
                "DualReportingReconciliationConfig loaded with %d "
                "validation warning(s):\n%s",
                len(errors),
                "\n".join(f"  - {e}" for e in errors),
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate all configuration settings.

        Performs comprehensive checks (120+) across all configuration
        domains: general settings, database connectivity parameters,
        reconciliation defaults, materiality thresholds, quality scoring
        weights, trend analysis parameters, compliance frameworks,
        cache settings, API settings, logging, provenance, and
        performance tuning.

        Returns:
            A list of human-readable error strings. An empty list means
            all validation checks passed.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> errors = cfg.validate()
            >>> assert len(errors) == 0
        """
        errors: List[str] = []

        # -- General Settings --------------------------------------------
        errors.extend(self._validate_general_settings())

        # -- Database Settings -------------------------------------------
        errors.extend(self._validate_database_settings())

        # -- Reconciliation Defaults -------------------------------------
        errors.extend(self._validate_reconciliation_defaults())

        # -- Materiality Thresholds --------------------------------------
        errors.extend(self._validate_materiality_thresholds())

        # -- Quality Scoring ---------------------------------------------
        errors.extend(self._validate_quality_scoring())

        # -- Trend Analysis ----------------------------------------------
        errors.extend(self._validate_trend_analysis())

        # -- Compliance Settings -----------------------------------------
        errors.extend(self._validate_compliance_settings())

        # -- Cache Settings ----------------------------------------------
        errors.extend(self._validate_cache_settings())

        # -- API Settings ------------------------------------------------
        errors.extend(self._validate_api_settings())

        # -- Logging & Observability -------------------------------------
        errors.extend(self._validate_logging_settings())

        # -- Provenance Tracking -----------------------------------------
        errors.extend(self._validate_provenance_settings())

        # -- Performance Tuning ------------------------------------------
        errors.extend(self._validate_performance_settings())

        if errors:
            logger.warning(
                "DualReportingReconciliationConfig validation found "
                "%d error(s):\n%s",
                len(errors),
                "\n".join(f"  - {e}" for e in errors),
            )
        else:
            logger.debug(
                "DualReportingReconciliationConfig validation passed: "
                "all %d checks OK",
                self._count_validation_checks(),
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: General Settings
    # ------------------------------------------------------------------

    def _validate_general_settings(self) -> List[str]:
        """Validate general service settings.

        Checks service name and version are non-empty, log level is
        valid, environment is in the allowed set, batch size is within
        bounds, and tenant ID is non-empty.

        Returns:
            List of error strings for invalid general settings.
        """
        errors: List[str] = []

        # -- service_name non-empty --------------------------------------
        if not self.service_name:
            errors.append("service_name must not be empty")

        # -- service_name length bound -----------------------------------
        if self.service_name and len(self.service_name) > 128:
            errors.append(
                f"service_name must be <= 128 characters, "
                f"got {len(self.service_name)}"
            )

        # -- version non-empty -------------------------------------------
        if not self.version:
            errors.append("version must not be empty")

        # -- version length bound ----------------------------------------
        if self.version and len(self.version) > 64:
            errors.append(
                f"version must be <= 64 characters, "
                f"got {len(self.version)}"
            )

        # -- log_level valid ---------------------------------------------
        normalised_log = self.log_level.upper()
        if normalised_log not in _VALID_LOG_LEVELS:
            errors.append(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got '{self.log_level}'"
            )

        # -- environment valid -------------------------------------------
        normalised_env = self.environment.lower()
        if normalised_env not in _VALID_ENVIRONMENTS:
            errors.append(
                f"environment must be one of "
                f"{sorted(_VALID_ENVIRONMENTS)}, "
                f"got '{self.environment}'"
            )

        # -- max_batch_size > 0 ------------------------------------------
        if self.max_batch_size <= 0:
            errors.append(
                f"max_batch_size must be > 0, "
                f"got {self.max_batch_size}"
            )

        # -- max_batch_size upper bound ----------------------------------
        if self.max_batch_size > 100_000:
            errors.append(
                f"max_batch_size must be <= 100000, "
                f"got {self.max_batch_size}"
            )

        # -- default_tenant non-empty ------------------------------------
        if not self.default_tenant:
            errors.append("default_tenant must not be empty")

        # -- default_tenant length bound ---------------------------------
        if self.default_tenant and len(self.default_tenant) > 128:
            errors.append(
                f"default_tenant must be <= 128 characters, "
                f"got {len(self.default_tenant)}"
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: Database Settings
    # ------------------------------------------------------------------

    def _validate_database_settings(self) -> List[str]:
        """Validate database connection settings.

        Checks host non-empty, port in valid TCP range (1-65535), name
        and user non-empty, pool sizes within reasonable bounds, and
        that pool_min does not exceed pool_max. SSL mode is validated
        against the PostgreSQL-accepted set. Table prefix is checked
        for alphanumeric + underscore format.

        Returns:
            List of error strings for invalid database settings.
        """
        errors: List[str] = []

        # -- db_host non-empty -------------------------------------------
        if not self.db_host:
            errors.append("db_host must not be empty")

        # -- db_port lower bound -----------------------------------------
        if self.db_port <= 0:
            errors.append(
                f"db_port must be > 0, got {self.db_port}"
            )

        # -- db_port upper bound -----------------------------------------
        if self.db_port > 65535:
            errors.append(
                f"db_port must be <= 65535, got {self.db_port}"
            )

        # -- db_name non-empty -------------------------------------------
        if not self.db_name:
            errors.append("db_name must not be empty")

        # -- db_user non-empty -------------------------------------------
        if not self.db_user:
            errors.append("db_user must not be empty")

        # -- db_pool_min lower bound -------------------------------------
        if self.db_pool_min < 0:
            errors.append(
                f"db_pool_min must be >= 0, got {self.db_pool_min}"
            )

        # -- db_pool_min upper bound -------------------------------------
        if self.db_pool_min > 100:
            errors.append(
                f"db_pool_min must be <= 100, got {self.db_pool_min}"
            )

        # -- db_pool_max lower bound -------------------------------------
        if self.db_pool_max <= 0:
            errors.append(
                f"db_pool_max must be > 0, got {self.db_pool_max}"
            )

        # -- db_pool_max upper bound -------------------------------------
        if self.db_pool_max > 500:
            errors.append(
                f"db_pool_max must be <= 500, got {self.db_pool_max}"
            )

        # -- db_pool_min <= db_pool_max ----------------------------------
        if self.db_pool_min > self.db_pool_max:
            errors.append(
                f"db_pool_min ({self.db_pool_min}) must be <= "
                f"db_pool_max ({self.db_pool_max})"
            )

        # -- db_ssl_mode valid -------------------------------------------
        normalised_ssl = self.db_ssl_mode.lower()
        if normalised_ssl not in _VALID_SSL_MODES:
            errors.append(
                f"db_ssl_mode must be one of {sorted(_VALID_SSL_MODES)}, "
                f"got '{self.db_ssl_mode}'"
            )

        # -- table_prefix non-empty --------------------------------------
        if not self.table_prefix:
            errors.append("table_prefix must not be empty")

        # -- table_prefix format (alphanumeric + underscore) -------------
        if self.table_prefix and not self.table_prefix.replace(
            "_", ""
        ).isalnum():
            errors.append(
                f"table_prefix must contain only alphanumeric "
                f"characters and underscores, "
                f"got '{self.table_prefix}'"
            )

        # -- table_prefix length bound -----------------------------------
        if self.table_prefix and len(self.table_prefix) > 32:
            errors.append(
                f"table_prefix must be <= 32 characters, "
                f"got {len(self.table_prefix)}"
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: Reconciliation Defaults
    # ------------------------------------------------------------------

    def _validate_reconciliation_defaults(self) -> List[str]:
        """Validate reconciliation calculation default settings.

        Checks GWP source, decimal places, max trace steps, variance
        precision, reporting year, max reconciliation records, rounding
        mode, and label lengths.

        Returns:
            List of error strings for invalid reconciliation defaults.
        """
        errors: List[str] = []

        # -- GWP source valid --------------------------------------------
        normalised_gwp = self.default_gwp_source.lower()
        if normalised_gwp not in _VALID_GWP_SOURCES:
            errors.append(
                f"default_gwp_source must be one of "
                f"{sorted(_VALID_GWP_SOURCES)}, "
                f"got '{self.default_gwp_source}'"
            )

        # -- decimal_places lower bound ----------------------------------
        if self.decimal_places < 0:
            errors.append(
                f"decimal_places must be >= 0, "
                f"got {self.decimal_places}"
            )

        # -- decimal_places upper bound ----------------------------------
        if self.decimal_places > 28:
            errors.append(
                f"decimal_places must be <= 28, "
                f"got {self.decimal_places}"
            )

        # -- max_trace_steps lower bound ---------------------------------
        if self.max_trace_steps <= 0:
            errors.append(
                f"max_trace_steps must be > 0, "
                f"got {self.max_trace_steps}"
            )

        # -- max_trace_steps upper bound ---------------------------------
        if self.max_trace_steps > 10000:
            errors.append(
                f"max_trace_steps must be <= 10000, "
                f"got {self.max_trace_steps}"
            )

        # -- variance_precision lower bound ------------------------------
        if self.variance_precision < 0:
            errors.append(
                f"variance_precision must be >= 0, "
                f"got {self.variance_precision}"
            )

        # -- variance_precision upper bound ------------------------------
        if self.variance_precision > 20:
            errors.append(
                f"variance_precision must be <= 20, "
                f"got {self.variance_precision}"
            )

        # -- location_based_label non-empty ------------------------------
        if not self.location_based_label:
            errors.append("location_based_label must not be empty")

        # -- location_based_label length bound ---------------------------
        if self.location_based_label and len(
            self.location_based_label
        ) > 128:
            errors.append(
                f"location_based_label must be <= 128 characters, "
                f"got {len(self.location_based_label)}"
            )

        # -- market_based_label non-empty --------------------------------
        if not self.market_based_label:
            errors.append("market_based_label must not be empty")

        # -- market_based_label length bound -----------------------------
        if self.market_based_label and len(
            self.market_based_label
        ) > 128:
            errors.append(
                f"market_based_label must be <= 128 characters, "
                f"got {len(self.market_based_label)}"
            )

        # -- default_reporting_year lower bound --------------------------
        if self.default_reporting_year < 1990:
            errors.append(
                f"default_reporting_year must be >= 1990, "
                f"got {self.default_reporting_year}"
            )

        # -- default_reporting_year upper bound --------------------------
        if self.default_reporting_year > 2100:
            errors.append(
                f"default_reporting_year must be <= 2100, "
                f"got {self.default_reporting_year}"
            )

        # -- max_reconciliation_records lower bound ----------------------
        if self.max_reconciliation_records <= 0:
            errors.append(
                f"max_reconciliation_records must be > 0, "
                f"got {self.max_reconciliation_records}"
            )

        # -- max_reconciliation_records upper bound ----------------------
        if self.max_reconciliation_records > 1_000_000:
            errors.append(
                f"max_reconciliation_records must be <= 1000000, "
                f"got {self.max_reconciliation_records}"
            )

        # -- rounding_mode valid -----------------------------------------
        normalised_rounding = self.rounding_mode.upper()
        if normalised_rounding not in _VALID_ROUNDING_MODES:
            errors.append(
                f"rounding_mode must be one of "
                f"{sorted(_VALID_ROUNDING_MODES)}, "
                f"got '{self.rounding_mode}'"
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: Materiality Thresholds
    # ------------------------------------------------------------------

    def _validate_materiality_thresholds(self) -> List[str]:
        """Validate materiality threshold settings.

        Checks that each threshold is non-negative, and that the four
        thresholds are strictly ordered: immaterial < minor < material
        < significant. This ordering is required by the Dual Reporting
        Reconciliation Engine to correctly classify variance magnitudes
        into materiality categories.

        Returns:
            List of error strings for invalid materiality thresholds.
        """
        errors: List[str] = []

        # -- immaterial_threshold lower bound -----------------------------
        if self.immaterial_threshold < Decimal("0"):
            errors.append(
                f"immaterial_threshold must be >= 0, "
                f"got {self.immaterial_threshold}"
            )

        # -- immaterial_threshold upper bound (sanity) -------------------
        if self.immaterial_threshold > Decimal("100"):
            errors.append(
                f"immaterial_threshold must be <= 100, "
                f"got {self.immaterial_threshold}"
            )

        # -- minor_threshold lower bound ---------------------------------
        if self.minor_threshold < Decimal("0"):
            errors.append(
                f"minor_threshold must be >= 0, "
                f"got {self.minor_threshold}"
            )

        # -- minor_threshold upper bound (sanity) ------------------------
        if self.minor_threshold > Decimal("500"):
            errors.append(
                f"minor_threshold must be <= 500, "
                f"got {self.minor_threshold}"
            )

        # -- material_threshold lower bound ------------------------------
        if self.material_threshold < Decimal("0"):
            errors.append(
                f"material_threshold must be >= 0, "
                f"got {self.material_threshold}"
            )

        # -- material_threshold upper bound (sanity) ---------------------
        if self.material_threshold > Decimal("1000"):
            errors.append(
                f"material_threshold must be <= 1000, "
                f"got {self.material_threshold}"
            )

        # -- significant_threshold lower bound ---------------------------
        if self.significant_threshold < Decimal("0"):
            errors.append(
                f"significant_threshold must be >= 0, "
                f"got {self.significant_threshold}"
            )

        # -- significant_threshold upper bound (sanity) ------------------
        if self.significant_threshold > Decimal("10000"):
            errors.append(
                f"significant_threshold must be <= 10000, "
                f"got {self.significant_threshold}"
            )

        # -- ordering: immaterial < minor --------------------------------
        if self.immaterial_threshold >= self.minor_threshold:
            errors.append(
                f"immaterial_threshold ({self.immaterial_threshold}) "
                f"must be < minor_threshold ({self.minor_threshold})"
            )

        # -- ordering: minor < material ----------------------------------
        if self.minor_threshold >= self.material_threshold:
            errors.append(
                f"minor_threshold ({self.minor_threshold}) "
                f"must be < material_threshold "
                f"({self.material_threshold})"
            )

        # -- ordering: material < significant ----------------------------
        if self.material_threshold >= self.significant_threshold:
            errors.append(
                f"material_threshold ({self.material_threshold}) "
                f"must be < significant_threshold "
                f"({self.significant_threshold})"
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: Quality Scoring
    # ------------------------------------------------------------------

    def _validate_quality_scoring(self) -> List[str]:
        """Validate quality scoring weight settings.

        Checks that each weight is in the range [0, 1], that the four
        weights sum to exactly ``Decimal("1.00")``, and that the
        assurance threshold is in (0, 1].

        Returns:
            List of error strings for invalid quality scoring settings.
        """
        errors: List[str] = []

        # -- completeness_weight lower bound -----------------------------
        if self.completeness_weight < Decimal("0"):
            errors.append(
                f"completeness_weight must be >= 0, "
                f"got {self.completeness_weight}"
            )

        # -- completeness_weight upper bound -----------------------------
        if self.completeness_weight > Decimal("1"):
            errors.append(
                f"completeness_weight must be <= 1, "
                f"got {self.completeness_weight}"
            )

        # -- consistency_weight lower bound ------------------------------
        if self.consistency_weight < Decimal("0"):
            errors.append(
                f"consistency_weight must be >= 0, "
                f"got {self.consistency_weight}"
            )

        # -- consistency_weight upper bound ------------------------------
        if self.consistency_weight > Decimal("1"):
            errors.append(
                f"consistency_weight must be <= 1, "
                f"got {self.consistency_weight}"
            )

        # -- accuracy_weight lower bound ---------------------------------
        if self.accuracy_weight < Decimal("0"):
            errors.append(
                f"accuracy_weight must be >= 0, "
                f"got {self.accuracy_weight}"
            )

        # -- accuracy_weight upper bound ---------------------------------
        if self.accuracy_weight > Decimal("1"):
            errors.append(
                f"accuracy_weight must be <= 1, "
                f"got {self.accuracy_weight}"
            )

        # -- transparency_weight lower bound -----------------------------
        if self.transparency_weight < Decimal("0"):
            errors.append(
                f"transparency_weight must be >= 0, "
                f"got {self.transparency_weight}"
            )

        # -- transparency_weight upper bound -----------------------------
        if self.transparency_weight > Decimal("1"):
            errors.append(
                f"transparency_weight must be <= 1, "
                f"got {self.transparency_weight}"
            )

        # -- quality weights must sum to 1.00 ----------------------------
        weight_sum = (
            self.completeness_weight
            + self.consistency_weight
            + self.accuracy_weight
            + self.transparency_weight
        )
        if weight_sum != Decimal("1.00"):
            errors.append(
                f"Quality weights must sum to exactly 1.00, "
                f"got {weight_sum} "
                f"(completeness={self.completeness_weight}, "
                f"consistency={self.consistency_weight}, "
                f"accuracy={self.accuracy_weight}, "
                f"transparency={self.transparency_weight})"
            )

        # -- assurance_threshold lower bound -----------------------------
        if self.assurance_threshold <= Decimal("0"):
            errors.append(
                f"assurance_threshold must be > 0, "
                f"got {self.assurance_threshold}"
            )

        # -- assurance_threshold upper bound -----------------------------
        if self.assurance_threshold > Decimal("1"):
            errors.append(
                f"assurance_threshold must be <= 1, "
                f"got {self.assurance_threshold}"
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: Trend Analysis
    # ------------------------------------------------------------------

    def _validate_trend_analysis(self) -> List[str]:
        """Validate trend analysis settings.

        Checks min/max periods are valid, stable threshold is
        non-negative, confidence level is in (0, 1), and that
        min_periods does not exceed max_periods.

        Returns:
            List of error strings for invalid trend analysis settings.
        """
        errors: List[str] = []

        # -- trend_min_periods lower bound -------------------------------
        if self.trend_min_periods < 1:
            errors.append(
                f"trend_min_periods must be >= 1, "
                f"got {self.trend_min_periods}"
            )

        # -- trend_min_periods upper bound -------------------------------
        if self.trend_min_periods > 100:
            errors.append(
                f"trend_min_periods must be <= 100, "
                f"got {self.trend_min_periods}"
            )

        # -- trend_max_periods lower bound -------------------------------
        if self.trend_max_periods < 1:
            errors.append(
                f"trend_max_periods must be >= 1, "
                f"got {self.trend_max_periods}"
            )

        # -- trend_max_periods upper bound -------------------------------
        if self.trend_max_periods > 100:
            errors.append(
                f"trend_max_periods must be <= 100, "
                f"got {self.trend_max_periods}"
            )

        # -- trend_min_periods <= trend_max_periods ----------------------
        if self.trend_min_periods > self.trend_max_periods:
            errors.append(
                f"trend_min_periods ({self.trend_min_periods}) must "
                f"be <= trend_max_periods ({self.trend_max_periods})"
            )

        # -- stable_threshold lower bound --------------------------------
        if self.stable_threshold < Decimal("0"):
            errors.append(
                f"stable_threshold must be >= 0, "
                f"got {self.stable_threshold}"
            )

        # -- stable_threshold upper bound --------------------------------
        if self.stable_threshold > Decimal("100"):
            errors.append(
                f"stable_threshold must be <= 100, "
                f"got {self.stable_threshold}"
            )

        # -- trend_confidence_level lower bound --------------------------
        if self.trend_confidence_level <= Decimal("0"):
            errors.append(
                f"trend_confidence_level must be > 0, "
                f"got {self.trend_confidence_level}"
            )

        # -- trend_confidence_level upper bound --------------------------
        if self.trend_confidence_level >= Decimal("1"):
            errors.append(
                f"trend_confidence_level must be < 1, "
                f"got {self.trend_confidence_level}"
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: Compliance Settings
    # ------------------------------------------------------------------

    def _validate_compliance_settings(self) -> List[str]:
        """Validate compliance framework settings.

        Checks that enabled frameworks are all recognised identifiers,
        checks for duplicates, and validates that strict mode and
        fail-on-non-compliant have consistent prerequisites.

        Returns:
            List of error strings for invalid compliance settings.
        """
        errors: List[str] = []

        # -- frameworks list non-empty -----------------------------------
        if not self.enabled_frameworks:
            errors.append("enabled_frameworks must not be empty")
        else:
            # -- each framework must be valid ----------------------------
            for fw in self.enabled_frameworks:
                normalised = fw.lower()
                if normalised not in _VALID_FRAMEWORKS:
                    errors.append(
                        f"Framework '{fw}' is not valid; "
                        f"must be one of {sorted(_VALID_FRAMEWORKS)}"
                    )

            # -- check for duplicates ------------------------------------
            seen: set = set()
            for fw in self.enabled_frameworks:
                normalised = fw.lower()
                if normalised in seen:
                    errors.append(
                        f"Duplicate framework '{fw}' in "
                        f"enabled_frameworks"
                    )
                seen.add(normalised)

        # -- strict_mode requires at least one framework -----------------
        if self.strict_mode and not self.enabled_frameworks:
            errors.append(
                "strict_mode requires at least one framework "
                "in enabled_frameworks"
            )

        # -- fail_on_non_compliant requires strict_mode ------------------
        if self.fail_on_non_compliant and not self.strict_mode:
            errors.append(
                "fail_on_non_compliant requires "
                "strict_mode to be True"
            )

        # -- SBTi requires ghg_protocol ---------------------------------
        normalised_fws = [
            fw.lower() for fw in self.enabled_frameworks
        ]
        if "sbti" in normalised_fws:
            if "ghg_protocol" not in normalised_fws:
                errors.append(
                    "SBTi framework requires 'ghg_protocol' "
                    "in enabled_frameworks"
                )

        # -- RE100 requires ghg_protocol ---------------------------------
        if "re100" in normalised_fws:
            if "ghg_protocol" not in normalised_fws:
                errors.append(
                    "RE100 framework requires 'ghg_protocol' "
                    "in enabled_frameworks"
                )

        return errors

    # ------------------------------------------------------------------
    # Validation: Cache Settings
    # ------------------------------------------------------------------

    def _validate_cache_settings(self) -> List[str]:
        """Validate Redis/cache settings.

        Checks Redis URL non-empty when caching enabled, cache TTL
        within bounds.

        Returns:
            List of error strings for invalid cache settings.
        """
        errors: List[str] = []

        # -- redis_url non-empty when caching enabled --------------------
        if self.enable_caching and not self.redis_url:
            errors.append(
                "redis_url must not be empty when "
                "enable_caching is True"
            )

        # -- redis_url format check --------------------------------------
        if self.redis_url and not (
            self.redis_url.startswith("redis://")
            or self.redis_url.startswith("rediss://")
        ):
            url_preview = (
                f"'{self.redis_url[:30]}...'"
                if len(self.redis_url) > 30
                else f"'{self.redis_url}'"
            )
            errors.append(
                f"redis_url must start with 'redis://' or "
                f"'rediss://', got {url_preview}"
            )

        # -- cache_ttl lower bound ---------------------------------------
        if self.cache_ttl <= 0:
            errors.append(
                f"cache_ttl must be > 0, got {self.cache_ttl}"
            )

        # -- cache_ttl upper bound ---------------------------------------
        if self.cache_ttl > 86400:
            errors.append(
                f"cache_ttl must be <= 86400 (24 hours), "
                f"got {self.cache_ttl}"
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: API Settings
    # ------------------------------------------------------------------

    def _validate_api_settings(self) -> List[str]:
        """Validate API settings.

        Checks API prefix is non-empty and starts with slash, rate limit
        is within reasonable bounds, page sizes are consistent, and CORS
        origins list is non-empty.

        Returns:
            List of error strings for invalid API settings.
        """
        errors: List[str] = []

        # -- api_prefix non-empty ----------------------------------------
        if not self.api_prefix:
            errors.append("api_prefix must not be empty")

        # -- api_prefix starts with / ------------------------------------
        if self.api_prefix and not self.api_prefix.startswith("/"):
            errors.append(
                f"api_prefix must start with '/', "
                f"got '{self.api_prefix}'"
            )

        # -- rate_limit lower bound --------------------------------------
        if self.rate_limit <= 0:
            errors.append(
                f"rate_limit must be > 0, "
                f"got {self.rate_limit}"
            )

        # -- rate_limit upper bound --------------------------------------
        if self.rate_limit > 100_000:
            errors.append(
                f"rate_limit must be <= 100000, "
                f"got {self.rate_limit}"
            )

        # -- page_size lower bound ---------------------------------------
        if self.page_size <= 0:
            errors.append(
                f"page_size must be > 0, "
                f"got {self.page_size}"
            )

        # -- page_size upper bound ---------------------------------------
        if self.page_size > 10000:
            errors.append(
                f"page_size must be <= 10000, "
                f"got {self.page_size}"
            )

        # -- max_page_size lower bound -----------------------------------
        if self.max_page_size <= 0:
            errors.append(
                f"max_page_size must be > 0, "
                f"got {self.max_page_size}"
            )

        # -- max_page_size upper bound -----------------------------------
        if self.max_page_size > 100_000:
            errors.append(
                f"max_page_size must be <= 100000, "
                f"got {self.max_page_size}"
            )

        # -- page_size <= max_page_size ----------------------------------
        if self.page_size > self.max_page_size:
            errors.append(
                f"page_size ({self.page_size}) must be <= "
                f"max_page_size ({self.max_page_size})"
            )

        # -- cors_origins non-empty --------------------------------------
        if not self.cors_origins:
            errors.append("cors_origins must not be empty")

        return errors

    # ------------------------------------------------------------------
    # Validation: Logging & Observability
    # ------------------------------------------------------------------

    def _validate_logging_settings(self) -> List[str]:
        """Validate logging and observability settings.

        Checks metrics prefix naming conventions.

        Returns:
            List of error strings for invalid logging settings.
        """
        errors: List[str] = []

        # -- metrics_prefix non-empty ------------------------------------
        if not self.metrics_prefix:
            errors.append("metrics_prefix must not be empty")

        # -- metrics_prefix format ---------------------------------------
        if self.metrics_prefix and not self.metrics_prefix.replace(
            "_", ""
        ).isalnum():
            errors.append(
                f"metrics_prefix must contain only alphanumeric "
                f"characters and underscores, "
                f"got '{self.metrics_prefix}'"
            )

        # -- metrics_prefix length bound ---------------------------------
        if self.metrics_prefix and len(self.metrics_prefix) > 32:
            errors.append(
                f"metrics_prefix must be <= 32 characters, "
                f"got {len(self.metrics_prefix)}"
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: Provenance Tracking
    # ------------------------------------------------------------------

    def _validate_provenance_settings(self) -> List[str]:
        """Validate provenance tracking settings.

        Checks that genesis hash is provided when provenance is enabled
        and that the hash length is within bounds.

        Returns:
            List of error strings for invalid provenance settings.
        """
        errors: List[str] = []

        # -- genesis_hash required when provenance enabled ---------------
        if self.enable_provenance and not self.genesis_hash:
            errors.append(
                "genesis_hash must not be empty when "
                "enable_provenance is True"
            )

        # -- genesis_hash length bound -----------------------------------
        if self.genesis_hash and len(self.genesis_hash) > 256:
            errors.append(
                f"genesis_hash must be <= 256 characters, "
                f"got {len(self.genesis_hash)}"
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: Performance Tuning
    # ------------------------------------------------------------------

    def _validate_performance_settings(self) -> List[str]:
        """Validate performance tuning settings.

        Checks worker threads and health check interval are within
        reasonable bounds for production deployments.

        Returns:
            List of error strings for invalid performance settings.
        """
        errors: List[str] = []

        # -- worker_threads lower bound ----------------------------------
        if self.worker_threads <= 0:
            errors.append(
                f"worker_threads must be > 0, "
                f"got {self.worker_threads}"
            )

        # -- worker_threads upper bound ----------------------------------
        if self.worker_threads > 64:
            errors.append(
                f"worker_threads must be <= 64, "
                f"got {self.worker_threads}"
            )

        # -- health_check_interval lower bound ---------------------------
        if self.health_check_interval <= 0:
            errors.append(
                f"health_check_interval must be > 0, "
                f"got {self.health_check_interval}"
            )

        # -- health_check_interval upper bound ---------------------------
        if self.health_check_interval > 3600:
            errors.append(
                f"health_check_interval must be <= 3600, "
                f"got {self.health_check_interval}"
            )

        return errors

    @staticmethod
    def _count_validation_checks() -> int:
        """Return the approximate number of validation checks performed.

        Returns:
            Count of individual validation assertions.
        """
        # General: 10, Database: 14, Reconciliation: 17,
        # Materiality: 11, Quality: 11, Trend: 10,
        # Compliance: 8, Cache: 4, API: 11, Logging: 3,
        # Provenance: 2, Performance: 4
        return 125

    # ------------------------------------------------------------------
    # Accessor Methods: General
    # ------------------------------------------------------------------

    def get_service_name(self) -> str:
        """Return the service name for tracing and identification.

        Returns:
            Service name string.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.get_service_name()
            'dual-reporting-reconciliation-service'
        """
        return self.service_name

    def get_version(self) -> str:
        """Return the service version string.

        Returns:
            Version string.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.get_version()
            '1.0.0'
        """
        return self.version

    def get_log_level(self) -> str:
        """Return the logging level.

        Returns:
            Log level string (e.g. "INFO", "DEBUG").

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.get_log_level()
            'INFO'
        """
        return self.log_level

    def get_environment(self) -> str:
        """Return the deployment environment.

        Returns:
            Environment string (development, staging, or production).

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.get_environment()
            'development'
        """
        return self.environment

    def get_general_config(self) -> Dict[str, Any]:
        """Return general configuration as a dictionary.

        Returns:
            Dictionary containing all general settings.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> gen = cfg.get_general_config()
            >>> gen["service_name"]
            'dual-reporting-reconciliation-service'
        """
        return {
            "service_name": self.service_name,
            "version": self.version,
            "log_level": self.log_level,
            "environment": self.environment,
            "max_batch_size": self.max_batch_size,
            "default_tenant": self.default_tenant,
            "enabled": self.enabled,
        }

    # ------------------------------------------------------------------
    # Accessor Methods: Database
    # ------------------------------------------------------------------

    def get_db_dsn(self) -> str:
        """Build a PostgreSQL connection DSN from individual settings.

        Constructs a standard PostgreSQL connection URL from the
        configured host, port, database, user, and password fields.
        The password is URL-encoded to handle special characters.
        The SSL mode is appended as a query parameter.

        Returns:
            PostgreSQL connection DSN string.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> dsn = cfg.get_db_dsn()
            >>> dsn.startswith("postgresql://")
            True
        """
        if self.db_password:
            encoded_password = quote_plus(self.db_password)
            auth = f"{self.db_user}:{encoded_password}"
        else:
            auth = self.db_user

        url = (
            f"postgresql://{auth}@{self.db_host}:{self.db_port}"
            f"/{self.db_name}"
        )

        if self.db_ssl_mode:
            url += f"?sslmode={self.db_ssl_mode}"

        return url

    def get_async_db_dsn(self) -> str:
        """Build an async PostgreSQL connection DSN for asyncpg/psycopg.

        Identical to :meth:`get_db_dsn` but uses the
        ``postgresql+asyncpg://`` scheme for async driver compatibility.

        Returns:
            Async PostgreSQL connection DSN string.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> dsn = cfg.get_async_db_dsn()
            >>> dsn.startswith("postgresql+asyncpg://")
            True
        """
        if self.db_password:
            encoded_password = quote_plus(self.db_password)
            auth = f"{self.db_user}:{encoded_password}"
        else:
            auth = self.db_user

        url = (
            f"postgresql+asyncpg://{auth}@{self.db_host}:{self.db_port}"
            f"/{self.db_name}"
        )

        if self.db_ssl_mode and self.db_ssl_mode != "disable":
            url += f"?ssl={self.db_ssl_mode}"

        return url

    def get_db_pool_config(self) -> Dict[str, Any]:
        """Return database connection pool parameters.

        Returns:
            Dictionary containing pool configuration suitable for
            passing to psycopg_pool or similar connection pool libraries.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> params = cfg.get_db_pool_config()
            >>> params["min_size"]
            2
        """
        return {
            "min_size": self.db_pool_min,
            "max_size": self.db_pool_max,
            "conninfo": self.get_db_dsn(),
        }

    def get_table_prefix(self) -> str:
        """Return the database table name prefix.

        Returns:
            Table prefix string (e.g. "drr_").

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.get_table_prefix()
            'drr_'
        """
        return self.table_prefix

    def get_database_config(self) -> Dict[str, Any]:
        """Return complete database configuration as a dictionary.

        Sensitive password field is redacted. Includes DSN, pool
        parameters, SSL mode, and table prefix.

        Returns:
            Dictionary of all database-related settings.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> db_cfg = cfg.get_database_config()
            >>> db_cfg["host"]
            'localhost'
        """
        return {
            "host": self.db_host,
            "port": self.db_port,
            "name": self.db_name,
            "user": self.db_user,
            "password": "***" if self.db_password else "",
            "pool_min": self.db_pool_min,
            "pool_max": self.db_pool_max,
            "ssl_mode": self.db_ssl_mode,
            "table_prefix": self.table_prefix,
            "dsn": self.get_db_dsn(),
        }

    # ------------------------------------------------------------------
    # Accessor Methods: Reconciliation Defaults
    # ------------------------------------------------------------------

    def get_default_gwp_source(self) -> str:
        """Return the default GWP assessment report source.

        Returns:
            GWP source string (e.g. "ar5").

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.get_default_gwp_source()
            'ar5'
        """
        return self.default_gwp_source

    def get_decimal_places(self) -> int:
        """Return the number of decimal places for calculations.

        Returns:
            Integer decimal places count.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.get_decimal_places()
            8
        """
        return self.decimal_places

    def get_reconciliation_config(self) -> Dict[str, Any]:
        """Return reconciliation engine configuration as a dictionary.

        Returns:
            Dictionary containing all reconciliation-related settings
            suitable for initialising the reconciliation engines.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> recon = cfg.get_reconciliation_config()
            >>> recon["gwp_source"]
            'ar5'
        """
        return {
            "gwp_source": self.default_gwp_source,
            "decimal_places": self.decimal_places,
            "include_trends": self.include_trends,
            "include_quality": self.include_quality,
            "include_compliance": self.include_compliance,
            "max_trace_steps": self.max_trace_steps,
            "location_based_label": self.location_based_label,
            "market_based_label": self.market_based_label,
            "variance_precision": self.variance_precision,
            "enable_cross_scope_recon": self.enable_cross_scope_recon,
            "enable_temporal_alignment": (
                self.enable_temporal_alignment
            ),
            "enable_boundary_validation": (
                self.enable_boundary_validation
            ),
            "default_reporting_year": self.default_reporting_year,
            "max_reconciliation_records": (
                self.max_reconciliation_records
            ),
            "rounding_mode": self.rounding_mode,
            "max_batch_size": self.max_batch_size,
        }

    # ------------------------------------------------------------------
    # Accessor Methods: Materiality Thresholds
    # ------------------------------------------------------------------

    def get_materiality_thresholds(self) -> Dict[str, Decimal]:
        """Return materiality threshold configuration as a dictionary.

        The thresholds define percentage variance bands used to classify
        differences between location-based and market-based emissions
        into materiality categories. The returned values are guaranteed
        to be ordered: immaterial < minor < material < significant.

        Returns:
            Dictionary mapping category names to Decimal threshold
            percentages.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> thresholds = cfg.get_materiality_thresholds()
            >>> thresholds["immaterial"]
            Decimal('5')
            >>> thresholds["significant"]
            Decimal('100')
        """
        return {
            "immaterial": self.immaterial_threshold,
            "minor": self.minor_threshold,
            "material": self.material_threshold,
            "significant": self.significant_threshold,
        }

    def classify_variance(self, variance_pct: Decimal) -> str:
        """Classify a variance percentage into a materiality category.

        Uses the configured materiality thresholds to determine the
        appropriate category for a given absolute variance percentage.

        Args:
            variance_pct: Absolute variance percentage to classify.
                Must be non-negative.

        Returns:
            Materiality category string: "immaterial", "minor",
            "material", "significant", or "critical".

        Raises:
            ValueError: If variance_pct is negative.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.classify_variance(Decimal("3.5"))
            'immaterial'
            >>> cfg.classify_variance(Decimal("25"))
            'material'
            >>> cfg.classify_variance(Decimal("150"))
            'critical'
        """
        if variance_pct < Decimal("0"):
            raise ValueError(
                f"variance_pct must be >= 0, got {variance_pct}"
            )

        if variance_pct <= self.immaterial_threshold:
            return "immaterial"
        elif variance_pct <= self.minor_threshold:
            return "minor"
        elif variance_pct <= self.material_threshold:
            return "material"
        elif variance_pct <= self.significant_threshold:
            return "significant"
        else:
            return "critical"

    # ------------------------------------------------------------------
    # Accessor Methods: Quality Scoring
    # ------------------------------------------------------------------

    def get_quality_weights(self) -> Dict[str, Decimal]:
        """Return quality scoring weights as a dictionary.

        The four quality dimension weights represent the relative
        importance of each dimension in computing the overall data
        quality score. The weights are guaranteed to sum to exactly
        ``Decimal("1.00")`` by configuration validation.

        Returns:
            Dictionary mapping dimension names to Decimal weights.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> weights = cfg.get_quality_weights()
            >>> weights["completeness"]
            Decimal('0.30')
            >>> sum(weights.values()) == Decimal("1.00")
            True
        """
        return {
            "completeness": self.completeness_weight,
            "consistency": self.consistency_weight,
            "accuracy": self.accuracy_weight,
            "transparency": self.transparency_weight,
        }

    def get_assurance_threshold(self) -> Decimal:
        """Return the assurance-ready quality score threshold.

        Reconciliation results with an overall quality score at or
        above this threshold are considered assurance-ready and
        suitable for external audit review.

        Returns:
            Decimal assurance threshold (0 to 1 scale).

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.get_assurance_threshold()
            Decimal('0.90')
        """
        return self.assurance_threshold

    def get_quality_config(self) -> Dict[str, Any]:
        """Return quality scoring configuration as a dictionary.

        Includes weights for all four quality dimensions and the
        assurance-ready threshold. Decimal values are converted to
        strings for JSON-safe serialisation.

        Returns:
            Dictionary containing all quality-scoring settings.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> qc = cfg.get_quality_config()
            >>> qc["assurance_threshold"]
            '0.90'
        """
        return {
            "completeness_weight": str(self.completeness_weight),
            "consistency_weight": str(self.consistency_weight),
            "accuracy_weight": str(self.accuracy_weight),
            "transparency_weight": str(self.transparency_weight),
            "assurance_threshold": str(self.assurance_threshold),
            "include_quality": self.include_quality,
        }

    def is_assurance_ready(self, score: Decimal) -> bool:
        """Check if a quality score meets the assurance-ready threshold.

        Args:
            score: Overall quality score (0 to 1 scale).

        Returns:
            True if the score meets or exceeds the assurance threshold.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.is_assurance_ready(Decimal("0.95"))
            True
            >>> cfg.is_assurance_ready(Decimal("0.85"))
            False
        """
        return score >= self.assurance_threshold

    # ------------------------------------------------------------------
    # Accessor Methods: Trend Analysis
    # ------------------------------------------------------------------

    def get_trend_config(self) -> Dict[str, Any]:
        """Return trend analysis configuration as a dictionary.

        Includes period bounds, stability threshold, confidence level,
        and seasonality flag. Decimal values are converted to strings
        for JSON-safe serialisation.

        Returns:
            Dictionary containing all trend-analysis settings.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> trend = cfg.get_trend_config()
            >>> trend["min_periods"]
            2
            >>> trend["stable_threshold"]
            '2.0'
        """
        return {
            "min_periods": self.trend_min_periods,
            "max_periods": self.trend_max_periods,
            "stable_threshold": str(self.stable_threshold),
            "confidence_level": str(self.trend_confidence_level),
            "seasonality_enabled": self.trend_seasonality_enabled,
            "include_trends": self.include_trends,
        }

    def get_trend_min_periods(self) -> int:
        """Return the minimum number of periods for trend analysis.

        Returns:
            Integer minimum period count.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.get_trend_min_periods()
            2
        """
        return self.trend_min_periods

    def get_trend_max_periods(self) -> int:
        """Return the maximum number of periods for trend analysis.

        Returns:
            Integer maximum period count.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.get_trend_max_periods()
            10
        """
        return self.trend_max_periods

    # ------------------------------------------------------------------
    # Accessor Methods: Compliance
    # ------------------------------------------------------------------

    def get_enabled_frameworks(self) -> List[str]:
        """Return a copy of the enabled compliance frameworks list.

        Returns:
            List of enabled framework identifiers.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> "cdp" in cfg.get_enabled_frameworks()
            True
        """
        return list(self.enabled_frameworks)

    def is_framework_enabled(self, framework: str) -> bool:
        """Check if a specific compliance framework is enabled.

        Performs a case-insensitive comparison against the configured
        list of enabled frameworks.

        Args:
            framework: Framework identifier to check (e.g. "cdp",
                "ghg_protocol", "re100").

        Returns:
            True if the framework is in the enabled list, False otherwise.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.is_framework_enabled("cdp")
            True
            >>> cfg.is_framework_enabled("unknown_framework")
            False
        """
        normalised = framework.lower()
        return normalised in [
            fw.lower() for fw in self.enabled_frameworks
        ]

    def get_compliance_config(self) -> Dict[str, Any]:
        """Return compliance configuration as a dictionary.

        Returns:
            Dictionary containing all compliance-related settings.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> comp = cfg.get_compliance_config()
            >>> comp["strict_mode"]
            False
        """
        return {
            "enabled_frameworks": list(self.enabled_frameworks),
            "strict_mode": self.strict_mode,
            "fail_on_non_compliant": self.fail_on_non_compliant,
            "include_compliance": self.include_compliance,
        }

    # ------------------------------------------------------------------
    # Accessor Methods: API
    # ------------------------------------------------------------------

    def get_api_prefix(self) -> str:
        """Return the REST API route prefix.

        Returns:
            API prefix string.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.get_api_prefix()
            '/api/v1/dual-reporting'
        """
        return self.api_prefix

    def get_api_config(self) -> Dict[str, Any]:
        """Return API configuration as a dictionary.

        Returns:
            Dictionary of API-related settings suitable for FastAPI
            application construction.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> api_cfg = cfg.get_api_config()
            >>> api_cfg["prefix"]
            '/api/v1/dual-reporting'
        """
        return {
            "prefix": self.api_prefix,
            "rate_limit": self.rate_limit,
            "page_size": self.page_size,
            "max_page_size": self.max_page_size,
            "cors_origins": list(self.cors_origins),
            "enable_docs": self.enable_docs,
        }

    # ------------------------------------------------------------------
    # Accessor Methods: Cache
    # ------------------------------------------------------------------

    def get_cache_config(self) -> Dict[str, Any]:
        """Return Redis/cache configuration as a dictionary.

        Returns:
            Dictionary containing all cache-related settings suitable
            for initialising a Redis connection.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cache = cfg.get_cache_config()
            >>> cache["ttl"]
            3600
        """
        return {
            "url": self.redis_url,
            "ttl": self.cache_ttl,
            "enabled": self.enable_caching,
        }

    # ------------------------------------------------------------------
    # Accessor Methods: Rounding Mode
    # ------------------------------------------------------------------

    def get_rounding_mode(self) -> str:
        """Return the Python Decimal rounding mode constant.

        Maps the configured rounding mode string to the corresponding
        ``decimal`` module constant for use in ``Decimal.quantize()``.

        Returns:
            Decimal rounding mode constant string.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.get_rounding_mode()
            'ROUND_HALF_UP'
        """
        normalised = self.rounding_mode.upper()
        return _ROUNDING_MODE_MAP.get(normalised, ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Accessor Methods: Observability
    # ------------------------------------------------------------------

    def get_observability_config(self) -> Dict[str, Any]:
        """Return observability configuration as a dictionary.

        Returns:
            Dictionary of observability-related settings for metrics
            and tracing initialisation.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> obs_cfg = cfg.get_observability_config()
            >>> obs_cfg["metrics_prefix"]
            'gl_drr_'
        """
        return {
            "log_level": self.log_level,
            "enable_metrics": self.enable_metrics,
            "metrics_prefix": self.metrics_prefix,
            "enable_tracing": self.enable_tracing,
            "service_name": self.service_name,
        }

    # ------------------------------------------------------------------
    # Accessor Methods: Feature Flags
    # ------------------------------------------------------------------

    def get_feature_flags(self) -> Dict[str, bool]:
        """Return all feature flags as a dictionary.

        Returns:
            Dictionary mapping feature flag name to boolean state.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> flags = cfg.get_feature_flags()
            >>> flags["include_trends"]
            True
        """
        return {
            "include_trends": self.include_trends,
            "include_quality": self.include_quality,
            "include_compliance": self.include_compliance,
            "enable_cross_scope_recon": self.enable_cross_scope_recon,
            "enable_temporal_alignment": (
                self.enable_temporal_alignment
            ),
            "enable_boundary_validation": (
                self.enable_boundary_validation
            ),
            "trend_seasonality_enabled": (
                self.trend_seasonality_enabled
            ),
            "strict_mode": self.strict_mode,
            "fail_on_non_compliant": self.fail_on_non_compliant,
            "enable_caching": self.enable_caching,
            "enable_metrics": self.enable_metrics,
            "enable_tracing": self.enable_tracing,
            "enable_provenance": self.enable_provenance,
            "enable_auth": self.enable_auth,
            "enable_docs": self.enable_docs,
            "enable_background_tasks": self.enable_background_tasks,
        }

    # ------------------------------------------------------------------
    # Accessor Methods: Provenance
    # ------------------------------------------------------------------

    def get_provenance_config(self) -> Dict[str, Any]:
        """Return provenance tracking configuration as a dictionary.

        Returns:
            Dictionary of provenance-related settings.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> prov = cfg.get_provenance_config()
            >>> prov["genesis_hash"]
            'GL-MRV-X-024-DUAL-REPORTING-RECONCILIATION-GENESIS'
        """
        return {
            "enabled": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            "max_trace_steps": self.max_trace_steps,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a plain Python dictionary.

        The returned dictionary is safe to pass to ``json.dumps``,
        ``yaml.dump``, or any structured logging framework. Sensitive
        fields (``db_password``) are redacted to prevent accidental
        credential leakage. Decimal values are converted to strings
        for JSON compatibility.

        Returns:
            Dictionary representation of the configuration with sensitive
            fields redacted and Decimal values as strings.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> d = cfg.to_dict()
            >>> d["default_gwp_source"]
            'ar5'
            >>> d["db_password"]
            '***'
        """
        return {
            # -- Master switch -------------------------------------------
            "enabled": self.enabled,
            # -- 1. General Settings -------------------------------------
            "service_name": self.service_name,
            "version": self.version,
            "log_level": self.log_level,
            "environment": self.environment,
            "max_batch_size": self.max_batch_size,
            "default_tenant": self.default_tenant,
            # -- 2. Database Settings ------------------------------------
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_name": self.db_name,
            "db_user": self.db_user,
            "db_password": "***" if self.db_password else "",
            "db_pool_min": self.db_pool_min,
            "db_pool_max": self.db_pool_max,
            "db_ssl_mode": self.db_ssl_mode,
            "table_prefix": self.table_prefix,
            # -- 3. Reconciliation Defaults ------------------------------
            "default_gwp_source": self.default_gwp_source,
            "decimal_places": self.decimal_places,
            "include_trends": self.include_trends,
            "include_quality": self.include_quality,
            "include_compliance": self.include_compliance,
            "max_trace_steps": self.max_trace_steps,
            "location_based_label": self.location_based_label,
            "market_based_label": self.market_based_label,
            "variance_precision": self.variance_precision,
            "enable_cross_scope_recon": self.enable_cross_scope_recon,
            "enable_temporal_alignment": (
                self.enable_temporal_alignment
            ),
            "enable_boundary_validation": (
                self.enable_boundary_validation
            ),
            "default_reporting_year": self.default_reporting_year,
            "max_reconciliation_records": (
                self.max_reconciliation_records
            ),
            "rounding_mode": self.rounding_mode,
            # -- 4. Materiality Thresholds -------------------------------
            "immaterial_threshold": str(self.immaterial_threshold),
            "minor_threshold": str(self.minor_threshold),
            "material_threshold": str(self.material_threshold),
            "significant_threshold": str(self.significant_threshold),
            # -- 5. Quality Scoring --------------------------------------
            "completeness_weight": str(self.completeness_weight),
            "consistency_weight": str(self.consistency_weight),
            "accuracy_weight": str(self.accuracy_weight),
            "transparency_weight": str(self.transparency_weight),
            "assurance_threshold": str(self.assurance_threshold),
            # -- 6. Trend Analysis ---------------------------------------
            "trend_min_periods": self.trend_min_periods,
            "trend_max_periods": self.trend_max_periods,
            "stable_threshold": str(self.stable_threshold),
            "trend_confidence_level": str(
                self.trend_confidence_level
            ),
            "trend_seasonality_enabled": (
                self.trend_seasonality_enabled
            ),
            # -- 7. Compliance Settings ----------------------------------
            "enabled_frameworks": list(self.enabled_frameworks),
            "strict_mode": self.strict_mode,
            "fail_on_non_compliant": self.fail_on_non_compliant,
            # -- 8. Redis/Cache Settings ---------------------------------
            "redis_url": self.redis_url,
            "cache_ttl": self.cache_ttl,
            "enable_caching": self.enable_caching,
            # -- 9. API Settings -----------------------------------------
            "api_prefix": self.api_prefix,
            "rate_limit": self.rate_limit,
            "page_size": self.page_size,
            "max_page_size": self.max_page_size,
            # -- Logging & Observability ---------------------------------
            "enable_metrics": self.enable_metrics,
            "metrics_prefix": self.metrics_prefix,
            "enable_tracing": self.enable_tracing,
            # -- Provenance Tracking -------------------------------------
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # -- Auth & Background Tasks ---------------------------------
            "enable_auth": self.enable_auth,
            "worker_threads": self.worker_threads,
            "enable_background_tasks": self.enable_background_tasks,
            "health_check_interval": self.health_check_interval,
            # -- CORS & Docs ---------------------------------------------
            "cors_origins": list(self.cors_origins),
            "enable_docs": self.enable_docs,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any]
    ) -> DualReportingReconciliationConfig:
        """Deserialise a configuration from a dictionary.

        Creates a new DualReportingReconciliationConfig instance and
        populates it from the provided dictionary. The singleton is
        reset first to allow the new configuration to be installed.
        Keys not present in the dictionary retain their
        environment-loaded defaults.

        String-valued Decimal fields are automatically converted back
        to Decimal objects.

        Args:
            data: Dictionary of configuration key-value pairs. Keys
                correspond to attribute names on the config object.

        Returns:
            A new DualReportingReconciliationConfig instance with
            values from the dictionary.

        Example:
            >>> d = {"default_gwp_source": "ar6", "decimal_places": 12}
            >>> cfg = DualReportingReconciliationConfig.from_dict(d)
            >>> cfg.default_gwp_source
            'ar6'
            >>> cfg.decimal_places
            12
        """
        cls.reset()
        instance = cls()
        instance._apply_dict(data)

        logger.info(
            "DualReportingReconciliationConfig loaded from dict: "
            "%d keys applied",
            len(data),
        )
        return instance

    def _apply_dict(self, data: Dict[str, Any]) -> None:
        """Apply dictionary values to the configuration instance.

        Only applies values for known attribute names. Unknown keys
        are logged as warnings and skipped. Redacted password fields
        (``'***'``) are skipped to avoid overwriting real credentials.
        String representations of Decimal fields are converted back
        to Decimal objects.

        Args:
            data: Dictionary of configuration key-value pairs.
        """
        known_attrs = set(self.to_dict().keys())

        # Fields that should be stored as Decimal
        decimal_fields = frozenset({
            "immaterial_threshold",
            "minor_threshold",
            "material_threshold",
            "significant_threshold",
            "completeness_weight",
            "consistency_weight",
            "accuracy_weight",
            "transparency_weight",
            "assurance_threshold",
            "stable_threshold",
            "trend_confidence_level",
        })

        for key, value in data.items():
            if key in known_attrs:
                if key in ("db_password",):
                    if value == "***":
                        continue
                if key in decimal_fields and isinstance(value, str):
                    try:
                        value = Decimal(value)
                    except InvalidOperation:
                        logger.warning(
                            "Cannot convert '%s' to Decimal for key "
                            "'%s', skipping",
                            value,
                            key,
                        )
                        continue
                setattr(self, key, value)
            else:
                logger.warning(
                    "Unknown configuration key '%s' in from_dict, "
                    "skipping",
                    key,
                )

    # ------------------------------------------------------------------
    # JSON serialisation helpers
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Serialise the configuration to a JSON string.

        Sensitive fields are redacted in the output. Uses the same
        redaction rules as :meth:`to_dict`.

        Args:
            indent: JSON indentation level. Defaults to 2.

        Returns:
            JSON string representation.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> json_str = cfg.to_json()
            >>> '"default_gwp_source": "ar5"' in json_str
            True
        """
        return json.dumps(
            self.to_dict(), indent=indent, sort_keys=False
        )

    @classmethod
    def from_json(
        cls, json_str: str
    ) -> DualReportingReconciliationConfig:
        """Deserialise a configuration from a JSON string.

        Args:
            json_str: JSON string containing configuration key-value
                pairs.

        Returns:
            A new DualReportingReconciliationConfig instance.

        Example:
            >>> json_str = '{"default_gwp_source": "ar6"}'
            >>> cfg = DualReportingReconciliationConfig.from_json(
            ...     json_str
            ... )
            >>> cfg.default_gwp_source
            'ar6'
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Normalisation helper
    # ------------------------------------------------------------------

    def normalise(self) -> None:
        """Normalise configuration values to canonical forms.

        Converts string enumerations to their expected case (e.g. GWP
        source to lowercase, log level to uppercase, SSL mode to
        lowercase, environment to lowercase). This method is idempotent.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.default_gwp_source = "AR5"
            >>> cfg.normalise()
            >>> cfg.default_gwp_source
            'ar5'
        """
        # GWP source -> lowercase
        self.default_gwp_source = self.default_gwp_source.lower()

        # Environment -> lowercase
        self.environment = self.environment.lower()

        # Log level -> uppercase
        self.log_level = self.log_level.upper()

        # SSL mode -> lowercase
        self.db_ssl_mode = self.db_ssl_mode.lower()

        # Frameworks -> lowercase
        self.enabled_frameworks = [
            fw.lower() for fw in self.enabled_frameworks
        ]

        # Table prefix -> lowercase
        self.table_prefix = self.table_prefix.lower()

        # Metrics prefix -> lowercase (convention)
        self.metrics_prefix = self.metrics_prefix.lower()

        # Rounding mode -> uppercase
        self.rounding_mode = self.rounding_mode.upper()

        logger.debug(
            "DualReportingReconciliationConfig normalised: "
            "gwp=%s, log_level=%s, ssl_mode=%s, env=%s, "
            "rounding=%s",
            self.default_gwp_source,
            self.log_level,
            self.db_ssl_mode,
            self.environment,
            self.rounding_mode,
        )

    # ------------------------------------------------------------------
    # Merge with overrides
    # ------------------------------------------------------------------

    def merge(self, overrides: Dict[str, Any]) -> None:
        """Merge override values into the current configuration.

        Only known attributes are applied. Unknown keys are logged
        as warnings. Sensitive fields cannot be set to the redacted
        placeholder ``'***'``.

        Args:
            overrides: Dictionary of override key-value pairs.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> cfg.merge({"decimal_places": 12})
            >>> cfg.decimal_places
            12
        """
        self._apply_dict(overrides)
        logger.debug(
            "DualReportingReconciliationConfig merged %d overrides",
            len(overrides),
        )

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> Dict[str, Any]:
        """Return a deep copy of the configuration as a dictionary.

        Unlike :meth:`to_dict`, this method does NOT redact sensitive
        fields. It is intended for internal use where the full
        configuration needs to be preserved (e.g. serialisation to
        encrypted storage).

        Returns:
            Dictionary with all values including sensitive fields.
        """
        d = self.to_dict()
        d["db_password"] = self.db_password
        return d

    # ------------------------------------------------------------------
    # Summary for health checks
    # ------------------------------------------------------------------

    def health_summary(self) -> Dict[str, Any]:
        """Return a health check summary of the configuration.

        Includes validation status, enabled feature counts, and
        key operational parameters. Suitable for inclusion in
        ``/health`` endpoint responses.

        Returns:
            Dictionary with health-relevant configuration summary.

        Example:
            >>> cfg = DualReportingReconciliationConfig()
            >>> summary = cfg.health_summary()
            >>> summary["validation_status"]
            'PASS'
        """
        errors = self.validate()
        flags = self.get_feature_flags()
        enabled_flag_count = sum(1 for v in flags.values() if v)

        return {
            "agent": "dual-reporting-reconciliation",
            "agent_id": "AGENT-MRV-013",
            "gl_id": "GL-MRV-X-024",
            "enabled": self.enabled,
            "validation_status": "PASS" if not errors else "FAIL",
            "validation_errors": len(errors),
            "service_name": self.service_name,
            "version": self.version,
            "environment": self.environment,
            "gwp_source": self.default_gwp_source,
            "decimal_places": self.decimal_places,
            "max_batch_size": self.max_batch_size,
            "enabled_frameworks": len(self.enabled_frameworks),
            "enabled_features": enabled_flag_count,
            "total_features": len(flags),
            "db_host": self.db_host,
            "db_port": self.db_port,
            "worker_threads": self.worker_threads,
            "provenance_enabled": self.enable_provenance,
            "caching_enabled": self.enable_caching,
            "metrics_prefix": self.metrics_prefix,
            "materiality_thresholds": {
                "immaterial": str(self.immaterial_threshold),
                "minor": str(self.minor_threshold),
                "material": str(self.material_threshold),
                "significant": str(self.significant_threshold),
            },
            "quality_weights_sum": str(
                self.completeness_weight
                + self.consistency_weight
                + self.accuracy_weight
                + self.transparency_weight
            ),
            "assurance_threshold": str(self.assurance_threshold),
            "trend_min_periods": self.trend_min_periods,
            "trend_max_periods": self.trend_max_periods,
        }

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Sensitive fields (db_password) are replaced with ``'***'`` so
        that repr output is safe to include in log messages and
        exception tracebacks.

        Returns:
            String representation of the configuration.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"DualReportingReconciliationConfig({pairs})"

    def __str__(self) -> str:
        """Return a human-readable summary of the configuration.

        Returns:
            Multi-line string summary of key settings.
        """
        return (
            f"DualReportingReconciliationConfig("
            f"enabled={self.enabled}, "
            f"service={self.service_name}, "
            f"env={self.environment}, "
            f"gwp={self.default_gwp_source}, "
            f"decimal_places={self.decimal_places}, "
            f"materiality=[{self.immaterial_threshold}/"
            f"{self.minor_threshold}/{self.material_threshold}/"
            f"{self.significant_threshold}], "
            f"quality_weights=[{self.completeness_weight}/"
            f"{self.consistency_weight}/{self.accuracy_weight}/"
            f"{self.transparency_weight}], "
            f"assurance={self.assurance_threshold}, "
            f"batch={self.max_batch_size}, "
            f"frameworks={len(self.enabled_frameworks)}, "
            f"trends=[{self.trend_min_periods}-"
            f"{self.trend_max_periods}]"
            f")"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality by comparing all serialised values.

        Args:
            other: Object to compare against.

        Returns:
            True if other is a DualReportingReconciliationConfig with
            identical settings.
        """
        if not isinstance(other, DualReportingReconciliationConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def get_config() -> DualReportingReconciliationConfig:
    """Return the singleton DualReportingReconciliationConfig.

    Creates the instance from environment variables on first call.
    Subsequent calls return the cached singleton.

    Returns:
        DualReportingReconciliationConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.default_gwp_source
        'ar5'
    """
    return DualReportingReconciliationConfig()


def reset_config() -> None:
    """Reset the singleton for test teardown.

    The next call to :func:`get_config` will re-read environment
    variables and construct a fresh instance.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads GL_DRR_* env vars
    """
    DualReportingReconciliationConfig.reset()
    logger.debug(
        "DualReportingReconciliationConfig singleton reset via "
        "module-level reset_config()"
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "DualReportingReconciliationConfig",
    "get_config",
    "reset_config",
]
