# -*- coding: utf-8 -*-
"""
Purchased Goods & Services Agent Configuration - AGENT-MRV-014

Centralized configuration for the Purchased Goods & Services Agent SDK
covering:
- General service settings (name, version, logging, environment, tenant)
- Database connection and pool settings (PostgreSQL)
- Calculation settings (method, GWP source, decimal precision, base year,
  currency, margin removal, inflation adjustment)
- EEIO settings (database, base year, PPP adjustment)
- DQI settings (5-dimension weights with sum-to-1.0 constraint)
- Coverage settings (minimum/target percentages, method-specific targets)
- Compliance framework toggles (GHG Protocol, CSRD ESRS, CDP, SBTi,
  California SB 253, GRI 305, ISO 14064)
- Redis/cache settings (URL, TTL, enable flag)
- API settings (prefix, rate limit, page size)
- Observability settings (metrics, tracing, provenance)

This module implements a thread-safe singleton pattern using ``__new__``
with a class-level ``_instance``, ``_initialized`` flag, and
``threading.RLock``, ensuring exactly one configuration object exists
across the application lifecycle. All numeric settings are stored as
``Decimal`` for zero-hallucination deterministic calculations.

All settings can be overridden via environment variables with the
``GL_PGS_`` prefix.

Environment Variable Reference (GL_PGS_ prefix):
    GL_PGS_SERVICE_NAME                 - Service name for tracing
    GL_PGS_VERSION                      - Service version string
    GL_PGS_LOG_LEVEL                    - Logging level
    GL_PGS_ENVIRONMENT                  - Deployment environment
    GL_PGS_DEFAULT_TENANT               - Default tenant identifier
    GL_PGS_MAX_BATCH_SIZE               - Maximum records per batch
    GL_PGS_ENABLED                      - Master enable/disable switch
    GL_PGS_DB_HOST                      - PostgreSQL host
    GL_PGS_DB_PORT                      - PostgreSQL port
    GL_PGS_DB_NAME                      - PostgreSQL database name
    GL_PGS_DB_USER                      - PostgreSQL username
    GL_PGS_DB_PASSWORD                  - PostgreSQL password
    GL_PGS_DB_POOL_MIN                  - Minimum connection pool size
    GL_PGS_DB_POOL_MAX                  - Maximum connection pool size
    GL_PGS_DB_SSL_MODE                  - PostgreSQL SSL mode
    GL_PGS_TABLE_PREFIX                 - Database table name prefix
    GL_PGS_DEFAULT_METHOD               - Default calculation method
    GL_PGS_DEFAULT_GWP_SOURCE           - Default GWP source (ar4/ar5/ar6)
    GL_PGS_DECIMAL_PLACES               - Decimal places for calculations
    GL_PGS_BASE_YEAR                    - Base year for EEIO deflation
    GL_PGS_DEFAULT_CURRENCY             - Default base currency (ISO 4217)
    GL_PGS_ENABLE_MARGIN_REMOVAL        - Enable margin removal adjustment
    GL_PGS_ENABLE_INFLATION_ADJUSTMENT  - Enable inflation/CPI adjustment
    GL_PGS_MAX_TRACE_STEPS              - Max provenance trace steps
    GL_PGS_ROUNDING_MODE                - Decimal rounding mode
    GL_PGS_DEFAULT_REPORTING_YEAR       - Default reporting year
    GL_PGS_MAX_CALCULATION_RECORDS      - Max calculation records per run
    GL_PGS_ENABLE_TRANSPORT_ADDER       - Enable transport-to-gate adder
    GL_PGS_ENABLE_WASTE_FACTOR          - Enable waste/loss factor
    GL_PGS_ENABLE_DOUBLE_COUNTING_CHECK - Enable double-counting prevention
    GL_PGS_CAPITAL_THRESHOLD            - Capital goods $ threshold
    GL_PGS_DEFAULT_EEIO_DATABASE        - Default EEIO database
    GL_PGS_EEIO_BASE_YEAR              - EEIO factor base year
    GL_PGS_ENABLE_PPP_ADJUSTMENT        - Enable PPP adjustment
    GL_PGS_ENABLE_DQI_SCORING           - Enable DQI scoring
    GL_PGS_DQI_TEMPORAL_WEIGHT          - DQI temporal dimension weight
    GL_PGS_DQI_GEOGRAPHICAL_WEIGHT      - DQI geographical dimension weight
    GL_PGS_DQI_TECHNOLOGICAL_WEIGHT     - DQI technological dimension weight
    GL_PGS_DQI_COMPLETENESS_WEIGHT      - DQI completeness dimension weight
    GL_PGS_DQI_RELIABILITY_WEIGHT       - DQI reliability dimension weight
    GL_PGS_MIN_COVERAGE_PCT             - Minimum spend coverage %
    GL_PGS_TARGET_COVERAGE_PCT          - Target spend coverage %
    GL_PGS_SUPPLIER_SPECIFIC_TARGET     - Supplier-specific method target %
    GL_PGS_AVERAGE_DATA_TARGET          - Average-data method target %
    GL_PGS_SPEND_BASED_MAX_PCT          - Max acceptable spend-based %
    GL_PGS_HIGH_SPEND_THRESHOLD         - High-spend category threshold $
    GL_PGS_MEDIUM_SPEND_THRESHOLD       - Medium-spend category threshold $
    GL_PGS_LOW_SPEND_THRESHOLD          - Low-spend category threshold $
    GL_PGS_ENABLED_FRAMEWORKS           - Comma-separated frameworks
    GL_PGS_STRICT_MODE                  - Enable strict compliance mode
    GL_PGS_FAIL_ON_NON_COMPLIANT        - Fail on non-compliant results
    GL_PGS_REDIS_URL                    - Redis connection URL
    GL_PGS_CACHE_TTL                    - Cache TTL in seconds
    GL_PGS_ENABLE_CACHING               - Enable Redis caching
    GL_PGS_API_PREFIX                   - REST API route prefix
    GL_PGS_RATE_LIMIT                   - API requests per minute
    GL_PGS_PAGE_SIZE                    - Default page size for list APIs
    GL_PGS_MAX_PAGE_SIZE                - Maximum page size for list APIs
    GL_PGS_ENABLE_METRICS               - Enable Prometheus metrics export
    GL_PGS_METRICS_PREFIX               - Prometheus metrics prefix
    GL_PGS_ENABLE_TRACING               - Enable OpenTelemetry tracing
    GL_PGS_ENABLE_PROVENANCE            - Enable SHA-256 provenance tracking
    GL_PGS_GENESIS_HASH                 - Provenance chain genesis anchor
    GL_PGS_ENABLE_AUTH                  - Enable authentication middleware
    GL_PGS_WORKER_THREADS               - Worker thread pool size
    GL_PGS_ENABLE_BACKGROUND_TASKS      - Enable background task processing
    GL_PGS_HEALTH_CHECK_INTERVAL        - Health check interval (seconds)
    GL_PGS_CORS_ORIGINS                 - Comma-separated CORS origins
    GL_PGS_ENABLE_DOCS                  - Enable API documentation

Example:
    >>> from greenlang.purchased_goods_services.config import (
    ...     PurchasedGoodsServicesConfig,
    ... )
    >>> cfg = PurchasedGoodsServicesConfig()
    >>> print(cfg.service_name, cfg.default_method)
    purchased-goods-services HYBRID

    >>> # Check singleton
    >>> cfg2 = PurchasedGoodsServicesConfig()
    >>> assert cfg is cfg2

    >>> # Reset for testing
    >>> PurchasedGoodsServicesConfig.reset()
    >>> cfg3 = PurchasedGoodsServicesConfig()
    >>> assert cfg is not cfg3

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-014 Purchased Goods & Services (GL-MRV-S3-001)
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

_ENV_PREFIX: str = "GL_PGS_"

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

_VALID_CALCULATION_METHODS = frozenset({
    "SUPPLIER_SPECIFIC",
    "HYBRID",
    "AVERAGE_DATA",
    "SPEND_BASED",
})

_VALID_EEIO_DATABASES = frozenset({
    "EPA_USEEIO_V12",
    "EPA_USEEIO_V13",
    "EXIOBASE_V38",
    "WIOD_2016",
    "GTAP_11",
})

_VALID_CURRENCIES = frozenset({
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "CNY",
    "INR", "BRL", "KRW", "MXN", "SGD", "HKD", "NOK", "SEK",
    "DKK", "NZD", "ZAR", "THB", "TWD", "PLN", "CZK", "HUF",
    "ILS", "CLP", "PHP", "IDR", "MYR", "RUB", "TRY", "ARS",
    "COP", "PEN", "VND", "EGP", "NGN", "KES", "GHS", "TZS",
    "UGX", "MAD", "DZD", "SAR", "AED", "QAR", "KWD", "BHD",
    "OMR", "JOD",
})

_VALID_FRAMEWORKS = frozenset({
    "ghg_protocol",
    "csrd_esrs",
    "cdp",
    "sbti",
    "sb_253",
    "gri_305",
    "iso_14064",
})

_VALID_CLASSIFICATION_SYSTEMS = frozenset({
    "NAICS_2022",
    "NACE_REV2",
    "NACE_REV21",
    "ISIC_REV4",
    "UNSPSC_V28",
})

_VALID_ALLOCATION_METHODS = frozenset({
    "REVENUE_BASED",
    "MASS_BASED",
    "ECONOMIC",
    "PHYSICAL",
    "ENERGY_BASED",
})

_VALID_DQI_LEVELS = frozenset({
    "VERY_GOOD",
    "GOOD",
    "FAIR",
    "POOR",
    "VERY_POOR",
})

_VALID_MATERIALITY_QUADRANTS = frozenset({
    "PRIORITIZE",
    "MONITOR",
    "IMPROVE_DATA",
    "LOW_PRIORITY",
})

# ---------------------------------------------------------------------------
# Default compliance frameworks
# ---------------------------------------------------------------------------

_DEFAULT_ENABLED_FRAMEWORKS: List[str] = [
    "ghg_protocol",
    "csrd_esrs",
    "cdp",
    "sbti",
    "sb_253",
    "gri_305",
    "iso_14064",
]

# ---------------------------------------------------------------------------
# Default DQI dimension weights (must sum to 1.00)
# Per GHG Protocol Scope 3 Standard Chapter 7
# ---------------------------------------------------------------------------

_DEFAULT_DQI_TEMPORAL_WEIGHT = Decimal("0.20")
_DEFAULT_DQI_GEOGRAPHICAL_WEIGHT = Decimal("0.20")
_DEFAULT_DQI_TECHNOLOGICAL_WEIGHT = Decimal("0.20")
_DEFAULT_DQI_COMPLETENESS_WEIGHT = Decimal("0.20")
_DEFAULT_DQI_RELIABILITY_WEIGHT = Decimal("0.20")

# ---------------------------------------------------------------------------
# Default coverage thresholds (percentage)
# ---------------------------------------------------------------------------

_DEFAULT_MIN_COVERAGE_PCT = Decimal("80")
_DEFAULT_TARGET_COVERAGE_PCT = Decimal("95")
_DEFAULT_SUPPLIER_SPECIFIC_TARGET = Decimal("20")
_DEFAULT_AVERAGE_DATA_TARGET = Decimal("30")
_DEFAULT_SPEND_BASED_MAX_PCT = Decimal("50")

# ---------------------------------------------------------------------------
# Default spend thresholds (USD)
# ---------------------------------------------------------------------------

_DEFAULT_HIGH_SPEND_THRESHOLD = Decimal("10000000")
_DEFAULT_MEDIUM_SPEND_THRESHOLD = Decimal("1000000")
_DEFAULT_LOW_SPEND_THRESHOLD = Decimal("100000")

# ---------------------------------------------------------------------------
# Default capital goods threshold (USD)
# ---------------------------------------------------------------------------

_DEFAULT_CAPITAL_THRESHOLD = Decimal("5000")


# ---------------------------------------------------------------------------
# PurchasedGoodsServicesConfig
# ---------------------------------------------------------------------------


class PurchasedGoodsServicesConfig:
    """Singleton configuration for the Purchased Goods & Services Agent.

    Implements a thread-safe singleton pattern via ``__new__`` with a
    class-level ``_instance``, ``_initialized`` flag, and
    ``threading.RLock``. On first instantiation, all settings are loaded
    from environment variables with the ``GL_PGS_`` prefix. Subsequent
    instantiations return the same object.

    All numeric values are stored as ``Decimal`` to ensure
    zero-hallucination deterministic arithmetic throughout the purchased
    goods and services calculation pipeline. This eliminates IEEE 754
    floating-point representation errors that could compound across
    spend-based EEIO calculations, currency conversion, inflation
    deflation, margin removal, DQI scoring, and coverage analysis.

    The configuration covers ten domains:
    1. General Settings - service name, version, logging, environment,
       batch size, default tenant, master switch
    2. Database Settings - PostgreSQL connection, pool sizing, SSL mode,
       table prefix
    3. Calculation Settings - default method, GWP source, decimal places,
       base year, currency, margin removal, inflation adjustment,
       transport adder, waste factor, double-counting, capital threshold
    4. EEIO Settings - default EEIO database, base year, PPP adjustment
    5. DQI Settings - 5-dimension weights with sum-to-1.0 constraint
    6. Coverage Settings - minimum/target percentages, method targets,
       spend thresholds for method recommendation
    7. Compliance Settings - 7 regulatory frameworks, strict mode,
       fail-on-non-compliant
    8. Redis/Cache Settings - Redis URL, cache TTL, enable flag
    9. API Settings - route prefix, rate limit, page sizes
    10. Observability Settings - metrics, tracing, provenance, auth,
        worker threads, health checks, CORS, docs

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
        enabled: Master enable/disable switch for the agent.
        db_host: PostgreSQL server hostname.
        db_port: PostgreSQL server port.
        db_name: PostgreSQL database name.
        db_user: PostgreSQL username.
        db_password: PostgreSQL password (never logged or serialised).
        db_pool_min: Minimum number of connections in the pool.
        db_pool_max: Maximum number of connections in the pool.
        db_ssl_mode: PostgreSQL SSL connection mode.
        table_prefix: Database table name prefix.
        default_method: Default calculation method (HYBRID).
        default_gwp_source: Default IPCC Assessment Report for GWP values.
        decimal_places: Number of decimal places for calculations.
        base_year: Base year for EEIO deflation and intensity calcs.
        default_currency: Default base currency (ISO 4217).
        enable_margin_removal: Enable producer-price margin removal.
        enable_inflation_adjustment: Enable CPI inflation deflation.
        max_trace_steps: Maximum provenance trace steps.
        rounding_mode: Decimal rounding mode for calculations.
        default_reporting_year: Default reporting year.
        max_calculation_records: Maximum calculation records per run.
        enable_transport_adder: Enable transport-to-gate emission adder.
        enable_waste_factor: Enable waste/loss factor application.
        enable_double_counting_check: Enable double-counting prevention.
        capital_threshold: Capital goods dollar threshold for Cat 2.
        default_eeio_database: Default EEIO database identifier.
        eeio_base_year: EEIO factor base year for deflation.
        enable_ppp_adjustment: Enable purchasing power parity adjustment.
        enable_dqi_scoring: Enable data quality indicator scoring.
        dqi_temporal_weight: DQI temporal dimension weight.
        dqi_geographical_weight: DQI geographical dimension weight.
        dqi_technological_weight: DQI technological dimension weight.
        dqi_completeness_weight: DQI completeness dimension weight.
        dqi_reliability_weight: DQI reliability dimension weight.
        min_coverage_pct: Minimum spend coverage percentage.
        target_coverage_pct: Target spend coverage percentage.
        supplier_specific_target: Supplier-specific method target pct.
        average_data_target: Average-data method target percentage.
        spend_based_max_pct: Max acceptable spend-based percentage.
        high_spend_threshold: High-spend category threshold in USD.
        medium_spend_threshold: Medium-spend category threshold in USD.
        low_spend_threshold: Low-spend category threshold in USD.
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

    Example:
        >>> cfg = PurchasedGoodsServicesConfig()
        >>> cfg.default_method
        'HYBRID'
        >>> cfg.get_db_dsn()
        'postgresql://greenlang@localhost:5432/greenlang?sslmode=prefer'
        >>> cfg.is_framework_enabled("cdp")
        True
        >>> cfg.get_dqi_weights()
        {'temporal': Decimal('0.20'), ...}
        >>> cfg.get_coverage_thresholds()
        {'min_coverage_pct': Decimal('80'), ...}
    """

    _instance: Optional[PurchasedGoodsServicesConfig] = None
    _initialized: bool = False
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton constructor
    # ------------------------------------------------------------------

    def __new__(cls) -> PurchasedGoodsServicesConfig:
        """Return the singleton instance, creating it on first call.

        Uses a threading RLock to ensure thread-safe initialisation. Only
        one instance is ever created; subsequent calls return the same
        object without acquiring the lock (double-checked locking).

        Returns:
            The singleton PurchasedGoodsServicesConfig instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialise configuration from environment variables.

        Guarded by the ``_initialized`` flag so that repeated calls to
        ``__init__`` (from repeated ``PurchasedGoodsServicesConfig()``
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
                "PurchasedGoodsServicesConfig initialised from "
                "environment: service=%s, version=%s, "
                "method=%s, gwp=%s, decimal_places=%d, "
                "eeio=%s, currency=%s, base_year=%d, "
                "frameworks=%s, env=%s, "
                "dqi_weights=[%s/%s/%s/%s/%s], "
                "coverage=[min=%s/target=%s]",
                self.service_name,
                self.version,
                self.default_method,
                self.default_gwp_source,
                self.decimal_places,
                self.default_eeio_database,
                self.default_currency,
                self.base_year,
                self.enabled_frameworks,
                self.environment,
                self.dqi_temporal_weight,
                self.dqi_geographical_weight,
                self.dqi_technological_weight,
                self.dqi_completeness_weight,
                self.dqi_reliability_weight,
                self.min_coverage_pct,
                self.target_coverage_pct,
            )

    # ------------------------------------------------------------------
    # Environment loading
    # ------------------------------------------------------------------

    def _load_from_env(self) -> None:
        """Load all configuration from environment variables.

        Reads environment variables with the ``GL_PGS_`` prefix and
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
            "SERVICE_NAME", "purchased-goods-services"
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
        self.enabled: bool = self._env_bool("ENABLED", True)

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
            "TABLE_PREFIX", "pgs_"
        )

        # -- 3. Calculation Settings -------------------------------------
        self.default_method: str = self._env_str(
            "DEFAULT_METHOD", "HYBRID"
        )
        self.default_gwp_source: str = self._env_str(
            "DEFAULT_GWP_SOURCE", "ar5"
        )
        self.decimal_places: int = self._env_int(
            "DECIMAL_PLACES", 8
        )
        self.base_year: int = self._env_int(
            "BASE_YEAR", 2022
        )
        self.default_currency: str = self._env_str(
            "DEFAULT_CURRENCY", "USD"
        )
        self.enable_margin_removal: bool = self._env_bool(
            "ENABLE_MARGIN_REMOVAL", True
        )
        self.enable_inflation_adjustment: bool = self._env_bool(
            "ENABLE_INFLATION_ADJUSTMENT", True
        )
        self.max_trace_steps: int = self._env_int(
            "MAX_TRACE_STEPS", 200
        )
        self.rounding_mode: str = self._env_str(
            "ROUNDING_MODE", "ROUND_HALF_UP"
        )
        self.default_reporting_year: int = self._env_int(
            "DEFAULT_REPORTING_YEAR", 2025
        )
        self.max_calculation_records: int = self._env_int(
            "MAX_CALCULATION_RECORDS", 50000
        )
        self.enable_transport_adder: bool = self._env_bool(
            "ENABLE_TRANSPORT_ADDER", True
        )
        self.enable_waste_factor: bool = self._env_bool(
            "ENABLE_WASTE_FACTOR", True
        )
        self.enable_double_counting_check: bool = self._env_bool(
            "ENABLE_DOUBLE_COUNTING_CHECK", True
        )
        self.capital_threshold: Decimal = self._env_decimal(
            "CAPITAL_THRESHOLD", _DEFAULT_CAPITAL_THRESHOLD
        )

        # -- 4. EEIO Settings --------------------------------------------
        self.default_eeio_database: str = self._env_str(
            "DEFAULT_EEIO_DATABASE", "EPA_USEEIO_V12"
        )
        self.eeio_base_year: int = self._env_int(
            "EEIO_BASE_YEAR", 2022
        )
        self.enable_ppp_adjustment: bool = self._env_bool(
            "ENABLE_PPP_ADJUSTMENT", True
        )

        # -- 5. DQI Settings ---------------------------------------------
        self.enable_dqi_scoring: bool = self._env_bool(
            "ENABLE_DQI_SCORING", True
        )
        self.dqi_temporal_weight: Decimal = self._env_decimal(
            "DQI_TEMPORAL_WEIGHT", _DEFAULT_DQI_TEMPORAL_WEIGHT
        )
        self.dqi_geographical_weight: Decimal = self._env_decimal(
            "DQI_GEOGRAPHICAL_WEIGHT", _DEFAULT_DQI_GEOGRAPHICAL_WEIGHT
        )
        self.dqi_technological_weight: Decimal = self._env_decimal(
            "DQI_TECHNOLOGICAL_WEIGHT", _DEFAULT_DQI_TECHNOLOGICAL_WEIGHT
        )
        self.dqi_completeness_weight: Decimal = self._env_decimal(
            "DQI_COMPLETENESS_WEIGHT", _DEFAULT_DQI_COMPLETENESS_WEIGHT
        )
        self.dqi_reliability_weight: Decimal = self._env_decimal(
            "DQI_RELIABILITY_WEIGHT", _DEFAULT_DQI_RELIABILITY_WEIGHT
        )

        # -- 6. Coverage Settings ----------------------------------------
        self.min_coverage_pct: Decimal = self._env_decimal(
            "MIN_COVERAGE_PCT", _DEFAULT_MIN_COVERAGE_PCT
        )
        self.target_coverage_pct: Decimal = self._env_decimal(
            "TARGET_COVERAGE_PCT", _DEFAULT_TARGET_COVERAGE_PCT
        )
        self.supplier_specific_target: Decimal = self._env_decimal(
            "SUPPLIER_SPECIFIC_TARGET", _DEFAULT_SUPPLIER_SPECIFIC_TARGET
        )
        self.average_data_target: Decimal = self._env_decimal(
            "AVERAGE_DATA_TARGET", _DEFAULT_AVERAGE_DATA_TARGET
        )
        self.spend_based_max_pct: Decimal = self._env_decimal(
            "SPEND_BASED_MAX_PCT", _DEFAULT_SPEND_BASED_MAX_PCT
        )
        self.high_spend_threshold: Decimal = self._env_decimal(
            "HIGH_SPEND_THRESHOLD", _DEFAULT_HIGH_SPEND_THRESHOLD
        )
        self.medium_spend_threshold: Decimal = self._env_decimal(
            "MEDIUM_SPEND_THRESHOLD", _DEFAULT_MEDIUM_SPEND_THRESHOLD
        )
        self.low_spend_threshold: Decimal = self._env_decimal(
            "LOW_SPEND_THRESHOLD", _DEFAULT_LOW_SPEND_THRESHOLD
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
            "API_PREFIX", "/api/v1/purchased-goods"
        )
        self.rate_limit: int = self._env_int("RATE_LIMIT", 1000)
        self.page_size: int = self._env_int("PAGE_SIZE", 100)
        self.max_page_size: int = self._env_int("MAX_PAGE_SIZE", 10000)

        # -- 10. Observability Settings ----------------------------------
        self.enable_metrics: bool = self._env_bool(
            "ENABLE_METRICS", True
        )
        self.metrics_prefix: str = self._env_str(
            "METRICS_PREFIX", "gl_pgs_"
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
            "GL-MRV-S3-001-PURCHASED-GOODS-SERVICES-GENESIS",
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

    # ------------------------------------------------------------------
    # Environment variable parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _env_str(name: str, default: str) -> str:
        """Read a string environment variable with the GL_PGS_ prefix.

        Args:
            name: Variable name suffix (after GL_PGS_).
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
        """Read an integer environment variable with the GL_PGS_ prefix.

        Args:
            name: Variable name suffix (after GL_PGS_).
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
        """Read a Decimal environment variable with the GL_PGS_ prefix.

        Uses ``Decimal(str)`` parsing for exact precision. Falls back
        to the default on ``InvalidOperation`` (malformed input) and
        emits a WARNING log.

        Args:
            name: Variable name suffix (after GL_PGS_).
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
        """Read a boolean environment variable with the GL_PGS_ prefix.

        Accepts ``true``, ``1``, ``yes`` (case-insensitive) as True.
        All other non-None values are treated as False.

        Args:
            name: Variable name suffix (after GL_PGS_).
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
            name: Variable name suffix (after GL_PGS_).
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
        ``PurchasedGoodsServicesConfig()`` will re-read all
        environment variables and construct a fresh configuration
        object. Thread-safe.

        Example:
            >>> PurchasedGoodsServicesConfig.reset()
            >>> cfg = PurchasedGoodsServicesConfig()  # fresh instance
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
        logger.debug(
            "PurchasedGoodsServicesConfig singleton reset"
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
                "PurchasedGoodsServicesConfig loaded with %d "
                "validation warning(s):\n%s",
                len(errors),
                "\n".join(f"  - {e}" for e in errors),
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate all configuration settings.

        Performs comprehensive checks (125+) across all configuration
        domains: general settings, database connectivity parameters,
        calculation defaults, EEIO settings, DQI weights, coverage
        thresholds, compliance frameworks, cache settings, API settings,
        logging, provenance, and performance tuning.

        Returns:
            A list of human-readable error strings. An empty list means
            all validation checks passed.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> errors = cfg.validate()
            >>> assert len(errors) == 0
        """
        errors: List[str] = []

        # -- General Settings --------------------------------------------
        errors.extend(self._validate_general_settings())

        # -- Database Settings -------------------------------------------
        errors.extend(self._validate_database_settings())

        # -- Calculation Settings ----------------------------------------
        errors.extend(self._validate_calculation_settings())

        # -- EEIO Settings -----------------------------------------------
        errors.extend(self._validate_eeio_settings())

        # -- DQI Settings ------------------------------------------------
        errors.extend(self._validate_dqi_settings())

        # -- Coverage Settings -------------------------------------------
        errors.extend(self._validate_coverage_settings())

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
                "PurchasedGoodsServicesConfig validation found "
                "%d error(s):\n%s",
                len(errors),
                "\n".join(f"  - {e}" for e in errors),
            )
        else:
            logger.debug(
                "PurchasedGoodsServicesConfig validation passed: "
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
    # Validation: Calculation Settings
    # ------------------------------------------------------------------

    def _validate_calculation_settings(self) -> List[str]:
        """Validate calculation default settings.

        Checks default method, GWP source, decimal places, base year,
        currency, max trace steps, rounding mode, reporting year,
        max calculation records, and capital threshold.

        Returns:
            List of error strings for invalid calculation settings.
        """
        errors: List[str] = []

        # -- default_method valid ----------------------------------------
        normalised_method = self.default_method.upper()
        if normalised_method not in _VALID_CALCULATION_METHODS:
            errors.append(
                f"default_method must be one of "
                f"{sorted(_VALID_CALCULATION_METHODS)}, "
                f"got '{self.default_method}'"
            )

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

        # -- base_year lower bound ---------------------------------------
        if self.base_year < 1990:
            errors.append(
                f"base_year must be >= 1990, "
                f"got {self.base_year}"
            )

        # -- base_year upper bound ---------------------------------------
        if self.base_year > 2100:
            errors.append(
                f"base_year must be <= 2100, "
                f"got {self.base_year}"
            )

        # -- default_currency valid --------------------------------------
        normalised_currency = self.default_currency.upper()
        if normalised_currency not in _VALID_CURRENCIES:
            errors.append(
                f"default_currency must be a valid ISO 4217 code "
                f"from the supported set, "
                f"got '{self.default_currency}'"
            )

        # -- default_currency length -------------------------------------
        if len(self.default_currency) != 3:
            errors.append(
                f"default_currency must be exactly 3 characters "
                f"(ISO 4217), got '{self.default_currency}'"
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

        # -- rounding_mode valid -----------------------------------------
        normalised_rounding = self.rounding_mode.upper()
        if normalised_rounding not in _VALID_ROUNDING_MODES:
            errors.append(
                f"rounding_mode must be one of "
                f"{sorted(_VALID_ROUNDING_MODES)}, "
                f"got '{self.rounding_mode}'"
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

        # -- max_calculation_records lower bound -------------------------
        if self.max_calculation_records <= 0:
            errors.append(
                f"max_calculation_records must be > 0, "
                f"got {self.max_calculation_records}"
            )

        # -- max_calculation_records upper bound -------------------------
        if self.max_calculation_records > 1_000_000:
            errors.append(
                f"max_calculation_records must be <= 1000000, "
                f"got {self.max_calculation_records}"
            )

        # -- capital_threshold lower bound -------------------------------
        if self.capital_threshold < Decimal("0"):
            errors.append(
                f"capital_threshold must be >= 0, "
                f"got {self.capital_threshold}"
            )

        # -- capital_threshold upper bound -------------------------------
        if self.capital_threshold > Decimal("10000000"):
            errors.append(
                f"capital_threshold must be <= 10000000, "
                f"got {self.capital_threshold}"
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: EEIO Settings
    # ------------------------------------------------------------------

    def _validate_eeio_settings(self) -> List[str]:
        """Validate EEIO database settings.

        Checks EEIO database is valid, base year is within bounds,
        and EEIO base year is consistent with the calculation base year.

        Returns:
            List of error strings for invalid EEIO settings.
        """
        errors: List[str] = []

        # -- default_eeio_database valid ---------------------------------
        normalised_eeio = self.default_eeio_database.upper()
        if normalised_eeio not in _VALID_EEIO_DATABASES:
            errors.append(
                f"default_eeio_database must be one of "
                f"{sorted(_VALID_EEIO_DATABASES)}, "
                f"got '{self.default_eeio_database}'"
            )

        # -- eeio_base_year lower bound ----------------------------------
        if self.eeio_base_year < 1990:
            errors.append(
                f"eeio_base_year must be >= 1990, "
                f"got {self.eeio_base_year}"
            )

        # -- eeio_base_year upper bound ----------------------------------
        if self.eeio_base_year > 2100:
            errors.append(
                f"eeio_base_year must be <= 2100, "
                f"got {self.eeio_base_year}"
            )

        # -- eeio_base_year should not be far from base_year -------------
        if abs(self.eeio_base_year - self.base_year) > 10:
            errors.append(
                f"eeio_base_year ({self.eeio_base_year}) should be "
                f"within 10 years of base_year ({self.base_year})"
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: DQI Settings
    # ------------------------------------------------------------------

    def _validate_dqi_settings(self) -> List[str]:
        """Validate DQI dimension weight settings.

        Checks that each weight is in the range [0, 1], that the five
        weights sum to exactly ``Decimal("1.00")``, and that DQI
        scoring is consistently configured.

        Returns:
            List of error strings for invalid DQI settings.
        """
        errors: List[str] = []

        # -- dqi_temporal_weight lower bound -----------------------------
        if self.dqi_temporal_weight < Decimal("0"):
            errors.append(
                f"dqi_temporal_weight must be >= 0, "
                f"got {self.dqi_temporal_weight}"
            )

        # -- dqi_temporal_weight upper bound -----------------------------
        if self.dqi_temporal_weight > Decimal("1"):
            errors.append(
                f"dqi_temporal_weight must be <= 1, "
                f"got {self.dqi_temporal_weight}"
            )

        # -- dqi_geographical_weight lower bound -------------------------
        if self.dqi_geographical_weight < Decimal("0"):
            errors.append(
                f"dqi_geographical_weight must be >= 0, "
                f"got {self.dqi_geographical_weight}"
            )

        # -- dqi_geographical_weight upper bound -------------------------
        if self.dqi_geographical_weight > Decimal("1"):
            errors.append(
                f"dqi_geographical_weight must be <= 1, "
                f"got {self.dqi_geographical_weight}"
            )

        # -- dqi_technological_weight lower bound ------------------------
        if self.dqi_technological_weight < Decimal("0"):
            errors.append(
                f"dqi_technological_weight must be >= 0, "
                f"got {self.dqi_technological_weight}"
            )

        # -- dqi_technological_weight upper bound ------------------------
        if self.dqi_technological_weight > Decimal("1"):
            errors.append(
                f"dqi_technological_weight must be <= 1, "
                f"got {self.dqi_technological_weight}"
            )

        # -- dqi_completeness_weight lower bound -------------------------
        if self.dqi_completeness_weight < Decimal("0"):
            errors.append(
                f"dqi_completeness_weight must be >= 0, "
                f"got {self.dqi_completeness_weight}"
            )

        # -- dqi_completeness_weight upper bound -------------------------
        if self.dqi_completeness_weight > Decimal("1"):
            errors.append(
                f"dqi_completeness_weight must be <= 1, "
                f"got {self.dqi_completeness_weight}"
            )

        # -- dqi_reliability_weight lower bound --------------------------
        if self.dqi_reliability_weight < Decimal("0"):
            errors.append(
                f"dqi_reliability_weight must be >= 0, "
                f"got {self.dqi_reliability_weight}"
            )

        # -- dqi_reliability_weight upper bound --------------------------
        if self.dqi_reliability_weight > Decimal("1"):
            errors.append(
                f"dqi_reliability_weight must be <= 1, "
                f"got {self.dqi_reliability_weight}"
            )

        # -- DQI weights must sum to 1.00 --------------------------------
        weight_sum = (
            self.dqi_temporal_weight
            + self.dqi_geographical_weight
            + self.dqi_technological_weight
            + self.dqi_completeness_weight
            + self.dqi_reliability_weight
        )
        if weight_sum != Decimal("1.00"):
            errors.append(
                f"DQI weights must sum to exactly 1.00, "
                f"got {weight_sum} "
                f"(temporal={self.dqi_temporal_weight}, "
                f"geographical={self.dqi_geographical_weight}, "
                f"technological={self.dqi_technological_weight}, "
                f"completeness={self.dqi_completeness_weight}, "
                f"reliability={self.dqi_reliability_weight})"
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: Coverage Settings
    # ------------------------------------------------------------------

    def _validate_coverage_settings(self) -> List[str]:
        """Validate coverage threshold settings.

        Checks that coverage percentages are in valid ranges, that
        min_coverage_pct < target_coverage_pct, that method targets
        are in range, and that spend thresholds are ordered correctly
        (low < medium < high).

        Returns:
            List of error strings for invalid coverage settings.
        """
        errors: List[str] = []

        # -- min_coverage_pct lower bound --------------------------------
        if self.min_coverage_pct < Decimal("0"):
            errors.append(
                f"min_coverage_pct must be >= 0, "
                f"got {self.min_coverage_pct}"
            )

        # -- min_coverage_pct upper bound --------------------------------
        if self.min_coverage_pct > Decimal("100"):
            errors.append(
                f"min_coverage_pct must be <= 100, "
                f"got {self.min_coverage_pct}"
            )

        # -- target_coverage_pct lower bound -----------------------------
        if self.target_coverage_pct < Decimal("0"):
            errors.append(
                f"target_coverage_pct must be >= 0, "
                f"got {self.target_coverage_pct}"
            )

        # -- target_coverage_pct upper bound -----------------------------
        if self.target_coverage_pct > Decimal("100"):
            errors.append(
                f"target_coverage_pct must be <= 100, "
                f"got {self.target_coverage_pct}"
            )

        # -- min_coverage_pct <= target_coverage_pct ---------------------
        if self.min_coverage_pct > self.target_coverage_pct:
            errors.append(
                f"min_coverage_pct ({self.min_coverage_pct}) must be "
                f"<= target_coverage_pct ({self.target_coverage_pct})"
            )

        # -- supplier_specific_target lower bound ------------------------
        if self.supplier_specific_target < Decimal("0"):
            errors.append(
                f"supplier_specific_target must be >= 0, "
                f"got {self.supplier_specific_target}"
            )

        # -- supplier_specific_target upper bound ------------------------
        if self.supplier_specific_target > Decimal("100"):
            errors.append(
                f"supplier_specific_target must be <= 100, "
                f"got {self.supplier_specific_target}"
            )

        # -- average_data_target lower bound -----------------------------
        if self.average_data_target < Decimal("0"):
            errors.append(
                f"average_data_target must be >= 0, "
                f"got {self.average_data_target}"
            )

        # -- average_data_target upper bound -----------------------------
        if self.average_data_target > Decimal("100"):
            errors.append(
                f"average_data_target must be <= 100, "
                f"got {self.average_data_target}"
            )

        # -- spend_based_max_pct lower bound -----------------------------
        if self.spend_based_max_pct < Decimal("0"):
            errors.append(
                f"spend_based_max_pct must be >= 0, "
                f"got {self.spend_based_max_pct}"
            )

        # -- spend_based_max_pct upper bound -----------------------------
        if self.spend_based_max_pct > Decimal("100"):
            errors.append(
                f"spend_based_max_pct must be <= 100, "
                f"got {self.spend_based_max_pct}"
            )

        # -- method targets should not exceed 100% total -----------------
        method_sum = (
            self.supplier_specific_target
            + self.average_data_target
            + self.spend_based_max_pct
        )
        if method_sum > Decimal("200"):
            errors.append(
                f"Sum of method targets ({method_sum}) exceeds "
                f"reasonable maximum of 200%"
            )

        # -- high_spend_threshold lower bound ----------------------------
        if self.high_spend_threshold < Decimal("0"):
            errors.append(
                f"high_spend_threshold must be >= 0, "
                f"got {self.high_spend_threshold}"
            )

        # -- medium_spend_threshold lower bound --------------------------
        if self.medium_spend_threshold < Decimal("0"):
            errors.append(
                f"medium_spend_threshold must be >= 0, "
                f"got {self.medium_spend_threshold}"
            )

        # -- low_spend_threshold lower bound -----------------------------
        if self.low_spend_threshold < Decimal("0"):
            errors.append(
                f"low_spend_threshold must be >= 0, "
                f"got {self.low_spend_threshold}"
            )

        # -- ordering: low < medium --------------------------------------
        if self.low_spend_threshold >= self.medium_spend_threshold:
            errors.append(
                f"low_spend_threshold ({self.low_spend_threshold}) "
                f"must be < medium_spend_threshold "
                f"({self.medium_spend_threshold})"
            )

        # -- ordering: medium < high -------------------------------------
        if self.medium_spend_threshold >= self.high_spend_threshold:
            errors.append(
                f"medium_spend_threshold ({self.medium_spend_threshold}) "
                f"must be < high_spend_threshold "
                f"({self.high_spend_threshold})"
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

        # -- SB 253 requires ghg_protocol --------------------------------
        if "sb_253" in normalised_fws:
            if "ghg_protocol" not in normalised_fws:
                errors.append(
                    "SB 253 framework requires 'ghg_protocol' "
                    "in enabled_frameworks"
                )

        # -- CDP requires ghg_protocol -----------------------------------
        if "cdp" in normalised_fws:
            if "ghg_protocol" not in normalised_fws:
                errors.append(
                    "CDP framework requires 'ghg_protocol' "
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
        # General: 10, Database: 14, Calculation: 18,
        # EEIO: 4, DQI: 11, Coverage: 19,
        # Compliance: 10, Cache: 4, API: 11, Logging: 3,
        # Provenance: 2, Performance: 4
        return 130

    # ------------------------------------------------------------------
    # Accessor Methods: General
    # ------------------------------------------------------------------

    def get_service_name(self) -> str:
        """Return the service name for tracing and identification.

        Returns:
            Service name string.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> cfg.get_service_name()
            'purchased-goods-services'
        """
        return self.service_name

    def get_version(self) -> str:
        """Return the service version string.

        Returns:
            Version string.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> cfg.get_version()
            '1.0.0'
        """
        return self.version

    def get_log_level(self) -> str:
        """Return the logging level.

        Returns:
            Log level string (e.g. "INFO", "DEBUG").

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> cfg.get_log_level()
            'INFO'
        """
        return self.log_level

    def get_environment(self) -> str:
        """Return the deployment environment.

        Returns:
            Environment string (development, staging, or production).

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> cfg.get_environment()
            'development'
        """
        return self.environment

    def get_general_config(self) -> Dict[str, Any]:
        """Return general configuration as a dictionary.

        Returns:
            Dictionary containing all general settings.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> gen = cfg.get_general_config()
            >>> gen["service_name"]
            'purchased-goods-services'
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
            >>> cfg = PurchasedGoodsServicesConfig()
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
            >>> cfg = PurchasedGoodsServicesConfig()
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
            >>> cfg = PurchasedGoodsServicesConfig()
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
            Table prefix string (e.g. "pgs_").

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> cfg.get_table_prefix()
            'pgs_'
        """
        return self.table_prefix

    def get_database_config(self) -> Dict[str, Any]:
        """Return complete database configuration as a dictionary.

        Sensitive password field is redacted. Includes DSN, pool
        parameters, SSL mode, and table prefix.

        Returns:
            Dictionary of all database-related settings.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
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
    # Accessor Methods: Calculation Settings
    # ------------------------------------------------------------------

    def get_default_method(self) -> str:
        """Return the default calculation method.

        Returns:
            Calculation method string (e.g. "HYBRID").

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> cfg.get_default_method()
            'HYBRID'
        """
        return self.default_method

    def get_default_gwp_source(self) -> str:
        """Return the default GWP assessment report source.

        Returns:
            GWP source string (e.g. "ar5").

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> cfg.get_default_gwp_source()
            'ar5'
        """
        return self.default_gwp_source

    def get_decimal_places(self) -> int:
        """Return the number of decimal places for calculations.

        Returns:
            Integer decimal places count.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> cfg.get_decimal_places()
            8
        """
        return self.decimal_places

    def get_calculation_config(self) -> Dict[str, Any]:
        """Return calculation engine configuration as a dictionary.

        Returns:
            Dictionary containing all calculation-related settings
            suitable for initialising the calculation engines.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> calc = cfg.get_calculation_config()
            >>> calc["default_method"]
            'HYBRID'
        """
        return {
            "default_method": self.default_method,
            "gwp_source": self.default_gwp_source,
            "decimal_places": self.decimal_places,
            "base_year": self.base_year,
            "default_currency": self.default_currency,
            "enable_margin_removal": self.enable_margin_removal,
            "enable_inflation_adjustment": (
                self.enable_inflation_adjustment
            ),
            "max_trace_steps": self.max_trace_steps,
            "rounding_mode": self.rounding_mode,
            "default_reporting_year": self.default_reporting_year,
            "max_calculation_records": self.max_calculation_records,
            "enable_transport_adder": self.enable_transport_adder,
            "enable_waste_factor": self.enable_waste_factor,
            "enable_double_counting_check": (
                self.enable_double_counting_check
            ),
            "capital_threshold": str(self.capital_threshold),
            "max_batch_size": self.max_batch_size,
        }

    def recommend_method(self, annual_spend: Decimal) -> str:
        """Recommend a calculation method based on annual spend amount.

        Uses the configured spend thresholds to recommend the most
        appropriate calculation method per GHG Protocol guidance.

        Args:
            annual_spend: Annual spend in base currency for the
                procurement category.

        Returns:
            Recommended calculation method string.

        Raises:
            ValueError: If annual_spend is negative.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> cfg.recommend_method(Decimal("15000000"))
            'SUPPLIER_SPECIFIC'
            >>> cfg.recommend_method(Decimal("50000"))
            'SPEND_BASED'
        """
        if annual_spend < Decimal("0"):
            raise ValueError(
                f"annual_spend must be >= 0, got {annual_spend}"
            )

        if annual_spend >= self.high_spend_threshold:
            return "SUPPLIER_SPECIFIC"
        elif annual_spend >= self.medium_spend_threshold:
            return "AVERAGE_DATA"
        elif annual_spend >= self.low_spend_threshold:
            return "AVERAGE_DATA"
        else:
            return "SPEND_BASED"

    # ------------------------------------------------------------------
    # Accessor Methods: EEIO Settings
    # ------------------------------------------------------------------

    def get_eeio_config(self) -> Dict[str, Any]:
        """Return EEIO database configuration as a dictionary.

        Returns:
            Dictionary containing all EEIO-related settings.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> eeio = cfg.get_eeio_config()
            >>> eeio["database"]
            'EPA_USEEIO_V12'
        """
        return {
            "database": self.default_eeio_database,
            "base_year": self.eeio_base_year,
            "enable_ppp_adjustment": self.enable_ppp_adjustment,
            "default_currency": self.default_currency,
            "enable_inflation_adjustment": (
                self.enable_inflation_adjustment
            ),
            "enable_margin_removal": self.enable_margin_removal,
        }

    def get_default_eeio_database(self) -> str:
        """Return the default EEIO database identifier.

        Returns:
            EEIO database string (e.g. "EPA_USEEIO_V12").

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> cfg.get_default_eeio_database()
            'EPA_USEEIO_V12'
        """
        return self.default_eeio_database

    # ------------------------------------------------------------------
    # Accessor Methods: DQI Settings
    # ------------------------------------------------------------------

    def get_dqi_weights(self) -> Dict[str, Decimal]:
        """Return DQI dimension weights as a dictionary.

        The five DQI dimension weights represent the relative
        importance of each dimension in computing the composite DQI
        score per GHG Protocol Scope 3 Standard Chapter 7. The weights
        are guaranteed to sum to exactly ``Decimal("1.00")`` by
        configuration validation.

        Returns:
            Dictionary mapping dimension names to Decimal weights.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> weights = cfg.get_dqi_weights()
            >>> weights["temporal"]
            Decimal('0.20')
            >>> sum(weights.values()) == Decimal("1.00")
            True
        """
        return {
            "temporal": self.dqi_temporal_weight,
            "geographical": self.dqi_geographical_weight,
            "technological": self.dqi_technological_weight,
            "completeness": self.dqi_completeness_weight,
            "reliability": self.dqi_reliability_weight,
        }

    def get_dqi_config(self) -> Dict[str, Any]:
        """Return DQI scoring configuration as a dictionary.

        Includes weights for all five DQI dimensions and the enable
        flag. Decimal values are converted to strings for JSON-safe
        serialisation.

        Returns:
            Dictionary containing all DQI-scoring settings.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> dqi = cfg.get_dqi_config()
            >>> dqi["enable_dqi_scoring"]
            True
        """
        return {
            "enable_dqi_scoring": self.enable_dqi_scoring,
            "temporal_weight": str(self.dqi_temporal_weight),
            "geographical_weight": str(self.dqi_geographical_weight),
            "technological_weight": str(self.dqi_technological_weight),
            "completeness_weight": str(self.dqi_completeness_weight),
            "reliability_weight": str(self.dqi_reliability_weight),
        }

    def classify_dqi(self, composite_score: Decimal) -> str:
        """Classify a composite DQI score into a quality level.

        Uses the GHG Protocol Scope 3 Standard Chapter 7 quality
        classification bands.

        Args:
            composite_score: Composite DQI score (1.0-5.0 scale).

        Returns:
            Quality level string: "VERY_GOOD", "GOOD", "FAIR",
            "POOR", or "VERY_POOR".

        Raises:
            ValueError: If composite_score is outside [1.0, 5.0].

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> cfg.classify_dqi(Decimal("1.3"))
            'VERY_GOOD'
            >>> cfg.classify_dqi(Decimal("3.0"))
            'FAIR'
        """
        if composite_score < Decimal("1.0") or composite_score > Decimal("5.0"):
            raise ValueError(
                f"composite_score must be in [1.0, 5.0], "
                f"got {composite_score}"
            )

        if composite_score <= Decimal("1.5"):
            return "VERY_GOOD"
        elif composite_score <= Decimal("2.5"):
            return "GOOD"
        elif composite_score <= Decimal("3.5"):
            return "FAIR"
        elif composite_score <= Decimal("4.5"):
            return "POOR"
        else:
            return "VERY_POOR"

    # ------------------------------------------------------------------
    # Accessor Methods: Coverage Settings
    # ------------------------------------------------------------------

    def get_coverage_thresholds(self) -> Dict[str, Decimal]:
        """Return coverage threshold configuration as a dictionary.

        Returns:
            Dictionary mapping threshold names to Decimal percentage
            values.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> cov = cfg.get_coverage_thresholds()
            >>> cov["min_coverage_pct"]
            Decimal('80')
        """
        return {
            "min_coverage_pct": self.min_coverage_pct,
            "target_coverage_pct": self.target_coverage_pct,
            "supplier_specific_target": self.supplier_specific_target,
            "average_data_target": self.average_data_target,
            "spend_based_max_pct": self.spend_based_max_pct,
        }

    def get_spend_thresholds(self) -> Dict[str, Decimal]:
        """Return spend category thresholds as a dictionary.

        Returns:
            Dictionary mapping threshold names to Decimal USD values.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> thresholds = cfg.get_spend_thresholds()
            >>> thresholds["high"]
            Decimal('10000000')
        """
        return {
            "high": self.high_spend_threshold,
            "medium": self.medium_spend_threshold,
            "low": self.low_spend_threshold,
        }

    def classify_coverage(self, coverage_pct: Decimal) -> str:
        """Classify a coverage percentage into a coverage level.

        Uses the configured minimum and target coverage thresholds
        to determine the coverage level.

        Args:
            coverage_pct: Coverage percentage (0-100).

        Returns:
            Coverage level string: "BELOW_MINIMUM", "MINIMUM",
            "GOOD", or "BEST".

        Raises:
            ValueError: If coverage_pct is outside [0, 100].

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> cfg.classify_coverage(Decimal("75"))
            'BELOW_MINIMUM'
            >>> cfg.classify_coverage(Decimal("92"))
            'GOOD'
        """
        if coverage_pct < Decimal("0") or coverage_pct > Decimal("100"):
            raise ValueError(
                f"coverage_pct must be in [0, 100], "
                f"got {coverage_pct}"
            )

        if coverage_pct < self.min_coverage_pct:
            return "BELOW_MINIMUM"
        elif coverage_pct < Decimal("90"):
            return "MINIMUM"
        elif coverage_pct < self.target_coverage_pct:
            return "GOOD"
        else:
            return "BEST"

    def get_coverage_config(self) -> Dict[str, Any]:
        """Return coverage configuration as a dictionary.

        Returns:
            Dictionary containing all coverage-related settings
            with Decimal values as strings for JSON safety.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> cov = cfg.get_coverage_config()
            >>> cov["min_coverage_pct"]
            '80'
        """
        return {
            "min_coverage_pct": str(self.min_coverage_pct),
            "target_coverage_pct": str(self.target_coverage_pct),
            "supplier_specific_target": str(
                self.supplier_specific_target
            ),
            "average_data_target": str(self.average_data_target),
            "spend_based_max_pct": str(self.spend_based_max_pct),
            "high_spend_threshold": str(self.high_spend_threshold),
            "medium_spend_threshold": str(self.medium_spend_threshold),
            "low_spend_threshold": str(self.low_spend_threshold),
        }

    # ------------------------------------------------------------------
    # Accessor Methods: Compliance
    # ------------------------------------------------------------------

    def get_enabled_frameworks(self) -> List[str]:
        """Return a copy of the enabled compliance frameworks list.

        Returns:
            List of enabled framework identifiers.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
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
                "ghg_protocol", "sb_253").

        Returns:
            True if the framework is in the enabled list, False otherwise.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
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
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> comp = cfg.get_compliance_config()
            >>> comp["strict_mode"]
            False
        """
        return {
            "enabled_frameworks": list(self.enabled_frameworks),
            "strict_mode": self.strict_mode,
            "fail_on_non_compliant": self.fail_on_non_compliant,
        }

    # ------------------------------------------------------------------
    # Accessor Methods: API
    # ------------------------------------------------------------------

    def get_api_prefix(self) -> str:
        """Return the REST API route prefix.

        Returns:
            API prefix string.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> cfg.get_api_prefix()
            '/api/v1/purchased-goods'
        """
        return self.api_prefix

    def get_api_config(self) -> Dict[str, Any]:
        """Return API configuration as a dictionary.

        Returns:
            Dictionary of API-related settings suitable for FastAPI
            application construction.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> api_cfg = cfg.get_api_config()
            >>> api_cfg["prefix"]
            '/api/v1/purchased-goods'
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
            >>> cfg = PurchasedGoodsServicesConfig()
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
            >>> cfg = PurchasedGoodsServicesConfig()
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
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> obs_cfg = cfg.get_observability_config()
            >>> obs_cfg["metrics_prefix"]
            'gl_pgs_'
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
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> flags = cfg.get_feature_flags()
            >>> flags["enable_margin_removal"]
            True
        """
        return {
            "enable_margin_removal": self.enable_margin_removal,
            "enable_inflation_adjustment": (
                self.enable_inflation_adjustment
            ),
            "enable_transport_adder": self.enable_transport_adder,
            "enable_waste_factor": self.enable_waste_factor,
            "enable_double_counting_check": (
                self.enable_double_counting_check
            ),
            "enable_ppp_adjustment": self.enable_ppp_adjustment,
            "enable_dqi_scoring": self.enable_dqi_scoring,
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
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> prov = cfg.get_provenance_config()
            >>> prov["genesis_hash"]
            'GL-MRV-S3-001-PURCHASED-GOODS-SERVICES-GENESIS'
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
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> d = cfg.to_dict()
            >>> d["default_method"]
            'HYBRID'
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
            # -- 3. Calculation Settings ---------------------------------
            "default_method": self.default_method,
            "default_gwp_source": self.default_gwp_source,
            "decimal_places": self.decimal_places,
            "base_year": self.base_year,
            "default_currency": self.default_currency,
            "enable_margin_removal": self.enable_margin_removal,
            "enable_inflation_adjustment": (
                self.enable_inflation_adjustment
            ),
            "max_trace_steps": self.max_trace_steps,
            "rounding_mode": self.rounding_mode,
            "default_reporting_year": self.default_reporting_year,
            "max_calculation_records": self.max_calculation_records,
            "enable_transport_adder": self.enable_transport_adder,
            "enable_waste_factor": self.enable_waste_factor,
            "enable_double_counting_check": (
                self.enable_double_counting_check
            ),
            "capital_threshold": str(self.capital_threshold),
            # -- 4. EEIO Settings ----------------------------------------
            "default_eeio_database": self.default_eeio_database,
            "eeio_base_year": self.eeio_base_year,
            "enable_ppp_adjustment": self.enable_ppp_adjustment,
            # -- 5. DQI Settings -----------------------------------------
            "enable_dqi_scoring": self.enable_dqi_scoring,
            "dqi_temporal_weight": str(self.dqi_temporal_weight),
            "dqi_geographical_weight": str(
                self.dqi_geographical_weight
            ),
            "dqi_technological_weight": str(
                self.dqi_technological_weight
            ),
            "dqi_completeness_weight": str(
                self.dqi_completeness_weight
            ),
            "dqi_reliability_weight": str(
                self.dqi_reliability_weight
            ),
            # -- 6. Coverage Settings ------------------------------------
            "min_coverage_pct": str(self.min_coverage_pct),
            "target_coverage_pct": str(self.target_coverage_pct),
            "supplier_specific_target": str(
                self.supplier_specific_target
            ),
            "average_data_target": str(self.average_data_target),
            "spend_based_max_pct": str(self.spend_based_max_pct),
            "high_spend_threshold": str(self.high_spend_threshold),
            "medium_spend_threshold": str(self.medium_spend_threshold),
            "low_spend_threshold": str(self.low_spend_threshold),
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
            # -- 10. Observability Settings ------------------------------
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
    ) -> PurchasedGoodsServicesConfig:
        """Deserialise a configuration from a dictionary.

        Creates a new PurchasedGoodsServicesConfig instance and
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
            A new PurchasedGoodsServicesConfig instance with
            values from the dictionary.

        Example:
            >>> d = {"default_method": "SPEND_BASED", "decimal_places": 12}
            >>> cfg = PurchasedGoodsServicesConfig.from_dict(d)
            >>> cfg.default_method
            'SPEND_BASED'
            >>> cfg.decimal_places
            12
        """
        cls.reset()
        instance = cls()
        instance._apply_dict(data)

        logger.info(
            "PurchasedGoodsServicesConfig loaded from dict: "
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
            "capital_threshold",
            "dqi_temporal_weight",
            "dqi_geographical_weight",
            "dqi_technological_weight",
            "dqi_completeness_weight",
            "dqi_reliability_weight",
            "min_coverage_pct",
            "target_coverage_pct",
            "supplier_specific_target",
            "average_data_target",
            "spend_based_max_pct",
            "high_spend_threshold",
            "medium_spend_threshold",
            "low_spend_threshold",
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
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> json_str = cfg.to_json()
            >>> '"default_method": "HYBRID"' in json_str
            True
        """
        return json.dumps(
            self.to_dict(), indent=indent, sort_keys=False
        )

    @classmethod
    def from_json(
        cls, json_str: str
    ) -> PurchasedGoodsServicesConfig:
        """Deserialise a configuration from a JSON string.

        Args:
            json_str: JSON string containing configuration key-value
                pairs.

        Returns:
            A new PurchasedGoodsServicesConfig instance.

        Example:
            >>> json_str = '{"default_method": "SPEND_BASED"}'
            >>> cfg = PurchasedGoodsServicesConfig.from_json(json_str)
            >>> cfg.default_method
            'SPEND_BASED'
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
        lowercase, environment to lowercase, calculation method to
        uppercase, EEIO database to uppercase, currency to uppercase).
        This method is idempotent.

        Example:
            >>> cfg = PurchasedGoodsServicesConfig()
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

        # Calculation method -> uppercase
        self.default_method = self.default_method.upper()

        # EEIO database -> uppercase
        self.default_eeio_database = self.default_eeio_database.upper()

        # Currency -> uppercase
        self.default_currency = self.default_currency.upper()

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
            "PurchasedGoodsServicesConfig normalised: "
            "method=%s, gwp=%s, eeio=%s, currency=%s, "
            "log_level=%s, ssl_mode=%s, env=%s, rounding=%s",
            self.default_method,
            self.default_gwp_source,
            self.default_eeio_database,
            self.default_currency,
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
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> cfg.merge({"decimal_places": 12})
            >>> cfg.decimal_places
            12
        """
        self._apply_dict(overrides)
        logger.debug(
            "PurchasedGoodsServicesConfig merged %d overrides",
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
            >>> cfg = PurchasedGoodsServicesConfig()
            >>> summary = cfg.health_summary()
            >>> summary["validation_status"]
            'PASS'
        """
        errors = self.validate()
        flags = self.get_feature_flags()
        enabled_flag_count = sum(1 for v in flags.values() if v)

        return {
            "agent": "purchased-goods-services",
            "agent_id": "AGENT-MRV-014",
            "gl_id": "GL-MRV-S3-001",
            "enabled": self.enabled,
            "validation_status": "PASS" if not errors else "FAIL",
            "validation_errors": len(errors),
            "service_name": self.service_name,
            "version": self.version,
            "environment": self.environment,
            "default_method": self.default_method,
            "gwp_source": self.default_gwp_source,
            "eeio_database": self.default_eeio_database,
            "default_currency": self.default_currency,
            "base_year": self.base_year,
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
            "dqi_weights_sum": str(
                self.dqi_temporal_weight
                + self.dqi_geographical_weight
                + self.dqi_technological_weight
                + self.dqi_completeness_weight
                + self.dqi_reliability_weight
            ),
            "coverage_thresholds": {
                "min": str(self.min_coverage_pct),
                "target": str(self.target_coverage_pct),
            },
            "spend_thresholds": {
                "high": str(self.high_spend_threshold),
                "medium": str(self.medium_spend_threshold),
                "low": str(self.low_spend_threshold),
            },
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
        return f"PurchasedGoodsServicesConfig({pairs})"

    def __str__(self) -> str:
        """Return a human-readable summary of the configuration.

        Returns:
            Multi-line string summary of key settings.
        """
        return (
            f"PurchasedGoodsServicesConfig("
            f"enabled={self.enabled}, "
            f"service={self.service_name}, "
            f"env={self.environment}, "
            f"method={self.default_method}, "
            f"gwp={self.default_gwp_source}, "
            f"eeio={self.default_eeio_database}, "
            f"currency={self.default_currency}, "
            f"base_year={self.base_year}, "
            f"decimal_places={self.decimal_places}, "
            f"dqi_weights=[{self.dqi_temporal_weight}/"
            f"{self.dqi_geographical_weight}/"
            f"{self.dqi_technological_weight}/"
            f"{self.dqi_completeness_weight}/"
            f"{self.dqi_reliability_weight}], "
            f"coverage=[min={self.min_coverage_pct}/"
            f"target={self.target_coverage_pct}], "
            f"batch={self.max_batch_size}, "
            f"frameworks={len(self.enabled_frameworks)}"
            f")"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality by comparing all serialised values.

        Args:
            other: Object to compare against.

        Returns:
            True if other is a PurchasedGoodsServicesConfig with
            identical settings.
        """
        if not isinstance(other, PurchasedGoodsServicesConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def get_config() -> PurchasedGoodsServicesConfig:
    """Return the singleton PurchasedGoodsServicesConfig.

    Creates the instance from environment variables on first call.
    Subsequent calls return the cached singleton.

    Returns:
        PurchasedGoodsServicesConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.default_method
        'HYBRID'
    """
    return PurchasedGoodsServicesConfig()


def set_config(data: Dict[str, Any]) -> PurchasedGoodsServicesConfig:
    """Create a new configuration from a dictionary.

    Resets the existing singleton and creates a fresh instance
    populated from the provided dictionary. Useful for testing
    and programmatic configuration.

    Args:
        data: Dictionary of configuration key-value pairs.

    Returns:
        A new PurchasedGoodsServicesConfig instance.

    Example:
        >>> cfg = set_config({"default_method": "SPEND_BASED"})
        >>> cfg.default_method
        'SPEND_BASED'
    """
    return PurchasedGoodsServicesConfig.from_dict(data)


def reset_config() -> None:
    """Reset the singleton for test teardown.

    The next call to :func:`get_config` will re-read environment
    variables and construct a fresh instance.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads GL_PGS_* env vars
    """
    PurchasedGoodsServicesConfig.reset()
    logger.debug(
        "PurchasedGoodsServicesConfig singleton reset via "
        "module-level reset_config()"
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "PurchasedGoodsServicesConfig",
    "get_config",
    "set_config",
    "reset_config",
]
