# -*- coding: utf-8 -*-
"""
Scope 2 Market-Based Emissions Agent Configuration - AGENT-MRV-010

Centralized configuration for the Scope 2 Market-Based Emissions Agent SDK
covering:
- Database connection and pool settings (PostgreSQL)
- Redis caching configuration
- Calculation settings (GWP source, allocation method, precision, vintage)
- Instrument settings (certificate retirement, tracking systems, unbundled)
- Residual mix factor settings (AIB, cache TTL, auto-update)
- Uncertainty quantification (Monte Carlo) parameters
- API routing, rate limiting, and CORS
- Regulatory compliance framework toggles (GHG Protocol, RE100, SBTi, etc.)
- Logging and observability (Prometheus metrics, OpenTelemetry tracing)
- Feature flags (dual reporting, certificate retirement, supplier factors,
  uncertainty)

This module implements a singleton pattern using ``__new__`` with a
class-level ``_instance`` and ``_initialized`` flag, ensuring exactly one
configuration object exists across the application lifecycle. All settings
can be overridden via environment variables with the ``GL_S2M_`` prefix.

Environment Variable Reference (GL_S2M_ prefix):
    GL_S2M_DB_HOST                      - PostgreSQL host
    GL_S2M_DB_PORT                      - PostgreSQL port
    GL_S2M_DB_NAME                      - PostgreSQL database name
    GL_S2M_DB_USER                      - PostgreSQL username
    GL_S2M_DB_PASSWORD                  - PostgreSQL password
    GL_S2M_DB_POOL_MIN                  - Minimum connection pool size
    GL_S2M_DB_POOL_MAX                  - Maximum connection pool size
    GL_S2M_DB_SSL_MODE                  - PostgreSQL SSL mode
    GL_S2M_REDIS_HOST                   - Redis host
    GL_S2M_REDIS_PORT                   - Redis port
    GL_S2M_REDIS_DB                     - Redis database number
    GL_S2M_REDIS_PASSWORD               - Redis password
    GL_S2M_REDIS_TTL                    - Redis default TTL (seconds)
    GL_S2M_DEFAULT_GWP_SOURCE           - Default GWP source (AR4/AR5/AR6)
    GL_S2M_DEFAULT_ALLOCATION_METHOD    - Allocation method for instruments
    GL_S2M_DECIMAL_PRECISION            - Decimal places for calculations
    GL_S2M_ROUNDING_MODE                - Decimal rounding mode
    GL_S2M_MAX_BATCH_SIZE               - Maximum records per batch
    GL_S2M_DEFAULT_VINTAGE_WINDOW       - Max vintage year window (years)
    GL_S2M_AUTO_RETIRE_ON_USE           - Auto-retire instruments on use
    GL_S2M_REQUIRE_TRACKING_SYSTEM      - Require tracking system for certs
    GL_S2M_ALLOW_UNBUNDLED_CERTS        - Allow unbundled certificates
    GL_S2M_MAX_INSTRUMENTS_PER_PURCHASE - Max instruments per purchase
    GL_S2M_VALIDATE_GEOGRAPHIC_MATCH    - Validate geographic match for certs
    GL_S2M_DEFAULT_RESIDUAL_SOURCE      - Default residual mix source
    GL_S2M_RESIDUAL_CACHE_TTL           - Residual mix factor cache TTL
    GL_S2M_AUTO_UPDATE_FACTORS          - Auto-update residual mix factors
    GL_S2M_DEFAULT_MC_ITERATIONS        - Monte Carlo iterations
    GL_S2M_DEFAULT_CONFIDENCE_LEVEL     - Confidence level (0.0-1.0)
    GL_S2M_RESIDUAL_UNCERTAINTY_PCT     - Residual mix uncertainty %
    GL_S2M_INSTRUMENT_UNCERTAINTY_PCT   - Instrument uncertainty %
    GL_S2M_API_PREFIX                   - REST API route prefix
    GL_S2M_API_RATE_LIMIT               - API requests per minute
    GL_S2M_CORS_ORIGINS                 - Comma-separated CORS origins
    GL_S2M_ENABLE_DOCS                  - Enable API documentation
    GL_S2M_ENABLED_FRAMEWORKS           - Comma-separated compliance frameworks
    GL_S2M_AUTO_COMPLIANCE_CHECK        - Auto compliance check on calculation
    GL_S2M_DUAL_REPORTING_REQUIRED      - Require dual reporting (location+market)
    GL_S2M_LOG_LEVEL                    - Logging level
    GL_S2M_ENABLE_METRICS               - Enable Prometheus metrics export
    GL_S2M_METRICS_PREFIX               - Prometheus metrics prefix
    GL_S2M_ENABLE_TRACING               - Enable OpenTelemetry tracing
    GL_S2M_SERVICE_NAME                 - Service name for tracing
    GL_S2M_ENABLE_DUAL_REPORTING        - Enable dual reporting feature
    GL_S2M_ENABLE_CERTIFICATE_RETIREMENT - Enable certificate retirement tracking
    GL_S2M_ENABLE_SUPPLIER_FACTORS      - Enable supplier-specific factors
    GL_S2M_ENABLE_UNCERTAINTY           - Enable uncertainty quantification
    GL_S2M_ENABLE_PROVENANCE            - Enable SHA-256 provenance tracking
    GL_S2M_GENESIS_HASH                 - Provenance chain genesis anchor
    GL_S2M_ENABLE_AUTH                  - Enable authentication middleware
    GL_S2M_WORKER_THREADS               - Worker thread pool size
    GL_S2M_ENABLE_BACKGROUND_TASKS      - Enable background task processing
    GL_S2M_HEALTH_CHECK_INTERVAL        - Health check interval (seconds)
    GL_S2M_ENABLED                      - Master enable/disable switch

Example:
    >>> from greenlang.scope2_market.config import Scope2MarketConfig
    >>> cfg = Scope2MarketConfig()
    >>> print(cfg.default_gwp_source, cfg.default_allocation_method)
    AR5 priority_based

    >>> # Check singleton
    >>> cfg2 = Scope2MarketConfig()
    >>> assert cfg is cfg2

    >>> # Reset for testing
    >>> Scope2MarketConfig.reset()
    >>> cfg3 = Scope2MarketConfig()
    >>> assert cfg is not cfg3

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-010 Scope 2 Market-Based Emissions (GL-MRV-SCOPE2-002)
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
import os
import threading
from decimal import ROUND_CEILING
from decimal import ROUND_DOWN
from decimal import ROUND_FLOOR
from decimal import ROUND_HALF_DOWN
from decimal import ROUND_HALF_EVEN
from decimal import ROUND_HALF_UP
from decimal import ROUND_UP
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX: str = "GL_S2M_"

# ---------------------------------------------------------------------------
# Valid enumeration values for configuration validation
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

_VALID_GWP_SOURCES = frozenset({"AR4", "AR5", "AR6"})

_VALID_ALLOCATION_METHODS = frozenset({
    "priority_based",
    "pro_rata",
    "proportional",
    "temporal_match",
    "geographic_match",
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

_VALID_SSL_MODES = frozenset({
    "disable",
    "allow",
    "prefer",
    "require",
    "verify-ca",
    "verify-full",
})

_VALID_RESIDUAL_SOURCES = frozenset({
    "aib",
    "green_e",
    "national",
    "epa_egrid",
    "iea",
    "custom",
})

_VALID_INSTRUMENT_TYPES = frozenset({
    "rec",
    "go",
    "i_rec",
    "lgc",
    "rego",
    "ppa",
    "vppa",
    "direct_line",
    "utility_green_tariff",
    "community_solar",
    "bundled_contract",
    "unbundled_eac",
})

_VALID_TRACKING_SYSTEMS = frozenset({
    "m_rets",
    "nar",
    "nepool_gis",
    "pjm_gats",
    "wregis",
    "aib_eecs",
    "rego_ofgem",
    "i_rec",
    "lgc_rec_registry",
    "custom",
})

_VALID_FRAMEWORKS = frozenset({
    "ghg_protocol_scope2",
    "iso_14064",
    "csrd_esrs",
    "cdp",
    "re100",
    "sbti",
    "green_e",
})

# ---------------------------------------------------------------------------
# Default compliance frameworks
# ---------------------------------------------------------------------------

_DEFAULT_ENABLED_FRAMEWORKS: List[str] = [
    "ghg_protocol_scope2",
    "iso_14064",
    "csrd_esrs",
    "cdp",
    "re100",
    "sbti",
    "green_e",
]

# ---------------------------------------------------------------------------
# Default instrument type hierarchy
# ---------------------------------------------------------------------------

_DEFAULT_INSTRUMENT_HIERARCHY: List[str] = [
    "direct_line",
    "ppa",
    "vppa",
    "utility_green_tariff",
    "bundled_contract",
    "rec",
    "go",
    "i_rec",
    "lgc",
    "rego",
    "unbundled_eac",
]


# ---------------------------------------------------------------------------
# Scope2MarketConfig
# ---------------------------------------------------------------------------


class Scope2MarketConfig:
    """Singleton configuration for the Scope 2 Market-Based Emissions Agent.

    Implements a singleton pattern via ``__new__`` with a class-level
    ``_instance`` and ``_initialized`` flag. On first instantiation, all
    settings are loaded from environment variables with the ``GL_S2M_``
    prefix. Subsequent instantiations return the same object.

    The configuration covers nine domains:
    1. Database Settings - PostgreSQL connection and pool sizing
    2. Redis Settings - cache host, port, database, TTL
    3. Calculation Settings - GWP source, allocation method, precision,
       vintage window
    4. Instrument Settings - certificate retirement, tracking systems,
       unbundled certificates, geographic matching
    5. Residual Mix Settings - default source, cache TTL, auto-update
    6. Uncertainty Settings - Monte Carlo parameters
    7. API Settings - prefix, rate limiting, CORS
    8. Compliance Settings - enabled regulatory frameworks, dual reporting
    9. Logging & Observability - log level, metrics, tracing

    The singleton can be reset for testing via :meth:`reset`. Configuration
    can be validated explicitly via :meth:`validate`, which returns a list
    of error strings (empty list means valid). Serialisation is supported
    via :meth:`to_dict` and :meth:`from_dict`.

    Attributes:
        db_host: PostgreSQL server hostname.
        db_port: PostgreSQL server port.
        db_name: PostgreSQL database name.
        db_user: PostgreSQL username.
        db_password: PostgreSQL password (never logged or serialised).
        db_pool_min: Minimum number of connections in the pool.
        db_pool_max: Maximum number of connections in the pool.
        db_ssl_mode: PostgreSQL SSL connection mode.
        redis_host: Redis server hostname.
        redis_port: Redis server port.
        redis_db: Redis database number.
        redis_password: Redis password (never logged or serialised).
        redis_ttl: Default Redis key TTL in seconds.
        default_gwp_source: Default IPCC Assessment Report for GWP values.
        default_allocation_method: Default allocation method for instruments.
        decimal_precision: Number of decimal places for calculations.
        rounding_mode: Python Decimal rounding mode name.
        max_batch_size: Maximum records per batch operation.
        default_vintage_window: Max vintage year window in years.
        auto_retire_on_use: Auto-retire instruments upon use in calculation.
        require_tracking_system: Require recognised tracking system for certs.
        allow_unbundled_certs: Allow unbundled energy attribute certificates.
        max_instruments_per_purchase: Max instruments per purchase record.
        validate_geographic_match: Validate geographic match for certificates.
        default_residual_source: Default residual mix factor source.
        residual_cache_ttl: Residual mix factor cache TTL in seconds.
        auto_update_factors: Auto-update residual mix factors on load.
        default_mc_iterations: Number of Monte Carlo iterations.
        default_confidence_level: Confidence level for uncertainty intervals.
        residual_uncertainty_pct: Default residual mix uncertainty percentage.
        instrument_uncertainty_pct: Default instrument uncertainty percentage.
        api_prefix: REST API URL prefix.
        api_rate_limit: Maximum API requests per minute.
        cors_origins: Allowed CORS origins.
        enable_docs: Enable interactive API documentation.
        enabled_frameworks: List of enabled compliance frameworks.
        auto_compliance_check: Run compliance check after each calculation.
        dual_reporting_required: Require dual reporting (location + market).
        log_level: Logging verbosity level.
        enable_metrics: Enable Prometheus metrics export.
        metrics_prefix: Prometheus metric name prefix.
        enable_tracing: Enable OpenTelemetry distributed tracing.
        service_name: Service name for tracing spans.
        enable_dual_reporting: Enable dual reporting feature.
        enable_certificate_retirement: Enable certificate retirement tracking.
        enable_supplier_factors: Enable supplier-specific emission factors.
        enable_uncertainty: Enable uncertainty quantification.
        enable_provenance: Enable SHA-256 provenance hash tracking.
        genesis_hash: Genesis anchor for the provenance chain.
        enable_auth: Enable authentication middleware.
        worker_threads: Thread pool size for parallel operations.
        enable_background_tasks: Enable background task processing.
        health_check_interval: Health check interval in seconds.
        enabled: Master enable/disable switch for the agent.

    Example:
        >>> cfg = Scope2MarketConfig()
        >>> cfg.default_gwp_source
        'AR5'
        >>> cfg.get_db_url()
        'postgresql://greenlang@localhost:5432/greenlang?sslmode=prefer'
        >>> cfg.is_framework_enabled("re100")
        True
        >>> cfg.get_enabled_frameworks()
        ['ghg_protocol_scope2', 'iso_14064', 'csrd_esrs', 'cdp', 're100', 'sbti', 'green_e']
    """

    _instance: Optional[Scope2MarketConfig] = None
    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton constructor
    # ------------------------------------------------------------------

    def __new__(cls) -> Scope2MarketConfig:
        """Return the singleton instance, creating it on first call.

        Uses a threading lock to ensure thread-safe initialisation. Only
        one instance is ever created; subsequent calls return the same
        object without acquiring the lock (double-checked locking).

        Returns:
            The singleton Scope2MarketConfig instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialise configuration from environment variables.

        Guarded by the ``_initialized`` flag so that repeated calls to
        ``__init__`` (from repeated ``Scope2MarketConfig()`` calls) do
        not re-read environment variables or overwrite customised values.
        """
        if self.__class__._initialized:
            return
        self._load_from_env()
        self.__class__._initialized = True
        logger.info(
            "Scope2MarketConfig initialised from environment: "
            "gwp=%s, allocation=%s, "
            "precision=%d, batch_size=%d, "
            "vintage_window=%d, "
            "residual_source=%s, frameworks=%s, "
            "metrics=%s, tracing=%s",
            self.default_gwp_source,
            self.default_allocation_method,
            self.decimal_precision,
            self.max_batch_size,
            self.default_vintage_window,
            self.default_residual_source,
            self.enabled_frameworks,
            self.enable_metrics,
            self.enable_tracing,
        )

    # ------------------------------------------------------------------
    # Environment loading
    # ------------------------------------------------------------------

    def _load_from_env(self) -> None:
        """Load all configuration from environment variables.

        Reads environment variables with the ``GL_S2M_`` prefix and
        populates all instance attributes. Each setting has a sensible
        default so the agent can start with zero environment configuration.

        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer and float values are parsed with fallback to defaults on
        malformed input, emitting a WARNING log.
        List values are parsed from comma-separated strings.
        """
        # -- 1. Database Settings ------------------------------------------------
        self.db_host: str = self._env_str("DB_HOST", "localhost")
        self.db_port: int = self._env_int("DB_PORT", 5432)
        self.db_name: str = self._env_str("DB_NAME", "greenlang")
        self.db_user: str = self._env_str("DB_USER", "greenlang")
        self.db_password: str = self._env_str("DB_PASSWORD", "")
        self.db_pool_min: int = self._env_int("DB_POOL_MIN", 2)
        self.db_pool_max: int = self._env_int("DB_POOL_MAX", 10)
        self.db_ssl_mode: str = self._env_str("DB_SSL_MODE", "prefer")

        # -- 2. Redis Settings ---------------------------------------------------
        self.redis_host: str = self._env_str("REDIS_HOST", "localhost")
        self.redis_port: int = self._env_int("REDIS_PORT", 6379)
        self.redis_db: int = self._env_int("REDIS_DB", 0)
        self.redis_password: str = self._env_str("REDIS_PASSWORD", "")
        self.redis_ttl: int = self._env_int("REDIS_TTL", 3600)

        # -- 3. Calculation Settings ---------------------------------------------
        self.default_gwp_source: str = self._env_str(
            "DEFAULT_GWP_SOURCE", "AR5"
        )
        self.default_allocation_method: str = self._env_str(
            "DEFAULT_ALLOCATION_METHOD", "priority_based"
        )
        self.decimal_precision: int = self._env_int(
            "DECIMAL_PRECISION", 10
        )
        self.rounding_mode: str = self._env_str(
            "ROUNDING_MODE", "ROUND_HALF_UP"
        )
        self.max_batch_size: int = self._env_int("MAX_BATCH_SIZE", 1000)
        self.default_vintage_window: int = self._env_int(
            "DEFAULT_VINTAGE_WINDOW", 1
        )

        # -- 4. Instrument Settings ----------------------------------------------
        self.auto_retire_on_use: bool = self._env_bool(
            "AUTO_RETIRE_ON_USE", True
        )
        self.require_tracking_system: bool = self._env_bool(
            "REQUIRE_TRACKING_SYSTEM", True
        )
        self.allow_unbundled_certs: bool = self._env_bool(
            "ALLOW_UNBUNDLED_CERTS", True
        )
        self.max_instruments_per_purchase: int = self._env_int(
            "MAX_INSTRUMENTS_PER_PURCHASE", 50
        )
        self.validate_geographic_match: bool = self._env_bool(
            "VALIDATE_GEOGRAPHIC_MATCH", True
        )
        self.default_instrument_hierarchy: List[str] = self._env_list(
            "DEFAULT_INSTRUMENT_HIERARCHY",
            _DEFAULT_INSTRUMENT_HIERARCHY,
        )
        self.allowed_instrument_types: List[str] = self._env_list(
            "ALLOWED_INSTRUMENT_TYPES",
            list(_VALID_INSTRUMENT_TYPES),
        )
        self.allowed_tracking_systems: List[str] = self._env_list(
            "ALLOWED_TRACKING_SYSTEMS",
            list(_VALID_TRACKING_SYSTEMS),
        )

        # -- 5. Residual Mix Settings --------------------------------------------
        self.default_residual_source: str = self._env_str(
            "DEFAULT_RESIDUAL_SOURCE", "aib"
        )
        self.residual_cache_ttl: int = self._env_int(
            "RESIDUAL_CACHE_TTL", 3600
        )
        self.auto_update_factors: bool = self._env_bool(
            "AUTO_UPDATE_FACTORS", False
        )
        self.residual_data_year: int = self._env_int(
            "RESIDUAL_DATA_YEAR", 2024
        )
        self.fallback_to_location_based: bool = self._env_bool(
            "FALLBACK_TO_LOCATION_BASED", True
        )

        # -- 6. Uncertainty Settings ---------------------------------------------
        self.default_mc_iterations: int = self._env_int(
            "DEFAULT_MC_ITERATIONS", 5000
        )
        self.default_confidence_level: float = self._env_float(
            "DEFAULT_CONFIDENCE_LEVEL", 0.95
        )
        self.residual_uncertainty_pct: float = self._env_float(
            "RESIDUAL_UNCERTAINTY_PCT", 0.10
        )
        self.instrument_uncertainty_pct: float = self._env_float(
            "INSTRUMENT_UNCERTAINTY_PCT", 0.05
        )

        # -- 7. API Settings -----------------------------------------------------
        self.api_prefix: str = self._env_str(
            "API_PREFIX", "/api/v1/scope2-market"
        )
        self.api_rate_limit: int = self._env_int("API_RATE_LIMIT", 100)
        self.cors_origins: List[str] = self._env_list(
            "CORS_ORIGINS", ["*"]
        )
        self.enable_docs: bool = self._env_bool("ENABLE_DOCS", True)

        # -- 8. Compliance Settings ----------------------------------------------
        self.enabled_frameworks: List[str] = self._env_list(
            "ENABLED_FRAMEWORKS",
            _DEFAULT_ENABLED_FRAMEWORKS,
        )
        self.auto_compliance_check: bool = self._env_bool(
            "AUTO_COMPLIANCE_CHECK", True
        )
        self.dual_reporting_required: bool = self._env_bool(
            "DUAL_REPORTING_REQUIRED", True
        )

        # -- 9. Logging & Observability ------------------------------------------
        self.log_level: str = self._env_str("LOG_LEVEL", "INFO")
        self.enable_metrics: bool = self._env_bool("ENABLE_METRICS", True)
        self.metrics_prefix: str = self._env_str(
            "METRICS_PREFIX", "gl_s2m_"
        )
        self.enable_tracing: bool = self._env_bool(
            "ENABLE_TRACING", True
        )
        self.service_name: str = self._env_str(
            "SERVICE_NAME", "scope2-market-service"
        )

        # -- Feature Flags -------------------------------------------------------
        self.enable_dual_reporting: bool = self._env_bool(
            "ENABLE_DUAL_REPORTING", True
        )
        self.enable_certificate_retirement: bool = self._env_bool(
            "ENABLE_CERTIFICATE_RETIREMENT", True
        )
        self.enable_supplier_factors: bool = self._env_bool(
            "ENABLE_SUPPLIER_FACTORS", False
        )
        self.enable_uncertainty: bool = self._env_bool(
            "ENABLE_UNCERTAINTY", True
        )

        # -- Provenance Tracking -------------------------------------------------
        self.enable_provenance: bool = self._env_bool(
            "ENABLE_PROVENANCE", True
        )
        self.genesis_hash: str = self._env_str(
            "GENESIS_HASH",
            "GL-MRV-X-010-SCOPE2-MARKET-GENESIS",
        )

        # -- Auth & Background Tasks ---------------------------------------------
        self.enable_auth: bool = self._env_bool("ENABLE_AUTH", True)
        self.worker_threads: int = self._env_int("WORKER_THREADS", 4)
        self.enable_background_tasks: bool = self._env_bool(
            "ENABLE_BACKGROUND_TASKS", True
        )
        self.health_check_interval: int = self._env_int(
            "HEALTH_CHECK_INTERVAL", 30
        )

        # -- Master switch -------------------------------------------------------
        self.enabled: bool = self._env_bool("ENABLED", True)

    # ------------------------------------------------------------------
    # Environment variable parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _env_str(name: str, default: str) -> str:
        """Read a string environment variable with the GL_S2M_ prefix.

        Args:
            name: Variable name suffix (after GL_S2M_).
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
        """Read an integer environment variable with the GL_S2M_ prefix.

        Args:
            name: Variable name suffix (after GL_S2M_).
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
    def _env_float(name: str, default: float) -> float:
        """Read a float environment variable with the GL_S2M_ prefix.

        Args:
            name: Variable name suffix (after GL_S2M_).
            default: Default value if not set or parse fails.

        Returns:
            Parsed float value or the default.
        """
        val = os.environ.get(f"{_ENV_PREFIX}{name}")
        if val is None:
            return default
        try:
            return float(val.strip())
        except ValueError:
            logger.warning(
                "Invalid float for %s%s=%r, using default %f",
                _ENV_PREFIX,
                name,
                val,
                default,
            )
            return default

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        """Read a boolean environment variable with the GL_S2M_ prefix.

        Accepts ``true``, ``1``, ``yes`` (case-insensitive) as True.
        All other non-None values are treated as False.

        Args:
            name: Variable name suffix (after GL_S2M_).
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
            name: Variable name suffix (after GL_S2M_).
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
        ``Scope2MarketConfig()`` will re-read all environment variables
        and construct a fresh configuration object. Thread-safe.

        Example:
            >>> Scope2MarketConfig.reset()
            >>> cfg = Scope2MarketConfig()  # fresh instance
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
        logger.debug("Scope2MarketConfig singleton reset")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate all configuration settings.

        Performs comprehensive checks across all configuration domains:
        database connectivity parameters, Redis parameters, calculation
        settings, instrument settings, residual mix settings, uncertainty
        parameters, API settings, compliance frameworks, logging levels,
        feature flags, provenance, and performance tuning.

        Returns:
            A list of human-readable error strings. An empty list means
            all validation checks passed.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> errors = cfg.validate()
            >>> assert len(errors) == 0
        """
        errors: List[str] = []

        # -- Database Settings -----------------------------------------------
        errors.extend(self._validate_database_settings())

        # -- Redis Settings --------------------------------------------------
        errors.extend(self._validate_redis_settings())

        # -- Calculation Settings --------------------------------------------
        errors.extend(self._validate_calculation_settings())

        # -- Instrument Settings ---------------------------------------------
        errors.extend(self._validate_instrument_settings())

        # -- Residual Mix Settings -------------------------------------------
        errors.extend(self._validate_residual_mix_settings())

        # -- Uncertainty Settings --------------------------------------------
        errors.extend(self._validate_uncertainty_settings())

        # -- API Settings ----------------------------------------------------
        errors.extend(self._validate_api_settings())

        # -- Compliance Settings ---------------------------------------------
        errors.extend(self._validate_compliance_settings())

        # -- Logging & Observability -----------------------------------------
        errors.extend(self._validate_logging_settings())

        # -- Feature Flags ---------------------------------------------------
        errors.extend(self._validate_feature_flags())

        # -- Provenance Tracking ---------------------------------------------
        errors.extend(self._validate_provenance_settings())

        # -- Performance Tuning ----------------------------------------------
        errors.extend(self._validate_performance_settings())

        if errors:
            logger.warning(
                "Scope2MarketConfig validation found %d error(s):\n%s",
                len(errors),
                "\n".join(f"  - {e}" for e in errors),
            )
        else:
            logger.debug(
                "Scope2MarketConfig validation passed: "
                "all %d checks OK",
                self._count_validation_checks(),
            )

        return errors

    def _validate_database_settings(self) -> List[str]:
        """Validate database connection settings.

        Checks host non-empty, port in valid TCP range (1-65535), name
        and user non-empty, pool sizes within reasonable bounds, and
        that pool_min does not exceed pool_max. SSL mode is validated
        against the PostgreSQL-accepted set.

        Returns:
            List of error strings for invalid database settings.
        """
        errors: List[str] = []

        if not self.db_host:
            errors.append("db_host must not be empty")

        if self.db_port <= 0:
            errors.append(
                f"db_port must be > 0, got {self.db_port}"
            )
        if self.db_port > 65535:
            errors.append(
                f"db_port must be <= 65535, got {self.db_port}"
            )

        if not self.db_name:
            errors.append("db_name must not be empty")

        if not self.db_user:
            errors.append("db_user must not be empty")

        if self.db_pool_min < 0:
            errors.append(
                f"db_pool_min must be >= 0, got {self.db_pool_min}"
            )
        if self.db_pool_min > 100:
            errors.append(
                f"db_pool_min must be <= 100, got {self.db_pool_min}"
            )

        if self.db_pool_max <= 0:
            errors.append(
                f"db_pool_max must be > 0, got {self.db_pool_max}"
            )
        if self.db_pool_max > 500:
            errors.append(
                f"db_pool_max must be <= 500, got {self.db_pool_max}"
            )

        if self.db_pool_min > self.db_pool_max:
            errors.append(
                f"db_pool_min ({self.db_pool_min}) must be <= "
                f"db_pool_max ({self.db_pool_max})"
            )

        normalised_ssl = self.db_ssl_mode.lower()
        if normalised_ssl not in _VALID_SSL_MODES:
            errors.append(
                f"db_ssl_mode must be one of {sorted(_VALID_SSL_MODES)}, "
                f"got '{self.db_ssl_mode}'"
            )

        return errors

    def _validate_redis_settings(self) -> List[str]:
        """Validate Redis connection settings.

        Checks host non-empty, port in valid TCP range (1-65535),
        database number within Redis bounds (0-15), and TTL within
        a reasonable range (1 to 7 days).

        Returns:
            List of error strings for invalid Redis settings.
        """
        errors: List[str] = []

        if not self.redis_host:
            errors.append("redis_host must not be empty")

        if self.redis_port <= 0:
            errors.append(
                f"redis_port must be > 0, got {self.redis_port}"
            )
        if self.redis_port > 65535:
            errors.append(
                f"redis_port must be <= 65535, got {self.redis_port}"
            )

        if self.redis_db < 0:
            errors.append(
                f"redis_db must be >= 0, got {self.redis_db}"
            )
        if self.redis_db > 15:
            errors.append(
                f"redis_db must be <= 15, got {self.redis_db}"
            )

        if self.redis_ttl <= 0:
            errors.append(
                f"redis_ttl must be > 0, got {self.redis_ttl}"
            )
        if self.redis_ttl > 604800:
            errors.append(
                f"redis_ttl must be <= 604800 (7 days), "
                f"got {self.redis_ttl}"
            )

        return errors

    def _validate_calculation_settings(self) -> List[str]:
        """Validate calculation settings.

        Checks GWP source is a valid IPCC AR edition, allocation method
        is a recognised strategy, decimal precision is within Decimal
        library bounds (0-28), rounding mode is a valid Python Decimal
        mode, batch size is within reasonable limits, and vintage window
        is positive.

        Returns:
            List of error strings for invalid calculation settings.
        """
        errors: List[str] = []

        # -- GWP source ------------------------------------------------------
        normalised_gwp = self.default_gwp_source.upper()
        if normalised_gwp not in _VALID_GWP_SOURCES:
            errors.append(
                f"default_gwp_source must be one of "
                f"{sorted(_VALID_GWP_SOURCES)}, "
                f"got '{self.default_gwp_source}'"
            )

        # -- Allocation method -----------------------------------------------
        normalised_alloc = self.default_allocation_method.lower()
        if normalised_alloc not in _VALID_ALLOCATION_METHODS:
            errors.append(
                f"default_allocation_method must be one of "
                f"{sorted(_VALID_ALLOCATION_METHODS)}, "
                f"got '{self.default_allocation_method}'"
            )

        # -- Decimal precision -----------------------------------------------
        if self.decimal_precision < 0:
            errors.append(
                f"decimal_precision must be >= 0, "
                f"got {self.decimal_precision}"
            )
        if self.decimal_precision > 28:
            errors.append(
                f"decimal_precision must be <= 28, "
                f"got {self.decimal_precision}"
            )

        # -- Rounding mode ---------------------------------------------------
        normalised_round = self.rounding_mode.upper()
        if normalised_round not in _VALID_ROUNDING_MODES:
            errors.append(
                f"rounding_mode must be one of "
                f"{sorted(_VALID_ROUNDING_MODES)}, "
                f"got '{self.rounding_mode}'"
            )

        # -- Max batch size --------------------------------------------------
        if self.max_batch_size <= 0:
            errors.append(
                f"max_batch_size must be > 0, got {self.max_batch_size}"
            )
        if self.max_batch_size > 100_000:
            errors.append(
                f"max_batch_size must be <= 100000, "
                f"got {self.max_batch_size}"
            )

        # -- Vintage window --------------------------------------------------
        if self.default_vintage_window < 0:
            errors.append(
                f"default_vintage_window must be >= 0, "
                f"got {self.default_vintage_window}"
            )
        if self.default_vintage_window > 10:
            errors.append(
                f"default_vintage_window must be <= 10 years, "
                f"got {self.default_vintage_window}"
            )

        return errors

    def _validate_instrument_settings(self) -> List[str]:
        """Validate instrument and certificate settings.

        Checks max instruments per purchase is within reasonable bounds,
        validates instrument hierarchy entries against the recognised
        instrument types, checks for duplicates in the hierarchy,
        validates allowed instrument types and tracking systems.

        Returns:
            List of error strings for invalid instrument settings.
        """
        errors: List[str] = []

        # -- Max instruments per purchase ------------------------------------
        if self.max_instruments_per_purchase <= 0:
            errors.append(
                f"max_instruments_per_purchase must be > 0, "
                f"got {self.max_instruments_per_purchase}"
            )
        if self.max_instruments_per_purchase > 10_000:
            errors.append(
                f"max_instruments_per_purchase must be <= 10000, "
                f"got {self.max_instruments_per_purchase}"
            )

        # -- Instrument hierarchy validation ---------------------------------
        if not self.default_instrument_hierarchy:
            errors.append(
                "default_instrument_hierarchy must not be empty"
            )
        else:
            for inst in self.default_instrument_hierarchy:
                normalised = inst.lower()
                if normalised not in _VALID_INSTRUMENT_TYPES:
                    errors.append(
                        f"Instrument type '{inst}' in hierarchy is not "
                        f"valid; must be one of "
                        f"{sorted(_VALID_INSTRUMENT_TYPES)}"
                    )

            # Check for duplicates
            seen: set = set()
            for inst in self.default_instrument_hierarchy:
                normalised = inst.lower()
                if normalised in seen:
                    errors.append(
                        f"Duplicate instrument type '{inst}' in hierarchy"
                    )
                seen.add(normalised)

        # -- Allowed instrument types ----------------------------------------
        if not self.allowed_instrument_types:
            errors.append(
                "allowed_instrument_types must not be empty"
            )
        else:
            for inst_type in self.allowed_instrument_types:
                normalised = inst_type.lower()
                if normalised not in _VALID_INSTRUMENT_TYPES:
                    errors.append(
                        f"Allowed instrument type '{inst_type}' is not "
                        f"valid; must be one of "
                        f"{sorted(_VALID_INSTRUMENT_TYPES)}"
                    )

        # -- Allowed tracking systems ----------------------------------------
        if self.require_tracking_system:
            if not self.allowed_tracking_systems:
                errors.append(
                    "allowed_tracking_systems must not be empty when "
                    "require_tracking_system is True"
                )
            else:
                for ts in self.allowed_tracking_systems:
                    normalised = ts.lower()
                    if normalised not in _VALID_TRACKING_SYSTEMS:
                        errors.append(
                            f"Tracking system '{ts}' is not valid; "
                            f"must be one of "
                            f"{sorted(_VALID_TRACKING_SYSTEMS)}"
                        )

        # -- Geographic match requires tracking system -----------------------
        if self.validate_geographic_match and not self.require_tracking_system:
            errors.append(
                "validate_geographic_match requires "
                "require_tracking_system to be True"
            )

        # -- Unbundled certs warning with auto-retire ------------------------
        # This is a soft consistency check (still logged as error)
        if not self.allow_unbundled_certs and not self.auto_retire_on_use:
            errors.append(
                "When allow_unbundled_certs is False, auto_retire_on_use "
                "should be True to ensure bundled instruments are retired"
            )

        return errors

    def _validate_residual_mix_settings(self) -> List[str]:
        """Validate residual mix factor settings.

        Checks that the default residual source is a recognised provider,
        cache TTL is within reasonable bounds, and residual data year is
        within expected range.

        Returns:
            List of error strings for invalid residual mix settings.
        """
        errors: List[str] = []

        # -- Default residual source -----------------------------------------
        normalised_source = self.default_residual_source.lower()
        if normalised_source not in _VALID_RESIDUAL_SOURCES:
            errors.append(
                f"default_residual_source must be one of "
                f"{sorted(_VALID_RESIDUAL_SOURCES)}, "
                f"got '{self.default_residual_source}'"
            )

        # -- Residual cache TTL ----------------------------------------------
        if self.residual_cache_ttl <= 0:
            errors.append(
                f"residual_cache_ttl must be > 0, "
                f"got {self.residual_cache_ttl}"
            )
        if self.residual_cache_ttl > 604800:
            errors.append(
                f"residual_cache_ttl must be <= 604800 (7 days), "
                f"got {self.residual_cache_ttl}"
            )

        # -- Residual data year ----------------------------------------------
        if self.residual_data_year < 2000:
            errors.append(
                f"residual_data_year must be >= 2000, "
                f"got {self.residual_data_year}"
            )
        if self.residual_data_year > 2030:
            errors.append(
                f"residual_data_year must be <= 2030, "
                f"got {self.residual_data_year}"
            )

        return errors

    def _validate_uncertainty_settings(self) -> List[str]:
        """Validate uncertainty quantification settings.

        Checks Monte Carlo iterations, confidence level, and uncertainty
        percentages are within valid mathematical bounds.

        Returns:
            List of error strings for invalid uncertainty settings.
        """
        errors: List[str] = []

        # -- Monte Carlo iterations ------------------------------------------
        if self.default_mc_iterations <= 0:
            errors.append(
                f"default_mc_iterations must be > 0, "
                f"got {self.default_mc_iterations}"
            )
        if self.default_mc_iterations > 1_000_000:
            errors.append(
                f"default_mc_iterations must be <= 1000000, "
                f"got {self.default_mc_iterations}"
            )

        # -- Confidence level ------------------------------------------------
        if self.default_confidence_level <= 0.0:
            errors.append(
                f"default_confidence_level must be > 0.0, "
                f"got {self.default_confidence_level}"
            )
        if self.default_confidence_level >= 1.0:
            errors.append(
                f"default_confidence_level must be < 1.0, "
                f"got {self.default_confidence_level}"
            )

        # -- Residual uncertainty percentage ---------------------------------
        if self.residual_uncertainty_pct < 0.0:
            errors.append(
                f"residual_uncertainty_pct must be >= 0.0, "
                f"got {self.residual_uncertainty_pct}"
            )
        if self.residual_uncertainty_pct > 1.0:
            errors.append(
                f"residual_uncertainty_pct must be <= 1.0, "
                f"got {self.residual_uncertainty_pct}"
            )

        # -- Instrument uncertainty percentage -------------------------------
        if self.instrument_uncertainty_pct < 0.0:
            errors.append(
                f"instrument_uncertainty_pct must be >= 0.0, "
                f"got {self.instrument_uncertainty_pct}"
            )
        if self.instrument_uncertainty_pct > 1.0:
            errors.append(
                f"instrument_uncertainty_pct must be <= 1.0, "
                f"got {self.instrument_uncertainty_pct}"
            )

        # -- Residual should have higher uncertainty than instrument ----------
        if self.residual_uncertainty_pct < self.instrument_uncertainty_pct:
            errors.append(
                f"residual_uncertainty_pct ({self.residual_uncertainty_pct}) "
                f"should be >= instrument_uncertainty_pct "
                f"({self.instrument_uncertainty_pct}) per GHG Protocol "
                f"Scope 2 Guidance"
            )

        return errors

    def _validate_api_settings(self) -> List[str]:
        """Validate API settings.

        Checks API prefix is non-empty and starts with slash, rate limit
        is within reasonable bounds, and CORS origins list is non-empty.

        Returns:
            List of error strings for invalid API settings.
        """
        errors: List[str] = []

        if not self.api_prefix:
            errors.append("api_prefix must not be empty")

        if not self.api_prefix.startswith("/"):
            errors.append(
                f"api_prefix must start with '/', "
                f"got '{self.api_prefix}'"
            )

        if self.api_rate_limit <= 0:
            errors.append(
                f"api_rate_limit must be > 0, "
                f"got {self.api_rate_limit}"
            )
        if self.api_rate_limit > 10_000:
            errors.append(
                f"api_rate_limit must be <= 10000, "
                f"got {self.api_rate_limit}"
            )

        if not self.cors_origins:
            errors.append("cors_origins must not be empty")

        return errors

    def _validate_compliance_settings(self) -> List[str]:
        """Validate compliance framework settings.

        Checks that enabled frameworks are all recognised identifiers,
        checks for duplicates, and validates that dual reporting
        requirements are met (ghg_protocol_scope2 must be enabled for
        dual reporting).

        Returns:
            List of error strings for invalid compliance settings.
        """
        errors: List[str] = []

        if not self.enabled_frameworks:
            errors.append("enabled_frameworks must not be empty")
        else:
            for fw in self.enabled_frameworks:
                normalised = fw.lower()
                if normalised not in _VALID_FRAMEWORKS:
                    errors.append(
                        f"Framework '{fw}' is not valid; "
                        f"must be one of {sorted(_VALID_FRAMEWORKS)}"
                    )

            # Check for duplicates
            seen: set = set()
            for fw in self.enabled_frameworks:
                normalised = fw.lower()
                if normalised in seen:
                    errors.append(
                        f"Duplicate framework '{fw}' in enabled_frameworks"
                    )
                seen.add(normalised)

        # Dual reporting needs ghg_protocol_scope2
        if self.dual_reporting_required:
            normalised_fws = [fw.lower() for fw in self.enabled_frameworks]
            if "ghg_protocol_scope2" not in normalised_fws:
                errors.append(
                    "dual_reporting_required requires "
                    "'ghg_protocol_scope2' in enabled_frameworks"
                )

        # RE100 requires either sbti or green_e to also be enabled
        normalised_fws = [fw.lower() for fw in self.enabled_frameworks]
        if "re100" in normalised_fws:
            if "sbti" not in normalised_fws and "green_e" not in normalised_fws:
                errors.append(
                    "RE100 framework requires at least one of "
                    "'sbti' or 'green_e' in enabled_frameworks "
                    "for certificate quality assurance"
                )

        # SBTi requires ghg_protocol_scope2
        if "sbti" in normalised_fws:
            if "ghg_protocol_scope2" not in normalised_fws:
                errors.append(
                    "SBTi framework requires 'ghg_protocol_scope2' "
                    "in enabled_frameworks"
                )

        return errors

    def _validate_logging_settings(self) -> List[str]:
        """Validate logging and observability settings.

        Checks log level is a valid Python logging level, metrics prefix
        and service name are non-empty, and metrics prefix follows naming
        conventions (lowercase with underscores).

        Returns:
            List of error strings for invalid logging settings.
        """
        errors: List[str] = []

        normalised_log = self.log_level.upper()
        if normalised_log not in _VALID_LOG_LEVELS:
            errors.append(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got '{self.log_level}'"
            )

        if not self.metrics_prefix:
            errors.append("metrics_prefix must not be empty")

        if self.metrics_prefix and not self.metrics_prefix.replace(
            "_", ""
        ).isalnum():
            errors.append(
                f"metrics_prefix must contain only alphanumeric "
                f"characters and underscores, "
                f"got '{self.metrics_prefix}'"
            )

        if not self.service_name:
            errors.append("service_name must not be empty")

        if self.service_name and len(self.service_name) > 128:
            errors.append(
                f"service_name must be <= 128 characters, "
                f"got {len(self.service_name)}"
            )

        return errors

    def _validate_feature_flags(self) -> List[str]:
        """Validate feature flag consistency.

        Checks that feature flag combinations are logically consistent.
        For example, dual reporting feature requires dual reporting
        required setting, and certificate retirement feature requires
        appropriate instrument settings.

        Returns:
            List of error strings for invalid feature flag combinations.
        """
        errors: List[str] = []

        # Dual reporting feature without compliance setting is inconsistent
        if self.enable_dual_reporting and not self.dual_reporting_required:
            # Not an error, but log a warning about potential inconsistency
            logger.debug(
                "enable_dual_reporting is True but "
                "dual_reporting_required is False; dual reporting "
                "will be available but not enforced"
            )

        # Certificate retirement requires instrument tracking
        if (
            self.enable_certificate_retirement
            and not self.auto_retire_on_use
        ):
            errors.append(
                "enable_certificate_retirement requires "
                "auto_retire_on_use to be True"
            )

        # Supplier factors require uncertainty to be meaningful
        if self.enable_supplier_factors and not self.enable_uncertainty:
            logger.debug(
                "enable_supplier_factors is True but "
                "enable_uncertainty is False; supplier factor "
                "uncertainty will not be quantified"
            )

        return errors

    def _validate_provenance_settings(self) -> List[str]:
        """Validate provenance tracking settings.

        Checks that genesis hash is provided when provenance is enabled
        and that the hash length is within bounds.

        Returns:
            List of error strings for invalid provenance settings.
        """
        errors: List[str] = []

        if self.enable_provenance and not self.genesis_hash:
            errors.append(
                "genesis_hash must not be empty when "
                "enable_provenance is True"
            )

        if self.genesis_hash and len(self.genesis_hash) > 256:
            errors.append(
                f"genesis_hash must be <= 256 characters, "
                f"got {len(self.genesis_hash)}"
            )

        return errors

    def _validate_performance_settings(self) -> List[str]:
        """Validate performance tuning settings.

        Checks worker threads and health check interval are within
        reasonable bounds for production deployments.

        Returns:
            List of error strings for invalid performance settings.
        """
        errors: List[str] = []

        if self.worker_threads <= 0:
            errors.append(
                f"worker_threads must be > 0, "
                f"got {self.worker_threads}"
            )
        if self.worker_threads > 64:
            errors.append(
                f"worker_threads must be <= 64, "
                f"got {self.worker_threads}"
            )

        if self.health_check_interval <= 0:
            errors.append(
                f"health_check_interval must be > 0, "
                f"got {self.health_check_interval}"
            )
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
        # Database: 10, Redis: 7, Calculation: 10, Instrument: 12,
        # Residual Mix: 5, Uncertainty: 9, API: 5, Compliance: 7,
        # Logging: 5, Feature Flags: 1, Provenance: 2, Performance: 4
        return 77

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a plain Python dictionary.

        The returned dictionary is safe to pass to ``json.dumps``,
        ``yaml.dump``, or any structured logging framework. Sensitive
        fields (``db_password``, ``redis_password``) are redacted to
        prevent accidental credential leakage.

        Returns:
            Dictionary representation of the configuration with sensitive
            fields redacted.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> d = cfg.to_dict()
            >>> d["default_gwp_source"]
            'AR5'
            >>> d["db_password"]
            '***'
        """
        return {
            # -- Master switch -----------------------------------------------
            "enabled": self.enabled,
            # -- 1. Database Settings ----------------------------------------
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_name": self.db_name,
            "db_user": self.db_user,
            "db_password": "***" if self.db_password else "",
            "db_pool_min": self.db_pool_min,
            "db_pool_max": self.db_pool_max,
            "db_ssl_mode": self.db_ssl_mode,
            # -- 2. Redis Settings -------------------------------------------
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "redis_db": self.redis_db,
            "redis_password": "***" if self.redis_password else "",
            "redis_ttl": self.redis_ttl,
            # -- 3. Calculation Settings -------------------------------------
            "default_gwp_source": self.default_gwp_source,
            "default_allocation_method": self.default_allocation_method,
            "decimal_precision": self.decimal_precision,
            "rounding_mode": self.rounding_mode,
            "max_batch_size": self.max_batch_size,
            "default_vintage_window": self.default_vintage_window,
            # -- 4. Instrument Settings --------------------------------------
            "auto_retire_on_use": self.auto_retire_on_use,
            "require_tracking_system": self.require_tracking_system,
            "allow_unbundled_certs": self.allow_unbundled_certs,
            "max_instruments_per_purchase": (
                self.max_instruments_per_purchase
            ),
            "validate_geographic_match": self.validate_geographic_match,
            "default_instrument_hierarchy": list(
                self.default_instrument_hierarchy
            ),
            "allowed_instrument_types": list(
                self.allowed_instrument_types
            ),
            "allowed_tracking_systems": list(
                self.allowed_tracking_systems
            ),
            # -- 5. Residual Mix Settings ------------------------------------
            "default_residual_source": self.default_residual_source,
            "residual_cache_ttl": self.residual_cache_ttl,
            "auto_update_factors": self.auto_update_factors,
            "residual_data_year": self.residual_data_year,
            "fallback_to_location_based": self.fallback_to_location_based,
            # -- 6. Uncertainty Settings -------------------------------------
            "default_mc_iterations": self.default_mc_iterations,
            "default_confidence_level": self.default_confidence_level,
            "residual_uncertainty_pct": self.residual_uncertainty_pct,
            "instrument_uncertainty_pct": (
                self.instrument_uncertainty_pct
            ),
            # -- 7. API Settings ---------------------------------------------
            "api_prefix": self.api_prefix,
            "api_rate_limit": self.api_rate_limit,
            "cors_origins": list(self.cors_origins),
            "enable_docs": self.enable_docs,
            # -- 8. Compliance Settings --------------------------------------
            "enabled_frameworks": list(self.enabled_frameworks),
            "auto_compliance_check": self.auto_compliance_check,
            "dual_reporting_required": self.dual_reporting_required,
            # -- 9. Logging & Observability ----------------------------------
            "log_level": self.log_level,
            "enable_metrics": self.enable_metrics,
            "metrics_prefix": self.metrics_prefix,
            "enable_tracing": self.enable_tracing,
            "service_name": self.service_name,
            # -- Feature Flags -----------------------------------------------
            "enable_dual_reporting": self.enable_dual_reporting,
            "enable_certificate_retirement": (
                self.enable_certificate_retirement
            ),
            "enable_supplier_factors": self.enable_supplier_factors,
            "enable_uncertainty": self.enable_uncertainty,
            # -- Provenance Tracking -----------------------------------------
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # -- Auth & Background Tasks -------------------------------------
            "enable_auth": self.enable_auth,
            "worker_threads": self.worker_threads,
            "enable_background_tasks": self.enable_background_tasks,
            "health_check_interval": self.health_check_interval,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Scope2MarketConfig:
        """Deserialise a configuration from a dictionary.

        Creates a new Scope2MarketConfig instance and populates it
        from the provided dictionary. The singleton is reset first to
        allow the new configuration to be installed. Keys not present
        in the dictionary retain their environment-loaded defaults.

        Args:
            data: Dictionary of configuration key-value pairs. Keys
                correspond to attribute names on the config object.

        Returns:
            A new Scope2MarketConfig instance with values from the
            dictionary.

        Example:
            >>> d = {"default_gwp_source": "AR6", "decimal_precision": 12}
            >>> cfg = Scope2MarketConfig.from_dict(d)
            >>> cfg.default_gwp_source
            'AR6'
            >>> cfg.decimal_precision
            12
        """
        # Reset singleton to allow fresh construction
        cls.reset()

        # Create fresh instance (triggers _load_from_env via __init__)
        instance = cls()

        # Override with provided dictionary values
        instance._apply_dict(data)

        logger.info(
            "Scope2MarketConfig loaded from dict: %d keys applied",
            len(data),
        )
        return instance

    def _apply_dict(self, data: Dict[str, Any]) -> None:
        """Apply dictionary values to the configuration instance.

        Only applies values for known attribute names. Unknown keys
        are logged as warnings and skipped. Redacted password fields
        (``'***'``) are skipped to avoid overwriting real credentials.

        Args:
            data: Dictionary of configuration key-value pairs.
        """
        known_attrs = set(self.to_dict().keys())

        for key, value in data.items():
            if key in known_attrs:
                # Handle password fields - skip '***' redacted values
                if key in ("db_password", "redis_password"):
                    if value == "***":
                        continue
                setattr(self, key, value)
            else:
                logger.warning(
                    "Unknown configuration key '%s' in from_dict, skipping",
                    key,
                )

    # ------------------------------------------------------------------
    # Connection URL builders
    # ------------------------------------------------------------------

    def get_db_url(self) -> str:
        """Build a PostgreSQL connection URL from individual settings.

        Constructs a standard PostgreSQL connection URL from the
        configured host, port, database, user, and password fields.
        The password is URL-encoded to handle special characters.
        The SSL mode is appended as a query parameter.

        Returns:
            PostgreSQL connection URL string.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> url = cfg.get_db_url()
            >>> url.startswith("postgresql://")
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

        # Append SSL mode as query parameter
        if self.db_ssl_mode:
            url += f"?sslmode={self.db_ssl_mode}"

        return url

    def get_async_db_url(self) -> str:
        """Build an async PostgreSQL connection URL for asyncpg/psycopg.

        Identical to :meth:`get_db_url` but uses the
        ``postgresql+asyncpg://`` scheme for async driver compatibility.

        Returns:
            Async PostgreSQL connection URL string.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> url = cfg.get_async_db_url()
            >>> url.startswith("postgresql+asyncpg://")
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

    def get_redis_url(self) -> str:
        """Build a Redis connection URL from individual settings.

        Constructs a standard Redis connection URL from the configured
        host, port, database, and optional password fields.

        Returns:
            Redis connection URL string.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> url = cfg.get_redis_url()
            >>> url.startswith("redis://")
            True
        """
        if self.redis_password:
            encoded_password = quote_plus(self.redis_password)
            auth = f":{encoded_password}@"
        else:
            auth = ""

        return (
            f"redis://{auth}{self.redis_host}:{self.redis_port}"
            f"/{self.redis_db}"
        )

    # ------------------------------------------------------------------
    # Framework accessors
    # ------------------------------------------------------------------

    def get_enabled_frameworks(self) -> List[str]:
        """Return a copy of the enabled compliance frameworks list.

        Returns:
            List of enabled framework identifiers.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> "re100" in cfg.get_enabled_frameworks()
            True
        """
        return list(self.enabled_frameworks)

    def is_framework_enabled(self, framework: str) -> bool:
        """Check if a specific compliance framework is enabled.

        Performs a case-insensitive comparison against the configured
        list of enabled frameworks.

        Args:
            framework: Framework identifier to check (e.g. "re100",
                "ghg_protocol_scope2", "csrd_esrs").

        Returns:
            True if the framework is in the enabled list, False otherwise.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> cfg.is_framework_enabled("re100")
            True
            >>> cfg.is_framework_enabled("unknown_framework")
            False
        """
        normalised = framework.lower()
        return normalised in [
            fw.lower() for fw in self.enabled_frameworks
        ]

    # ------------------------------------------------------------------
    # Instrument hierarchy accessor
    # ------------------------------------------------------------------

    def get_instrument_hierarchy(self) -> List[str]:
        """Return the instrument type priority hierarchy.

        Returns a copy of the ordered list of instrument types from
        highest to lowest priority. The allocation engine uses this
        order when applying instruments to consumption data.

        Returns:
            Ordered list of instrument type identifiers.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> hierarchy = cfg.get_instrument_hierarchy()
            >>> hierarchy[0]
            'direct_line'
        """
        return list(self.default_instrument_hierarchy)

    # ------------------------------------------------------------------
    # Rounding mode accessor
    # ------------------------------------------------------------------

    def get_rounding_mode(self) -> str:
        """Return the Python Decimal rounding mode constant.

        Maps the configured rounding mode string to the corresponding
        ``decimal`` module constant for use in ``Decimal.quantize()``.

        Returns:
            Decimal rounding mode constant string.

        Raises:
            ValueError: If the configured rounding mode is not valid.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> cfg.get_rounding_mode()
            'ROUND_HALF_UP'
        """
        normalised = self.rounding_mode.upper()
        if normalised not in _ROUNDING_MODE_MAP:
            raise ValueError(
                f"Invalid rounding_mode '{self.rounding_mode}'; "
                f"must be one of {sorted(_ROUNDING_MODE_MAP.keys())}"
            )
        return _ROUNDING_MODE_MAP[normalised]

    # ------------------------------------------------------------------
    # Residual mix configuration accessor
    # ------------------------------------------------------------------

    def get_residual_mix_config(self) -> Dict[str, Any]:
        """Return residual mix factor configuration as a dictionary.

        Returns:
            Dictionary containing all residual-mix-related settings.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> rmx = cfg.get_residual_mix_config()
            >>> rmx["default_source"]
            'aib'
        """
        return {
            "default_source": self.default_residual_source,
            "cache_ttl": self.residual_cache_ttl,
            "auto_update": self.auto_update_factors,
            "data_year": self.residual_data_year,
            "fallback_to_location_based": self.fallback_to_location_based,
        }

    # ------------------------------------------------------------------
    # Instrument configuration accessor
    # ------------------------------------------------------------------

    def get_instrument_config(self) -> Dict[str, Any]:
        """Return instrument and certificate configuration as a dictionary.

        Returns:
            Dictionary containing all instrument-related settings suitable
            for initialising the InstrumentAllocationEngine.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> inst = cfg.get_instrument_config()
            >>> inst["auto_retire_on_use"]
            True
        """
        return {
            "auto_retire_on_use": self.auto_retire_on_use,
            "require_tracking_system": self.require_tracking_system,
            "allow_unbundled_certs": self.allow_unbundled_certs,
            "max_instruments_per_purchase": (
                self.max_instruments_per_purchase
            ),
            "validate_geographic_match": self.validate_geographic_match,
            "instrument_hierarchy": list(
                self.default_instrument_hierarchy
            ),
            "allowed_instrument_types": list(
                self.allowed_instrument_types
            ),
            "allowed_tracking_systems": list(
                self.allowed_tracking_systems
            ),
            "vintage_window": self.default_vintage_window,
        }

    # ------------------------------------------------------------------
    # Uncertainty parameter accessor
    # ------------------------------------------------------------------

    def get_uncertainty_params(self) -> Dict[str, Any]:
        """Return uncertainty quantification parameters as a dictionary.

        Returns:
            Dictionary containing all uncertainty-related settings.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> params = cfg.get_uncertainty_params()
            >>> params["mc_iterations"]
            5000
        """
        return {
            "mc_iterations": self.default_mc_iterations,
            "confidence_level": self.default_confidence_level,
            "residual_uncertainty_pct": self.residual_uncertainty_pct,
            "instrument_uncertainty_pct": (
                self.instrument_uncertainty_pct
            ),
            "enabled": self.enable_uncertainty,
        }

    # ------------------------------------------------------------------
    # Database pool parameters accessor
    # ------------------------------------------------------------------

    def get_db_pool_params(self) -> Dict[str, Any]:
        """Return database connection pool parameters.

        Returns:
            Dictionary containing pool configuration suitable for
            passing to psycopg_pool or similar connection pool libraries.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> params = cfg.get_db_pool_params()
            >>> params["min_size"]
            2
        """
        return {
            "min_size": self.db_pool_min,
            "max_size": self.db_pool_max,
            "conninfo": self.get_db_url(),
        }

    # ------------------------------------------------------------------
    # API configuration accessor
    # ------------------------------------------------------------------

    def get_api_config(self) -> Dict[str, Any]:
        """Return API configuration as a dictionary.

        Returns:
            Dictionary of API-related settings suitable for FastAPI
            application construction.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> api_cfg = cfg.get_api_config()
            >>> api_cfg["prefix"]
            '/api/v1/scope2-market'
        """
        return {
            "prefix": self.api_prefix,
            "rate_limit": self.api_rate_limit,
            "cors_origins": list(self.cors_origins),
            "enable_docs": self.enable_docs,
        }

    # ------------------------------------------------------------------
    # Observability configuration accessor
    # ------------------------------------------------------------------

    def get_observability_config(self) -> Dict[str, Any]:
        """Return observability configuration as a dictionary.

        Returns:
            Dictionary of observability-related settings for metrics
            and tracing initialisation.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> obs_cfg = cfg.get_observability_config()
            >>> obs_cfg["metrics_prefix"]
            'gl_s2m_'
        """
        return {
            "log_level": self.log_level,
            "enable_metrics": self.enable_metrics,
            "metrics_prefix": self.metrics_prefix,
            "enable_tracing": self.enable_tracing,
            "service_name": self.service_name,
        }

    # ------------------------------------------------------------------
    # Feature flag summary
    # ------------------------------------------------------------------

    def get_feature_flags(self) -> Dict[str, bool]:
        """Return all feature flags as a dictionary.

        Returns:
            Dictionary mapping feature flag name to boolean state.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> flags = cfg.get_feature_flags()
            >>> flags["enable_dual_reporting"]
            True
        """
        return {
            "enable_dual_reporting": self.enable_dual_reporting,
            "enable_certificate_retirement": (
                self.enable_certificate_retirement
            ),
            "enable_supplier_factors": self.enable_supplier_factors,
            "enable_uncertainty": self.enable_uncertainty,
            "enable_provenance": self.enable_provenance,
            "enable_auth": self.enable_auth,
            "enable_metrics": self.enable_metrics,
            "enable_tracing": self.enable_tracing,
            "enable_docs": self.enable_docs,
            "enable_background_tasks": self.enable_background_tasks,
            "auto_compliance_check": self.auto_compliance_check,
            "dual_reporting_required": self.dual_reporting_required,
            "auto_retire_on_use": self.auto_retire_on_use,
            "require_tracking_system": self.require_tracking_system,
            "allow_unbundled_certs": self.allow_unbundled_certs,
            "validate_geographic_match": self.validate_geographic_match,
            "auto_update_factors": self.auto_update_factors,
            "fallback_to_location_based": self.fallback_to_location_based,
        }

    # ------------------------------------------------------------------
    # Calculation configuration accessor
    # ------------------------------------------------------------------

    def get_calculation_config(self) -> Dict[str, Any]:
        """Return calculation engine configuration as a dictionary.

        Returns:
            Dictionary containing all calculation-related settings
            suitable for initialising the MarketBasedCalculationEngine.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> calc = cfg.get_calculation_config()
            >>> calc["allocation_method"]
            'priority_based'
        """
        return {
            "gwp_source": self.default_gwp_source,
            "allocation_method": self.default_allocation_method,
            "decimal_precision": self.decimal_precision,
            "rounding_mode": self.rounding_mode,
            "max_batch_size": self.max_batch_size,
            "vintage_window": self.default_vintage_window,
        }

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Sensitive fields (db_password, redis_password) are replaced with
        ``'***'`` so that repr output is safe to include in log messages
        and exception tracebacks.

        Returns:
            String representation of the configuration.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"Scope2MarketConfig({pairs})"

    def __str__(self) -> str:
        """Return a human-readable summary of the configuration.

        Returns:
            Multi-line string summary of key settings.
        """
        return (
            f"Scope2MarketConfig("
            f"enabled={self.enabled}, "
            f"gwp={self.default_gwp_source}, "
            f"allocation={self.default_allocation_method}, "
            f"precision={self.decimal_precision}, "
            f"vintage_window={self.default_vintage_window}, "
            f"batch={self.max_batch_size}, "
            f"frameworks={len(self.enabled_frameworks)}, "
            f"residual_source={self.default_residual_source}, "
            f"uncertainty={self.enable_uncertainty}"
            f")"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality by comparing all serialised values.

        Args:
            other: Object to compare against.

        Returns:
            True if other is a Scope2MarketConfig with identical settings.
        """
        if not isinstance(other, Scope2MarketConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()

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
            >>> cfg = Scope2MarketConfig()
            >>> json_str = cfg.to_json()
            >>> '"default_gwp_source": "AR5"' in json_str
            True
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, json_str: str) -> Scope2MarketConfig:
        """Deserialise a configuration from a JSON string.

        Args:
            json_str: JSON string containing configuration key-value pairs.

        Returns:
            A new Scope2MarketConfig instance.

        Example:
            >>> json_str = '{"default_gwp_source": "AR6"}'
            >>> cfg = Scope2MarketConfig.from_json(json_str)
            >>> cfg.default_gwp_source
            'AR6'
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Normalisation helper
    # ------------------------------------------------------------------

    def normalise(self) -> None:
        """Normalise configuration values to canonical forms.

        Converts string enumerations to their expected case (e.g. GWP
        source to uppercase, allocation method to lowercase, log level
        to uppercase, SSL mode to lowercase). This method is idempotent.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> cfg.default_gwp_source = "ar5"
            >>> cfg.normalise()
            >>> cfg.default_gwp_source
            'AR5'
        """
        # GWP source -> uppercase
        self.default_gwp_source = self.default_gwp_source.upper()

        # Allocation method -> lowercase
        self.default_allocation_method = (
            self.default_allocation_method.lower()
        )

        # Rounding mode -> uppercase
        self.rounding_mode = self.rounding_mode.upper()

        # Log level -> uppercase
        self.log_level = self.log_level.upper()

        # SSL mode -> lowercase
        self.db_ssl_mode = self.db_ssl_mode.lower()

        # Residual source -> lowercase
        self.default_residual_source = (
            self.default_residual_source.lower()
        )

        # Instrument hierarchy -> lowercase
        self.default_instrument_hierarchy = [
            s.lower() for s in self.default_instrument_hierarchy
        ]

        # Allowed instrument types -> lowercase
        self.allowed_instrument_types = [
            s.lower() for s in self.allowed_instrument_types
        ]

        # Allowed tracking systems -> lowercase
        self.allowed_tracking_systems = [
            s.lower() for s in self.allowed_tracking_systems
        ]

        # Frameworks -> lowercase
        self.enabled_frameworks = [
            fw.lower() for fw in self.enabled_frameworks
        ]

        # Metrics prefix -> lowercase (convention)
        self.metrics_prefix = self.metrics_prefix.lower()

        logger.debug(
            "Scope2MarketConfig normalised: gwp=%s, allocation=%s, "
            "log_level=%s, ssl_mode=%s, residual_source=%s",
            self.default_gwp_source,
            self.default_allocation_method,
            self.log_level,
            self.db_ssl_mode,
            self.default_residual_source,
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
            >>> cfg = Scope2MarketConfig()
            >>> cfg.merge({"decimal_precision": 12})
            >>> cfg.decimal_precision
            12
        """
        self._apply_dict(overrides)
        logger.debug(
            "Scope2MarketConfig merged %d overrides",
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
        # Restore actual sensitive values
        d["db_password"] = self.db_password
        d["redis_password"] = self.redis_password
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
            >>> cfg = Scope2MarketConfig()
            >>> summary = cfg.health_summary()
            >>> summary["validation_status"]
            'PASS'
        """
        errors = self.validate()
        flags = self.get_feature_flags()
        enabled_flag_count = sum(1 for v in flags.values() if v)

        return {
            "agent": "scope2-market",
            "agent_id": "AGENT-MRV-010",
            "enabled": self.enabled,
            "validation_status": "PASS" if not errors else "FAIL",
            "validation_errors": len(errors),
            "gwp_source": self.default_gwp_source,
            "allocation_method": self.default_allocation_method,
            "decimal_precision": self.decimal_precision,
            "max_batch_size": self.max_batch_size,
            "vintage_window": self.default_vintage_window,
            "residual_source": self.default_residual_source,
            "enabled_frameworks": len(self.enabled_frameworks),
            "enabled_features": enabled_flag_count,
            "total_features": len(flags),
            "instrument_hierarchy_depth": len(
                self.default_instrument_hierarchy
            ),
            "db_host": self.db_host,
            "db_port": self.db_port,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "worker_threads": self.worker_threads,
            "provenance_enabled": self.enable_provenance,
            "metrics_prefix": self.metrics_prefix,
            "service_name": self.service_name,
            "dual_reporting_required": self.dual_reporting_required,
            "certificate_retirement": self.enable_certificate_retirement,
        }

    # ------------------------------------------------------------------
    # Market-based specific accessors
    # ------------------------------------------------------------------

    def get_allocation_config(self) -> Dict[str, Any]:
        """Return the instrument allocation configuration.

        Combines calculation settings and instrument settings into a
        single dictionary suitable for initialising the allocation
        engine used in market-based Scope 2 calculations.

        Returns:
            Dictionary containing all allocation-relevant settings.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> alloc = cfg.get_allocation_config()
            >>> alloc["method"]
            'priority_based'
        """
        return {
            "method": self.default_allocation_method,
            "vintage_window": self.default_vintage_window,
            "instrument_hierarchy": list(
                self.default_instrument_hierarchy
            ),
            "auto_retire_on_use": self.auto_retire_on_use,
            "allow_unbundled_certs": self.allow_unbundled_certs,
            "validate_geographic_match": self.validate_geographic_match,
            "max_instruments_per_purchase": (
                self.max_instruments_per_purchase
            ),
        }

    def get_dual_reporting_config(self) -> Dict[str, Any]:
        """Return dual reporting configuration.

        Provides the settings needed to coordinate location-based and
        market-based calculations for GHG Protocol Scope 2 Guidance
        dual reporting requirements.

        Returns:
            Dictionary containing dual reporting settings.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> dr = cfg.get_dual_reporting_config()
            >>> dr["required"]
            True
        """
        return {
            "enabled": self.enable_dual_reporting,
            "required": self.dual_reporting_required,
            "fallback_to_location_based": self.fallback_to_location_based,
        }

    def get_certificate_config(self) -> Dict[str, Any]:
        """Return certificate management configuration.

        Provides settings for energy attribute certificate (EAC)
        management including retirement tracking, geographic validation,
        and tracking system requirements.

        Returns:
            Dictionary containing certificate management settings.

        Example:
            >>> cfg = Scope2MarketConfig()
            >>> cert = cfg.get_certificate_config()
            >>> cert["retirement_enabled"]
            True
        """
        return {
            "retirement_enabled": self.enable_certificate_retirement,
            "auto_retire_on_use": self.auto_retire_on_use,
            "require_tracking_system": self.require_tracking_system,
            "allow_unbundled": self.allow_unbundled_certs,
            "validate_geographic_match": self.validate_geographic_match,
            "max_per_purchase": self.max_instruments_per_purchase,
            "allowed_types": list(self.allowed_instrument_types),
            "allowed_tracking_systems": list(
                self.allowed_tracking_systems
            ),
            "vintage_window": self.default_vintage_window,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def get_config() -> Scope2MarketConfig:
    """Return the singleton Scope2MarketConfig.

    Convenience function that delegates to the singleton constructor.
    Thread-safe via the class-level lock in ``__new__``.

    Returns:
        Scope2MarketConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.default_gwp_source
        'AR5'
    """
    return Scope2MarketConfig()


def set_config(
    overrides: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Scope2MarketConfig:
    """Reset and re-create the singleton with optional overrides.

    Resets the singleton, creates a fresh instance from environment
    variables, then applies any provided overrides. This is the
    primary entry point for test setup.

    Args:
        overrides: Dictionary of configuration overrides.
        **kwargs: Additional keyword overrides (merged with overrides).

    Returns:
        The new Scope2MarketConfig singleton.

    Example:
        >>> cfg = set_config(default_gwp_source="AR6")
        >>> cfg.default_gwp_source
        'AR6'
    """
    Scope2MarketConfig.reset()
    cfg = Scope2MarketConfig()

    merged: Dict[str, Any] = {}
    if overrides:
        merged.update(overrides)
    merged.update(kwargs)

    if merged:
        cfg._apply_dict(merged)

    logger.info(
        "Scope2MarketConfig set with %d overrides: "
        "enabled=%s, gwp=%s, allocation=%s, "
        "precision=%d, batch_size=%d, "
        "vintage_window=%d",
        len(merged),
        cfg.enabled,
        cfg.default_gwp_source,
        cfg.default_allocation_method,
        cfg.decimal_precision,
        cfg.max_batch_size,
        cfg.default_vintage_window,
    )
    return cfg


def reset_config() -> None:
    """Reset the singleton Scope2MarketConfig to None.

    The next call to :func:`get_config` or ``Scope2MarketConfig()``
    will re-read environment variables and construct a fresh instance.
    Intended for test teardown to prevent state leakage.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # fresh instance from env vars
    """
    Scope2MarketConfig.reset()


def validate_config() -> List[str]:
    """Validate the current singleton configuration.

    Convenience function that calls :meth:`Scope2MarketConfig.validate`
    on the current singleton instance.

    Returns:
        List of validation error strings (empty if valid).

    Example:
        >>> errors = validate_config()
        >>> assert len(errors) == 0
    """
    return get_config().validate()


# ---------------------------------------------------------------------------
# Constants for external consumers
# ---------------------------------------------------------------------------

#: Default enabled compliance frameworks
DEFAULT_ENABLED_FRAMEWORKS: List[str] = list(_DEFAULT_ENABLED_FRAMEWORKS)

#: Default instrument type hierarchy
DEFAULT_INSTRUMENT_HIERARCHY: List[str] = list(
    _DEFAULT_INSTRUMENT_HIERARCHY
)

#: Valid GWP sources
VALID_GWP_SOURCES: frozenset = _VALID_GWP_SOURCES

#: Valid allocation methods
VALID_ALLOCATION_METHODS: frozenset = _VALID_ALLOCATION_METHODS

#: Valid rounding modes
VALID_ROUNDING_MODES: frozenset = _VALID_ROUNDING_MODES

#: Valid SSL modes
VALID_SSL_MODES: frozenset = _VALID_SSL_MODES

#: Valid residual mix sources
VALID_RESIDUAL_SOURCES: frozenset = _VALID_RESIDUAL_SOURCES

#: Valid instrument types
VALID_INSTRUMENT_TYPES: frozenset = _VALID_INSTRUMENT_TYPES

#: Valid tracking systems
VALID_TRACKING_SYSTEMS: frozenset = _VALID_TRACKING_SYSTEMS

#: Valid compliance frameworks
VALID_FRAMEWORKS: frozenset = _VALID_FRAMEWORKS

#: Environment variable prefix
ENV_PREFIX: str = _ENV_PREFIX


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    # Core class
    "Scope2MarketConfig",
    # Module-level functions
    "get_config",
    "set_config",
    "reset_config",
    "validate_config",
    # Constants
    "DEFAULT_ENABLED_FRAMEWORKS",
    "DEFAULT_INSTRUMENT_HIERARCHY",
    "VALID_GWP_SOURCES",
    "VALID_ALLOCATION_METHODS",
    "VALID_ROUNDING_MODES",
    "VALID_SSL_MODES",
    "VALID_RESIDUAL_SOURCES",
    "VALID_INSTRUMENT_TYPES",
    "VALID_TRACKING_SYSTEMS",
    "VALID_FRAMEWORKS",
    "ENV_PREFIX",
]
