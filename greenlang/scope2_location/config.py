# -*- coding: utf-8 -*-
"""
Scope 2 Location-Based Emissions Agent Configuration - AGENT-MRV-009

Centralized configuration for the Scope 2 Location-Based Emissions Agent SDK
covering:
- Database connection and pool settings (PostgreSQL)
- Redis caching configuration
- Calculation settings (GWP source, time granularity, T&D losses, precision)
- Grid emission factor source hierarchy and data year defaults
- Uncertainty quantification (Monte Carlo) parameters
- API routing, rate limiting, and CORS
- Regulatory compliance framework toggles
- Logging and observability (Prometheus metrics, OpenTelemetry tracing)
- Feature flags (hourly/monthly/custom factors, uncertainty)

This module implements a singleton pattern using ``__new__`` with a
class-level ``_instance`` and ``_initialized`` flag, ensuring exactly one
configuration object exists across the application lifecycle. All settings
can be overridden via environment variables with the ``GL_S2L_`` prefix.

Environment Variable Reference (GL_S2L_ prefix):
    GL_S2L_DB_HOST                      - PostgreSQL host
    GL_S2L_DB_PORT                      - PostgreSQL port
    GL_S2L_DB_NAME                      - PostgreSQL database name
    GL_S2L_DB_USER                      - PostgreSQL username
    GL_S2L_DB_PASSWORD                  - PostgreSQL password
    GL_S2L_DB_POOL_MIN                  - Minimum connection pool size
    GL_S2L_DB_POOL_MAX                  - Maximum connection pool size
    GL_S2L_DB_SSL_MODE                  - PostgreSQL SSL mode
    GL_S2L_REDIS_HOST                   - Redis host
    GL_S2L_REDIS_PORT                   - Redis port
    GL_S2L_REDIS_DB                     - Redis database number
    GL_S2L_REDIS_PASSWORD               - Redis password
    GL_S2L_REDIS_TTL                    - Redis default TTL (seconds)
    GL_S2L_DEFAULT_GWP                  - Default GWP source (AR4/AR5/AR6)
    GL_S2L_DEFAULT_TIME_GRANULARITY     - Time granularity (annual/monthly/hourly)
    GL_S2L_INCLUDE_TD_LOSSES            - Include T&D losses by default
    GL_S2L_DECIMAL_PRECISION            - Decimal places for calculations
    GL_S2L_ROUNDING_MODE                - Decimal rounding mode
    GL_S2L_MAX_BATCH_SIZE               - Maximum records per batch
    GL_S2L_DEFAULT_EF_SOURCE_HIERARCHY  - Comma-separated EF source hierarchy
    GL_S2L_BIOGENIC_CO2_SEPARATE        - Track biogenic CO2 separately
    GL_S2L_EGRID_DATA_YEAR              - eGRID reference data year
    GL_S2L_IEA_DATA_YEAR                - IEA reference data year
    GL_S2L_DEFRA_DATA_YEAR              - DEFRA reference data year
    GL_S2L_AUTO_UPDATE_FACTORS          - Auto-update grid factors
    GL_S2L_FACTOR_CACHE_TTL             - Grid factor cache TTL (seconds)
    GL_S2L_DEFAULT_MC_ITERATIONS        - Monte Carlo iterations
    GL_S2L_DEFAULT_CONFIDENCE_LEVEL     - Confidence level (0.0-1.0)
    GL_S2L_EF_UNCERTAINTY_PCT           - Emission factor uncertainty %
    GL_S2L_ACTIVITY_DATA_UNCERTAINTY_PCT - Activity data uncertainty %
    GL_S2L_API_PREFIX                   - REST API route prefix
    GL_S2L_API_RATE_LIMIT               - API requests per minute
    GL_S2L_CORS_ORIGINS                 - Comma-separated CORS origins
    GL_S2L_ENABLE_DOCS                  - Enable API documentation
    GL_S2L_ENABLED_FRAMEWORKS           - Comma-separated compliance frameworks
    GL_S2L_AUTO_COMPLIANCE_CHECK        - Auto compliance check on calculation
    GL_S2L_DUAL_REPORTING_ENABLED       - Enable dual reporting (location+market)
    GL_S2L_LOG_LEVEL                    - Logging level
    GL_S2L_ENABLE_METRICS               - Enable Prometheus metrics export
    GL_S2L_METRICS_PREFIX               - Prometheus metrics prefix
    GL_S2L_ENABLE_TRACING               - Enable OpenTelemetry tracing
    GL_S2L_SERVICE_NAME                 - Service name for tracing
    GL_S2L_ENABLE_HOURLY_FACTORS        - Enable hourly grid factors
    GL_S2L_ENABLE_MONTHLY_FACTORS       - Enable monthly grid factors
    GL_S2L_ENABLE_CUSTOM_FACTORS        - Enable custom grid factors
    GL_S2L_ENABLE_UNCERTAINTY           - Enable uncertainty quantification
    GL_S2L_ENABLE_PROVENANCE            - Enable SHA-256 provenance tracking
    GL_S2L_GENESIS_HASH                 - Provenance chain genesis anchor
    GL_S2L_ENABLE_AUTH                  - Enable authentication middleware
    GL_S2L_WORKER_THREADS               - Worker thread pool size
    GL_S2L_ENABLE_BACKGROUND_TASKS      - Enable background task processing
    GL_S2L_HEALTH_CHECK_INTERVAL        - Health check interval (seconds)
    GL_S2L_ENABLED                      - Master enable/disable switch

Example:
    >>> from greenlang.scope2_location.config import Scope2LocationConfig
    >>> cfg = Scope2LocationConfig()
    >>> print(cfg.default_gwp_source, cfg.default_time_granularity)
    AR5 annual

    >>> # Check singleton
    >>> cfg2 = Scope2LocationConfig()
    >>> assert cfg is cfg2

    >>> # Reset for testing
    >>> Scope2LocationConfig.reset()
    >>> cfg3 = Scope2LocationConfig()
    >>> assert cfg is not cfg3

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-009 Scope 2 Location-Based Emissions (GL-MRV-SCOPE2-001)
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

_ENV_PREFIX: str = "GL_S2L_"

# ---------------------------------------------------------------------------
# Valid enumeration values for configuration validation
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

_VALID_GWP_SOURCES = frozenset({"AR4", "AR5", "AR6"})

_VALID_TIME_GRANULARITIES = frozenset({"annual", "monthly", "hourly"})

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

_VALID_EF_SOURCES = frozenset({
    "custom",
    "national",
    "egrid",
    "iea",
    "ipcc",
    "defra",
    "aib",
    "epa",
})

_VALID_FRAMEWORKS = frozenset({
    "ghg_protocol_scope2",
    "ipcc_2006",
    "iso_14064",
    "csrd_esrs",
    "epa_ghgrp",
    "defra",
    "cdp",
})

# ---------------------------------------------------------------------------
# Default EF source hierarchy
# ---------------------------------------------------------------------------

_DEFAULT_EF_SOURCE_HIERARCHY: List[str] = [
    "custom",
    "national",
    "egrid",
    "iea",
    "ipcc",
]

# ---------------------------------------------------------------------------
# Default compliance frameworks
# ---------------------------------------------------------------------------

_DEFAULT_ENABLED_FRAMEWORKS: List[str] = [
    "ghg_protocol_scope2",
    "ipcc_2006",
    "iso_14064",
    "csrd_esrs",
    "epa_ghgrp",
    "defra",
    "cdp",
]


# ---------------------------------------------------------------------------
# Scope2LocationConfig
# ---------------------------------------------------------------------------


class Scope2LocationConfig:
    """Singleton configuration for the Scope 2 Location-Based Emissions Agent.

    Implements a singleton pattern via ``__new__`` with a class-level
    ``_instance`` and ``_initialized`` flag. On first instantiation, all
    settings are loaded from environment variables with the ``GL_S2L_``
    prefix. Subsequent instantiations return the same object.

    The configuration covers nine domains:
    1. Database Settings - PostgreSQL connection and pool sizing
    2. Redis Settings - cache host, port, database, TTL
    3. Calculation Settings - GWP source, precision, EF hierarchy
    4. Grid Factor Settings - data years for eGRID/IEA/DEFRA
    5. Uncertainty Settings - Monte Carlo parameters
    6. API Settings - prefix, rate limiting, CORS
    7. Compliance Settings - enabled regulatory frameworks
    8. Logging & Observability - log level, metrics, tracing
    9. Feature Flags - hourly/monthly/custom factors, uncertainty

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
        default_time_granularity: Default temporal granularity for calculations.
        include_td_losses_default: Whether to include T&D losses by default.
        decimal_precision: Number of decimal places for calculations.
        rounding_mode: Python Decimal rounding mode name.
        max_batch_size: Maximum records per batch operation.
        default_ef_source_hierarchy: Ordered list of EF sources to query.
        biogenic_co2_separate: Track biogenic CO2 separately from fossil.
        egrid_data_year: eGRID reference data year.
        iea_data_year: IEA reference data year.
        defra_data_year: DEFRA reference data year.
        auto_update_factors: Automatically update grid factors on load.
        factor_cache_ttl: Grid factor cache TTL in seconds.
        default_mc_iterations: Number of Monte Carlo iterations.
        default_confidence_level: Confidence level for uncertainty intervals.
        ef_uncertainty_pct: Default emission factor uncertainty percentage.
        activity_data_uncertainty_pct: Default activity data uncertainty %.
        api_prefix: REST API URL prefix.
        api_rate_limit: Maximum API requests per minute.
        cors_origins: Allowed CORS origins.
        enable_docs: Enable interactive API documentation.
        enabled_frameworks: List of enabled compliance frameworks.
        auto_compliance_check: Run compliance check after each calculation.
        dual_reporting_enabled: Enable dual reporting (location + market).
        log_level: Logging verbosity level.
        enable_metrics: Enable Prometheus metrics export.
        metrics_prefix: Prometheus metric name prefix.
        enable_tracing: Enable OpenTelemetry distributed tracing.
        service_name: Service name for tracing spans.
        enable_hourly_factors: Enable hourly grid emission factors.
        enable_monthly_factors: Enable monthly grid emission factors.
        enable_custom_factors: Enable custom grid emission factors.
        enable_uncertainty: Enable uncertainty quantification.
        enable_provenance: Enable SHA-256 provenance hash tracking.
        genesis_hash: Genesis anchor for the provenance chain.
        enable_auth: Enable authentication middleware.
        worker_threads: Thread pool size for parallel operations.
        enable_background_tasks: Enable background task processing.
        health_check_interval: Health check interval in seconds.
        enabled: Master enable/disable switch for the agent.

    Example:
        >>> cfg = Scope2LocationConfig()
        >>> cfg.default_gwp_source
        'AR5'
        >>> cfg.get_db_url()
        'postgresql://greenlang@localhost:5432/greenlang?sslmode=prefer'
        >>> cfg.is_framework_enabled("cdp")
        True
        >>> cfg.get_ef_hierarchy()
        ['custom', 'national', 'egrid', 'iea', 'ipcc']
    """

    _instance: Optional[Scope2LocationConfig] = None
    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton constructor
    # ------------------------------------------------------------------

    def __new__(cls) -> Scope2LocationConfig:
        """Return the singleton instance, creating it on first call.

        Uses a threading lock to ensure thread-safe initialisation. Only
        one instance is ever created; subsequent calls return the same
        object without acquiring the lock (double-checked locking).

        Returns:
            The singleton Scope2LocationConfig instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialise configuration from environment variables.

        Guarded by the ``_initialized`` flag so that repeated calls to
        ``__init__`` (from repeated ``Scope2LocationConfig()`` calls) do
        not re-read environment variables or overwrite customised values.
        """
        if self.__class__._initialized:
            return
        self._load_from_env()
        self.__class__._initialized = True
        logger.info(
            "Scope2LocationConfig initialised from environment: "
            "gwp=%s, granularity=%s, td_losses=%s, "
            "precision=%d, batch_size=%d, "
            "ef_hierarchy=%s, frameworks=%s, "
            "metrics=%s, tracing=%s",
            self.default_gwp_source,
            self.default_time_granularity,
            self.include_td_losses_default,
            self.decimal_precision,
            self.max_batch_size,
            self.default_ef_source_hierarchy,
            self.enabled_frameworks,
            self.enable_metrics,
            self.enable_tracing,
        )

    # ------------------------------------------------------------------
    # Environment loading
    # ------------------------------------------------------------------

    def _load_from_env(self) -> None:
        """Load all configuration from environment variables.

        Reads environment variables with the ``GL_S2L_`` prefix and
        populates all instance attributes. Each setting has a sensible
        default so the agent can start with zero environment configuration.

        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer and float values are parsed with fallback to defaults on
        malformed input, emitting a WARNING log.
        List values are parsed from comma-separated strings.
        """
        # -- Database Settings -----------------------------------------------
        self.db_host: str = self._env_str("DB_HOST", "localhost")
        self.db_port: int = self._env_int("DB_PORT", 5432)
        self.db_name: str = self._env_str("DB_NAME", "greenlang")
        self.db_user: str = self._env_str("DB_USER", "greenlang")
        self.db_password: str = self._env_str("DB_PASSWORD", "")
        self.db_pool_min: int = self._env_int("DB_POOL_MIN", 2)
        self.db_pool_max: int = self._env_int("DB_POOL_MAX", 10)
        self.db_ssl_mode: str = self._env_str("DB_SSL_MODE", "prefer")

        # -- Redis Settings --------------------------------------------------
        self.redis_host: str = self._env_str("REDIS_HOST", "localhost")
        self.redis_port: int = self._env_int("REDIS_PORT", 6379)
        self.redis_db: int = self._env_int("REDIS_DB", 0)
        self.redis_password: str = self._env_str("REDIS_PASSWORD", "")
        self.redis_ttl: int = self._env_int("REDIS_TTL", 3600)

        # -- Calculation Settings --------------------------------------------
        self.default_gwp_source: str = self._env_str(
            "DEFAULT_GWP", "AR5"
        )
        self.default_time_granularity: str = self._env_str(
            "DEFAULT_TIME_GRANULARITY", "annual"
        )
        self.include_td_losses_default: bool = self._env_bool(
            "INCLUDE_TD_LOSSES", True
        )
        self.decimal_precision: int = self._env_int(
            "DECIMAL_PRECISION", 10
        )
        self.rounding_mode: str = self._env_str(
            "ROUNDING_MODE", "ROUND_HALF_UP"
        )
        self.max_batch_size: int = self._env_int("MAX_BATCH_SIZE", 1000)
        self.default_ef_source_hierarchy: List[str] = self._env_list(
            "DEFAULT_EF_SOURCE_HIERARCHY",
            _DEFAULT_EF_SOURCE_HIERARCHY,
        )
        self.biogenic_co2_separate: bool = self._env_bool(
            "BIOGENIC_CO2_SEPARATE", True
        )

        # -- Grid Factor Settings --------------------------------------------
        self.egrid_data_year: int = self._env_int("EGRID_DATA_YEAR", 2022)
        self.iea_data_year: int = self._env_int("IEA_DATA_YEAR", 2024)
        self.defra_data_year: int = self._env_int("DEFRA_DATA_YEAR", 2024)
        self.auto_update_factors: bool = self._env_bool(
            "AUTO_UPDATE_FACTORS", False
        )
        self.factor_cache_ttl: int = self._env_int(
            "FACTOR_CACHE_TTL", 86400
        )

        # -- Uncertainty Settings --------------------------------------------
        self.default_mc_iterations: int = self._env_int(
            "DEFAULT_MC_ITERATIONS", 10000
        )
        self.default_confidence_level: float = self._env_float(
            "DEFAULT_CONFIDENCE_LEVEL", 0.95
        )
        self.ef_uncertainty_pct: float = self._env_float(
            "EF_UNCERTAINTY_PCT", 0.10
        )
        self.activity_data_uncertainty_pct: float = self._env_float(
            "ACTIVITY_DATA_UNCERTAINTY_PCT", 0.05
        )

        # -- API Settings ----------------------------------------------------
        self.api_prefix: str = self._env_str(
            "API_PREFIX", "/api/v1/scope2-location"
        )
        self.api_rate_limit: int = self._env_int("API_RATE_LIMIT", 100)
        self.cors_origins: List[str] = self._env_list(
            "CORS_ORIGINS", ["*"]
        )
        self.enable_docs: bool = self._env_bool("ENABLE_DOCS", True)

        # -- Compliance Settings ---------------------------------------------
        self.enabled_frameworks: List[str] = self._env_list(
            "ENABLED_FRAMEWORKS",
            _DEFAULT_ENABLED_FRAMEWORKS,
        )
        self.auto_compliance_check: bool = self._env_bool(
            "AUTO_COMPLIANCE_CHECK", False
        )
        self.dual_reporting_enabled: bool = self._env_bool(
            "DUAL_REPORTING_ENABLED", True
        )

        # -- Logging & Observability -----------------------------------------
        self.log_level: str = self._env_str("LOG_LEVEL", "INFO")
        self.enable_metrics: bool = self._env_bool("ENABLE_METRICS", True)
        self.metrics_prefix: str = self._env_str(
            "METRICS_PREFIX", "gl_s2l"
        )
        self.enable_tracing: bool = self._env_bool(
            "ENABLE_TRACING", True
        )
        self.service_name: str = self._env_str(
            "SERVICE_NAME", "scope2-location-service"
        )

        # -- Feature Flags ---------------------------------------------------
        self.enable_hourly_factors: bool = self._env_bool(
            "ENABLE_HOURLY_FACTORS", False
        )
        self.enable_monthly_factors: bool = self._env_bool(
            "ENABLE_MONTHLY_FACTORS", True
        )
        self.enable_custom_factors: bool = self._env_bool(
            "ENABLE_CUSTOM_FACTORS", True
        )
        self.enable_uncertainty: bool = self._env_bool(
            "ENABLE_UNCERTAINTY", True
        )

        # -- Provenance Tracking ---------------------------------------------
        self.enable_provenance: bool = self._env_bool(
            "ENABLE_PROVENANCE", True
        )
        self.genesis_hash: str = self._env_str(
            "GENESIS_HASH",
            "GL-MRV-X-009-SCOPE2-LOCATION-GENESIS",
        )

        # -- Auth & Background Tasks -----------------------------------------
        self.enable_auth: bool = self._env_bool("ENABLE_AUTH", True)
        self.worker_threads: int = self._env_int("WORKER_THREADS", 4)
        self.enable_background_tasks: bool = self._env_bool(
            "ENABLE_BACKGROUND_TASKS", True
        )
        self.health_check_interval: int = self._env_int(
            "HEALTH_CHECK_INTERVAL", 30
        )

        # -- Master switch ---------------------------------------------------
        self.enabled: bool = self._env_bool("ENABLED", True)

    # ------------------------------------------------------------------
    # Environment variable parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _env_str(name: str, default: str) -> str:
        """Read a string environment variable with the GL_S2L_ prefix.

        Args:
            name: Variable name suffix (after GL_S2L_).
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
        """Read an integer environment variable with the GL_S2L_ prefix.

        Args:
            name: Variable name suffix (after GL_S2L_).
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
        """Read a float environment variable with the GL_S2L_ prefix.

        Args:
            name: Variable name suffix (after GL_S2L_).
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
        """Read a boolean environment variable with the GL_S2L_ prefix.

        Accepts ``true``, ``1``, ``yes`` (case-insensitive) as True.
        All other non-None values are treated as False.

        Args:
            name: Variable name suffix (after GL_S2L_).
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
            name: Variable name suffix (after GL_S2L_).
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
        ``Scope2LocationConfig()`` will re-read all environment variables
        and construct a fresh configuration object. Thread-safe.

        Example:
            >>> Scope2LocationConfig.reset()
            >>> cfg = Scope2LocationConfig()  # fresh instance
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
        logger.debug("Scope2LocationConfig singleton reset")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate all configuration settings.

        Performs comprehensive checks across all configuration domains:
        database connectivity parameters, Redis parameters, calculation
        settings, grid factor years, uncertainty parameters, API settings,
        compliance frameworks, logging levels, feature flags, provenance,
        and performance tuning.

        Returns:
            A list of human-readable error strings. An empty list means
            all validation checks passed.

        Example:
            >>> cfg = Scope2LocationConfig()
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

        # -- Grid Factor Settings --------------------------------------------
        errors.extend(self._validate_grid_factor_settings())

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
                "Scope2LocationConfig validation found %d error(s):\n%s",
                len(errors),
                "\n".join(f"  - {e}" for e in errors),
            )
        else:
            logger.debug(
                "Scope2LocationConfig validation passed: "
                "all %d checks OK",
                self._count_validation_checks(),
            )

        return errors

    def _validate_database_settings(self) -> List[str]:
        """Validate database connection settings.

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

        # -- Time granularity ------------------------------------------------
        normalised_gran = self.default_time_granularity.lower()
        if normalised_gran not in _VALID_TIME_GRANULARITIES:
            errors.append(
                f"default_time_granularity must be one of "
                f"{sorted(_VALID_TIME_GRANULARITIES)}, "
                f"got '{self.default_time_granularity}'"
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

        # -- EF source hierarchy ---------------------------------------------
        if not self.default_ef_source_hierarchy:
            errors.append(
                "default_ef_source_hierarchy must not be empty"
            )
        else:
            for source in self.default_ef_source_hierarchy:
                normalised = source.lower()
                if normalised not in _VALID_EF_SOURCES:
                    errors.append(
                        f"EF source '{source}' in hierarchy is not valid; "
                        f"must be one of {sorted(_VALID_EF_SOURCES)}"
                    )

            # Check for duplicates
            seen = set()
            for source in self.default_ef_source_hierarchy:
                normalised = source.lower()
                if normalised in seen:
                    errors.append(
                        f"Duplicate EF source '{source}' in hierarchy"
                    )
                seen.add(normalised)

        return errors

    def _validate_grid_factor_settings(self) -> List[str]:
        """Validate grid emission factor settings.

        Returns:
            List of error strings for invalid grid factor settings.
        """
        errors: List[str] = []

        # -- eGRID data year -------------------------------------------------
        if self.egrid_data_year < 2000:
            errors.append(
                f"egrid_data_year must be >= 2000, "
                f"got {self.egrid_data_year}"
            )
        if self.egrid_data_year > 2030:
            errors.append(
                f"egrid_data_year must be <= 2030, "
                f"got {self.egrid_data_year}"
            )

        # -- IEA data year ---------------------------------------------------
        if self.iea_data_year < 2000:
            errors.append(
                f"iea_data_year must be >= 2000, "
                f"got {self.iea_data_year}"
            )
        if self.iea_data_year > 2030:
            errors.append(
                f"iea_data_year must be <= 2030, "
                f"got {self.iea_data_year}"
            )

        # -- DEFRA data year -------------------------------------------------
        if self.defra_data_year < 2000:
            errors.append(
                f"defra_data_year must be >= 2000, "
                f"got {self.defra_data_year}"
            )
        if self.defra_data_year > 2030:
            errors.append(
                f"defra_data_year must be <= 2030, "
                f"got {self.defra_data_year}"
            )

        # -- Factor cache TTL ------------------------------------------------
        if self.factor_cache_ttl <= 0:
            errors.append(
                f"factor_cache_ttl must be > 0, "
                f"got {self.factor_cache_ttl}"
            )
        if self.factor_cache_ttl > 604800:
            errors.append(
                f"factor_cache_ttl must be <= 604800 (7 days), "
                f"got {self.factor_cache_ttl}"
            )

        return errors

    def _validate_uncertainty_settings(self) -> List[str]:
        """Validate uncertainty quantification settings.

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

        # -- EF uncertainty percentage ---------------------------------------
        if self.ef_uncertainty_pct < 0.0:
            errors.append(
                f"ef_uncertainty_pct must be >= 0.0, "
                f"got {self.ef_uncertainty_pct}"
            )
        if self.ef_uncertainty_pct > 1.0:
            errors.append(
                f"ef_uncertainty_pct must be <= 1.0, "
                f"got {self.ef_uncertainty_pct}"
            )

        # -- Activity data uncertainty percentage ----------------------------
        if self.activity_data_uncertainty_pct < 0.0:
            errors.append(
                f"activity_data_uncertainty_pct must be >= 0.0, "
                f"got {self.activity_data_uncertainty_pct}"
            )
        if self.activity_data_uncertainty_pct > 1.0:
            errors.append(
                f"activity_data_uncertainty_pct must be <= 1.0, "
                f"got {self.activity_data_uncertainty_pct}"
            )

        return errors

    def _validate_api_settings(self) -> List[str]:
        """Validate API settings.

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
            seen = set()
            for fw in self.enabled_frameworks:
                normalised = fw.lower()
                if normalised in seen:
                    errors.append(
                        f"Duplicate framework '{fw}' in enabled_frameworks"
                    )
                seen.add(normalised)

        # Dual reporting needs ghg_protocol_scope2
        if self.dual_reporting_enabled:
            normalised_fws = [fw.lower() for fw in self.enabled_frameworks]
            if "ghg_protocol_scope2" not in normalised_fws:
                errors.append(
                    "dual_reporting_enabled requires "
                    "'ghg_protocol_scope2' in enabled_frameworks"
                )

        return errors

    def _validate_logging_settings(self) -> List[str]:
        """Validate logging and observability settings.

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

        if not self.service_name:
            errors.append("service_name must not be empty")

        return errors

    def _validate_feature_flags(self) -> List[str]:
        """Validate feature flag consistency.

        Returns:
            List of error strings for invalid feature flag combinations.
        """
        errors: List[str] = []

        # Hourly factors require monthly factors to be enabled
        if self.enable_hourly_factors and not self.enable_monthly_factors:
            errors.append(
                "enable_hourly_factors requires "
                "enable_monthly_factors to be True"
            )

        return errors

    def _validate_provenance_settings(self) -> List[str]:
        """Validate provenance tracking settings.

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
        # Database: 10, Redis: 5, Calculation: 10, Grid: 7,
        # Uncertainty: 8, API: 4, Compliance: 4, Logging: 3,
        # Feature Flags: 1, Provenance: 2, Performance: 4
        return 58

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
            >>> cfg = Scope2LocationConfig()
            >>> d = cfg.to_dict()
            >>> d["default_gwp_source"]
            'AR5'
            >>> d["db_password"]
            '***'
        """
        return {
            # -- Master switch -----------------------------------------------
            "enabled": self.enabled,
            # -- Database Settings -------------------------------------------
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_name": self.db_name,
            "db_user": self.db_user,
            "db_password": "***" if self.db_password else "",
            "db_pool_min": self.db_pool_min,
            "db_pool_max": self.db_pool_max,
            "db_ssl_mode": self.db_ssl_mode,
            # -- Redis Settings ----------------------------------------------
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "redis_db": self.redis_db,
            "redis_password": "***" if self.redis_password else "",
            "redis_ttl": self.redis_ttl,
            # -- Calculation Settings ----------------------------------------
            "default_gwp_source": self.default_gwp_source,
            "default_time_granularity": self.default_time_granularity,
            "include_td_losses_default": self.include_td_losses_default,
            "decimal_precision": self.decimal_precision,
            "rounding_mode": self.rounding_mode,
            "max_batch_size": self.max_batch_size,
            "default_ef_source_hierarchy": list(
                self.default_ef_source_hierarchy
            ),
            "biogenic_co2_separate": self.biogenic_co2_separate,
            # -- Grid Factor Settings ----------------------------------------
            "egrid_data_year": self.egrid_data_year,
            "iea_data_year": self.iea_data_year,
            "defra_data_year": self.defra_data_year,
            "auto_update_factors": self.auto_update_factors,
            "factor_cache_ttl": self.factor_cache_ttl,
            # -- Uncertainty Settings ----------------------------------------
            "default_mc_iterations": self.default_mc_iterations,
            "default_confidence_level": self.default_confidence_level,
            "ef_uncertainty_pct": self.ef_uncertainty_pct,
            "activity_data_uncertainty_pct": (
                self.activity_data_uncertainty_pct
            ),
            # -- API Settings ------------------------------------------------
            "api_prefix": self.api_prefix,
            "api_rate_limit": self.api_rate_limit,
            "cors_origins": list(self.cors_origins),
            "enable_docs": self.enable_docs,
            # -- Compliance Settings -----------------------------------------
            "enabled_frameworks": list(self.enabled_frameworks),
            "auto_compliance_check": self.auto_compliance_check,
            "dual_reporting_enabled": self.dual_reporting_enabled,
            # -- Logging & Observability -------------------------------------
            "log_level": self.log_level,
            "enable_metrics": self.enable_metrics,
            "metrics_prefix": self.metrics_prefix,
            "enable_tracing": self.enable_tracing,
            "service_name": self.service_name,
            # -- Feature Flags -----------------------------------------------
            "enable_hourly_factors": self.enable_hourly_factors,
            "enable_monthly_factors": self.enable_monthly_factors,
            "enable_custom_factors": self.enable_custom_factors,
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
    def from_dict(cls, data: Dict[str, Any]) -> Scope2LocationConfig:
        """Deserialise a configuration from a dictionary.

        Creates a new Scope2LocationConfig instance and populates it
        from the provided dictionary. The singleton is reset first to
        allow the new configuration to be installed. Keys not present
        in the dictionary retain their environment-loaded defaults.

        Args:
            data: Dictionary of configuration key-value pairs. Keys
                correspond to attribute names on the config object.

        Returns:
            A new Scope2LocationConfig instance with values from the
            dictionary.

        Example:
            >>> d = {"default_gwp_source": "AR6", "decimal_precision": 12}
            >>> cfg = Scope2LocationConfig.from_dict(d)
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
            "Scope2LocationConfig loaded from dict: %d keys applied",
            len(data),
        )
        return instance

    def _apply_dict(self, data: Dict[str, Any]) -> None:
        """Apply dictionary values to the configuration instance.

        Only applies values for known attribute names. Unknown keys
        are logged as warnings and skipped.

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
            >>> cfg = Scope2LocationConfig()
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
            >>> cfg = Scope2LocationConfig()
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
            >>> cfg = Scope2LocationConfig()
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
    # EF hierarchy accessor
    # ------------------------------------------------------------------

    def get_ef_hierarchy(self) -> List[str]:
        """Return the emission factor source hierarchy.

        Returns a copy of the ordered list of emission factor sources
        to query, from highest to lowest priority. The calculation
        engine queries sources in this order and uses the first
        available factor.

        Returns:
            Ordered list of emission factor source identifiers.

        Example:
            >>> cfg = Scope2LocationConfig()
            >>> cfg.get_ef_hierarchy()
            ['custom', 'national', 'egrid', 'iea', 'ipcc']
        """
        return list(self.default_ef_source_hierarchy)

    # ------------------------------------------------------------------
    # Framework accessor
    # ------------------------------------------------------------------

    def is_framework_enabled(self, framework: str) -> bool:
        """Check if a specific compliance framework is enabled.

        Performs a case-insensitive comparison against the configured
        list of enabled frameworks.

        Args:
            framework: Framework identifier to check (e.g. "cdp",
                "ghg_protocol_scope2", "csrd_esrs").

        Returns:
            True if the framework is in the enabled list, False otherwise.

        Example:
            >>> cfg = Scope2LocationConfig()
            >>> cfg.is_framework_enabled("cdp")
            True
            >>> cfg.is_framework_enabled("unknown_framework")
            False
        """
        normalised = framework.lower()
        return normalised in [
            fw.lower() for fw in self.enabled_frameworks
        ]

    def get_enabled_frameworks(self) -> List[str]:
        """Return a copy of the enabled compliance frameworks list.

        Returns:
            List of enabled framework identifiers.

        Example:
            >>> cfg = Scope2LocationConfig()
            >>> "cdp" in cfg.get_enabled_frameworks()
            True
        """
        return list(self.enabled_frameworks)

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
            >>> cfg = Scope2LocationConfig()
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
    # Grid factor data year accessors
    # ------------------------------------------------------------------

    def get_grid_factor_years(self) -> Dict[str, int]:
        """Return a dictionary of grid factor data years by source.

        Returns:
            Dictionary mapping source name to data year.

        Example:
            >>> cfg = Scope2LocationConfig()
            >>> years = cfg.get_grid_factor_years()
            >>> years["egrid"]
            2022
        """
        return {
            "egrid": self.egrid_data_year,
            "iea": self.iea_data_year,
            "defra": self.defra_data_year,
        }

    # ------------------------------------------------------------------
    # Uncertainty parameter accessor
    # ------------------------------------------------------------------

    def get_uncertainty_params(self) -> Dict[str, Any]:
        """Return uncertainty quantification parameters as a dictionary.

        Returns:
            Dictionary containing all uncertainty-related settings.

        Example:
            >>> cfg = Scope2LocationConfig()
            >>> params = cfg.get_uncertainty_params()
            >>> params["mc_iterations"]
            10000
        """
        return {
            "mc_iterations": self.default_mc_iterations,
            "confidence_level": self.default_confidence_level,
            "ef_uncertainty_pct": self.ef_uncertainty_pct,
            "activity_data_uncertainty_pct": (
                self.activity_data_uncertainty_pct
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
            >>> cfg = Scope2LocationConfig()
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
            >>> cfg = Scope2LocationConfig()
            >>> api_cfg = cfg.get_api_config()
            >>> api_cfg["prefix"]
            '/api/v1/scope2-location'
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
            >>> cfg = Scope2LocationConfig()
            >>> obs_cfg = cfg.get_observability_config()
            >>> obs_cfg["metrics_prefix"]
            'gl_s2l'
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
            >>> cfg = Scope2LocationConfig()
            >>> flags = cfg.get_feature_flags()
            >>> flags["enable_monthly_factors"]
            True
        """
        return {
            "enable_hourly_factors": self.enable_hourly_factors,
            "enable_monthly_factors": self.enable_monthly_factors,
            "enable_custom_factors": self.enable_custom_factors,
            "enable_uncertainty": self.enable_uncertainty,
            "enable_provenance": self.enable_provenance,
            "enable_auth": self.enable_auth,
            "enable_metrics": self.enable_metrics,
            "enable_tracing": self.enable_tracing,
            "enable_docs": self.enable_docs,
            "enable_background_tasks": self.enable_background_tasks,
            "auto_compliance_check": self.auto_compliance_check,
            "dual_reporting_enabled": self.dual_reporting_enabled,
            "biogenic_co2_separate": self.biogenic_co2_separate,
            "include_td_losses_default": self.include_td_losses_default,
            "auto_update_factors": self.auto_update_factors,
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
        return f"Scope2LocationConfig({pairs})"

    def __str__(self) -> str:
        """Return a human-readable summary of the configuration.

        Returns:
            Multi-line string summary of key settings.
        """
        return (
            f"Scope2LocationConfig("
            f"enabled={self.enabled}, "
            f"gwp={self.default_gwp_source}, "
            f"granularity={self.default_time_granularity}, "
            f"td_losses={self.include_td_losses_default}, "
            f"precision={self.decimal_precision}, "
            f"batch={self.max_batch_size}, "
            f"frameworks={len(self.enabled_frameworks)}, "
            f"uncertainty={self.enable_uncertainty}"
            f")"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality by comparing all serialised values.

        Args:
            other: Object to compare against.

        Returns:
            True if other is a Scope2LocationConfig with identical settings.
        """
        if not isinstance(other, Scope2LocationConfig):
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
            >>> cfg = Scope2LocationConfig()
            >>> json_str = cfg.to_json()
            >>> '"default_gwp_source": "AR5"' in json_str
            True
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, json_str: str) -> Scope2LocationConfig:
        """Deserialise a configuration from a JSON string.

        Args:
            json_str: JSON string containing configuration key-value pairs.

        Returns:
            A new Scope2LocationConfig instance.

        Example:
            >>> json_str = '{"default_gwp_source": "AR6"}'
            >>> cfg = Scope2LocationConfig.from_json(json_str)
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
        source to uppercase, time granularity to lowercase, log level
        to uppercase, SSL mode to lowercase). This method is idempotent.

        Example:
            >>> cfg = Scope2LocationConfig()
            >>> cfg.default_gwp_source = "ar5"
            >>> cfg.normalise()
            >>> cfg.default_gwp_source
            'AR5'
        """
        # GWP source -> uppercase
        self.default_gwp_source = self.default_gwp_source.upper()

        # Time granularity -> lowercase
        self.default_time_granularity = (
            self.default_time_granularity.lower()
        )

        # Rounding mode -> uppercase
        self.rounding_mode = self.rounding_mode.upper()

        # Log level -> uppercase
        self.log_level = self.log_level.upper()

        # SSL mode -> lowercase
        self.db_ssl_mode = self.db_ssl_mode.lower()

        # EF hierarchy -> lowercase
        self.default_ef_source_hierarchy = [
            s.lower() for s in self.default_ef_source_hierarchy
        ]

        # Frameworks -> lowercase
        self.enabled_frameworks = [
            fw.lower() for fw in self.enabled_frameworks
        ]

        # Metrics prefix -> lowercase (convention)
        self.metrics_prefix = self.metrics_prefix.lower()

        logger.debug(
            "Scope2LocationConfig normalised: gwp=%s, granularity=%s, "
            "log_level=%s, ssl_mode=%s",
            self.default_gwp_source,
            self.default_time_granularity,
            self.log_level,
            self.db_ssl_mode,
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
            >>> cfg = Scope2LocationConfig()
            >>> cfg.merge({"decimal_precision": 12})
            >>> cfg.decimal_precision
            12
        """
        self._apply_dict(overrides)
        logger.debug(
            "Scope2LocationConfig merged %d overrides",
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
            >>> cfg = Scope2LocationConfig()
            >>> summary = cfg.health_summary()
            >>> summary["validation_status"]
            'PASS'
        """
        errors = self.validate()
        flags = self.get_feature_flags()
        enabled_flag_count = sum(1 for v in flags.values() if v)

        return {
            "agent": "scope2-location",
            "agent_id": "AGENT-MRV-009",
            "enabled": self.enabled,
            "validation_status": "PASS" if not errors else "FAIL",
            "validation_errors": len(errors),
            "gwp_source": self.default_gwp_source,
            "time_granularity": self.default_time_granularity,
            "decimal_precision": self.decimal_precision,
            "max_batch_size": self.max_batch_size,
            "enabled_frameworks": len(self.enabled_frameworks),
            "enabled_features": enabled_flag_count,
            "total_features": len(flags),
            "ef_hierarchy_depth": len(self.default_ef_source_hierarchy),
            "db_host": self.db_host,
            "db_port": self.db_port,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "worker_threads": self.worker_threads,
            "provenance_enabled": self.enable_provenance,
            "metrics_prefix": self.metrics_prefix,
            "service_name": self.service_name,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def get_config() -> Scope2LocationConfig:
    """Return the singleton Scope2LocationConfig.

    Convenience function that delegates to the singleton constructor.
    Thread-safe via the class-level lock in ``__new__``.

    Returns:
        Scope2LocationConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.default_gwp_source
        'AR5'
    """
    return Scope2LocationConfig()


def set_config(
    overrides: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Scope2LocationConfig:
    """Reset and re-create the singleton with optional overrides.

    Resets the singleton, creates a fresh instance from environment
    variables, then applies any provided overrides. This is the
    primary entry point for test setup.

    Args:
        overrides: Dictionary of configuration overrides.
        **kwargs: Additional keyword overrides (merged with overrides).

    Returns:
        The new Scope2LocationConfig singleton.

    Example:
        >>> cfg = set_config(default_gwp_source="AR6")
        >>> cfg.default_gwp_source
        'AR6'
    """
    Scope2LocationConfig.reset()
    cfg = Scope2LocationConfig()

    merged: Dict[str, Any] = {}
    if overrides:
        merged.update(overrides)
    merged.update(kwargs)

    if merged:
        cfg._apply_dict(merged)

    logger.info(
        "Scope2LocationConfig set with %d overrides: "
        "enabled=%s, gwp=%s, granularity=%s, "
        "precision=%d, batch_size=%d",
        len(merged),
        cfg.enabled,
        cfg.default_gwp_source,
        cfg.default_time_granularity,
        cfg.decimal_precision,
        cfg.max_batch_size,
    )
    return cfg


def reset_config() -> None:
    """Reset the singleton Scope2LocationConfig to None.

    The next call to :func:`get_config` or ``Scope2LocationConfig()``
    will re-read environment variables and construct a fresh instance.
    Intended for test teardown to prevent state leakage.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # fresh instance from env vars
    """
    Scope2LocationConfig.reset()


def validate_config() -> List[str]:
    """Validate the current singleton configuration.

    Convenience function that calls :meth:`Scope2LocationConfig.validate`
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

#: Default EF source hierarchy for documentation and schema generation
DEFAULT_EF_SOURCE_HIERARCHY: List[str] = list(_DEFAULT_EF_SOURCE_HIERARCHY)

#: Default enabled compliance frameworks
DEFAULT_ENABLED_FRAMEWORKS: List[str] = list(_DEFAULT_ENABLED_FRAMEWORKS)

#: Valid GWP sources
VALID_GWP_SOURCES: frozenset = _VALID_GWP_SOURCES

#: Valid time granularities
VALID_TIME_GRANULARITIES: frozenset = _VALID_TIME_GRANULARITIES

#: Valid rounding modes
VALID_ROUNDING_MODES: frozenset = _VALID_ROUNDING_MODES

#: Valid SSL modes
VALID_SSL_MODES: frozenset = _VALID_SSL_MODES

#: Valid EF sources
VALID_EF_SOURCES: frozenset = _VALID_EF_SOURCES

#: Valid compliance frameworks
VALID_FRAMEWORKS: frozenset = _VALID_FRAMEWORKS

#: Environment variable prefix
ENV_PREFIX: str = _ENV_PREFIX


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    # Core class
    "Scope2LocationConfig",
    # Module-level functions
    "get_config",
    "set_config",
    "reset_config",
    "validate_config",
    # Constants
    "DEFAULT_EF_SOURCE_HIERARCHY",
    "DEFAULT_ENABLED_FRAMEWORKS",
    "VALID_GWP_SOURCES",
    "VALID_TIME_GRANULARITIES",
    "VALID_ROUNDING_MODES",
    "VALID_SSL_MODES",
    "VALID_EF_SOURCES",
    "VALID_FRAMEWORKS",
    "ENV_PREFIX",
]
