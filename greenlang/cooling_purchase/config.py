# -*- coding: utf-8 -*-
"""
Cooling Purchase Agent Configuration - AGENT-MRV-012

Centralized configuration for the Cooling Purchase Agent SDK covering:
- General service settings (name, version, logging, environment, tenant)
- Database connection and pool settings (PostgreSQL)
- Calculation defaults (GWP source, tier, COP source, decimal places,
  distribution loss, auxiliary percentage, parasitic ratio, IPLV)
- Electric chiller defaults (AHRI 550/590 part-load weights, condenser
  type, COP bounds)
- Thermal energy storage (TES) defaults (ice, chilled water, PCM
  round-trip efficiencies, temporal shifting)
- Uncertainty quantification (Monte Carlo) parameters with per-tier
  default uncertainties
- Regulatory compliance framework toggles (GHG Protocol, ISO 14064,
  CSRD ESRS, CDP, SBTi, ASHRAE 90.1, EU F-Gas)
- Redis/cache settings (URL, TTL, enable flag)
- API settings (prefix, rate limit, page sizes)

This module implements a thread-safe singleton pattern using ``__new__``
with a class-level ``_instance``, ``_initialized`` flag, and
``threading.RLock``, ensuring exactly one configuration object exists
across the application lifecycle. All numeric settings are stored as
``Decimal`` for zero-hallucination deterministic calculations.

All settings can be overridden via environment variables with the
``GL_CP_`` prefix.

Environment Variable Reference (GL_CP_ prefix):
    GL_CP_SERVICE_NAME                  - Service name for tracing
    GL_CP_VERSION                       - Service version string
    GL_CP_LOG_LEVEL                     - Logging level
    GL_CP_ENVIRONMENT                   - Deployment environment
    GL_CP_MAX_BATCH_SIZE                - Maximum records per batch
    GL_CP_DEFAULT_TENANT                - Default tenant identifier
    GL_CP_DB_HOST                       - PostgreSQL host
    GL_CP_DB_PORT                       - PostgreSQL port
    GL_CP_DB_NAME                       - PostgreSQL database name
    GL_CP_DB_USER                       - PostgreSQL username
    GL_CP_DB_PASSWORD                   - PostgreSQL password
    GL_CP_DB_POOL_MIN                   - Minimum connection pool size
    GL_CP_DB_POOL_MAX                   - Maximum connection pool size
    GL_CP_DB_SSL_MODE                   - PostgreSQL SSL mode
    GL_CP_TABLE_PREFIX                  - Database table name prefix
    GL_CP_DEFAULT_GWP_SOURCE            - Default GWP source (ar4/ar5/ar6)
    GL_CP_DEFAULT_TIER                  - Default data quality tier
    GL_CP_DEFAULT_COP_SOURCE            - Default COP data source
    GL_CP_DECIMAL_PLACES                - Decimal places for calculations
    GL_CP_MAX_TRACE_STEPS               - Max provenance trace steps
    GL_CP_USE_IPLV_DEFAULT              - Use IPLV for part-load default
    GL_CP_DEFAULT_AUXILIARY_PCT         - Default auxiliary energy fraction
    GL_CP_DEFAULT_PARASITIC_RATIO       - Default parasitic load ratio
    GL_CP_DEFAULT_DISTRIBUTION_LOSS     - Default distribution loss fraction
    GL_CP_AHRI_100_WEIGHT              - AHRI 100% load weight
    GL_CP_AHRI_75_WEIGHT               - AHRI 75% load weight
    GL_CP_AHRI_50_WEIGHT               - AHRI 50% load weight
    GL_CP_AHRI_25_WEIGHT               - AHRI 25% load weight
    GL_CP_DEFAULT_CONDENSER_TYPE        - Default condenser type
    GL_CP_MIN_COP                       - Minimum valid COP
    GL_CP_MAX_COP                       - Maximum valid COP
    GL_CP_ICE_ROUND_TRIP_EFF            - Ice TES round-trip efficiency
    GL_CP_CW_ROUND_TRIP_EFF             - Chilled water TES round-trip eff
    GL_CP_PCM_ROUND_TRIP_EFF            - PCM TES round-trip efficiency
    GL_CP_TES_ENABLE_TEMPORAL_SHIFTING  - Enable TES temporal shifting
    GL_CP_MONTE_CARLO_ITERATIONS        - Monte Carlo iterations
    GL_CP_CONFIDENCE_LEVEL              - Confidence level (0.0-1.0)
    GL_CP_DEFAULT_SEED                  - Random seed for reproducibility
    GL_CP_TIER1_UNCERTAINTY             - Tier 1 default uncertainty (0-1)
    GL_CP_TIER2_UNCERTAINTY             - Tier 2 default uncertainty (0-1)
    GL_CP_TIER3_UNCERTAINTY             - Tier 3 default uncertainty (0-1)
    GL_CP_ENABLED_FRAMEWORKS            - Comma-separated frameworks
    GL_CP_STRICT_MODE                   - Enable strict compliance mode
    GL_CP_FAIL_ON_NON_COMPLIANT         - Fail on non-compliant results
    GL_CP_REDIS_URL                     - Redis connection URL
    GL_CP_CACHE_TTL                     - Cache TTL in seconds
    GL_CP_ENABLE_CACHING                - Enable Redis caching
    GL_CP_API_PREFIX                    - REST API route prefix
    GL_CP_RATE_LIMIT                    - API requests per minute
    GL_CP_PAGE_SIZE                     - Default page size for list APIs
    GL_CP_MAX_PAGE_SIZE                 - Maximum page size for list APIs
    GL_CP_ENABLE_METRICS                - Enable Prometheus metrics export
    GL_CP_METRICS_PREFIX                - Prometheus metrics prefix
    GL_CP_ENABLE_TRACING                - Enable OpenTelemetry tracing
    GL_CP_ENABLE_PROVENANCE             - Enable SHA-256 provenance tracking
    GL_CP_GENESIS_HASH                  - Provenance chain genesis anchor
    GL_CP_ENABLE_AUTH                   - Enable authentication middleware
    GL_CP_WORKER_THREADS                - Worker thread pool size
    GL_CP_ENABLE_BACKGROUND_TASKS       - Enable background task processing
    GL_CP_HEALTH_CHECK_INTERVAL         - Health check interval (seconds)
    GL_CP_CORS_ORIGINS                  - Comma-separated CORS origins
    GL_CP_ENABLE_DOCS                   - Enable API documentation
    GL_CP_ENABLED                       - Master enable/disable switch

Example:
    >>> from greenlang.cooling_purchase.config import CoolingPurchaseConfig
    >>> cfg = CoolingPurchaseConfig()
    >>> print(cfg.service_name, cfg.default_gwp_source)
    cooling-purchase-service ar5

    >>> # Check singleton
    >>> cfg2 = CoolingPurchaseConfig()
    >>> assert cfg is cfg2

    >>> # Reset for testing
    >>> CoolingPurchaseConfig.reset()
    >>> cfg3 = CoolingPurchaseConfig()
    >>> assert cfg is not cfg3

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-012 Cooling Purchase Agent (GL-MRV-X-023)
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

_ENV_PREFIX: str = "GL_CP_"

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

_VALID_TIERS = frozenset({
    "tier_1",
    "tier_2",
    "tier_3",
})

_VALID_COP_SOURCES = frozenset({
    "nameplate",
    "measured",
    "ahri_certified",
    "manufacturer",
    "default",
})

_VALID_CONDENSER_TYPES = frozenset({
    "water_cooled",
    "air_cooled",
    "evaporative",
})

_VALID_TES_TYPES = frozenset({
    "ice",
    "chilled_water",
    "pcm",
})

_VALID_CHILLER_TYPES = frozenset({
    "centrifugal",
    "screw",
    "scroll",
    "reciprocating",
    "absorption_single_effect",
    "absorption_double_effect",
    "magnetic_bearing",
    "vsd_centrifugal",
})

_VALID_COOLING_TECHNOLOGIES = frozenset({
    "electric_chiller",
    "absorption_chiller",
    "district_cooling",
    "free_cooling",
    "evaporative_cooling",
    "thermal_energy_storage",
    "ground_source_heat_pump",
    "air_source_heat_pump",
    "water_source_heat_pump",
    "hybrid_cooling",
})

_VALID_UNCERTAINTY_METHODS = frozenset({
    "monte_carlo",
    "analytical",
    "bootstrap",
    "latin_hypercube",
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

_VALID_FRAMEWORKS = frozenset({
    "ghg_protocol",
    "iso_14064",
    "csrd_esrs",
    "cdp",
    "sbti",
    "ashrae_90_1",
    "eu_fgas",
})

# ---------------------------------------------------------------------------
# Default compliance frameworks
# ---------------------------------------------------------------------------

_DEFAULT_ENABLED_FRAMEWORKS: List[str] = [
    "ghg_protocol",
    "iso_14064",
    "csrd_esrs",
    "cdp",
    "sbti",
    "ashrae_90_1",
    "eu_fgas",
]


# ---------------------------------------------------------------------------
# CoolingPurchaseConfig
# ---------------------------------------------------------------------------


class CoolingPurchaseConfig:
    """Singleton configuration for the Cooling Purchase Agent.

    Implements a thread-safe singleton pattern via ``__new__`` with a
    class-level ``_instance``, ``_initialized`` flag, and
    ``threading.RLock``. On first instantiation, all settings are loaded
    from environment variables with the ``GL_CP_`` prefix. Subsequent
    instantiations return the same object.

    All numeric values are stored as ``Decimal`` to ensure
    zero-hallucination deterministic arithmetic throughout the cooling
    purchase calculation pipeline. This eliminates IEEE 754 floating-point
    representation errors that could compound across chiller efficiency,
    AHRI part-load, TES round-trip, and distribution loss calculations.

    The configuration covers nine domains:
    1. General Settings - service name, version, logging, environment,
       batch size, default tenant
    2. Database Settings - PostgreSQL connection, pool sizing, SSL mode,
       table prefix
    3. Calculation Defaults - GWP source, data quality tier, COP source,
       decimal places, trace steps, IPLV flag, auxiliary percentage,
       parasitic ratio, distribution loss
    4. Electric Chiller Defaults - AHRI 550/590 part-load weights (100%,
       75%, 50%, 25%), condenser type, COP min/max bounds
    5. TES Defaults - ice/chilled-water/PCM round-trip efficiencies,
       temporal shifting enable flag
    6. Uncertainty Settings - Monte Carlo method, iterations, confidence
       level, seed, per-tier uncertainty percentages
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
        default_tier: Default IPCC data quality tier.
        default_cop_source: Default source for COP values.
        decimal_places: Number of decimal places for calculations.
        max_trace_steps: Maximum provenance trace steps.
        use_iplv_default: Use IPLV for part-load calculations by default.
        default_auxiliary_pct: Default auxiliary energy fraction (0-1).
        default_parasitic_ratio: Default parasitic load ratio (0-1).
        default_distribution_loss: Default distribution loss fraction (0-1).
        ahri_100_weight: AHRI 550/590 weight at 100% load.
        ahri_75_weight: AHRI 550/590 weight at 75% load.
        ahri_50_weight: AHRI 550/590 weight at 50% load.
        ahri_25_weight: AHRI 550/590 weight at 25% load.
        default_condenser_type: Default chiller condenser type.
        min_cop: Minimum valid COP value.
        max_cop: Maximum valid COP value.
        ice_round_trip_eff: Ice TES round-trip efficiency.
        cw_round_trip_eff: Chilled water TES round-trip efficiency.
        pcm_round_trip_eff: PCM TES round-trip efficiency.
        tes_enable_temporal_shifting: Enable temporal shifting for TES.
        monte_carlo_iterations: Number of Monte Carlo iterations.
        confidence_level: Confidence level for uncertainty intervals.
        default_seed: Random seed for reproducibility.
        tier1_uncertainty: Tier 1 default uncertainty fraction.
        tier2_uncertainty: Tier 2 default uncertainty fraction.
        tier3_uncertainty: Tier 3 default uncertainty fraction.
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
        >>> cfg = CoolingPurchaseConfig()
        >>> cfg.default_gwp_source
        'ar5'
        >>> cfg.get_db_dsn()
        'postgresql://greenlang@localhost:5432/greenlang?sslmode=prefer'
        >>> cfg.is_framework_enabled("cdp")
        True
        >>> cfg.get_ahri_weights()
        {'100': Decimal('0.01'), '75': Decimal('0.42'), ...}
    """

    _instance: Optional[CoolingPurchaseConfig] = None
    _initialized: bool = False
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton constructor
    # ------------------------------------------------------------------

    def __new__(cls) -> CoolingPurchaseConfig:
        """Return the singleton instance, creating it on first call.

        Uses a threading RLock to ensure thread-safe initialisation. Only
        one instance is ever created; subsequent calls return the same
        object without acquiring the lock (double-checked locking).

        Returns:
            The singleton CoolingPurchaseConfig instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialise configuration from environment variables.

        Guarded by the ``_initialized`` flag so that repeated calls to
        ``__init__`` (from repeated ``CoolingPurchaseConfig()`` calls)
        do not re-read environment variables or overwrite customised
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
                "CoolingPurchaseConfig initialised from environment: "
                "service=%s, version=%s, "
                "gwp=%s, tier=%s, "
                "cop_source=%s, condenser=%s, "
                "frameworks=%s, env=%s",
                self.service_name,
                self.version,
                self.default_gwp_source,
                self.default_tier,
                self.default_cop_source,
                self.default_condenser_type,
                self.enabled_frameworks,
                self.environment,
            )

    # ------------------------------------------------------------------
    # Environment loading
    # ------------------------------------------------------------------

    def _load_from_env(self) -> None:
        """Load all configuration from environment variables.

        Reads environment variables with the ``GL_CP_`` prefix and
        populates all instance attributes. Each setting has a sensible
        default so the agent can start with zero environment configuration.

        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed with fallback to defaults on
        malformed input, emitting a WARNING log.
        Decimal values are parsed from string representations for
        exact precision; malformed input falls back to defaults.
        List values are parsed from comma-separated strings.
        """
        # -- 1. General Settings -------------------------------------------------
        self.service_name: str = self._env_str(
            "SERVICE_NAME", "cooling-purchase-service"
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

        # -- 2. Database Settings ------------------------------------------------
        self.db_host: str = self._env_str("DB_HOST", "localhost")
        self.db_port: int = self._env_int("DB_PORT", 5432)
        self.db_name: str = self._env_str("DB_NAME", "greenlang")
        self.db_user: str = self._env_str("DB_USER", "greenlang")
        self.db_password: str = self._env_str("DB_PASSWORD", "")
        self.db_pool_min: int = self._env_int("DB_POOL_MIN", 2)
        self.db_pool_max: int = self._env_int("DB_POOL_MAX", 10)
        self.db_ssl_mode: str = self._env_str("DB_SSL_MODE", "prefer")
        self.table_prefix: str = self._env_str(
            "TABLE_PREFIX", "gl_cp_"
        )

        # -- 3. Calculation Defaults ---------------------------------------------
        self.default_gwp_source: str = self._env_str(
            "DEFAULT_GWP_SOURCE", "ar5"
        )
        self.default_tier: str = self._env_str(
            "DEFAULT_TIER", "tier_2"
        )
        self.default_cop_source: str = self._env_str(
            "DEFAULT_COP_SOURCE", "nameplate"
        )
        self.decimal_places: int = self._env_int(
            "DECIMAL_PLACES", 8
        )
        self.max_trace_steps: int = self._env_int(
            "MAX_TRACE_STEPS", 200
        )
        self.use_iplv_default: bool = self._env_bool(
            "USE_IPLV_DEFAULT", True
        )
        self.default_auxiliary_pct: Decimal = self._env_decimal(
            "DEFAULT_AUXILIARY_PCT", Decimal("0.05")
        )
        self.default_parasitic_ratio: Decimal = self._env_decimal(
            "DEFAULT_PARASITIC_RATIO", Decimal("0.05")
        )
        self.default_distribution_loss: Decimal = self._env_decimal(
            "DEFAULT_DISTRIBUTION_LOSS", Decimal("0.08")
        )

        # -- 4. Electric Chiller Defaults ----------------------------------------
        self.ahri_100_weight: Decimal = self._env_decimal(
            "AHRI_100_WEIGHT", Decimal("0.01")
        )
        self.ahri_75_weight: Decimal = self._env_decimal(
            "AHRI_75_WEIGHT", Decimal("0.42")
        )
        self.ahri_50_weight: Decimal = self._env_decimal(
            "AHRI_50_WEIGHT", Decimal("0.45")
        )
        self.ahri_25_weight: Decimal = self._env_decimal(
            "AHRI_25_WEIGHT", Decimal("0.12")
        )
        self.default_condenser_type: str = self._env_str(
            "DEFAULT_CONDENSER_TYPE", "water_cooled"
        )
        self.min_cop: Decimal = self._env_decimal(
            "MIN_COP", Decimal("0.3")
        )
        self.max_cop: Decimal = self._env_decimal(
            "MAX_COP", Decimal("35.0")
        )

        # -- 5. TES Defaults -----------------------------------------------------
        self.ice_round_trip_eff: Decimal = self._env_decimal(
            "ICE_ROUND_TRIP_EFF", Decimal("0.85")
        )
        self.cw_round_trip_eff: Decimal = self._env_decimal(
            "CW_ROUND_TRIP_EFF", Decimal("0.95")
        )
        self.pcm_round_trip_eff: Decimal = self._env_decimal(
            "PCM_ROUND_TRIP_EFF", Decimal("0.90")
        )
        self.tes_enable_temporal_shifting: bool = self._env_bool(
            "TES_ENABLE_TEMPORAL_SHIFTING", True
        )

        # -- 6. Uncertainty Settings ---------------------------------------------
        self.monte_carlo_iterations: int = self._env_int(
            "MONTE_CARLO_ITERATIONS", 10000
        )
        self.confidence_level: Decimal = self._env_decimal(
            "CONFIDENCE_LEVEL", Decimal("0.95")
        )
        self.default_seed: int = self._env_int("DEFAULT_SEED", 42)
        self.tier1_uncertainty: Decimal = self._env_decimal(
            "TIER1_UNCERTAINTY", Decimal("0.40")
        )
        self.tier2_uncertainty: Decimal = self._env_decimal(
            "TIER2_UNCERTAINTY", Decimal("0.20")
        )
        self.tier3_uncertainty: Decimal = self._env_decimal(
            "TIER3_UNCERTAINTY", Decimal("0.10")
        )

        # -- 7. Compliance Settings ----------------------------------------------
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

        # -- 8. Redis/Cache Settings ---------------------------------------------
        self.redis_url: str = self._env_str(
            "REDIS_URL", "redis://localhost:6379/0"
        )
        self.cache_ttl: int = self._env_int("CACHE_TTL", 3600)
        self.enable_caching: bool = self._env_bool(
            "ENABLE_CACHING", True
        )

        # -- 9. API Settings -----------------------------------------------------
        self.api_prefix: str = self._env_str(
            "API_PREFIX", "/api/v1/cooling-purchase"
        )
        self.rate_limit: int = self._env_int("RATE_LIMIT", 1000)
        self.page_size: int = self._env_int("PAGE_SIZE", 100)
        self.max_page_size: int = self._env_int("MAX_PAGE_SIZE", 10000)

        # -- Logging & Observability ---------------------------------------------
        self.enable_metrics: bool = self._env_bool(
            "ENABLE_METRICS", True
        )
        self.metrics_prefix: str = self._env_str(
            "METRICS_PREFIX", "gl_cp_"
        )
        self.enable_tracing: bool = self._env_bool(
            "ENABLE_TRACING", True
        )

        # -- Provenance Tracking -------------------------------------------------
        self.enable_provenance: bool = self._env_bool(
            "ENABLE_PROVENANCE", True
        )
        self.genesis_hash: str = self._env_str(
            "GENESIS_HASH",
            "GL-MRV-X-023-COOLING-PURCHASE-GENESIS",
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

        # -- CORS & Docs ---------------------------------------------------------
        self.cors_origins: List[str] = self._env_list(
            "CORS_ORIGINS", ["*"]
        )
        self.enable_docs: bool = self._env_bool("ENABLE_DOCS", True)

        # -- Master switch -------------------------------------------------------
        self.enabled: bool = self._env_bool("ENABLED", True)

    # ------------------------------------------------------------------
    # Environment variable parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _env_str(name: str, default: str) -> str:
        """Read a string environment variable with the GL_CP_ prefix.

        Args:
            name: Variable name suffix (after GL_CP_).
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
        """Read an integer environment variable with the GL_CP_ prefix.

        Args:
            name: Variable name suffix (after GL_CP_).
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
        """Read a Decimal environment variable with the GL_CP_ prefix.

        Uses ``Decimal(str)`` parsing for exact precision. Falls back
        to the default on ``InvalidOperation`` (malformed input) and
        emits a WARNING log.

        Args:
            name: Variable name suffix (after GL_CP_).
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
        """Read a boolean environment variable with the GL_CP_ prefix.

        Accepts ``true``, ``1``, ``yes`` (case-insensitive) as True.
        All other non-None values are treated as False.

        Args:
            name: Variable name suffix (after GL_CP_).
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
            name: Variable name suffix (after GL_CP_).
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
        ``CoolingPurchaseConfig()`` will re-read all environment
        variables and construct a fresh configuration object. Thread-safe.

        Example:
            >>> CoolingPurchaseConfig.reset()
            >>> cfg = CoolingPurchaseConfig()  # fresh instance
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
        logger.debug("CoolingPurchaseConfig singleton reset")

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
                "CoolingPurchaseConfig loaded with %d validation "
                "warning(s):\n%s",
                len(errors),
                "\n".join(f"  - {e}" for e in errors),
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate all configuration settings.

        Performs comprehensive checks (90+) across all configuration
        domains: general settings, database connectivity parameters,
        calculation defaults, electric chiller defaults, TES defaults,
        uncertainty parameters, compliance frameworks, cache settings,
        API settings, logging, provenance, and performance tuning.

        Returns:
            A list of human-readable error strings. An empty list means
            all validation checks passed.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> errors = cfg.validate()
            >>> assert len(errors) == 0
        """
        errors: List[str] = []

        # -- General Settings ------------------------------------------------
        errors.extend(self._validate_general_settings())

        # -- Database Settings -----------------------------------------------
        errors.extend(self._validate_database_settings())

        # -- Calculation Defaults --------------------------------------------
        errors.extend(self._validate_calculation_defaults())

        # -- Electric Chiller Defaults ---------------------------------------
        errors.extend(self._validate_electric_chiller_defaults())

        # -- TES Defaults ----------------------------------------------------
        errors.extend(self._validate_tes_defaults())

        # -- Uncertainty Settings --------------------------------------------
        errors.extend(self._validate_uncertainty_settings())

        # -- Compliance Settings ---------------------------------------------
        errors.extend(self._validate_compliance_settings())

        # -- Cache Settings --------------------------------------------------
        errors.extend(self._validate_cache_settings())

        # -- API Settings ----------------------------------------------------
        errors.extend(self._validate_api_settings())

        # -- Logging & Observability -----------------------------------------
        errors.extend(self._validate_logging_settings())

        # -- Provenance Tracking ---------------------------------------------
        errors.extend(self._validate_provenance_settings())

        # -- Performance Tuning ----------------------------------------------
        errors.extend(self._validate_performance_settings())

        if errors:
            logger.warning(
                "CoolingPurchaseConfig validation found %d error(s):\n%s",
                len(errors),
                "\n".join(f"  - {e}" for e in errors),
            )
        else:
            logger.debug(
                "CoolingPurchaseConfig validation passed: "
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

        # -- service_name non-empty ------------------------------------------
        if not self.service_name:
            errors.append("service_name must not be empty")

        # -- service_name length bound ---------------------------------------
        if self.service_name and len(self.service_name) > 128:
            errors.append(
                f"service_name must be <= 128 characters, "
                f"got {len(self.service_name)}"
            )

        # -- version non-empty -----------------------------------------------
        if not self.version:
            errors.append("version must not be empty")

        # -- version length bound --------------------------------------------
        if self.version and len(self.version) > 64:
            errors.append(
                f"version must be <= 64 characters, "
                f"got {len(self.version)}"
            )

        # -- log_level valid -------------------------------------------------
        normalised_log = self.log_level.upper()
        if normalised_log not in _VALID_LOG_LEVELS:
            errors.append(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got '{self.log_level}'"
            )

        # -- environment valid -----------------------------------------------
        normalised_env = self.environment.lower()
        if normalised_env not in _VALID_ENVIRONMENTS:
            errors.append(
                f"environment must be one of "
                f"{sorted(_VALID_ENVIRONMENTS)}, "
                f"got '{self.environment}'"
            )

        # -- max_batch_size > 0 ----------------------------------------------
        if self.max_batch_size <= 0:
            errors.append(
                f"max_batch_size must be > 0, "
                f"got {self.max_batch_size}"
            )

        # -- max_batch_size upper bound --------------------------------------
        if self.max_batch_size > 100_000:
            errors.append(
                f"max_batch_size must be <= 100000, "
                f"got {self.max_batch_size}"
            )

        # -- default_tenant non-empty ----------------------------------------
        if not self.default_tenant:
            errors.append("default_tenant must not be empty")

        # -- default_tenant length bound -------------------------------------
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

        # -- db_host non-empty -----------------------------------------------
        if not self.db_host:
            errors.append("db_host must not be empty")

        # -- db_port lower bound ---------------------------------------------
        if self.db_port <= 0:
            errors.append(
                f"db_port must be > 0, got {self.db_port}"
            )

        # -- db_port upper bound ---------------------------------------------
        if self.db_port > 65535:
            errors.append(
                f"db_port must be <= 65535, got {self.db_port}"
            )

        # -- db_name non-empty -----------------------------------------------
        if not self.db_name:
            errors.append("db_name must not be empty")

        # -- db_user non-empty -----------------------------------------------
        if not self.db_user:
            errors.append("db_user must not be empty")

        # -- db_pool_min lower bound -----------------------------------------
        if self.db_pool_min < 0:
            errors.append(
                f"db_pool_min must be >= 0, got {self.db_pool_min}"
            )

        # -- db_pool_min upper bound -----------------------------------------
        if self.db_pool_min > 100:
            errors.append(
                f"db_pool_min must be <= 100, got {self.db_pool_min}"
            )

        # -- db_pool_max lower bound -----------------------------------------
        if self.db_pool_max <= 0:
            errors.append(
                f"db_pool_max must be > 0, got {self.db_pool_max}"
            )

        # -- db_pool_max upper bound -----------------------------------------
        if self.db_pool_max > 500:
            errors.append(
                f"db_pool_max must be <= 500, got {self.db_pool_max}"
            )

        # -- db_pool_min <= db_pool_max --------------------------------------
        if self.db_pool_min > self.db_pool_max:
            errors.append(
                f"db_pool_min ({self.db_pool_min}) must be <= "
                f"db_pool_max ({self.db_pool_max})"
            )

        # -- db_ssl_mode valid -----------------------------------------------
        normalised_ssl = self.db_ssl_mode.lower()
        if normalised_ssl not in _VALID_SSL_MODES:
            errors.append(
                f"db_ssl_mode must be one of {sorted(_VALID_SSL_MODES)}, "
                f"got '{self.db_ssl_mode}'"
            )

        # -- table_prefix non-empty ------------------------------------------
        if not self.table_prefix:
            errors.append("table_prefix must not be empty")

        # -- table_prefix format (alphanumeric + underscore) -----------------
        if self.table_prefix and not self.table_prefix.replace(
            "_", ""
        ).isalnum():
            errors.append(
                f"table_prefix must contain only alphanumeric "
                f"characters and underscores, "
                f"got '{self.table_prefix}'"
            )

        # -- table_prefix length bound ---------------------------------------
        if self.table_prefix and len(self.table_prefix) > 32:
            errors.append(
                f"table_prefix must be <= 32 characters, "
                f"got {len(self.table_prefix)}"
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: Calculation Defaults
    # ------------------------------------------------------------------

    def _validate_calculation_defaults(self) -> List[str]:
        """Validate calculation default settings.

        Checks GWP source, data quality tier, COP source, decimal places,
        max trace steps, auxiliary percentage, parasitic ratio, and
        distribution loss bounds.

        Returns:
            List of error strings for invalid calculation defaults.
        """
        errors: List[str] = []

        # -- GWP source valid ------------------------------------------------
        normalised_gwp = self.default_gwp_source.lower()
        if normalised_gwp not in _VALID_GWP_SOURCES:
            errors.append(
                f"default_gwp_source must be one of "
                f"{sorted(_VALID_GWP_SOURCES)}, "
                f"got '{self.default_gwp_source}'"
            )

        # -- Default tier valid ----------------------------------------------
        normalised_tier = self.default_tier.lower()
        if normalised_tier not in _VALID_TIERS:
            errors.append(
                f"default_tier must be one of "
                f"{sorted(_VALID_TIERS)}, "
                f"got '{self.default_tier}'"
            )

        # -- COP source valid ------------------------------------------------
        normalised_cop_src = self.default_cop_source.lower()
        if normalised_cop_src not in _VALID_COP_SOURCES:
            errors.append(
                f"default_cop_source must be one of "
                f"{sorted(_VALID_COP_SOURCES)}, "
                f"got '{self.default_cop_source}'"
            )

        # -- decimal_places lower bound --------------------------------------
        if self.decimal_places < 0:
            errors.append(
                f"decimal_places must be >= 0, "
                f"got {self.decimal_places}"
            )

        # -- decimal_places upper bound --------------------------------------
        if self.decimal_places > 28:
            errors.append(
                f"decimal_places must be <= 28, "
                f"got {self.decimal_places}"
            )

        # -- max_trace_steps lower bound -------------------------------------
        if self.max_trace_steps <= 0:
            errors.append(
                f"max_trace_steps must be > 0, "
                f"got {self.max_trace_steps}"
            )

        # -- max_trace_steps upper bound -------------------------------------
        if self.max_trace_steps > 10000:
            errors.append(
                f"max_trace_steps must be <= 10000, "
                f"got {self.max_trace_steps}"
            )

        # -- default_auxiliary_pct lower bound --------------------------------
        if self.default_auxiliary_pct < Decimal("0"):
            errors.append(
                f"default_auxiliary_pct must be >= 0, "
                f"got {self.default_auxiliary_pct}"
            )

        # -- default_auxiliary_pct upper bound --------------------------------
        if self.default_auxiliary_pct > Decimal("1"):
            errors.append(
                f"default_auxiliary_pct must be <= 1, "
                f"got {self.default_auxiliary_pct}"
            )

        # -- default_parasitic_ratio lower bound ------------------------------
        if self.default_parasitic_ratio < Decimal("0"):
            errors.append(
                f"default_parasitic_ratio must be >= 0, "
                f"got {self.default_parasitic_ratio}"
            )

        # -- default_parasitic_ratio upper bound ------------------------------
        if self.default_parasitic_ratio > Decimal("1"):
            errors.append(
                f"default_parasitic_ratio must be <= 1, "
                f"got {self.default_parasitic_ratio}"
            )

        # -- default_distribution_loss lower bound ----------------------------
        if self.default_distribution_loss < Decimal("0"):
            errors.append(
                f"default_distribution_loss must be >= 0, "
                f"got {self.default_distribution_loss}"
            )

        # -- default_distribution_loss upper bound ----------------------------
        if self.default_distribution_loss > Decimal("1"):
            errors.append(
                f"default_distribution_loss must be <= 1, "
                f"got {self.default_distribution_loss}"
            )

        # -- combined auxiliary + parasitic + distribution < 1 ----------------
        combined_losses = (
            self.default_auxiliary_pct
            + self.default_parasitic_ratio
            + self.default_distribution_loss
        )
        if combined_losses >= Decimal("1"):
            errors.append(
                f"Combined losses (auxiliary {self.default_auxiliary_pct} + "
                f"parasitic {self.default_parasitic_ratio} + "
                f"distribution {self.default_distribution_loss} = "
                f"{combined_losses}) must be < 1.0"
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: Electric Chiller Defaults
    # ------------------------------------------------------------------

    def _validate_electric_chiller_defaults(self) -> List[str]:
        """Validate electric chiller default settings.

        Checks AHRI 550/590 part-load weights sum to exactly 1.00,
        each weight is in [0, 1], condenser type is valid, and COP
        min/max bounds are consistent.

        Returns:
            List of error strings for invalid chiller defaults.
        """
        errors: List[str] = []

        # -- AHRI 100% weight bounds -----------------------------------------
        if self.ahri_100_weight < Decimal("0"):
            errors.append(
                f"ahri_100_weight must be >= 0, "
                f"got {self.ahri_100_weight}"
            )
        if self.ahri_100_weight > Decimal("1"):
            errors.append(
                f"ahri_100_weight must be <= 1, "
                f"got {self.ahri_100_weight}"
            )

        # -- AHRI 75% weight bounds ------------------------------------------
        if self.ahri_75_weight < Decimal("0"):
            errors.append(
                f"ahri_75_weight must be >= 0, "
                f"got {self.ahri_75_weight}"
            )
        if self.ahri_75_weight > Decimal("1"):
            errors.append(
                f"ahri_75_weight must be <= 1, "
                f"got {self.ahri_75_weight}"
            )

        # -- AHRI 50% weight bounds ------------------------------------------
        if self.ahri_50_weight < Decimal("0"):
            errors.append(
                f"ahri_50_weight must be >= 0, "
                f"got {self.ahri_50_weight}"
            )
        if self.ahri_50_weight > Decimal("1"):
            errors.append(
                f"ahri_50_weight must be <= 1, "
                f"got {self.ahri_50_weight}"
            )

        # -- AHRI 25% weight bounds ------------------------------------------
        if self.ahri_25_weight < Decimal("0"):
            errors.append(
                f"ahri_25_weight must be >= 0, "
                f"got {self.ahri_25_weight}"
            )
        if self.ahri_25_weight > Decimal("1"):
            errors.append(
                f"ahri_25_weight must be <= 1, "
                f"got {self.ahri_25_weight}"
            )

        # -- AHRI weights sum to 1.00 ----------------------------------------
        ahri_sum = (
            self.ahri_100_weight
            + self.ahri_75_weight
            + self.ahri_50_weight
            + self.ahri_25_weight
        )
        if ahri_sum != Decimal("1.00"):
            errors.append(
                f"AHRI weights must sum to exactly 1.00, "
                f"got {ahri_sum} "
                f"(100%={self.ahri_100_weight}, "
                f"75%={self.ahri_75_weight}, "
                f"50%={self.ahri_50_weight}, "
                f"25%={self.ahri_25_weight})"
            )

        # -- condenser type valid --------------------------------------------
        normalised_condenser = self.default_condenser_type.lower()
        if normalised_condenser not in _VALID_CONDENSER_TYPES:
            errors.append(
                f"default_condenser_type must be one of "
                f"{sorted(_VALID_CONDENSER_TYPES)}, "
                f"got '{self.default_condenser_type}'"
            )

        # -- min_cop lower bound ---------------------------------------------
        if self.min_cop < Decimal("0"):
            errors.append(
                f"min_cop must be >= 0, got {self.min_cop}"
            )

        # -- min_cop upper sanity check --------------------------------------
        if self.min_cop > Decimal("10"):
            errors.append(
                f"min_cop must be <= 10 (sanity check), "
                f"got {self.min_cop}"
            )

        # -- max_cop lower bound ---------------------------------------------
        if self.max_cop <= Decimal("0"):
            errors.append(
                f"max_cop must be > 0, got {self.max_cop}"
            )

        # -- max_cop upper sanity check --------------------------------------
        if self.max_cop > Decimal("100"):
            errors.append(
                f"max_cop must be <= 100 (sanity check), "
                f"got {self.max_cop}"
            )

        # -- min_cop < max_cop -----------------------------------------------
        if self.min_cop >= self.max_cop:
            errors.append(
                f"min_cop ({self.min_cop}) must be < "
                f"max_cop ({self.max_cop})"
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: TES Defaults
    # ------------------------------------------------------------------

    def _validate_tes_defaults(self) -> List[str]:
        """Validate thermal energy storage default settings.

        Checks that each TES round-trip efficiency is in the range
        (0, 1] and that ice efficiency is less than or equal to
        chilled water efficiency (physical constraint: ice storage
        has higher losses than chilled water due to phase change).

        Returns:
            List of error strings for invalid TES defaults.
        """
        errors: List[str] = []

        # -- ice_round_trip_eff lower bound ----------------------------------
        if self.ice_round_trip_eff <= Decimal("0"):
            errors.append(
                f"ice_round_trip_eff must be > 0, "
                f"got {self.ice_round_trip_eff}"
            )

        # -- ice_round_trip_eff upper bound ----------------------------------
        if self.ice_round_trip_eff > Decimal("1"):
            errors.append(
                f"ice_round_trip_eff must be <= 1, "
                f"got {self.ice_round_trip_eff}"
            )

        # -- cw_round_trip_eff lower bound -----------------------------------
        if self.cw_round_trip_eff <= Decimal("0"):
            errors.append(
                f"cw_round_trip_eff must be > 0, "
                f"got {self.cw_round_trip_eff}"
            )

        # -- cw_round_trip_eff upper bound -----------------------------------
        if self.cw_round_trip_eff > Decimal("1"):
            errors.append(
                f"cw_round_trip_eff must be <= 1, "
                f"got {self.cw_round_trip_eff}"
            )

        # -- pcm_round_trip_eff lower bound ----------------------------------
        if self.pcm_round_trip_eff <= Decimal("0"):
            errors.append(
                f"pcm_round_trip_eff must be > 0, "
                f"got {self.pcm_round_trip_eff}"
            )

        # -- pcm_round_trip_eff upper bound ----------------------------------
        if self.pcm_round_trip_eff > Decimal("1"):
            errors.append(
                f"pcm_round_trip_eff must be <= 1, "
                f"got {self.pcm_round_trip_eff}"
            )

        # -- ice <= cw (physical constraint) ---------------------------------
        if self.ice_round_trip_eff > self.cw_round_trip_eff:
            errors.append(
                f"ice_round_trip_eff ({self.ice_round_trip_eff}) should "
                f"be <= cw_round_trip_eff ({self.cw_round_trip_eff}) "
                f"(ice storage has higher losses than chilled water)"
            )

        # -- pcm between ice and cw (typical physical constraint) ------------
        if self.pcm_round_trip_eff < self.ice_round_trip_eff:
            errors.append(
                f"pcm_round_trip_eff ({self.pcm_round_trip_eff}) should "
                f"be >= ice_round_trip_eff ({self.ice_round_trip_eff}) "
                f"(PCM typically has lower losses than ice)"
            )

        if self.pcm_round_trip_eff > self.cw_round_trip_eff:
            errors.append(
                f"pcm_round_trip_eff ({self.pcm_round_trip_eff}) should "
                f"be <= cw_round_trip_eff ({self.cw_round_trip_eff}) "
                f"(PCM typically has higher losses than chilled water)"
            )

        return errors

    # ------------------------------------------------------------------
    # Validation: Uncertainty Settings
    # ------------------------------------------------------------------

    def _validate_uncertainty_settings(self) -> List[str]:
        """Validate uncertainty quantification settings.

        Checks Monte Carlo iterations, confidence level, random seed,
        and per-tier uncertainty fractions. Also validates the tier
        ordering constraint: tier 1 >= tier 2 >= tier 3.

        Returns:
            List of error strings for invalid uncertainty settings.
        """
        errors: List[str] = []

        # -- monte_carlo_iterations lower bound ------------------------------
        if self.monte_carlo_iterations <= 0:
            errors.append(
                f"monte_carlo_iterations must be > 0, "
                f"got {self.monte_carlo_iterations}"
            )

        # -- monte_carlo_iterations upper bound ------------------------------
        if self.monte_carlo_iterations > 1_000_000:
            errors.append(
                f"monte_carlo_iterations must be <= 1000000, "
                f"got {self.monte_carlo_iterations}"
            )

        # -- confidence_level lower bound ------------------------------------
        if self.confidence_level <= Decimal("0"):
            errors.append(
                f"confidence_level must be > 0, "
                f"got {self.confidence_level}"
            )

        # -- confidence_level upper bound ------------------------------------
        if self.confidence_level >= Decimal("1"):
            errors.append(
                f"confidence_level must be < 1, "
                f"got {self.confidence_level}"
            )

        # -- default_seed non-negative ---------------------------------------
        if self.default_seed < 0:
            errors.append(
                f"default_seed must be >= 0, got {self.default_seed}"
            )

        # -- tier1_uncertainty lower bound -----------------------------------
        if self.tier1_uncertainty < Decimal("0"):
            errors.append(
                f"tier1_uncertainty must be >= 0, "
                f"got {self.tier1_uncertainty}"
            )

        # -- tier1_uncertainty upper bound -----------------------------------
        if self.tier1_uncertainty > Decimal("1"):
            errors.append(
                f"tier1_uncertainty must be <= 1, "
                f"got {self.tier1_uncertainty}"
            )

        # -- tier2_uncertainty lower bound -----------------------------------
        if self.tier2_uncertainty < Decimal("0"):
            errors.append(
                f"tier2_uncertainty must be >= 0, "
                f"got {self.tier2_uncertainty}"
            )

        # -- tier2_uncertainty upper bound -----------------------------------
        if self.tier2_uncertainty > Decimal("1"):
            errors.append(
                f"tier2_uncertainty must be <= 1, "
                f"got {self.tier2_uncertainty}"
            )

        # -- tier3_uncertainty lower bound -----------------------------------
        if self.tier3_uncertainty < Decimal("0"):
            errors.append(
                f"tier3_uncertainty must be >= 0, "
                f"got {self.tier3_uncertainty}"
            )

        # -- tier3_uncertainty upper bound -----------------------------------
        if self.tier3_uncertainty > Decimal("1"):
            errors.append(
                f"tier3_uncertainty must be <= 1, "
                f"got {self.tier3_uncertainty}"
            )

        # -- tier ordering: tier1 >= tier2 >= tier3 --------------------------
        if self.tier1_uncertainty < self.tier2_uncertainty:
            errors.append(
                f"tier1_uncertainty ({self.tier1_uncertainty}) must be >= "
                f"tier2_uncertainty ({self.tier2_uncertainty}) "
                f"(higher tier = lower uncertainty)"
            )

        if self.tier2_uncertainty < self.tier3_uncertainty:
            errors.append(
                f"tier2_uncertainty ({self.tier2_uncertainty}) must be >= "
                f"tier3_uncertainty ({self.tier3_uncertainty}) "
                f"(higher tier = lower uncertainty)"
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

        # -- frameworks list non-empty ---------------------------------------
        if not self.enabled_frameworks:
            errors.append("enabled_frameworks must not be empty")
        else:
            # -- each framework must be valid --------------------------------
            for fw in self.enabled_frameworks:
                normalised = fw.lower()
                if normalised not in _VALID_FRAMEWORKS:
                    errors.append(
                        f"Framework '{fw}' is not valid; "
                        f"must be one of {sorted(_VALID_FRAMEWORKS)}"
                    )

            # -- check for duplicates ----------------------------------------
            seen: set = set()
            for fw in self.enabled_frameworks:
                normalised = fw.lower()
                if normalised in seen:
                    errors.append(
                        f"Duplicate framework '{fw}' in "
                        f"enabled_frameworks"
                    )
                seen.add(normalised)

        # -- strict_mode requires at least one framework ---------------------
        if self.strict_mode and not self.enabled_frameworks:
            errors.append(
                "strict_mode requires at least one framework "
                "in enabled_frameworks"
            )

        # -- fail_on_non_compliant requires strict_mode ----------------------
        if self.fail_on_non_compliant and not self.strict_mode:
            errors.append(
                "fail_on_non_compliant requires "
                "strict_mode to be True"
            )

        # -- SBTi requires ghg_protocol -------------------------------------
        normalised_fws = [
            fw.lower() for fw in self.enabled_frameworks
        ]
        if "sbti" in normalised_fws:
            if "ghg_protocol" not in normalised_fws:
                errors.append(
                    "SBTi framework requires 'ghg_protocol' "
                    "in enabled_frameworks"
                )

        # -- ASHRAE 90.1 requires condenser type to be set -------------------
        if "ashrae_90_1" in normalised_fws:
            if not self.default_condenser_type:
                errors.append(
                    "ASHRAE 90.1 framework requires "
                    "default_condenser_type to be set"
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

        # -- redis_url non-empty when caching enabled ------------------------
        if self.enable_caching and not self.redis_url:
            errors.append(
                "redis_url must not be empty when "
                "enable_caching is True"
            )

        # -- redis_url format check ------------------------------------------
        if self.redis_url and not (
            self.redis_url.startswith("redis://")
            or self.redis_url.startswith("rediss://")
        ):
            errors.append(
                f"redis_url must start with 'redis://' or 'rediss://', "
                f"got '{self.redis_url[:30]}...'"
                if len(self.redis_url) > 30
                else f"redis_url must start with 'redis://' or 'rediss://', "
                f"got '{self.redis_url}'"
            )

        # -- cache_ttl lower bound -------------------------------------------
        if self.cache_ttl <= 0:
            errors.append(
                f"cache_ttl must be > 0, got {self.cache_ttl}"
            )

        # -- cache_ttl upper bound -------------------------------------------
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

        # -- api_prefix non-empty --------------------------------------------
        if not self.api_prefix:
            errors.append("api_prefix must not be empty")

        # -- api_prefix starts with / ----------------------------------------
        if self.api_prefix and not self.api_prefix.startswith("/"):
            errors.append(
                f"api_prefix must start with '/', "
                f"got '{self.api_prefix}'"
            )

        # -- rate_limit lower bound ------------------------------------------
        if self.rate_limit <= 0:
            errors.append(
                f"rate_limit must be > 0, "
                f"got {self.rate_limit}"
            )

        # -- rate_limit upper bound ------------------------------------------
        if self.rate_limit > 100_000:
            errors.append(
                f"rate_limit must be <= 100000, "
                f"got {self.rate_limit}"
            )

        # -- page_size lower bound -------------------------------------------
        if self.page_size <= 0:
            errors.append(
                f"page_size must be > 0, "
                f"got {self.page_size}"
            )

        # -- page_size upper bound -------------------------------------------
        if self.page_size > 10000:
            errors.append(
                f"page_size must be <= 10000, "
                f"got {self.page_size}"
            )

        # -- max_page_size lower bound ---------------------------------------
        if self.max_page_size <= 0:
            errors.append(
                f"max_page_size must be > 0, "
                f"got {self.max_page_size}"
            )

        # -- max_page_size upper bound ---------------------------------------
        if self.max_page_size > 100_000:
            errors.append(
                f"max_page_size must be <= 100000, "
                f"got {self.max_page_size}"
            )

        # -- page_size <= max_page_size --------------------------------------
        if self.page_size > self.max_page_size:
            errors.append(
                f"page_size ({self.page_size}) must be <= "
                f"max_page_size ({self.max_page_size})"
            )

        # -- cors_origins non-empty ------------------------------------------
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

        # -- metrics_prefix non-empty ----------------------------------------
        if not self.metrics_prefix:
            errors.append("metrics_prefix must not be empty")

        # -- metrics_prefix format -------------------------------------------
        if self.metrics_prefix and not self.metrics_prefix.replace(
            "_", ""
        ).isalnum():
            errors.append(
                f"metrics_prefix must contain only alphanumeric "
                f"characters and underscores, "
                f"got '{self.metrics_prefix}'"
            )

        # -- metrics_prefix length bound -------------------------------------
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

        # -- genesis_hash required when provenance enabled -------------------
        if self.enable_provenance and not self.genesis_hash:
            errors.append(
                "genesis_hash must not be empty when "
                "enable_provenance is True"
            )

        # -- genesis_hash length bound ---------------------------------------
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

        # -- worker_threads lower bound --------------------------------------
        if self.worker_threads <= 0:
            errors.append(
                f"worker_threads must be > 0, "
                f"got {self.worker_threads}"
            )

        # -- worker_threads upper bound --------------------------------------
        if self.worker_threads > 64:
            errors.append(
                f"worker_threads must be <= 64, "
                f"got {self.worker_threads}"
            )

        # -- health_check_interval lower bound -------------------------------
        if self.health_check_interval <= 0:
            errors.append(
                f"health_check_interval must be > 0, "
                f"got {self.health_check_interval}"
            )

        # -- health_check_interval upper bound -------------------------------
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
        # General: 10, Database: 14, Calculation: 15, Chiller: 14,
        # TES: 9, Uncertainty: 14, Compliance: 7, Cache: 4,
        # API: 11, Logging: 3, Provenance: 2, Performance: 4
        return 107

    # ------------------------------------------------------------------
    # Accessor Methods
    # ------------------------------------------------------------------

    def get_service_name(self) -> str:
        """Return the service name for tracing and identification.

        Returns:
            Service name string.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> cfg.get_service_name()
            'cooling-purchase-service'
        """
        return self.service_name

    def get_version(self) -> str:
        """Return the service version string.

        Returns:
            Version string.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> cfg.get_version()
            '1.0.0'
        """
        return self.version

    def get_log_level(self) -> str:
        """Return the logging level.

        Returns:
            Log level string (e.g. "INFO", "DEBUG").

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> cfg.get_log_level()
            'INFO'
        """
        return self.log_level

    def get_environment(self) -> str:
        """Return the deployment environment.

        Returns:
            Environment string (development, staging, or production).

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> cfg.get_environment()
            'development'
        """
        return self.environment

    # ------------------------------------------------------------------
    # Database accessors
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
            >>> cfg = CoolingPurchaseConfig()
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
            >>> cfg = CoolingPurchaseConfig()
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
            >>> cfg = CoolingPurchaseConfig()
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
            Table prefix string (e.g. "gl_cp_").

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> cfg.get_table_prefix()
            'gl_cp_'
        """
        return self.table_prefix

    # ------------------------------------------------------------------
    # Calculation accessors
    # ------------------------------------------------------------------

    def get_default_gwp_source(self) -> str:
        """Return the default GWP assessment report source.

        Returns:
            GWP source string (e.g. "ar5").

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> cfg.get_default_gwp_source()
            'ar5'
        """
        return self.default_gwp_source

    def get_default_tier(self) -> str:
        """Return the default data quality tier.

        Returns:
            Tier string (e.g. "tier_2").

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> cfg.get_default_tier()
            'tier_2'
        """
        return self.default_tier

    def get_decimal_places(self) -> int:
        """Return the number of decimal places for calculations.

        Returns:
            Integer decimal places count.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> cfg.get_decimal_places()
            8
        """
        return self.decimal_places

    def get_calculation_config(self) -> Dict[str, Any]:
        """Return calculation engine configuration as a dictionary.

        Returns:
            Dictionary containing all calculation-related settings
            suitable for initialising the cooling calculation engines.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> calc = cfg.get_calculation_config()
            >>> calc["gwp_source"]
            'ar5'
        """
        return {
            "gwp_source": self.default_gwp_source,
            "data_quality_tier": self.default_tier,
            "cop_source": self.default_cop_source,
            "decimal_places": self.decimal_places,
            "max_trace_steps": self.max_trace_steps,
            "use_iplv": self.use_iplv_default,
            "auxiliary_pct": str(self.default_auxiliary_pct),
            "parasitic_ratio": str(self.default_parasitic_ratio),
            "distribution_loss": str(self.default_distribution_loss),
            "max_batch_size": self.max_batch_size,
        }

    # ------------------------------------------------------------------
    # Electric chiller accessors
    # ------------------------------------------------------------------

    def get_ahri_weights(self) -> Dict[str, Decimal]:
        """Return AHRI 550/590 part-load weights as a dictionary.

        The AHRI IPLV (Integrated Part-Load Value) weights represent
        the fraction of operating hours at each load condition per
        AHRI Standard 550/590. The four weights must sum to exactly
        ``Decimal("1.00")``.

        Returns:
            Dictionary mapping load percentage strings to Decimal weights.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> weights = cfg.get_ahri_weights()
            >>> weights["100"]
            Decimal('0.01')
            >>> weights["75"]
            Decimal('0.42')
            >>> sum(weights.values()) == Decimal("1.00")
            True
        """
        return {
            "100": self.ahri_100_weight,
            "75": self.ahri_75_weight,
            "50": self.ahri_50_weight,
            "25": self.ahri_25_weight,
        }

    def get_chiller_config(self) -> Dict[str, Any]:
        """Return electric chiller configuration as a dictionary.

        Returns:
            Dictionary containing all electric-chiller-related settings
            suitable for initialising the ChillerCalculationEngine.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> chiller = cfg.get_chiller_config()
            >>> chiller["condenser_type"]
            'water_cooled'
        """
        return {
            "ahri_weights": self.get_ahri_weights(),
            "condenser_type": self.default_condenser_type,
            "min_cop": str(self.min_cop),
            "max_cop": str(self.max_cop),
            "use_iplv": self.use_iplv_default,
            "cop_source": self.default_cop_source,
        }

    # ------------------------------------------------------------------
    # TES accessors
    # ------------------------------------------------------------------

    def get_tes_efficiency(self, tes_type: str) -> Decimal:
        """Return the round-trip efficiency for a specific TES type.

        Args:
            tes_type: TES technology type. Must be one of "ice",
                "chilled_water", or "pcm" (case-insensitive).

        Returns:
            Decimal round-trip efficiency for the requested TES type.

        Raises:
            ValueError: If tes_type is not a recognised TES technology.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> cfg.get_tes_efficiency("ice")
            Decimal('0.85')
            >>> cfg.get_tes_efficiency("chilled_water")
            Decimal('0.95')
            >>> cfg.get_tes_efficiency("pcm")
            Decimal('0.90')
        """
        normalised = tes_type.lower().strip()
        tes_map: Dict[str, Decimal] = {
            "ice": self.ice_round_trip_eff,
            "chilled_water": self.cw_round_trip_eff,
            "pcm": self.pcm_round_trip_eff,
        }
        if normalised not in tes_map:
            raise ValueError(
                f"Unknown TES type '{tes_type}'; must be one of "
                f"{sorted(tes_map.keys())}"
            )
        return tes_map[normalised]

    def get_tes_config(self) -> Dict[str, Any]:
        """Return TES configuration as a dictionary.

        Returns:
            Dictionary containing all TES-related settings suitable
            for initialising the TESCalculationEngine.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> tes = cfg.get_tes_config()
            >>> tes["ice_efficiency"]
            '0.85'
        """
        return {
            "ice_efficiency": str(self.ice_round_trip_eff),
            "chilled_water_efficiency": str(self.cw_round_trip_eff),
            "pcm_efficiency": str(self.pcm_round_trip_eff),
            "enable_temporal_shifting": self.tes_enable_temporal_shifting,
        }

    # ------------------------------------------------------------------
    # Uncertainty accessors
    # ------------------------------------------------------------------

    def get_uncertainty_config(self) -> Dict[str, Any]:
        """Return uncertainty quantification parameters as a dictionary.

        All Decimal values are converted to strings for JSON-safe
        serialisation while preserving exact precision.

        Returns:
            Dictionary containing all uncertainty-related settings.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> params = cfg.get_uncertainty_config()
            >>> params["method"]
            'monte_carlo'
            >>> params["confidence_level"]
            '0.95'
        """
        return {
            "method": "monte_carlo",
            "iterations": self.monte_carlo_iterations,
            "confidence_level": str(self.confidence_level),
            "seed": self.default_seed,
            "tier1_uncertainty": str(self.tier1_uncertainty),
            "tier2_uncertainty": str(self.tier2_uncertainty),
            "tier3_uncertainty": str(self.tier3_uncertainty),
        }

    def get_uncertainty_for_tier(self, tier: str) -> Decimal:
        """Return the default uncertainty fraction for a given tier.

        Args:
            tier: Data quality tier (e.g. "tier_1", "tier_2", "tier_3").
                Case-insensitive.

        Returns:
            Decimal uncertainty fraction for the requested tier.

        Raises:
            ValueError: If tier is not a recognised data quality tier.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> cfg.get_uncertainty_for_tier("tier_1")
            Decimal('0.40')
            >>> cfg.get_uncertainty_for_tier("tier_3")
            Decimal('0.10')
        """
        normalised = tier.lower().strip()
        tier_map: Dict[str, Decimal] = {
            "tier_1": self.tier1_uncertainty,
            "tier_2": self.tier2_uncertainty,
            "tier_3": self.tier3_uncertainty,
        }
        if normalised not in tier_map:
            raise ValueError(
                f"Unknown tier '{tier}'; must be one of "
                f"{sorted(tier_map.keys())}"
            )
        return tier_map[normalised]

    # ------------------------------------------------------------------
    # Compliance accessors
    # ------------------------------------------------------------------

    def get_enabled_frameworks(self) -> List[str]:
        """Return a copy of the enabled compliance frameworks list.

        Returns:
            List of enabled framework identifiers.

        Example:
            >>> cfg = CoolingPurchaseConfig()
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
                "ghg_protocol", "ashrae_90_1").

        Returns:
            True if the framework is in the enabled list, False otherwise.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> cfg.is_framework_enabled("cdp")
            True
            >>> cfg.is_framework_enabled("unknown_framework")
            False
        """
        normalised = framework.lower()
        return normalised in [
            fw.lower() for fw in self.enabled_frameworks
        ]

    # ------------------------------------------------------------------
    # API accessors
    # ------------------------------------------------------------------

    def get_api_prefix(self) -> str:
        """Return the REST API route prefix.

        Returns:
            API prefix string.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> cfg.get_api_prefix()
            '/api/v1/cooling-purchase'
        """
        return self.api_prefix

    def get_api_config(self) -> Dict[str, Any]:
        """Return API configuration as a dictionary.

        Returns:
            Dictionary of API-related settings suitable for FastAPI
            application construction.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> api_cfg = cfg.get_api_config()
            >>> api_cfg["prefix"]
            '/api/v1/cooling-purchase'
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
    # Cache accessors
    # ------------------------------------------------------------------

    def get_cache_config(self) -> Dict[str, Any]:
        """Return Redis/cache configuration as a dictionary.

        Returns:
            Dictionary containing all cache-related settings suitable
            for initialising a Redis connection.

        Example:
            >>> cfg = CoolingPurchaseConfig()
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
    # Rounding mode accessor
    # ------------------------------------------------------------------

    def get_rounding_mode(self) -> str:
        """Return the Python Decimal rounding mode constant.

        Maps ``ROUND_HALF_UP`` to the corresponding ``decimal`` module
        constant for use in ``Decimal.quantize()``.

        Returns:
            Decimal rounding mode constant string.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> cfg.get_rounding_mode()
            'ROUND_HALF_UP'
        """
        return _ROUNDING_MODE_MAP.get("ROUND_HALF_UP", ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Observability configuration accessor
    # ------------------------------------------------------------------

    def get_observability_config(self) -> Dict[str, Any]:
        """Return observability configuration as a dictionary.

        Returns:
            Dictionary of observability-related settings for metrics
            and tracing initialisation.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> obs_cfg = cfg.get_observability_config()
            >>> obs_cfg["metrics_prefix"]
            'gl_cp_'
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
            >>> cfg = CoolingPurchaseConfig()
            >>> flags = cfg.get_feature_flags()
            >>> flags["use_iplv_default"]
            True
        """
        return {
            "use_iplv_default": self.use_iplv_default,
            "tes_enable_temporal_shifting": (
                self.tes_enable_temporal_shifting
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
            >>> cfg = CoolingPurchaseConfig()
            >>> d = cfg.to_dict()
            >>> d["default_gwp_source"]
            'ar5'
            >>> d["db_password"]
            '***'
        """
        return {
            # -- Master switch -----------------------------------------------
            "enabled": self.enabled,
            # -- 1. General Settings -----------------------------------------
            "service_name": self.service_name,
            "version": self.version,
            "log_level": self.log_level,
            "environment": self.environment,
            "max_batch_size": self.max_batch_size,
            "default_tenant": self.default_tenant,
            # -- 2. Database Settings ----------------------------------------
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_name": self.db_name,
            "db_user": self.db_user,
            "db_password": "***" if self.db_password else "",
            "db_pool_min": self.db_pool_min,
            "db_pool_max": self.db_pool_max,
            "db_ssl_mode": self.db_ssl_mode,
            "table_prefix": self.table_prefix,
            # -- 3. Calculation Defaults -------------------------------------
            "default_gwp_source": self.default_gwp_source,
            "default_tier": self.default_tier,
            "default_cop_source": self.default_cop_source,
            "decimal_places": self.decimal_places,
            "max_trace_steps": self.max_trace_steps,
            "use_iplv_default": self.use_iplv_default,
            "default_auxiliary_pct": str(self.default_auxiliary_pct),
            "default_parasitic_ratio": str(
                self.default_parasitic_ratio
            ),
            "default_distribution_loss": str(
                self.default_distribution_loss
            ),
            # -- 4. Electric Chiller Defaults --------------------------------
            "ahri_100_weight": str(self.ahri_100_weight),
            "ahri_75_weight": str(self.ahri_75_weight),
            "ahri_50_weight": str(self.ahri_50_weight),
            "ahri_25_weight": str(self.ahri_25_weight),
            "default_condenser_type": self.default_condenser_type,
            "min_cop": str(self.min_cop),
            "max_cop": str(self.max_cop),
            # -- 5. TES Defaults ---------------------------------------------
            "ice_round_trip_eff": str(self.ice_round_trip_eff),
            "cw_round_trip_eff": str(self.cw_round_trip_eff),
            "pcm_round_trip_eff": str(self.pcm_round_trip_eff),
            "tes_enable_temporal_shifting": (
                self.tes_enable_temporal_shifting
            ),
            # -- 6. Uncertainty Settings -------------------------------------
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "confidence_level": str(self.confidence_level),
            "default_seed": self.default_seed,
            "tier1_uncertainty": str(self.tier1_uncertainty),
            "tier2_uncertainty": str(self.tier2_uncertainty),
            "tier3_uncertainty": str(self.tier3_uncertainty),
            # -- 7. Compliance Settings --------------------------------------
            "enabled_frameworks": list(self.enabled_frameworks),
            "strict_mode": self.strict_mode,
            "fail_on_non_compliant": self.fail_on_non_compliant,
            # -- 8. Redis/Cache Settings -------------------------------------
            "redis_url": self.redis_url,
            "cache_ttl": self.cache_ttl,
            "enable_caching": self.enable_caching,
            # -- 9. API Settings ---------------------------------------------
            "api_prefix": self.api_prefix,
            "rate_limit": self.rate_limit,
            "page_size": self.page_size,
            "max_page_size": self.max_page_size,
            # -- Logging & Observability -------------------------------------
            "enable_metrics": self.enable_metrics,
            "metrics_prefix": self.metrics_prefix,
            "enable_tracing": self.enable_tracing,
            # -- Provenance Tracking -----------------------------------------
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # -- Auth & Background Tasks -------------------------------------
            "enable_auth": self.enable_auth,
            "worker_threads": self.worker_threads,
            "enable_background_tasks": self.enable_background_tasks,
            "health_check_interval": self.health_check_interval,
            # -- CORS & Docs -------------------------------------------------
            "cors_origins": list(self.cors_origins),
            "enable_docs": self.enable_docs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CoolingPurchaseConfig:
        """Deserialise a configuration from a dictionary.

        Creates a new CoolingPurchaseConfig instance and populates it
        from the provided dictionary. The singleton is reset first to
        allow the new configuration to be installed. Keys not present
        in the dictionary retain their environment-loaded defaults.

        String-valued Decimal fields are automatically converted back
        to Decimal objects.

        Args:
            data: Dictionary of configuration key-value pairs. Keys
                correspond to attribute names on the config object.

        Returns:
            A new CoolingPurchaseConfig instance with values from the
            dictionary.

        Example:
            >>> d = {"default_gwp_source": "ar6", "decimal_places": 12}
            >>> cfg = CoolingPurchaseConfig.from_dict(d)
            >>> cfg.default_gwp_source
            'ar6'
            >>> cfg.decimal_places
            12
        """
        cls.reset()
        instance = cls()
        instance._apply_dict(data)

        logger.info(
            "CoolingPurchaseConfig loaded from dict: %d keys applied",
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
            "default_auxiliary_pct",
            "default_parasitic_ratio",
            "default_distribution_loss",
            "ahri_100_weight",
            "ahri_75_weight",
            "ahri_50_weight",
            "ahri_25_weight",
            "min_cop",
            "max_cop",
            "ice_round_trip_eff",
            "cw_round_trip_eff",
            "pcm_round_trip_eff",
            "confidence_level",
            "tier1_uncertainty",
            "tier2_uncertainty",
            "tier3_uncertainty",
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
            >>> cfg = CoolingPurchaseConfig()
            >>> json_str = cfg.to_json()
            >>> '"default_gwp_source": "ar5"' in json_str
            True
        """
        return json.dumps(
            self.to_dict(), indent=indent, sort_keys=False
        )

    @classmethod
    def from_json(cls, json_str: str) -> CoolingPurchaseConfig:
        """Deserialise a configuration from a JSON string.

        Args:
            json_str: JSON string containing configuration key-value
                pairs.

        Returns:
            A new CoolingPurchaseConfig instance.

        Example:
            >>> json_str = '{"default_gwp_source": "ar6"}'
            >>> cfg = CoolingPurchaseConfig.from_json(json_str)
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
        source to lowercase, condenser type to lowercase, log level to
        uppercase, SSL mode to lowercase, environment to lowercase).
        This method is idempotent.

        Example:
            >>> cfg = CoolingPurchaseConfig()
            >>> cfg.default_gwp_source = "AR5"
            >>> cfg.normalise()
            >>> cfg.default_gwp_source
            'ar5'
        """
        # GWP source -> lowercase
        self.default_gwp_source = self.default_gwp_source.lower()

        # Default tier -> lowercase
        self.default_tier = self.default_tier.lower()

        # COP source -> lowercase
        self.default_cop_source = self.default_cop_source.lower()

        # Condenser type -> lowercase
        self.default_condenser_type = (
            self.default_condenser_type.lower()
        )

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

        logger.debug(
            "CoolingPurchaseConfig normalised: gwp=%s, tier=%s, "
            "log_level=%s, ssl_mode=%s, env=%s, condenser=%s",
            self.default_gwp_source,
            self.default_tier,
            self.log_level,
            self.db_ssl_mode,
            self.environment,
            self.default_condenser_type,
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
            >>> cfg = CoolingPurchaseConfig()
            >>> cfg.merge({"decimal_places": 12})
            >>> cfg.decimal_places
            12
        """
        self._apply_dict(overrides)
        logger.debug(
            "CoolingPurchaseConfig merged %d overrides",
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
            >>> cfg = CoolingPurchaseConfig()
            >>> summary = cfg.health_summary()
            >>> summary["validation_status"]
            'PASS'
        """
        errors = self.validate()
        flags = self.get_feature_flags()
        enabled_flag_count = sum(1 for v in flags.values() if v)

        return {
            "agent": "cooling-purchase",
            "agent_id": "AGENT-MRV-012",
            "gl_id": "GL-MRV-X-023",
            "enabled": self.enabled,
            "validation_status": "PASS" if not errors else "FAIL",
            "validation_errors": len(errors),
            "service_name": self.service_name,
            "version": self.version,
            "environment": self.environment,
            "gwp_source": self.default_gwp_source,
            "data_quality_tier": self.default_tier,
            "cop_source": self.default_cop_source,
            "condenser_type": self.default_condenser_type,
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
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "ahri_weights_sum": str(
                self.ahri_100_weight
                + self.ahri_75_weight
                + self.ahri_50_weight
                + self.ahri_25_weight
            ),
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
        return f"CoolingPurchaseConfig({pairs})"

    def __str__(self) -> str:
        """Return a human-readable summary of the configuration.

        Returns:
            Multi-line string summary of key settings.
        """
        return (
            f"CoolingPurchaseConfig("
            f"enabled={self.enabled}, "
            f"service={self.service_name}, "
            f"env={self.environment}, "
            f"gwp={self.default_gwp_source}, "
            f"tier={self.default_tier}, "
            f"condenser={self.default_condenser_type}, "
            f"cop_range=[{self.min_cop}-{self.max_cop}], "
            f"batch={self.max_batch_size}, "
            f"frameworks={len(self.enabled_frameworks)}, "
            f"monte_carlo={self.monte_carlo_iterations}"
            f")"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality by comparing all serialised values.

        Args:
            other: Object to compare against.

        Returns:
            True if other is a CoolingPurchaseConfig with identical
            settings.
        """
        if not isinstance(other, CoolingPurchaseConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def get_config() -> CoolingPurchaseConfig:
    """Return the singleton CoolingPurchaseConfig.

    Convenience function that delegates to the singleton constructor.
    Thread-safe via the class-level lock in ``__new__``.

    Returns:
        CoolingPurchaseConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.default_gwp_source
        'ar5'
    """
    return CoolingPurchaseConfig()


def set_config(
    overrides: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> CoolingPurchaseConfig:
    """Reset and re-create the singleton with optional overrides.

    Resets the singleton, creates a fresh instance from environment
    variables, then applies any provided overrides. This is the
    primary entry point for test setup.

    Args:
        overrides: Dictionary of configuration overrides.
        **kwargs: Additional keyword overrides (merged with overrides).

    Returns:
        The new CoolingPurchaseConfig singleton.

    Example:
        >>> cfg = set_config(default_gwp_source="ar6")
        >>> cfg.default_gwp_source
        'ar6'
    """
    CoolingPurchaseConfig.reset()
    cfg = CoolingPurchaseConfig()

    merged: Dict[str, Any] = {}
    if overrides:
        merged.update(overrides)
    merged.update(kwargs)

    if merged:
        cfg._apply_dict(merged)

    logger.info(
        "CoolingPurchaseConfig set with %d overrides: "
        "enabled=%s, gwp=%s, tier=%s, "
        "condenser=%s, batch_size=%d",
        len(merged),
        cfg.enabled,
        cfg.default_gwp_source,
        cfg.default_tier,
        cfg.default_condenser_type,
        cfg.max_batch_size,
    )
    return cfg


def reset_config() -> None:
    """Reset the singleton CoolingPurchaseConfig to None.

    The next call to :func:`get_config` or
    ``CoolingPurchaseConfig()`` will re-read environment variables
    and construct a fresh instance. Intended for test teardown to
    prevent state leakage.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # fresh instance from env vars
    """
    CoolingPurchaseConfig.reset()


def validate_config() -> List[str]:
    """Validate the current singleton configuration.

    Convenience function that calls
    :meth:`CoolingPurchaseConfig.validate` on the current singleton
    instance.

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

#: Valid GWP sources
VALID_GWP_SOURCES: frozenset = _VALID_GWP_SOURCES

#: Valid data quality tiers
VALID_TIERS: frozenset = _VALID_TIERS

#: Valid COP sources
VALID_COP_SOURCES: frozenset = _VALID_COP_SOURCES

#: Valid condenser types
VALID_CONDENSER_TYPES: frozenset = _VALID_CONDENSER_TYPES

#: Valid TES types
VALID_TES_TYPES: frozenset = _VALID_TES_TYPES

#: Valid chiller types
VALID_CHILLER_TYPES: frozenset = _VALID_CHILLER_TYPES

#: Valid cooling technologies
VALID_COOLING_TECHNOLOGIES: frozenset = _VALID_COOLING_TECHNOLOGIES

#: Valid uncertainty methods
VALID_UNCERTAINTY_METHODS: frozenset = _VALID_UNCERTAINTY_METHODS

#: Valid rounding modes
VALID_ROUNDING_MODES: frozenset = _VALID_ROUNDING_MODES

#: Valid SSL modes
VALID_SSL_MODES: frozenset = _VALID_SSL_MODES

#: Valid compliance frameworks
VALID_FRAMEWORKS: frozenset = _VALID_FRAMEWORKS

#: Valid environments
VALID_ENVIRONMENTS: frozenset = _VALID_ENVIRONMENTS

#: Valid log levels
VALID_LOG_LEVELS: frozenset = _VALID_LOG_LEVELS

#: Environment variable prefix
ENV_PREFIX: str = _ENV_PREFIX


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    # Core class
    "CoolingPurchaseConfig",
    # Module-level functions
    "get_config",
    "set_config",
    "reset_config",
    "validate_config",
    # Constants
    "DEFAULT_ENABLED_FRAMEWORKS",
    "VALID_GWP_SOURCES",
    "VALID_TIERS",
    "VALID_COP_SOURCES",
    "VALID_CONDENSER_TYPES",
    "VALID_TES_TYPES",
    "VALID_CHILLER_TYPES",
    "VALID_COOLING_TECHNOLOGIES",
    "VALID_UNCERTAINTY_METHODS",
    "VALID_ROUNDING_MODES",
    "VALID_SSL_MODES",
    "VALID_FRAMEWORKS",
    "VALID_ENVIRONMENTS",
    "VALID_LOG_LEVELS",
    "ENV_PREFIX",
]
