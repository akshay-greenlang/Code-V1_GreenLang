# -*- coding: utf-8 -*-
"""
Fuel & Energy Activities Agent Configuration - AGENT-MRV-016

Centralized configuration for the Fuel & Energy Activities Agent SDK covering:
- Database, cache, and connection defaults
- GHG Protocol Scope 3 Category 3 methodology defaults
- Well-to-Tank (WTT) emission factors for fuel upstream
- Transmission & Distribution (T&D) loss factors for electricity
- Upstream electricity emissions for purchased energy
- Decimal precision for emission calculations
- Capacity limits (fuel types, electricity sources, activity records)
- Monte Carlo uncertainty analysis parameters
- Feature toggles (WTT factors, upstream electricity, T&D losses,
  compliance checking, uncertainty, provenance, metrics)
- API configuration (prefix, pagination)
- Background task and worker thread settings

All settings can be overridden via environment variables with the
``GL_FEA_`` prefix (e.g. ``GL_FEA_DEFAULT_GWP_SOURCE``,
``GL_FEA_DEFAULT_WTT_SOURCE``).

Environment Variable Reference (GL_FEA_ prefix):
    GL_FEA_ENABLED                          - Enable/disable agent
    GL_FEA_DATABASE_URL                     - PostgreSQL connection URL
    GL_FEA_REDIS_URL                        - Redis connection URL
    GL_FEA_MAX_BATCH_SIZE                   - Maximum records per batch
    GL_FEA_DEFAULT_GWP_SOURCE               - Default GWP source (AR4/AR5/AR6/AR6_20YR)
    GL_FEA_DEFAULT_CALCULATION_METHOD       - Default method (HYBRID/WTT_ONLY/TD_ONLY/UPSTREAM_ELECTRICITY)
    GL_FEA_DEFAULT_WTT_SOURCE               - Default WTT source (DEFRA/EPA/ECOINVENT/IEA/CUSTOM)
    GL_FEA_DEFAULT_TD_SOURCE                - Default T&D loss source (IEA/EIA/NATIONAL_GRID/CUSTOM)
    GL_FEA_DECIMAL_PRECISION                - Decimal places for calculations
    GL_FEA_BASE_YEAR                        - Base year for emission factors
    GL_FEA_DEFAULT_CURRENCY                 - Default currency (USD/EUR/GBP)
    GL_FEA_ENABLE_WTT_FACTORS               - Enable well-to-tank emissions
    GL_FEA_ENABLE_UPSTREAM_ELECTRICITY      - Enable upstream electricity emissions
    GL_FEA_ENABLE_TD_LOSSES                 - Enable transmission & distribution losses
    GL_FEA_MONTE_CARLO_ITERATIONS           - Monte Carlo simulation iterations
    GL_FEA_MONTE_CARLO_SEED                 - Random seed for reproducibility
    GL_FEA_CONFIDENCE_LEVELS                - Comma-separated confidence levels
    GL_FEA_ENABLE_COMPLIANCE_CHECKING       - Enable compliance checking
    GL_FEA_ENABLE_UNCERTAINTY               - Enable uncertainty quantification
    GL_FEA_ENABLE_PROVENANCE                - Enable SHA-256 provenance chain
    GL_FEA_ENABLE_METRICS                   - Enable Prometheus metrics export
    GL_FEA_MAX_FUEL_TYPES                   - Maximum fuel type registrations
    GL_FEA_MAX_ELECTRICITY_SOURCES          - Maximum electricity source registrations
    GL_FEA_MAX_ACTIVITY_RECORDS             - Maximum activity records per batch
    GL_FEA_CACHE_TTL_SECONDS                - Cache time-to-live in seconds
    GL_FEA_API_PREFIX                       - REST API route prefix
    GL_FEA_API_MAX_PAGE_SIZE                - Maximum API page size
    GL_FEA_API_DEFAULT_PAGE_SIZE            - Default API page size
    GL_FEA_LOG_LEVEL                        - Logging level
    GL_FEA_WORKER_THREADS                   - Worker thread pool size
    GL_FEA_ENABLE_BACKGROUND_TASKS          - Enable background task processing
    GL_FEA_HEALTH_CHECK_INTERVAL            - Health check interval seconds
    GL_FEA_GENESIS_HASH                     - Genesis anchor for provenance chain

Example:
    >>> from greenlang.fuel_energy_activities.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_gwp_source, cfg.default_calculation_method)
    AR6 HYBRID

    >>> # Override for testing
    >>> from greenlang.fuel_energy_activities.config import set_config, reset_config
    >>> from greenlang.fuel_energy_activities.config import FuelEnergyActivitiesConfig
    >>> set_config(FuelEnergyActivitiesConfig(enable_wtt_factors=False))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-016 Fuel & Energy Activities (GL-MRV-S3-003)
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_FEA_"

# ---------------------------------------------------------------------------
# Valid enumeration values for configuration validation
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

_VALID_GWP_SOURCES = frozenset({"AR4", "AR5", "AR6", "AR6_20YR"})

_VALID_CALCULATION_METHODS = frozenset({
    "HYBRID",
    "WTT_ONLY",
    "TD_ONLY",
    "UPSTREAM_ELECTRICITY",
})

_VALID_WTT_SOURCES = frozenset({
    "DEFRA",
    "EPA",
    "ECOINVENT",
    "IEA",
    "CUSTOM",
})

_VALID_TD_SOURCES = frozenset({
    "IEA",
    "EIA",
    "NATIONAL_GRID",
    "CUSTOM",
})

_VALID_CURRENCIES = frozenset({
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CNY",
    "INR",
})

_VALID_ROUNDING_MODES = frozenset({
    "ROUND_HALF_UP",
    "ROUND_HALF_DOWN",
    "ROUND_HALF_EVEN",
    "ROUND_UP",
    "ROUND_DOWN",
})


# ---------------------------------------------------------------------------
# Configuration sections as dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ServiceConfig:
    """Service-level configuration.

    Attributes:
        name: Service name for logging and metrics.
        version: Service version (SemVer).
        log_level: Logging verbosity (DEBUG/INFO/WARNING/ERROR/CRITICAL).
        environment: Deployment environment (dev/staging/prod).
        default_tenant: Default tenant ID for multi-tenancy.
    """

    name: str = "fuel-energy-activities"
    version: str = "1.0.0"
    log_level: str = "INFO"
    environment: str = "production"
    default_tenant: str = "default"


@dataclass
class DatabaseConfig:
    """Database connection configuration.

    Attributes:
        host: PostgreSQL host.
        port: PostgreSQL port.
        name: Database name.
        user: Database user.
        password: Database password (redacted in logs).
        pool_min: Minimum connection pool size.
        pool_max: Maximum connection pool size.
        ssl_mode: SSL mode (disable/allow/prefer/require/verify-ca/verify-full).
        table_prefix: Table prefix for namespacing (gl_fea_).
    """

    host: str = "localhost"
    port: int = 5432
    name: str = "greenlang"
    user: str = "greenlang"
    password: str = ""
    pool_min: int = 2
    pool_max: int = 10
    ssl_mode: str = "prefer"
    table_prefix: str = "gl_fea_"


@dataclass
class CalculationConfig:
    """Calculation methodology configuration.

    Attributes:
        default_method: Default calculation method (HYBRID/WTT_ONLY/TD_ONLY/UPSTREAM_ELECTRICITY).
        default_gwp: Default GWP source (AR4/AR5/AR6/AR6_20YR).
        decimal_places: Decimal precision for calculations.
        rounding_mode: Rounding mode (ROUND_HALF_UP/ROUND_HALF_DOWN/etc.).
        base_year: Base year for emission factors.
        default_currency: Default currency for monetary-based factors.
        enable_wtt_factors: Enable well-to-tank upstream emissions.
        enable_upstream_electricity: Enable upstream electricity emissions.
        enable_td_losses: Enable transmission & distribution losses.
        default_td_source: Default T&D loss source (IEA/EIA/NATIONAL_GRID/CUSTOM).
        default_wtt_source: Default WTT source (DEFRA/EPA/ECOINVENT/IEA/CUSTOM).
    """

    default_method: str = "HYBRID"
    default_gwp: str = "AR6"
    decimal_places: int = 8
    rounding_mode: str = "ROUND_HALF_UP"
    base_year: int = 2024
    default_currency: str = "USD"
    enable_wtt_factors: bool = True
    enable_upstream_electricity: bool = True
    enable_td_losses: bool = True
    default_td_source: str = "IEA"
    default_wtt_source: str = "DEFRA"


@dataclass
class DQIConfig:
    """Data Quality Indicator (DQI) configuration.

    Attributes:
        temporal_weight: Weight for temporal representativeness (0.0-1.0).
        geographical_weight: Weight for geographical representativeness (0.0-1.0).
        technological_weight: Weight for technological representativeness (0.0-1.0).
        completeness_weight: Weight for data completeness (0.0-1.0).
        reliability_weight: Weight for data reliability (0.0-1.0).
        min_coverage: Minimum data coverage threshold (0.0-1.0).
        target_coverage: Target data coverage for high quality (0.0-1.0).
    """

    temporal_weight: float = 0.20
    geographical_weight: float = 0.20
    technological_weight: float = 0.20
    completeness_weight: float = 0.20
    reliability_weight: float = 0.20
    min_coverage: float = 0.70
    target_coverage: float = 0.90


@dataclass
class ComplianceConfig:
    """Compliance framework configuration.

    Attributes:
        enabled_frameworks: List of enabled regulatory frameworks.
        strict_mode: Reject calculations that fail compliance checks.
    """

    enabled_frameworks: List[str] = field(default_factory=lambda: [
        "GHG_PROTOCOL_SCOPE3",
        "ISO_14064_1",
        "CSRD",
        "SEC_CLIMATE_DISCLOSURE",
        "TCFD",
        "CDP",
    ])
    strict_mode: bool = False


@dataclass
class CacheConfig:
    """Cache configuration.

    Attributes:
        redis_url: Redis connection URL.
        ttl: Cache TTL in seconds.
        enabled: Enable caching.
    """

    redis_url: str = ""
    ttl: int = 3600
    enabled: bool = True


@dataclass
class APIConfig:
    """API configuration.

    Attributes:
        prefix: API route prefix.
        rate_limit: Rate limit (requests per minute).
        page_size: Default page size for pagination.
        max_page_size: Maximum page size for pagination.
    """

    prefix: str = "/api/v1/fuel-energy-activities"
    rate_limit: int = 1000
    page_size: int = 20
    max_page_size: int = 100


@dataclass
class MetricsConfig:
    """Metrics and observability configuration.

    Attributes:
        enabled: Enable Prometheus metrics export.
        prefix: Metrics prefix (gl_fea_).
        enable_tracing: Enable OpenTelemetry tracing.
    """

    enabled: bool = True
    prefix: str = "gl_fea_"
    enable_tracing: bool = True


@dataclass
class ProvenanceConfig:
    """Provenance tracking configuration.

    Attributes:
        enabled: Enable SHA-256 provenance chain.
        genesis_hash: Genesis anchor for provenance chain.
    """

    enabled: bool = True
    genesis_hash: str = "GL-MRV-S3-003-FUEL-ENERGY-ACTIVITIES-GENESIS"


# ---------------------------------------------------------------------------
# Main configuration class
# ---------------------------------------------------------------------------


@dataclass
class FuelEnergyActivitiesConfig:
    """Complete configuration for the GreenLang Fuel & Energy Activities Agent SDK.

    This configuration covers Scope 3 Category 3: Fuel- and Energy-Related
    Activities Not Included in Scope 1 or Scope 2, including:
    - Upstream emissions of purchased fuels (well-to-tank)
    - Upstream emissions of purchased electricity (generation)
    - Transmission and distribution losses (electricity/steam/heating/cooling)

    Attributes are grouped by concern: service, database, calculation,
    data quality, compliance, cache, API, metrics, and provenance.

    All attributes can be overridden via environment variables using the
    ``GL_FEA_`` prefix (e.g. ``GL_FEA_DEFAULT_GWP_SOURCE=AR6``).

    Attributes:
        enabled: Master switch to enable/disable the agent.
        service: Service-level configuration.
        database: Database connection configuration.
        calculation: Calculation methodology configuration.
        dqi: Data quality indicator configuration.
        compliance: Compliance framework configuration.
        cache: Cache configuration.
        api: API configuration.
        metrics: Metrics and observability configuration.
        provenance: Provenance tracking configuration.
        max_batch_size: Maximum records per batch.
        max_fuel_types: Maximum fuel type registrations.
        max_electricity_sources: Maximum electricity source registrations.
        max_activity_records: Maximum activity records per batch.
        monte_carlo_iterations: Monte Carlo simulation iterations.
        monte_carlo_seed: Random seed for reproducibility.
        confidence_levels: Comma-separated confidence levels.
        worker_threads: Worker thread pool size.
        enable_background_tasks: Enable background task processing.
        health_check_interval: Health check interval in seconds.
    """

    # -- Feature flag --------------------------------------------------------
    enabled: bool = True

    # -- Configuration sections ----------------------------------------------
    service: ServiceConfig = field(default_factory=ServiceConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    calculation: CalculationConfig = field(default_factory=CalculationConfig)
    dqi: DQIConfig = field(default_factory=DQIConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    api: APIConfig = field(default_factory=APIConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)

    # -- Capacity limits -----------------------------------------------------
    max_batch_size: int = 500
    max_fuel_types: int = 200
    max_electricity_sources: int = 100
    max_activity_records: int = 10_000

    # -- Monte Carlo uncertainty analysis ------------------------------------
    monte_carlo_iterations: int = 5_000
    monte_carlo_seed: int = 42
    confidence_levels: str = "90,95,99"

    # -- Performance tuning --------------------------------------------------
    worker_threads: int = 4
    enable_background_tasks: bool = True
    health_check_interval: int = 30

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialization.

        Performs range checks on all numeric fields, enumeration checks on
        string fields (GWP source, calculation method, WTT source, T&D source,
        currency, log level), and normalization of values.

        Raises:
            ValueError: If any configuration value is outside its valid
                range or violates a constraint. The exception message
                lists all detected errors, not just the first one.
        """
        errors: list[str] = []

        # -- Service configuration -------------------------------------------
        normalized_log = self.service.log_level.upper()
        if normalized_log not in _VALID_LOG_LEVELS:
            errors.append(
                f"service.log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got '{self.service.log_level}'"
            )
        else:
            self.service.log_level = normalized_log

        if not self.service.name:
            errors.append("service.name must not be empty")

        if not self.service.version:
            errors.append("service.version must not be empty")

        # -- Database configuration ------------------------------------------
        if self.database.port <= 0 or self.database.port > 65535:
            errors.append(
                f"database.port must be in range 1-65535, "
                f"got {self.database.port}"
            )

        if self.database.pool_min <= 0:
            errors.append(
                f"database.pool_min must be > 0, "
                f"got {self.database.pool_min}"
            )

        if self.database.pool_max <= 0:
            errors.append(
                f"database.pool_max must be > 0, "
                f"got {self.database.pool_max}"
            )

        if self.database.pool_min > self.database.pool_max:
            errors.append(
                f"database.pool_min ({self.database.pool_min}) "
                f"must be <= pool_max ({self.database.pool_max})"
            )

        # -- Calculation configuration ---------------------------------------
        normalized_gwp = self.calculation.default_gwp.upper()
        if normalized_gwp not in _VALID_GWP_SOURCES:
            errors.append(
                f"calculation.default_gwp must be one of "
                f"{sorted(_VALID_GWP_SOURCES)}, "
                f"got '{self.calculation.default_gwp}'"
            )
        else:
            self.calculation.default_gwp = normalized_gwp

        normalized_method = self.calculation.default_method.upper()
        if normalized_method not in _VALID_CALCULATION_METHODS:
            errors.append(
                f"calculation.default_method must be one of "
                f"{sorted(_VALID_CALCULATION_METHODS)}, "
                f"got '{self.calculation.default_method}'"
            )
        else:
            self.calculation.default_method = normalized_method

        normalized_wtt = self.calculation.default_wtt_source.upper()
        if normalized_wtt not in _VALID_WTT_SOURCES:
            errors.append(
                f"calculation.default_wtt_source must be one of "
                f"{sorted(_VALID_WTT_SOURCES)}, "
                f"got '{self.calculation.default_wtt_source}'"
            )
        else:
            self.calculation.default_wtt_source = normalized_wtt

        normalized_td = self.calculation.default_td_source.upper()
        if normalized_td not in _VALID_TD_SOURCES:
            errors.append(
                f"calculation.default_td_source must be one of "
                f"{sorted(_VALID_TD_SOURCES)}, "
                f"got '{self.calculation.default_td_source}'"
            )
        else:
            self.calculation.default_td_source = normalized_td

        normalized_currency = self.calculation.default_currency.upper()
        if normalized_currency not in _VALID_CURRENCIES:
            errors.append(
                f"calculation.default_currency must be one of "
                f"{sorted(_VALID_CURRENCIES)}, "
                f"got '{self.calculation.default_currency}'"
            )
        else:
            self.calculation.default_currency = normalized_currency

        normalized_rounding = self.calculation.rounding_mode.upper()
        if normalized_rounding not in _VALID_ROUNDING_MODES:
            errors.append(
                f"calculation.rounding_mode must be one of "
                f"{sorted(_VALID_ROUNDING_MODES)}, "
                f"got '{self.calculation.rounding_mode}'"
            )
        else:
            self.calculation.rounding_mode = normalized_rounding

        if self.calculation.decimal_places < 0:
            errors.append(
                f"calculation.decimal_places must be >= 0, "
                f"got {self.calculation.decimal_places}"
            )
        if self.calculation.decimal_places > 20:
            errors.append(
                f"calculation.decimal_places must be <= 20, "
                f"got {self.calculation.decimal_places}"
            )

        if self.calculation.base_year < 1990 or self.calculation.base_year > 2100:
            errors.append(
                f"calculation.base_year must be in range 1990-2100, "
                f"got {self.calculation.base_year}"
            )

        # -- DQI configuration -----------------------------------------------
        for field_name, value in [
            ("temporal_weight", self.dqi.temporal_weight),
            ("geographical_weight", self.dqi.geographical_weight),
            ("technological_weight", self.dqi.technological_weight),
            ("completeness_weight", self.dqi.completeness_weight),
            ("reliability_weight", self.dqi.reliability_weight),
        ]:
            if not (0.0 <= value <= 1.0):
                errors.append(
                    f"dqi.{field_name} must be in range [0.0, 1.0], "
                    f"got {value}"
                )

        total_weight = (
            self.dqi.temporal_weight +
            self.dqi.geographical_weight +
            self.dqi.technological_weight +
            self.dqi.completeness_weight +
            self.dqi.reliability_weight
        )
        if not (0.99 <= total_weight <= 1.01):
            errors.append(
                f"dqi weights must sum to 1.0, got {total_weight}"
            )

        if not (0.0 <= self.dqi.min_coverage <= 1.0):
            errors.append(
                f"dqi.min_coverage must be in range [0.0, 1.0], "
                f"got {self.dqi.min_coverage}"
            )

        if not (0.0 <= self.dqi.target_coverage <= 1.0):
            errors.append(
                f"dqi.target_coverage must be in range [0.0, 1.0], "
                f"got {self.dqi.target_coverage}"
            )

        if self.dqi.min_coverage > self.dqi.target_coverage:
            errors.append(
                f"dqi.min_coverage ({self.dqi.min_coverage}) "
                f"must be <= target_coverage ({self.dqi.target_coverage})"
            )

        # -- Capacity limits -------------------------------------------------
        for field_name, value, upper in [
            ("max_batch_size", self.max_batch_size, 100_000),
            ("max_fuel_types", self.max_fuel_types, 10_000),
            ("max_electricity_sources", self.max_electricity_sources, 5_000),
            ("max_activity_records", self.max_activity_records, 1_000_000),
        ]:
            if value <= 0:
                errors.append(
                    f"{field_name} must be > 0, got {value}"
                )
            if value > upper:
                errors.append(
                    f"{field_name} must be <= {upper}, got {value}"
                )

        # -- Monte Carlo -----------------------------------------------------
        if self.monte_carlo_iterations <= 0:
            errors.append(
                f"monte_carlo_iterations must be > 0, "
                f"got {self.monte_carlo_iterations}"
            )
        if self.monte_carlo_iterations > 1_000_000:
            errors.append(
                f"monte_carlo_iterations must be <= 1000000, "
                f"got {self.monte_carlo_iterations}"
            )
        if self.monte_carlo_seed < 0:
            errors.append(
                f"monte_carlo_seed must be >= 0, "
                f"got {self.monte_carlo_seed}"
            )

        # -- Confidence levels -----------------------------------------------
        try:
            levels = [
                float(x.strip())
                for x in self.confidence_levels.split(",")
                if x.strip()
            ]
            for lvl in levels:
                if not (0.0 < lvl < 100.0):
                    errors.append(
                        f"Each confidence level must be in (0, 100), "
                        f"got {lvl}"
                    )
        except ValueError:
            errors.append(
                f"confidence_levels must be comma-separated floats, "
                f"got '{self.confidence_levels}'"
            )

        # -- Cache configuration ---------------------------------------------
        if self.cache.ttl <= 0:
            errors.append(
                f"cache.ttl must be > 0, got {self.cache.ttl}"
            )

        # -- API configuration -----------------------------------------------
        if self.api.max_page_size <= 0:
            errors.append(
                f"api.max_page_size must be > 0, "
                f"got {self.api.max_page_size}"
            )
        if self.api.page_size <= 0:
            errors.append(
                f"api.page_size must be > 0, "
                f"got {self.api.page_size}"
            )
        if self.api.page_size > self.api.max_page_size:
            errors.append(
                f"api.page_size ({self.api.page_size}) "
                f"must be <= max_page_size ({self.api.max_page_size})"
            )
        if self.api.rate_limit <= 0:
            errors.append(
                f"api.rate_limit must be > 0, got {self.api.rate_limit}"
            )

        # -- Performance tuning ----------------------------------------------
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

        # -- Provenance ------------------------------------------------------
        if not self.provenance.genesis_hash:
            errors.append("provenance.genesis_hash must not be empty")

        if errors:
            raise ValueError(
                "FuelEnergyActivitiesConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "FuelEnergyActivitiesConfig validated successfully: "
            "enabled=%s, gwp=%s, method=%s, wtt_source=%s, "
            "td_source=%s, decimal_places=%d, base_year=%d, "
            "currency=%s, enable_wtt=%s, enable_upstream_elec=%s, "
            "enable_td_losses=%s, max_batch_size=%d, "
            "monte_carlo_iterations=%d, confidence_levels=%s, "
            "provenance=%s, metrics=%s",
            self.enabled,
            self.calculation.default_gwp,
            self.calculation.default_method,
            self.calculation.default_wtt_source,
            self.calculation.default_td_source,
            self.calculation.decimal_places,
            self.calculation.base_year,
            self.calculation.default_currency,
            self.calculation.enable_wtt_factors,
            self.calculation.enable_upstream_electricity,
            self.calculation.enable_td_losses,
            self.max_batch_size,
            self.monte_carlo_iterations,
            self.confidence_levels,
            self.provenance.enabled,
            self.metrics.enabled,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> FuelEnergyActivitiesConfig:
        """Build a FuelEnergyActivitiesConfig from environment variables.

        Every field can be overridden via ``GL_FEA_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated FuelEnergyActivitiesConfig instance, validated via
            ``__post_init__``.

        Example:
            >>> import os
            >>> os.environ["GL_FEA_DEFAULT_GWP_SOURCE"] = "AR6"
            >>> cfg = FuelEnergyActivitiesConfig.from_env()
            >>> cfg.calculation.default_gwp
            'AR6'
        """
        prefix = _ENV_PREFIX

        def _env(name: str, default: Any = None) -> Optional[str]:
            return os.environ.get(f"{prefix}{name}", default)

        def _bool(name: str, default: bool) -> bool:
            val = _env(name)
            if val is None:
                return default
            return val.strip().lower() in ("true", "1", "yes")

        def _int(name: str, default: int) -> int:
            val = _env(name)
            if val is None:
                return default
            try:
                return int(val.strip())
            except ValueError:
                logger.warning(
                    "Invalid integer for %s%s=%r, using default %d",
                    prefix,
                    name,
                    val,
                    default,
                )
                return default

        def _float(name: str, default: float) -> float:
            val = _env(name)
            if val is None:
                return default
            try:
                return float(val.strip())
            except ValueError:
                logger.warning(
                    "Invalid float for %s%s=%r, using default %f",
                    prefix,
                    name,
                    val,
                    default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val.strip()

        def _list(name: str, default: List[str]) -> List[str]:
            val = _env(name)
            if val is None:
                return default
            return [x.strip() for x in val.split(",") if x.strip()]

        # Build section configs
        service = ServiceConfig(
            name=_str("SERVICE_NAME", ServiceConfig.name),
            version=_str("SERVICE_VERSION", ServiceConfig.version),
            log_level=_str("LOG_LEVEL", ServiceConfig.log_level),
            environment=_str("ENVIRONMENT", ServiceConfig.environment),
            default_tenant=_str("DEFAULT_TENANT", ServiceConfig.default_tenant),
        )

        database = DatabaseConfig(
            host=_str("DATABASE_HOST", DatabaseConfig.host),
            port=_int("DATABASE_PORT", DatabaseConfig.port),
            name=_str("DATABASE_NAME", DatabaseConfig.name),
            user=_str("DATABASE_USER", DatabaseConfig.user),
            password=_str("DATABASE_PASSWORD", DatabaseConfig.password),
            pool_min=_int("DATABASE_POOL_MIN", DatabaseConfig.pool_min),
            pool_max=_int("DATABASE_POOL_MAX", DatabaseConfig.pool_max),
            ssl_mode=_str("DATABASE_SSL_MODE", DatabaseConfig.ssl_mode),
            table_prefix=_str("TABLE_PREFIX", DatabaseConfig.table_prefix),
        )

        calculation = CalculationConfig(
            default_method=_str("DEFAULT_CALCULATION_METHOD", CalculationConfig.default_method),
            default_gwp=_str("DEFAULT_GWP_SOURCE", CalculationConfig.default_gwp),
            decimal_places=_int("DECIMAL_PRECISION", CalculationConfig.decimal_places),
            rounding_mode=_str("ROUNDING_MODE", CalculationConfig.rounding_mode),
            base_year=_int("BASE_YEAR", CalculationConfig.base_year),
            default_currency=_str("DEFAULT_CURRENCY", CalculationConfig.default_currency),
            enable_wtt_factors=_bool("ENABLE_WTT_FACTORS", CalculationConfig.enable_wtt_factors),
            enable_upstream_electricity=_bool("ENABLE_UPSTREAM_ELECTRICITY", CalculationConfig.enable_upstream_electricity),
            enable_td_losses=_bool("ENABLE_TD_LOSSES", CalculationConfig.enable_td_losses),
            default_td_source=_str("DEFAULT_TD_SOURCE", CalculationConfig.default_td_source),
            default_wtt_source=_str("DEFAULT_WTT_SOURCE", CalculationConfig.default_wtt_source),
        )

        dqi = DQIConfig(
            temporal_weight=_float("DQI_TEMPORAL_WEIGHT", DQIConfig.temporal_weight),
            geographical_weight=_float("DQI_GEOGRAPHICAL_WEIGHT", DQIConfig.geographical_weight),
            technological_weight=_float("DQI_TECHNOLOGICAL_WEIGHT", DQIConfig.technological_weight),
            completeness_weight=_float("DQI_COMPLETENESS_WEIGHT", DQIConfig.completeness_weight),
            reliability_weight=_float("DQI_RELIABILITY_WEIGHT", DQIConfig.reliability_weight),
            min_coverage=_float("DQI_MIN_COVERAGE", DQIConfig.min_coverage),
            target_coverage=_float("DQI_TARGET_COVERAGE", DQIConfig.target_coverage),
        )

        compliance = ComplianceConfig(
            enabled_frameworks=_list("ENABLED_FRAMEWORKS", ComplianceConfig.enabled_frameworks),
            strict_mode=_bool("STRICT_MODE", ComplianceConfig.strict_mode),
        )

        cache = CacheConfig(
            redis_url=_str("REDIS_URL", CacheConfig.redis_url),
            ttl=_int("CACHE_TTL_SECONDS", CacheConfig.ttl),
            enabled=_bool("CACHE_ENABLED", CacheConfig.enabled),
        )

        api = APIConfig(
            prefix=_str("API_PREFIX", APIConfig.prefix),
            rate_limit=_int("API_RATE_LIMIT", APIConfig.rate_limit),
            page_size=_int("API_DEFAULT_PAGE_SIZE", APIConfig.page_size),
            max_page_size=_int("API_MAX_PAGE_SIZE", APIConfig.max_page_size),
        )

        metrics = MetricsConfig(
            enabled=_bool("ENABLE_METRICS", MetricsConfig.enabled),
            prefix=_str("METRICS_PREFIX", MetricsConfig.prefix),
            enable_tracing=_bool("ENABLE_TRACING", MetricsConfig.enable_tracing),
        )

        provenance = ProvenanceConfig(
            enabled=_bool("ENABLE_PROVENANCE", ProvenanceConfig.enabled),
            genesis_hash=_str("GENESIS_HASH", ProvenanceConfig.genesis_hash),
        )

        config = cls(
            # Feature flag
            enabled=_bool("ENABLED", cls.enabled),
            # Sections
            service=service,
            database=database,
            calculation=calculation,
            dqi=dqi,
            compliance=compliance,
            cache=cache,
            api=api,
            metrics=metrics,
            provenance=provenance,
            # Capacity limits
            max_batch_size=_int("MAX_BATCH_SIZE", cls.max_batch_size),
            max_fuel_types=_int("MAX_FUEL_TYPES", cls.max_fuel_types),
            max_electricity_sources=_int("MAX_ELECTRICITY_SOURCES", cls.max_electricity_sources),
            max_activity_records=_int("MAX_ACTIVITY_RECORDS", cls.max_activity_records),
            # Monte Carlo
            monte_carlo_iterations=_int("MONTE_CARLO_ITERATIONS", cls.monte_carlo_iterations),
            monte_carlo_seed=_int("MONTE_CARLO_SEED", cls.monte_carlo_seed),
            confidence_levels=_str("CONFIDENCE_LEVELS", cls.confidence_levels),
            # Performance
            worker_threads=_int("WORKER_THREADS", cls.worker_threads),
            enable_background_tasks=_bool("ENABLE_BACKGROUND_TASKS", cls.enable_background_tasks),
            health_check_interval=_int("HEALTH_CHECK_INTERVAL", cls.health_check_interval),
        )

        logger.info(
            "FuelEnergyActivitiesConfig loaded: "
            "enabled=%s, gwp=%s, method=%s, wtt_source=%s, "
            "td_source=%s, decimal_places=%d, base_year=%d, "
            "currency=%s, enable_wtt=%s, enable_upstream_elec=%s, "
            "enable_td_losses=%s, max_batch_size=%d, "
            "max_fuel_types=%d, max_electricity_sources=%d, "
            "max_activity_records=%d, monte_carlo_iterations=%d, "
            "confidence_levels=%s, cache_ttl=%ds, worker_threads=%d, "
            "background_tasks=%s, health_check=%ds, "
            "provenance=%s, metrics=%s",
            config.enabled,
            config.calculation.default_gwp,
            config.calculation.default_method,
            config.calculation.default_wtt_source,
            config.calculation.default_td_source,
            config.calculation.decimal_places,
            config.calculation.base_year,
            config.calculation.default_currency,
            config.calculation.enable_wtt_factors,
            config.calculation.enable_upstream_electricity,
            config.calculation.enable_td_losses,
            config.max_batch_size,
            config.max_fuel_types,
            config.max_electricity_sources,
            config.max_activity_records,
            config.monte_carlo_iterations,
            config.confidence_levels,
            config.cache.ttl,
            config.worker_threads,
            config.enable_background_tasks,
            config.health_check_interval,
            config.provenance.enabled,
            config.metrics.enabled,
        )
        return config

    # ------------------------------------------------------------------
    # Validation method
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate the configuration.

        This method is called automatically by __post_init__, but can
        be called manually after programmatic changes to configuration
        values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        # Validation is already performed in __post_init__
        # This method is provided for consistency with other agents
        self.__post_init__()

    # ------------------------------------------------------------------
    # Accessor properties
    # ------------------------------------------------------------------

    @property
    def database_url(self) -> str:
        """Build PostgreSQL connection URL from database config.

        Returns:
            PostgreSQL connection URL (credentials redacted in logs).
        """
        if self.database.password:
            return (
                f"postgresql://{self.database.user}:{self.database.password}@"
                f"{self.database.host}:{self.database.port}/{self.database.name}"
            )
        return (
            f"postgresql://{self.database.user}@"
            f"{self.database.host}:{self.database.port}/{self.database.name}"
        )

    @property
    def redis_url(self) -> str:
        """Get Redis connection URL.

        Returns:
            Redis connection URL.
        """
        return self.cache.redis_url

    @property
    def log_level(self) -> str:
        """Get logging level.

        Returns:
            Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL).
        """
        return self.service.log_level

    @property
    def default_gwp_source(self) -> str:
        """Get default GWP source.

        Returns:
            Default GWP source (AR4/AR5/AR6/AR6_20YR).
        """
        return self.calculation.default_gwp

    @property
    def default_calculation_method(self) -> str:
        """Get default calculation method.

        Returns:
            Default calculation method (HYBRID/WTT_ONLY/TD_ONLY/UPSTREAM_ELECTRICITY).
        """
        return self.calculation.default_method

    @property
    def default_wtt_source(self) -> str:
        """Get default WTT source.

        Returns:
            Default WTT source (DEFRA/EPA/ECOINVENT/IEA/CUSTOM).
        """
        return self.calculation.default_wtt_source

    @property
    def default_td_source(self) -> str:
        """Get default T&D loss source.

        Returns:
            Default T&D loss source (IEA/EIA/NATIONAL_GRID/CUSTOM).
        """
        return self.calculation.default_td_source

    @property
    def decimal_precision(self) -> int:
        """Get decimal precision.

        Returns:
            Decimal precision for calculations.
        """
        return self.calculation.decimal_places

    @property
    def enable_wtt_factors(self) -> bool:
        """Check if WTT factors are enabled.

        Returns:
            True if WTT factors are enabled, False otherwise.
        """
        return self.calculation.enable_wtt_factors

    @property
    def enable_upstream_electricity(self) -> bool:
        """Check if upstream electricity emissions are enabled.

        Returns:
            True if upstream electricity emissions are enabled, False otherwise.
        """
        return self.calculation.enable_upstream_electricity

    @property
    def enable_td_losses(self) -> bool:
        """Check if T&D losses are enabled.

        Returns:
            True if T&D losses are enabled, False otherwise.
        """
        return self.calculation.enable_td_losses

    @property
    def enable_compliance_checking(self) -> bool:
        """Check if compliance checking is enabled.

        Returns:
            True if compliance checking is enabled, False otherwise.
        """
        return bool(self.compliance.enabled_frameworks) and not self.compliance.strict_mode

    @property
    def enable_provenance(self) -> bool:
        """Check if provenance tracking is enabled.

        Returns:
            True if provenance tracking is enabled, False otherwise.
        """
        return self.provenance.enabled

    @property
    def enable_metrics(self) -> bool:
        """Check if metrics export is enabled.

        Returns:
            True if metrics export is enabled, False otherwise.
        """
        return self.metrics.enabled

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a plain Python dictionary.

        The returned dictionary is safe to pass to ``json.dumps``,
        ``yaml.dump``, or any structured logging framework. All values
        are JSON-serializable primitives (str, int, float, bool).

        Sensitive connection strings (database password, redis_url) are
        redacted to prevent accidental credential leakage in logs,
        exception tracebacks, and monitoring dashboards.

        Returns:
            Dictionary representation of the configuration with sensitive
            fields redacted.

        Example:
            >>> cfg = FuelEnergyActivitiesConfig()
            >>> d = cfg.to_dict()
            >>> d["calculation"]["default_method"]
            'HYBRID'
            >>> d["database"]["password"]  # redacted
            '***'
        """
        return {
            "enabled": self.enabled,
            "service": {
                "name": self.service.name,
                "version": self.service.version,
                "log_level": self.service.log_level,
                "environment": self.service.environment,
                "default_tenant": self.service.default_tenant,
            },
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "name": self.database.name,
                "user": self.database.user,
                "password": "***" if self.database.password else "",
                "pool_min": self.database.pool_min,
                "pool_max": self.database.pool_max,
                "ssl_mode": self.database.ssl_mode,
                "table_prefix": self.database.table_prefix,
            },
            "calculation": {
                "default_method": self.calculation.default_method,
                "default_gwp": self.calculation.default_gwp,
                "decimal_places": self.calculation.decimal_places,
                "rounding_mode": self.calculation.rounding_mode,
                "base_year": self.calculation.base_year,
                "default_currency": self.calculation.default_currency,
                "enable_wtt_factors": self.calculation.enable_wtt_factors,
                "enable_upstream_electricity": self.calculation.enable_upstream_electricity,
                "enable_td_losses": self.calculation.enable_td_losses,
                "default_td_source": self.calculation.default_td_source,
                "default_wtt_source": self.calculation.default_wtt_source,
            },
            "dqi": {
                "temporal_weight": self.dqi.temporal_weight,
                "geographical_weight": self.dqi.geographical_weight,
                "technological_weight": self.dqi.technological_weight,
                "completeness_weight": self.dqi.completeness_weight,
                "reliability_weight": self.dqi.reliability_weight,
                "min_coverage": self.dqi.min_coverage,
                "target_coverage": self.dqi.target_coverage,
            },
            "compliance": {
                "enabled_frameworks": self.compliance.enabled_frameworks,
                "strict_mode": self.compliance.strict_mode,
            },
            "cache": {
                "redis_url": "***" if self.cache.redis_url else "",
                "ttl": self.cache.ttl,
                "enabled": self.cache.enabled,
            },
            "api": {
                "prefix": self.api.prefix,
                "rate_limit": self.api.rate_limit,
                "page_size": self.api.page_size,
                "max_page_size": self.api.max_page_size,
            },
            "metrics": {
                "enabled": self.metrics.enabled,
                "prefix": self.metrics.prefix,
                "enable_tracing": self.metrics.enable_tracing,
            },
            "provenance": {
                "enabled": self.provenance.enabled,
                "genesis_hash": self.provenance.genesis_hash,
            },
            "max_batch_size": self.max_batch_size,
            "max_fuel_types": self.max_fuel_types,
            "max_electricity_sources": self.max_electricity_sources,
            "max_activity_records": self.max_activity_records,
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "monte_carlo_seed": self.monte_carlo_seed,
            "confidence_levels": self.confidence_levels,
            "worker_threads": self.worker_threads,
            "enable_background_tasks": self.enable_background_tasks,
            "health_check_interval": self.health_check_interval,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Sensitive fields (database password, redis_url) are replaced with
        ``'***'`` so that repr output is safe to include in log messages
        and exception tracebacks.

        Returns:
            String representation of the configuration.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"FuelEnergyActivitiesConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[FuelEnergyActivitiesConfig] = None
_config_lock = threading.RLock()


def get_config() -> FuelEnergyActivitiesConfig:
    """Return the singleton FuelEnergyActivitiesConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_FEA_*`` environment variables via
    :meth:`FuelEnergyActivitiesConfig.from_env`.

    Returns:
        FuelEnergyActivitiesConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.calculation.default_method
        'HYBRID'
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = FuelEnergyActivitiesConfig.from_env()
    return _config_instance


def set_config(config: FuelEnergyActivitiesConfig) -> None:
    """Replace the singleton FuelEnergyActivitiesConfig.

    Primarily intended for testing and dependency injection scenarios
    where a custom configuration must be supplied without relying on
    environment variables.

    Args:
        config: New :class:`FuelEnergyActivitiesConfig` to install as the
            singleton.

    Example:
        >>> cfg = FuelEnergyActivitiesConfig(calculation=CalculationConfig(enable_wtt_factors=False))
        >>> set_config(cfg)
        >>> assert not get_config().enable_wtt_factors
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "FuelEnergyActivitiesConfig replaced programmatically: "
        "enabled=%s, gwp=%s, method=%s, wtt_source=%s, "
        "td_source=%s, max_batch_size=%d, "
        "monte_carlo_iterations=%d",
        config.enabled,
        config.calculation.default_gwp,
        config.calculation.default_method,
        config.calculation.default_wtt_source,
        config.calculation.default_td_source,
        config.max_batch_size,
        config.monte_carlo_iterations,
    )


def reset_config() -> None:
    """Reset the singleton FuelEnergyActivitiesConfig to ``None``.

    The next call to :func:`get_config` will re-read environment variables
    and construct a fresh instance. Intended for test teardown to prevent
    state leakage between test cases.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads GL_FEA_* env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("FuelEnergyActivitiesConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "ServiceConfig",
    "DatabaseConfig",
    "CalculationConfig",
    "DQIConfig",
    "ComplianceConfig",
    "CacheConfig",
    "APIConfig",
    "MetricsConfig",
    "ProvenanceConfig",
    "FuelEnergyActivitiesConfig",
    "get_config",
    "set_config",
    "reset_config",
]
