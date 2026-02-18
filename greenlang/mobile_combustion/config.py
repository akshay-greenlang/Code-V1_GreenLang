# -*- coding: utf-8 -*-
"""
Mobile Combustion Agent Configuration - AGENT-MRV-003

Centralized configuration for the Mobile Combustion Agent SDK covering:
- Database, cache, and connection defaults
- GHG Protocol methodology defaults (GWP source, calculation method, tier)
- Vehicle fleet management parameters
- Distance and fuel economy unit defaults
- Monte Carlo uncertainty analysis parameters
- Biogenic carbon tracking toggle (biofuel blends)
- Compliance and regulatory framework defaults
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Prometheus metrics export toggle
- Performance tuning (batch sizes, timeouts, pool sizes)

All settings can be overridden via environment variables with the
``GL_MOBILE_COMBUSTION_`` prefix (e.g. ``GL_MOBILE_COMBUSTION_DEFAULT_GWP_SOURCE``,
``GL_MOBILE_COMBUSTION_MAX_BATCH_SIZE``).

Environment Variable Reference (GL_MOBILE_COMBUSTION_ prefix):
    GL_MOBILE_COMBUSTION_DATABASE_URL              - PostgreSQL connection URL
    GL_MOBILE_COMBUSTION_REDIS_URL                 - Redis connection URL
    GL_MOBILE_COMBUSTION_LOG_LEVEL                 - Logging level (DEBUG/INFO/WARNING/ERROR)
    GL_MOBILE_COMBUSTION_DEFAULT_GWP_SOURCE        - Default GWP source (AR4, AR5, AR6, AR6_20YR)
    GL_MOBILE_COMBUSTION_DEFAULT_CALCULATION_METHOD - Default method (FUEL_BASED, DISTANCE_BASED, SPEND_BASED)
    GL_MOBILE_COMBUSTION_MONTE_CARLO_ITERATIONS    - Monte Carlo simulation iterations
    GL_MOBILE_COMBUSTION_BATCH_SIZE                - Default batch processing size
    GL_MOBILE_COMBUSTION_MAX_BATCH_SIZE            - Maximum records per batch
    GL_MOBILE_COMBUSTION_CACHE_TTL_SECONDS         - Cache time-to-live in seconds
    GL_MOBILE_COMBUSTION_ENABLE_BIOGENIC_TRACKING  - Enable biogenic CO2 tracking
    GL_MOBILE_COMBUSTION_ENABLE_UNCERTAINTY        - Enable uncertainty quantification
    GL_MOBILE_COMBUSTION_ENABLE_COMPLIANCE         - Enable regulatory compliance checks
    GL_MOBILE_COMBUSTION_ENABLE_FLEET_MANAGEMENT   - Enable fleet management features
    GL_MOBILE_COMBUSTION_DECIMAL_PRECISION         - Decimal places for calculations
    GL_MOBILE_COMBUSTION_DEFAULT_VEHICLE_TYPE      - Default vehicle type
    GL_MOBILE_COMBUSTION_DEFAULT_FUEL_TYPE         - Default fuel type
    GL_MOBILE_COMBUSTION_DEFAULT_DISTANCE_UNIT     - Default distance unit (KM, MILES, NAUTICAL_MILES)
    GL_MOBILE_COMBUSTION_DEFAULT_FUEL_ECONOMY_UNIT - Default fuel economy unit
    GL_MOBILE_COMBUSTION_CONFIDENCE_LEVEL_90       - 90% confidence level threshold
    GL_MOBILE_COMBUSTION_CONFIDENCE_LEVEL_95       - 95% confidence level threshold
    GL_MOBILE_COMBUSTION_CONFIDENCE_LEVEL_99       - 99% confidence level threshold
    GL_MOBILE_COMBUSTION_DEFAULT_REGULATORY_FRAMEWORK - Default regulatory framework
    GL_MOBILE_COMBUSTION_MAX_VEHICLES_PER_FLEET    - Maximum vehicles per fleet
    GL_MOBILE_COMBUSTION_MAX_TRIPS_PER_QUERY       - Maximum trips per query
    GL_MOBILE_COMBUSTION_CALCULATION_TIMEOUT_SECONDS - Calculation timeout in seconds
    GL_MOBILE_COMBUSTION_ENABLE_METRICS            - Enable Prometheus metrics export
    GL_MOBILE_COMBUSTION_ENABLE_TRACING            - Enable OpenTelemetry tracing
    GL_MOBILE_COMBUSTION_ENABLE_PROVENANCE         - Enable SHA-256 provenance chain
    GL_MOBILE_COMBUSTION_GENESIS_HASH              - Genesis anchor string for provenance
    GL_MOBILE_COMBUSTION_POOL_SIZE                 - Database connection pool size
    GL_MOBILE_COMBUSTION_RATE_LIMIT                - Max API requests per minute

Example:
    >>> from greenlang.mobile_combustion.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_gwp_source, cfg.default_calculation_method)
    AR6 FUEL_BASED

    >>> # Override for testing
    >>> from greenlang.mobile_combustion.config import set_config, reset_config
    >>> from greenlang.mobile_combustion.config import MobileCombustionConfig
    >>> set_config(MobileCombustionConfig(default_gwp_source="AR5"))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_MOBILE_COMBUSTION_"

# ---------------------------------------------------------------------------
# Valid enumeration values for validation
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

_VALID_GWP_SOURCES = frozenset({"AR4", "AR5", "AR6", "AR6_20YR"})

_VALID_CALCULATION_METHODS = frozenset(
    {"FUEL_BASED", "DISTANCE_BASED", "SPEND_BASED"}
)

_VALID_DISTANCE_UNITS = frozenset({"KM", "MILES", "NAUTICAL_MILES"})

_VALID_FUEL_ECONOMY_UNITS = frozenset(
    {"L_PER_100KM", "MPG_US", "MPG_UK", "KM_PER_L"}
)

_VALID_REGULATORY_FRAMEWORKS = frozenset({
    "GHG_PROTOCOL",
    "ISO_14064",
    "CSRD_ESRS_E1",
    "EPA_40CFR98",
    "UK_SECR",
    "EU_ETS",
})


# ---------------------------------------------------------------------------
# MobileCombustionConfig
# ---------------------------------------------------------------------------


@dataclass
class MobileCombustionConfig:
    """Complete configuration for the GreenLang Mobile Combustion Agent SDK.

    Attributes are grouped by concern: connections, logging, GHG methodology
    defaults, vehicle defaults, distance/fuel economy defaults, uncertainty
    parameters, compliance settings, fleet management, performance tuning,
    provenance tracking, and metrics export.

    All attributes can be overridden via environment variables using the
    ``GL_MOBILE_COMBUSTION_`` prefix (e.g.
    ``GL_MOBILE_COMBUSTION_DEFAULT_CALCULATION_METHOD=DISTANCE_BASED``).

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage of
            vehicle records, trip data, emission factors, and calculation
            results.
        redis_url: Redis connection URL for caching emission factor lookups,
            fuel economy data, and distributed locks.
        log_level: Logging verbosity level. Accepts DEBUG, INFO, WARNING,
            ERROR, or CRITICAL.
        default_gwp_source: Default IPCC Assessment Report edition for
            Global Warming Potential values. Valid: AR4, AR5, AR6, AR6_20YR.
        default_calculation_method: Default calculation approach for
            mobile combustion emissions. FUEL_BASED uses actual fuel
            consumption; DISTANCE_BASED uses distance and fuel economy;
            SPEND_BASED uses expenditure-based emission factors.
        monte_carlo_iterations: Number of Monte Carlo simulation iterations
            for uncertainty quantification.
        batch_size: Default number of records to process in a single batch
            chunk for memory efficiency.
        max_batch_size: Maximum number of calculation input records that
            can be processed in a single batch request.
        cache_ttl_seconds: TTL (seconds) for cached emission factor and
            fuel economy lookups in Redis.
        enable_biogenic_tracking: When True, biogenic CO2 from biofuel
            blends (E10, E85, B5, B20, B100, SAF) is tracked separately
            per GHG Protocol guidance.
        enable_uncertainty: When True, Monte Carlo uncertainty quantification
            is performed for all calculation results.
        enable_compliance: When True, regulatory compliance checks are
            performed against the configured framework.
        enable_fleet_management: When True, fleet-level vehicle registration,
            trip tracking, and aggregation features are enabled.
        decimal_precision: Number of decimal places to retain in emission
            calculations for intermediate and final results.
        default_vehicle_type: Default vehicle type used when no explicit
            type is specified in calculation inputs.
        default_fuel_type: Default fuel type used when no explicit type
            is specified in calculation inputs.
        default_distance_unit: Default unit for distance measurements.
            Valid: KM, MILES, NAUTICAL_MILES.
        default_fuel_economy_unit: Default unit for fuel economy values.
            Valid: L_PER_100KM, MPG_US, MPG_UK, KM_PER_L.
        confidence_level_90: The 90th percentile confidence level threshold
            for uncertainty intervals.
        confidence_level_95: The 95th percentile confidence level threshold
            for uncertainty intervals.
        confidence_level_99: The 99th percentile confidence level threshold
            for uncertainty intervals.
        default_regulatory_framework: Default regulatory framework for
            compliance checks and reporting.
        max_vehicles_per_fleet: Maximum number of vehicles that can be
            registered in a single fleet.
        max_trips_per_query: Maximum number of trip records returned in
            a single query response.
        calculation_timeout_seconds: Maximum wall-clock time allowed for
            a single calculation before timeout.
        enable_metrics: When True, Prometheus metrics are exported under
            the ``gl_mc_`` prefix.
        enable_tracing: When True, OpenTelemetry distributed tracing is
            enabled for calculation operations.
        enable_provenance: Compute and store SHA-256 provenance hashes for
            all vehicle registrations, trip records, emission factor
            selections, and calculation operations.
        genesis_hash: Anchor string used as the root of every provenance
            chain. Uniquely identifies the Mobile Combustion agent.
        pool_size: PostgreSQL connection pool size for the agent.
        rate_limit: Maximum inbound API requests per minute.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- GHG methodology defaults --------------------------------------------
    default_gwp_source: str = "AR6"
    default_calculation_method: str = "FUEL_BASED"

    # -- Uncertainty analysis ------------------------------------------------
    monte_carlo_iterations: int = 5_000

    # -- Batch processing ----------------------------------------------------
    batch_size: int = 100
    max_batch_size: int = 1_000

    # -- Cache ---------------------------------------------------------------
    cache_ttl_seconds: int = 3_600

    # -- Feature toggles -----------------------------------------------------
    enable_biogenic_tracking: bool = True
    enable_uncertainty: bool = True
    enable_compliance: bool = True
    enable_fleet_management: bool = True

    # -- Calculation precision -----------------------------------------------
    decimal_precision: int = 8

    # -- Vehicle defaults ----------------------------------------------------
    default_vehicle_type: str = "PASSENGER_CAR_GASOLINE"
    default_fuel_type: str = "GASOLINE"

    # -- Distance / fuel economy defaults ------------------------------------
    default_distance_unit: str = "KM"
    default_fuel_economy_unit: str = "L_PER_100KM"

    # -- Confidence levels ---------------------------------------------------
    confidence_level_90: float = 0.90
    confidence_level_95: float = 0.95
    confidence_level_99: float = 0.99

    # -- Compliance ----------------------------------------------------------
    default_regulatory_framework: str = "GHG_PROTOCOL"

    # -- Fleet management ----------------------------------------------------
    max_vehicles_per_fleet: int = 10_000
    max_trips_per_query: int = 5_000

    # -- Performance ---------------------------------------------------------
    calculation_timeout_seconds: int = 30
    enable_metrics: bool = True
    enable_tracing: bool = True

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-MRV-X-003-MOBILE-COMBUSTION-GENESIS"

    # -- Connection pool and rate limiting -----------------------------------
    pool_size: int = 10
    rate_limit: int = 1_000

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialisation.

        Performs range checks on all numeric fields, enumeration checks on
        string fields (GWP source, calculation method, log level, distance
        unit, fuel economy unit, regulatory framework), and normalisation
        of values (e.g. log_level to uppercase).

        Raises:
            ValueError: If any configuration value is outside its valid
                range or violates a constraint. The exception message
                lists all detected errors, not just the first one.
        """
        errors: list[str] = []

        # -- Logging ---------------------------------------------------------
        normalised_log = self.log_level.upper()
        if normalised_log not in _VALID_LOG_LEVELS:
            errors.append(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got '{self.log_level}'"
            )
        else:
            self.log_level = normalised_log

        # -- GWP source ------------------------------------------------------
        normalised_gwp = self.default_gwp_source.upper()
        if normalised_gwp not in _VALID_GWP_SOURCES:
            errors.append(
                f"default_gwp_source must be one of "
                f"{sorted(_VALID_GWP_SOURCES)}, "
                f"got '{self.default_gwp_source}'"
            )
        else:
            self.default_gwp_source = normalised_gwp

        # -- Calculation method ----------------------------------------------
        normalised_method = self.default_calculation_method.upper()
        if normalised_method not in _VALID_CALCULATION_METHODS:
            errors.append(
                f"default_calculation_method must be one of "
                f"{sorted(_VALID_CALCULATION_METHODS)}, "
                f"got '{self.default_calculation_method}'"
            )
        else:
            self.default_calculation_method = normalised_method

        # -- Distance unit ---------------------------------------------------
        normalised_dist = self.default_distance_unit.upper()
        if normalised_dist not in _VALID_DISTANCE_UNITS:
            errors.append(
                f"default_distance_unit must be one of "
                f"{sorted(_VALID_DISTANCE_UNITS)}, "
                f"got '{self.default_distance_unit}'"
            )
        else:
            self.default_distance_unit = normalised_dist

        # -- Fuel economy unit -----------------------------------------------
        normalised_fe = self.default_fuel_economy_unit.upper()
        if normalised_fe not in _VALID_FUEL_ECONOMY_UNITS:
            errors.append(
                f"default_fuel_economy_unit must be one of "
                f"{sorted(_VALID_FUEL_ECONOMY_UNITS)}, "
                f"got '{self.default_fuel_economy_unit}'"
            )
        else:
            self.default_fuel_economy_unit = normalised_fe

        # -- Regulatory framework --------------------------------------------
        normalised_fw = self.default_regulatory_framework.upper()
        if normalised_fw not in _VALID_REGULATORY_FRAMEWORKS:
            errors.append(
                f"default_regulatory_framework must be one of "
                f"{sorted(_VALID_REGULATORY_FRAMEWORKS)}, "
                f"got '{self.default_regulatory_framework}'"
            )
        else:
            self.default_regulatory_framework = normalised_fw

        # -- Decimal precision -----------------------------------------------
        if self.decimal_precision < 0:
            errors.append(
                f"decimal_precision must be >= 0, "
                f"got {self.decimal_precision}"
            )
        if self.decimal_precision > 20:
            errors.append(
                f"decimal_precision must be <= 20, "
                f"got {self.decimal_precision}"
            )

        # -- Batch sizes -----------------------------------------------------
        if self.batch_size <= 0:
            errors.append(
                f"batch_size must be > 0, got {self.batch_size}"
            )
        if self.batch_size > self.max_batch_size:
            errors.append(
                f"batch_size ({self.batch_size}) must be <= "
                f"max_batch_size ({self.max_batch_size})"
            )
        if self.max_batch_size <= 0:
            errors.append(
                f"max_batch_size must be > 0, got {self.max_batch_size}"
            )
        if self.max_batch_size > 1_000_000:
            errors.append(
                f"max_batch_size must be <= 1000000, "
                f"got {self.max_batch_size}"
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

        # -- Cache TTL -------------------------------------------------------
        if self.cache_ttl_seconds <= 0:
            errors.append(
                f"cache_ttl_seconds must be > 0, "
                f"got {self.cache_ttl_seconds}"
            )

        # -- Confidence levels -----------------------------------------------
        for field_name, value in [
            ("confidence_level_90", self.confidence_level_90),
            ("confidence_level_95", self.confidence_level_95),
            ("confidence_level_99", self.confidence_level_99),
        ]:
            if not (0.0 < value < 1.0):
                errors.append(
                    f"{field_name} must be in (0.0, 1.0), got {value}"
                )

        # -- Fleet management ------------------------------------------------
        if self.max_vehicles_per_fleet <= 0:
            errors.append(
                f"max_vehicles_per_fleet must be > 0, "
                f"got {self.max_vehicles_per_fleet}"
            )
        if self.max_vehicles_per_fleet > 1_000_000:
            errors.append(
                f"max_vehicles_per_fleet must be <= 1000000, "
                f"got {self.max_vehicles_per_fleet}"
            )
        if self.max_trips_per_query <= 0:
            errors.append(
                f"max_trips_per_query must be > 0, "
                f"got {self.max_trips_per_query}"
            )
        if self.max_trips_per_query > 100_000:
            errors.append(
                f"max_trips_per_query must be <= 100000, "
                f"got {self.max_trips_per_query}"
            )

        # -- Performance -----------------------------------------------------
        if self.calculation_timeout_seconds <= 0:
            errors.append(
                f"calculation_timeout_seconds must be > 0, "
                f"got {self.calculation_timeout_seconds}"
            )
        if self.calculation_timeout_seconds > 600:
            errors.append(
                f"calculation_timeout_seconds must be <= 600, "
                f"got {self.calculation_timeout_seconds}"
            )

        # -- Provenance ------------------------------------------------------
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        # -- Connection pool and rate limiting --------------------------------
        if self.pool_size <= 0:
            errors.append(
                f"pool_size must be > 0, got {self.pool_size}"
            )
        if self.rate_limit <= 0:
            errors.append(
                f"rate_limit must be > 0, got {self.rate_limit}"
            )

        if errors:
            raise ValueError(
                "MobileCombustionConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "MobileCombustionConfig validated successfully: "
            "gwp_source=%s, method=%s, decimal_precision=%d, "
            "batch_size=%d, max_batch_size=%d, "
            "monte_carlo_iterations=%d, biogenic_tracking=%s, "
            "uncertainty=%s, compliance=%s, fleet=%s, "
            "vehicle_type=%s, fuel_type=%s, distance_unit=%s, "
            "fuel_economy_unit=%s, framework=%s, metrics=%s",
            self.default_gwp_source,
            self.default_calculation_method,
            self.decimal_precision,
            self.batch_size,
            self.max_batch_size,
            self.monte_carlo_iterations,
            self.enable_biogenic_tracking,
            self.enable_uncertainty,
            self.enable_compliance,
            self.enable_fleet_management,
            self.default_vehicle_type,
            self.default_fuel_type,
            self.default_distance_unit,
            self.default_fuel_economy_unit,
            self.default_regulatory_framework,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> MobileCombustionConfig:
        """Build a MobileCombustionConfig from environment variables.

        Every field can be overridden via
        ``GL_MOBILE_COMBUSTION_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated MobileCombustionConfig instance, validated via
            ``__post_init__``.

        Example:
            >>> import os
            >>> os.environ["GL_MOBILE_COMBUSTION_DEFAULT_GWP_SOURCE"] = "AR5"
            >>> cfg = MobileCombustionConfig.from_env()
            >>> cfg.default_gwp_source
            'AR5'
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

        config = cls(
            # Connections
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
            # GHG methodology defaults
            default_gwp_source=_str(
                "DEFAULT_GWP_SOURCE",
                cls.default_gwp_source,
            ),
            default_calculation_method=_str(
                "DEFAULT_CALCULATION_METHOD",
                cls.default_calculation_method,
            ),
            # Uncertainty analysis
            monte_carlo_iterations=_int(
                "MONTE_CARLO_ITERATIONS",
                cls.monte_carlo_iterations,
            ),
            # Batch processing
            batch_size=_int("BATCH_SIZE", cls.batch_size),
            max_batch_size=_int("MAX_BATCH_SIZE", cls.max_batch_size),
            # Cache
            cache_ttl_seconds=_int(
                "CACHE_TTL_SECONDS",
                cls.cache_ttl_seconds,
            ),
            # Feature toggles
            enable_biogenic_tracking=_bool(
                "ENABLE_BIOGENIC_TRACKING",
                cls.enable_biogenic_tracking,
            ),
            enable_uncertainty=_bool(
                "ENABLE_UNCERTAINTY",
                cls.enable_uncertainty,
            ),
            enable_compliance=_bool(
                "ENABLE_COMPLIANCE",
                cls.enable_compliance,
            ),
            enable_fleet_management=_bool(
                "ENABLE_FLEET_MANAGEMENT",
                cls.enable_fleet_management,
            ),
            # Calculation precision
            decimal_precision=_int(
                "DECIMAL_PRECISION",
                cls.decimal_precision,
            ),
            # Vehicle defaults
            default_vehicle_type=_str(
                "DEFAULT_VEHICLE_TYPE",
                cls.default_vehicle_type,
            ),
            default_fuel_type=_str(
                "DEFAULT_FUEL_TYPE",
                cls.default_fuel_type,
            ),
            # Distance / fuel economy defaults
            default_distance_unit=_str(
                "DEFAULT_DISTANCE_UNIT",
                cls.default_distance_unit,
            ),
            default_fuel_economy_unit=_str(
                "DEFAULT_FUEL_ECONOMY_UNIT",
                cls.default_fuel_economy_unit,
            ),
            # Confidence levels
            confidence_level_90=_float(
                "CONFIDENCE_LEVEL_90",
                cls.confidence_level_90,
            ),
            confidence_level_95=_float(
                "CONFIDENCE_LEVEL_95",
                cls.confidence_level_95,
            ),
            confidence_level_99=_float(
                "CONFIDENCE_LEVEL_99",
                cls.confidence_level_99,
            ),
            # Compliance
            default_regulatory_framework=_str(
                "DEFAULT_REGULATORY_FRAMEWORK",
                cls.default_regulatory_framework,
            ),
            # Fleet management
            max_vehicles_per_fleet=_int(
                "MAX_VEHICLES_PER_FLEET",
                cls.max_vehicles_per_fleet,
            ),
            max_trips_per_query=_int(
                "MAX_TRIPS_PER_QUERY",
                cls.max_trips_per_query,
            ),
            # Performance
            calculation_timeout_seconds=_int(
                "CALCULATION_TIMEOUT_SECONDS",
                cls.calculation_timeout_seconds,
            ),
            enable_metrics=_bool("ENABLE_METRICS", cls.enable_metrics),
            enable_tracing=_bool("ENABLE_TRACING", cls.enable_tracing),
            # Provenance tracking
            enable_provenance=_bool(
                "ENABLE_PROVENANCE",
                cls.enable_provenance,
            ),
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
            # Connection pool and rate limiting
            pool_size=_int("POOL_SIZE", cls.pool_size),
            rate_limit=_int("RATE_LIMIT", cls.rate_limit),
        )

        logger.info(
            "MobileCombustionConfig loaded: "
            "gwp_source=%s, method=%s, "
            "decimal_precision=%d, batch_size=%d, "
            "max_batch_size=%d, monte_carlo=%d, "
            "biogenic=%s, uncertainty=%s, compliance=%s, "
            "fleet=%s, vehicle_type=%s, fuel_type=%s, "
            "distance_unit=%s, fuel_economy_unit=%s, "
            "framework=%s, max_vehicles=%d, max_trips=%d, "
            "timeout=%ds, pool=%d, rate_limit=%d/min, "
            "metrics=%s, tracing=%s, provenance=%s",
            config.default_gwp_source,
            config.default_calculation_method,
            config.decimal_precision,
            config.batch_size,
            config.max_batch_size,
            config.monte_carlo_iterations,
            config.enable_biogenic_tracking,
            config.enable_uncertainty,
            config.enable_compliance,
            config.enable_fleet_management,
            config.default_vehicle_type,
            config.default_fuel_type,
            config.default_distance_unit,
            config.default_fuel_economy_unit,
            config.default_regulatory_framework,
            config.max_vehicles_per_fleet,
            config.max_trips_per_query,
            config.calculation_timeout_seconds,
            config.pool_size,
            config.rate_limit,
            config.enable_metrics,
            config.enable_tracing,
            config.enable_provenance,
        )
        return config

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a plain Python dictionary.

        The returned dictionary is safe to pass to ``json.dumps``,
        ``yaml.dump``, or any structured logging framework. All values
        are JSON-serialisable primitives (str, int, float, bool).

        Sensitive connection strings (``database_url``, ``redis_url``) are
        redacted to prevent accidental credential leakage in logs,
        exception tracebacks, and monitoring dashboards.

        Returns:
            Dictionary representation of the configuration with sensitive
            fields redacted.

        Example:
            >>> cfg = MobileCombustionConfig()
            >>> d = cfg.to_dict()
            >>> d["default_calculation_method"]
            'FUEL_BASED'
            >>> d["database_url"]  # redacted
            '***'
        """
        return {
            # -- Connections (redacted) --------------------------------------
            "database_url": "***" if self.database_url else "",
            "redis_url": "***" if self.redis_url else "",
            # -- Logging -----------------------------------------------------
            "log_level": self.log_level,
            # -- GHG methodology defaults -----------------------------------
            "default_gwp_source": self.default_gwp_source,
            "default_calculation_method": self.default_calculation_method,
            # -- Uncertainty analysis ----------------------------------------
            "monte_carlo_iterations": self.monte_carlo_iterations,
            # -- Batch processing --------------------------------------------
            "batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size,
            # -- Cache -------------------------------------------------------
            "cache_ttl_seconds": self.cache_ttl_seconds,
            # -- Feature toggles ---------------------------------------------
            "enable_biogenic_tracking": self.enable_biogenic_tracking,
            "enable_uncertainty": self.enable_uncertainty,
            "enable_compliance": self.enable_compliance,
            "enable_fleet_management": self.enable_fleet_management,
            # -- Calculation precision ---------------------------------------
            "decimal_precision": self.decimal_precision,
            # -- Vehicle defaults --------------------------------------------
            "default_vehicle_type": self.default_vehicle_type,
            "default_fuel_type": self.default_fuel_type,
            # -- Distance / fuel economy defaults ----------------------------
            "default_distance_unit": self.default_distance_unit,
            "default_fuel_economy_unit": self.default_fuel_economy_unit,
            # -- Confidence levels -------------------------------------------
            "confidence_level_90": self.confidence_level_90,
            "confidence_level_95": self.confidence_level_95,
            "confidence_level_99": self.confidence_level_99,
            # -- Compliance --------------------------------------------------
            "default_regulatory_framework": self.default_regulatory_framework,
            # -- Fleet management --------------------------------------------
            "max_vehicles_per_fleet": self.max_vehicles_per_fleet,
            "max_trips_per_query": self.max_trips_per_query,
            # -- Performance -------------------------------------------------
            "calculation_timeout_seconds": self.calculation_timeout_seconds,
            "enable_metrics": self.enable_metrics,
            "enable_tracing": self.enable_tracing,
            # -- Provenance tracking -----------------------------------------
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # -- Connection pool and rate limiting ---------------------------
            "pool_size": self.pool_size,
            "rate_limit": self.rate_limit,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Sensitive fields (database_url, redis_url) are replaced with
        ``'***'`` so that repr output is safe to include in log messages
        and exception tracebacks.

        Returns:
            String representation of the configuration.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"MobileCombustionConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[MobileCombustionConfig] = None
_config_lock = threading.Lock()


def get_config() -> MobileCombustionConfig:
    """Return the singleton MobileCombustionConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_MOBILE_COMBUSTION_*`` environment variables via
    :meth:`MobileCombustionConfig.from_env`.

    Returns:
        MobileCombustionConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.default_calculation_method
        'FUEL_BASED'
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = MobileCombustionConfig.from_env()
    return _config_instance


def set_config(config: MobileCombustionConfig) -> None:
    """Replace the singleton MobileCombustionConfig.

    Primarily intended for testing and dependency injection scenarios
    where a custom configuration must be supplied without relying on
    environment variables.

    Args:
        config: New :class:`MobileCombustionConfig` to install as the
            singleton.

    Example:
        >>> cfg = MobileCombustionConfig(default_calculation_method="DISTANCE_BASED")
        >>> set_config(cfg)
        >>> assert get_config().default_calculation_method == "DISTANCE_BASED"
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "MobileCombustionConfig replaced programmatically: "
        "gwp_source=%s, method=%s, batch_size=%d, "
        "max_batch_size=%d, monte_carlo=%d, "
        "biogenic=%s, fleet=%s",
        config.default_gwp_source,
        config.default_calculation_method,
        config.batch_size,
        config.max_batch_size,
        config.monte_carlo_iterations,
        config.enable_biogenic_tracking,
        config.enable_fleet_management,
    )


def reset_config() -> None:
    """Reset the singleton MobileCombustionConfig to ``None``.

    The next call to :func:`get_config` will re-read environment variables
    and construct a fresh instance. Intended for test teardown to prevent
    state leakage between test cases.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads GL_MOBILE_COMBUSTION_* env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("MobileCombustionConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "MobileCombustionConfig",
    "get_config",
    "set_config",
    "reset_config",
]
