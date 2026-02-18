# -*- coding: utf-8 -*-
"""
Stationary Combustion Agent Configuration - AGENT-MRV-001

Centralized configuration for the Stationary Combustion Agent SDK covering:
- Database, cache, and connection defaults
- GHG Protocol methodology defaults (GWP source, tier level, oxidation factor)
- Decimal precision for emission calculations
- Capacity limits (fuel types, emission factors, equipment, calculations)
- Monte Carlo uncertainty analysis parameters
- Biogenic CO2 tracking toggle
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Connection pool sizing, rate limiting, and cache TTL
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_STATIONARY_COMBUSTION_`` prefix (e.g. ``GL_STATIONARY_COMBUSTION_DEFAULT_TIER``,
``GL_STATIONARY_COMBUSTION_MAX_BATCH_SIZE``).

Environment Variable Reference (GL_STATIONARY_COMBUSTION_ prefix):
    GL_STATIONARY_COMBUSTION_DATABASE_URL           - PostgreSQL connection URL
    GL_STATIONARY_COMBUSTION_REDIS_URL              - Redis connection URL
    GL_STATIONARY_COMBUSTION_LOG_LEVEL              - Logging level (DEBUG/INFO/WARNING/ERROR)
    GL_STATIONARY_COMBUSTION_DEFAULT_GWP_SOURCE     - Default GWP source (AR4, AR5, AR6)
    GL_STATIONARY_COMBUSTION_DEFAULT_TIER           - Default calculation tier (1, 2, 3)
    GL_STATIONARY_COMBUSTION_DEFAULT_OXIDATION_FACTOR - Default oxidation factor (0.0-1.0)
    GL_STATIONARY_COMBUSTION_DECIMAL_PRECISION      - Decimal places for calculations
    GL_STATIONARY_COMBUSTION_MAX_BATCH_SIZE         - Maximum records per batch
    GL_STATIONARY_COMBUSTION_MAX_FUEL_TYPES         - Maximum fuel type definitions
    GL_STATIONARY_COMBUSTION_MAX_EMISSION_FACTORS   - Maximum emission factor records
    GL_STATIONARY_COMBUSTION_MAX_EQUIPMENT_PROFILES - Maximum equipment profiles
    GL_STATIONARY_COMBUSTION_MAX_CALCULATIONS       - Maximum stored calculation results
    GL_STATIONARY_COMBUSTION_MONTE_CARLO_ITERATIONS - Monte Carlo simulation iterations
    GL_STATIONARY_COMBUSTION_CONFIDENCE_LEVELS      - Comma-separated confidence levels
    GL_STATIONARY_COMBUSTION_ENABLE_BIOGENIC_TRACKING - Enable biogenic CO2 tracking
    GL_STATIONARY_COMBUSTION_ENABLE_PROVENANCE      - Enable SHA-256 provenance chain
    GL_STATIONARY_COMBUSTION_GENESIS_HASH           - Genesis anchor string for provenance
    GL_STATIONARY_COMBUSTION_ENABLE_METRICS         - Enable Prometheus metrics export
    GL_STATIONARY_COMBUSTION_POOL_SIZE              - Database connection pool size
    GL_STATIONARY_COMBUSTION_CACHE_TTL              - Cache time-to-live in seconds
    GL_STATIONARY_COMBUSTION_RATE_LIMIT             - Max API requests per minute

Example:
    >>> from greenlang.stationary_combustion.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_gwp_source, cfg.default_tier)
    AR6 1

    >>> # Override for testing
    >>> from greenlang.stationary_combustion.config import set_config, reset_config
    >>> from greenlang.stationary_combustion.config import StationaryCombustionConfig
    >>> set_config(StationaryCombustionConfig(default_tier=3, default_gwp_source="AR5"))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-001 Stationary Combustion (GL-MRV-SCOPE1-001)
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

_ENV_PREFIX = "GL_STATIONARY_COMBUSTION_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Valid GWP sources (IPCC Assessment Report editions)
# ---------------------------------------------------------------------------

_VALID_GWP_SOURCES = frozenset({"AR4", "AR5", "AR6"})

# ---------------------------------------------------------------------------
# Valid calculation tiers (GHG Protocol / IPCC)
# ---------------------------------------------------------------------------

_VALID_TIERS = frozenset({1, 2, 3})


# ---------------------------------------------------------------------------
# StationaryCombustionConfig
# ---------------------------------------------------------------------------


@dataclass
class StationaryCombustionConfig:
    """Complete configuration for the GreenLang Stationary Combustion Agent SDK.

    Attributes are grouped by concern: connections, logging, GHG methodology
    defaults, calculation precision, capacity limits, Monte Carlo parameters,
    biogenic tracking, provenance tracking, metrics export, and performance
    tuning.

    All attributes can be overridden via environment variables using the
    ``GL_STATIONARY_COMBUSTION_`` prefix (e.g.
    ``GL_STATIONARY_COMBUSTION_DEFAULT_TIER=3``).

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage of
            emission factors, fuel properties, equipment profiles, and
            calculation results.
        redis_url: Redis connection URL for caching emission factor lookups,
            heating value conversions, and distributed locks.
        log_level: Logging verbosity level. Accepts DEBUG, INFO, WARNING,
            ERROR, or CRITICAL.
        default_gwp_source: Default IPCC Assessment Report edition for
            Global Warming Potential values. Valid: AR4, AR5, AR6.
        default_tier: Default GHG Protocol / IPCC calculation tier.
            Tier 1 uses default factors; Tier 2 uses country/fuel-specific
            factors; Tier 3 uses facility-specific measurements. Valid: 1, 2, 3.
        default_oxidation_factor: Default fraction of carbon oxidized during
            combustion. 1.0 means complete oxidation. Valid range: 0.0-1.0.
        decimal_precision: Number of decimal places to retain in emission
            calculations for intermediate and final results.
        max_batch_size: Maximum number of combustion input records that
            can be processed in a single batch calculation request.
        max_fuel_types: Maximum fuel type definitions that can be registered
            in the system simultaneously.
        max_emission_factors: Maximum emission factor records retained in
            persistent storage.
        max_equipment_profiles: Maximum equipment profiles that can be
            registered for equipment-level Tier 2/3 calculations.
        max_calculations: Maximum stored calculation result records before
            automatic archival or purging.
        monte_carlo_iterations: Number of Monte Carlo simulation iterations
            for uncertainty quantification. Higher values yield more
            precise confidence intervals at the cost of computation time.
        confidence_levels: Comma-separated confidence level percentages for
            uncertainty analysis output (e.g. "90,95,99").
        enable_biogenic_tracking: When True, biogenic CO2 from biomass
            fuels is tracked separately per GHG Protocol guidance and
            excluded from Scope 1 totals.
        enable_provenance: Compute and store SHA-256 provenance hashes for
            all fuel registrations, emission factor selections, calculation
            steps, and batch operations.
        genesis_hash: Anchor string used as the root of every provenance
            chain. Uniquely identifies the Stationary Combustion agent.
        enable_metrics: When True, Prometheus metrics are exported under
            the ``gl_sc_`` prefix.
        pool_size: PostgreSQL connection pool size for the agent.
        cache_ttl: TTL (seconds) for cached emission factor and heating
            value lookups in Redis.
        rate_limit: Maximum inbound API requests per minute.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- GHG methodology defaults --------------------------------------------
    default_gwp_source: str = "AR6"
    default_tier: int = 1
    default_oxidation_factor: float = 1.0

    # -- Calculation precision -----------------------------------------------
    decimal_precision: int = 8

    # -- Capacity limits -----------------------------------------------------
    max_batch_size: int = 10_000
    max_fuel_types: int = 1_000
    max_emission_factors: int = 10_000
    max_equipment_profiles: int = 5_000
    max_calculations: int = 100_000

    # -- Monte Carlo uncertainty analysis ------------------------------------
    monte_carlo_iterations: int = 5_000
    confidence_levels: str = "90,95,99"

    # -- Biogenic tracking ---------------------------------------------------
    enable_biogenic_tracking: bool = True

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-MRV-X-001-STATIONARY-COMBUSTION-GENESIS"

    # -- Metrics export ------------------------------------------------------
    enable_metrics: bool = True

    # -- Performance tuning --------------------------------------------------
    pool_size: int = 10
    cache_ttl: int = 3600
    rate_limit: int = 1000

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialisation.

        Performs range checks on all numeric fields, enumeration checks on
        string fields (GWP source, log level), and normalisation of values
        (e.g. log_level to uppercase, gwp_source to uppercase).

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

        # -- Default tier ----------------------------------------------------
        if self.default_tier not in _VALID_TIERS:
            errors.append(
                f"default_tier must be one of {sorted(_VALID_TIERS)}, "
                f"got {self.default_tier}"
            )

        # -- Oxidation factor ------------------------------------------------
        if not (0.0 <= self.default_oxidation_factor <= 1.0):
            errors.append(
                f"default_oxidation_factor must be in [0.0, 1.0], "
                f"got {self.default_oxidation_factor}"
            )

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

        # -- Capacity limits -------------------------------------------------
        for field_name, value, upper in [
            ("max_batch_size", self.max_batch_size, 1_000_000),
            ("max_fuel_types", self.max_fuel_types, 100_000),
            ("max_emission_factors", self.max_emission_factors, 1_000_000),
            ("max_equipment_profiles", self.max_equipment_profiles, 500_000),
            ("max_calculations", self.max_calculations, 10_000_000),
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

        # -- Provenance ------------------------------------------------------
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        # -- Performance -----------------------------------------------------
        if self.pool_size <= 0:
            errors.append(
                f"pool_size must be > 0, got {self.pool_size}"
            )
        if self.cache_ttl <= 0:
            errors.append(
                f"cache_ttl must be > 0, got {self.cache_ttl}"
            )
        if self.rate_limit <= 0:
            errors.append(
                f"rate_limit must be > 0, got {self.rate_limit}"
            )

        if errors:
            raise ValueError(
                "StationaryCombustionConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "StationaryCombustionConfig validated successfully: "
            "gwp_source=%s, tier=%d, oxidation_factor=%.2f, "
            "decimal_precision=%d, max_batch_size=%d, "
            "monte_carlo_iterations=%d, confidence_levels=%s, "
            "biogenic_tracking=%s, provenance=%s, metrics=%s",
            self.default_gwp_source,
            self.default_tier,
            self.default_oxidation_factor,
            self.decimal_precision,
            self.max_batch_size,
            self.monte_carlo_iterations,
            self.confidence_levels,
            self.enable_biogenic_tracking,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> StationaryCombustionConfig:
        """Build a StationaryCombustionConfig from environment variables.

        Every field can be overridden via
        ``GL_STATIONARY_COMBUSTION_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated StationaryCombustionConfig instance, validated via
            ``__post_init__``.

        Example:
            >>> import os
            >>> os.environ["GL_STATIONARY_COMBUSTION_DEFAULT_TIER"] = "3"
            >>> cfg = StationaryCombustionConfig.from_env()
            >>> cfg.default_tier
            3
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
            default_tier=_int("DEFAULT_TIER", cls.default_tier),
            default_oxidation_factor=_float(
                "DEFAULT_OXIDATION_FACTOR",
                cls.default_oxidation_factor,
            ),
            # Calculation precision
            decimal_precision=_int(
                "DECIMAL_PRECISION",
                cls.decimal_precision,
            ),
            # Capacity limits
            max_batch_size=_int("MAX_BATCH_SIZE", cls.max_batch_size),
            max_fuel_types=_int("MAX_FUEL_TYPES", cls.max_fuel_types),
            max_emission_factors=_int(
                "MAX_EMISSION_FACTORS",
                cls.max_emission_factors,
            ),
            max_equipment_profiles=_int(
                "MAX_EQUIPMENT_PROFILES",
                cls.max_equipment_profiles,
            ),
            max_calculations=_int(
                "MAX_CALCULATIONS",
                cls.max_calculations,
            ),
            # Monte Carlo uncertainty analysis
            monte_carlo_iterations=_int(
                "MONTE_CARLO_ITERATIONS",
                cls.monte_carlo_iterations,
            ),
            confidence_levels=_str(
                "CONFIDENCE_LEVELS",
                cls.confidence_levels,
            ),
            # Biogenic tracking
            enable_biogenic_tracking=_bool(
                "ENABLE_BIOGENIC_TRACKING",
                cls.enable_biogenic_tracking,
            ),
            # Provenance tracking
            enable_provenance=_bool(
                "ENABLE_PROVENANCE",
                cls.enable_provenance,
            ),
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
            # Metrics export
            enable_metrics=_bool("ENABLE_METRICS", cls.enable_metrics),
            # Performance tuning
            pool_size=_int("POOL_SIZE", cls.pool_size),
            cache_ttl=_int("CACHE_TTL", cls.cache_ttl),
            rate_limit=_int("RATE_LIMIT", cls.rate_limit),
        )

        logger.info(
            "StationaryCombustionConfig loaded: "
            "gwp_source=%s, tier=%d, oxidation_factor=%.2f, "
            "decimal_precision=%d, max_batch_size=%d, "
            "max_fuel_types=%d, max_emission_factors=%d, "
            "max_equipment_profiles=%d, max_calculations=%d, "
            "monte_carlo_iterations=%d, confidence_levels=%s, "
            "biogenic_tracking=%s, provenance=%s, "
            "pool=%d, cache_ttl=%ds, rate_limit=%d/min, "
            "metrics=%s",
            config.default_gwp_source,
            config.default_tier,
            config.default_oxidation_factor,
            config.decimal_precision,
            config.max_batch_size,
            config.max_fuel_types,
            config.max_emission_factors,
            config.max_equipment_profiles,
            config.max_calculations,
            config.monte_carlo_iterations,
            config.confidence_levels,
            config.enable_biogenic_tracking,
            config.enable_provenance,
            config.pool_size,
            config.cache_ttl,
            config.rate_limit,
            config.enable_metrics,
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
            >>> cfg = StationaryCombustionConfig()
            >>> d = cfg.to_dict()
            >>> d["default_tier"]
            1
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
            "default_tier": self.default_tier,
            "default_oxidation_factor": self.default_oxidation_factor,
            # -- Calculation precision ---------------------------------------
            "decimal_precision": self.decimal_precision,
            # -- Capacity limits ---------------------------------------------
            "max_batch_size": self.max_batch_size,
            "max_fuel_types": self.max_fuel_types,
            "max_emission_factors": self.max_emission_factors,
            "max_equipment_profiles": self.max_equipment_profiles,
            "max_calculations": self.max_calculations,
            # -- Monte Carlo uncertainty analysis ----------------------------
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "confidence_levels": self.confidence_levels,
            # -- Biogenic tracking -------------------------------------------
            "enable_biogenic_tracking": self.enable_biogenic_tracking,
            # -- Provenance tracking -----------------------------------------
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # -- Metrics export ----------------------------------------------
            "enable_metrics": self.enable_metrics,
            # -- Performance tuning ------------------------------------------
            "pool_size": self.pool_size,
            "cache_ttl": self.cache_ttl,
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
        return f"StationaryCombustionConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[StationaryCombustionConfig] = None
_config_lock = threading.Lock()


def get_config() -> StationaryCombustionConfig:
    """Return the singleton StationaryCombustionConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_STATIONARY_COMBUSTION_*`` environment variables via
    :meth:`StationaryCombustionConfig.from_env`.

    Returns:
        StationaryCombustionConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.default_tier
        1
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = StationaryCombustionConfig.from_env()
    return _config_instance


def set_config(config: StationaryCombustionConfig) -> None:
    """Replace the singleton StationaryCombustionConfig.

    Primarily intended for testing and dependency injection scenarios
    where a custom configuration must be supplied without relying on
    environment variables.

    Args:
        config: New :class:`StationaryCombustionConfig` to install as the
            singleton.

    Example:
        >>> cfg = StationaryCombustionConfig(default_tier=3)
        >>> set_config(cfg)
        >>> assert get_config().default_tier == 3
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "StationaryCombustionConfig replaced programmatically: "
        "gwp_source=%s, tier=%d, oxidation_factor=%.2f, "
        "max_batch_size=%d, monte_carlo_iterations=%d, "
        "biogenic_tracking=%s",
        config.default_gwp_source,
        config.default_tier,
        config.default_oxidation_factor,
        config.max_batch_size,
        config.monte_carlo_iterations,
        config.enable_biogenic_tracking,
    )


def reset_config() -> None:
    """Reset the singleton StationaryCombustionConfig to ``None``.

    The next call to :func:`get_config` will re-read environment variables
    and construct a fresh instance. Intended for test teardown to prevent
    state leakage between test cases.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads GL_STATIONARY_COMBUSTION_* env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("StationaryCombustionConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "StationaryCombustionConfig",
    "get_config",
    "set_config",
    "reset_config",
]
