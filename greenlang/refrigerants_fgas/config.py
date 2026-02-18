# -*- coding: utf-8 -*-
"""
Refrigerants & F-Gas Agent Configuration - AGENT-MRV-002

Centralized configuration for the Refrigerants & F-Gas Agent SDK covering:
- Database, cache, and connection defaults
- GWP source and timeframe defaults (AR6, 100-year horizon)
- Calculation method defaults (equipment-based, mass-balance, screening)
- Capacity limits (refrigerants, equipment, calculations, blends, service events)
- Monte Carlo uncertainty analysis parameters
- Blend decomposition toggle
- Lifecycle tracking toggle
- Compliance checking and phase-down tracking
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Connection pool sizing, rate limiting, and cache TTL
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_REFRIGERANTS_FGAS_`` prefix (e.g. ``GL_REFRIGERANTS_FGAS_DEFAULT_GWP_SOURCE``,
``GL_REFRIGERANTS_FGAS_MAX_REFRIGERANTS``).

Environment Variable Reference (GL_REFRIGERANTS_FGAS_ prefix):
    GL_REFRIGERANTS_FGAS_DATABASE_URL                - PostgreSQL connection URL
    GL_REFRIGERANTS_FGAS_REDIS_URL                   - Redis connection URL
    GL_REFRIGERANTS_FGAS_LOG_LEVEL                   - Logging level (DEBUG/INFO/WARNING/ERROR)
    GL_REFRIGERANTS_FGAS_DEFAULT_GWP_SOURCE          - Default GWP source (AR4, AR5, AR6)
    GL_REFRIGERANTS_FGAS_DEFAULT_GWP_TIMEFRAME       - Default GWP timeframe (100yr, 20yr)
    GL_REFRIGERANTS_FGAS_DEFAULT_CALCULATION_METHOD   - Default calculation method
    GL_REFRIGERANTS_FGAS_MAX_REFRIGERANTS            - Maximum refrigerant definitions
    GL_REFRIGERANTS_FGAS_MAX_EQUIPMENT               - Maximum equipment profiles
    GL_REFRIGERANTS_FGAS_MAX_CALCULATIONS            - Maximum stored calculation results
    GL_REFRIGERANTS_FGAS_MAX_BLENDS                  - Maximum blend definitions
    GL_REFRIGERANTS_FGAS_MAX_SERVICE_EVENTS          - Maximum service event records
    GL_REFRIGERANTS_FGAS_DEFAULT_UNCERTAINTY_ITERATIONS - Monte Carlo iterations
    GL_REFRIGERANTS_FGAS_CONFIDENCE_LEVELS           - Comma-separated confidence levels
    GL_REFRIGERANTS_FGAS_PHASE_DOWN_BASELINE_YEAR    - HFC phase-down baseline year
    GL_REFRIGERANTS_FGAS_ENABLE_BLEND_DECOMPOSITION  - Enable blend decomposition
    GL_REFRIGERANTS_FGAS_ENABLE_LIFECYCLE_TRACKING   - Enable lifecycle tracking
    GL_REFRIGERANTS_FGAS_ENABLE_COMPLIANCE_CHECKING  - Enable compliance checking
    GL_REFRIGERANTS_FGAS_ENABLE_PROVENANCE           - Enable SHA-256 provenance chain
    GL_REFRIGERANTS_FGAS_GENESIS_HASH                - Genesis anchor string for provenance
    GL_REFRIGERANTS_FGAS_ENABLE_METRICS              - Enable Prometheus metrics export
    GL_REFRIGERANTS_FGAS_POOL_SIZE                   - Database connection pool size
    GL_REFRIGERANTS_FGAS_CACHE_TTL                   - Cache time-to-live in seconds
    GL_REFRIGERANTS_FGAS_RATE_LIMIT                  - Max API requests per minute

Example:
    >>> from greenlang.refrigerants_fgas.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_gwp_source, cfg.default_gwp_timeframe)
    AR6 100yr

    >>> # Override for testing
    >>> from greenlang.refrigerants_fgas.config import set_config, reset_config
    >>> from greenlang.refrigerants_fgas.config import RefrigerantsFGasConfig
    >>> set_config(RefrigerantsFGasConfig(default_gwp_source="AR5"))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-002 Refrigerants & F-Gas (GL-MRV-SCOPE1-002)
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

_ENV_PREFIX = "GL_REFRIGERANTS_FGAS_"

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
# Valid GWP timeframes
# ---------------------------------------------------------------------------

_VALID_GWP_TIMEFRAMES = frozenset({"100yr", "20yr"})

# ---------------------------------------------------------------------------
# Valid calculation methods
# ---------------------------------------------------------------------------

_VALID_CALCULATION_METHODS = frozenset({
    "equipment_based",
    "mass_balance",
    "screening",
    "direct_measurement",
    "top_down",
})


# ---------------------------------------------------------------------------
# RefrigerantsFGasConfig
# ---------------------------------------------------------------------------


@dataclass
class RefrigerantsFGasConfig:
    """Complete configuration for the GreenLang Refrigerants & F-Gas Agent SDK.

    Attributes are grouped by concern: connections, logging, GWP methodology
    defaults, calculation method, capacity limits, uncertainty parameters,
    feature toggles, provenance tracking, metrics export, and performance
    tuning.

    All attributes can be overridden via environment variables using the
    ``GL_REFRIGERANTS_FGAS_`` prefix (e.g.
    ``GL_REFRIGERANTS_FGAS_DEFAULT_GWP_SOURCE=AR5``).

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage of
            refrigerant properties, equipment profiles, service events,
            and calculation results.
        redis_url: Redis connection URL for caching GWP lookups,
            refrigerant property queries, and distributed locks.
        log_level: Logging verbosity level. Accepts DEBUG, INFO, WARNING,
            ERROR, or CRITICAL.
        default_gwp_source: Default IPCC Assessment Report edition for
            Global Warming Potential values. Valid: AR4, AR5, AR6.
        default_gwp_timeframe: Default GWP integration time horizon.
            Valid: 100yr, 20yr.
        default_calculation_method: Default calculation methodology.
            Valid: equipment_based, mass_balance, screening,
            direct_measurement, top_down.
        max_refrigerants: Maximum refrigerant type definitions that can
            be registered in the system simultaneously.
        max_equipment: Maximum equipment profiles that can be registered.
        max_calculations: Maximum stored calculation result records before
            automatic archival or purging.
        max_blends: Maximum blend definitions (multi-component refrigerant
            mixtures) that can be registered.
        max_service_events: Maximum service event records (installation,
            recharge, repair, recovery, etc.) retained in storage.
        default_uncertainty_iterations: Number of Monte Carlo simulation
            iterations for uncertainty quantification.
        confidence_levels: Comma-separated confidence level percentages for
            uncertainty analysis output (e.g. "90,95,99").
        phase_down_baseline_year: Baseline year for HFC phase-down schedule
            calculations per Kigali Amendment and EU F-Gas regulation.
        enable_blend_decomposition: When True, blended refrigerants are
            decomposed into constituent gases with weight-fraction GWP
            for accurate per-gas emission reporting.
        enable_lifecycle_tracking: When True, equipment lifecycle stages
            (installation, operating, end-of-life) are tracked for
            lifecycle-adjusted leak rate estimation.
        enable_compliance_checking: When True, regulatory compliance is
            checked against applicable frameworks (EU F-Gas, Kigali, etc.)
            and quota tracking is enabled.
        enable_provenance: Compute and store SHA-256 provenance hashes for
            all refrigerant registrations, equipment events, calculation
            steps, and pipeline operations.
        genesis_hash: Anchor string used as the root of every provenance
            chain. Uniquely identifies the Refrigerants & F-Gas agent.
        enable_metrics: When True, Prometheus metrics are exported under
            the ``gl_rf_`` prefix.
        pool_size: PostgreSQL connection pool size for the agent.
        cache_ttl: TTL (seconds) for cached refrigerant property and
            GWP lookups in Redis.
        rate_limit: Maximum inbound API requests per minute.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- GWP methodology defaults --------------------------------------------
    default_gwp_source: str = "AR6"
    default_gwp_timeframe: str = "100yr"

    # -- Calculation method --------------------------------------------------
    default_calculation_method: str = "equipment_based"

    # -- Capacity limits -----------------------------------------------------
    max_refrigerants: int = 50_000
    max_equipment: int = 100_000
    max_calculations: int = 1_000_000
    max_blends: int = 5_000
    max_service_events: int = 500_000

    # -- Monte Carlo uncertainty analysis ------------------------------------
    default_uncertainty_iterations: int = 5_000
    confidence_levels: str = "90,95,99"

    # -- Phase-down tracking -------------------------------------------------
    phase_down_baseline_year: int = 2015

    # -- Feature toggles -----------------------------------------------------
    enable_blend_decomposition: bool = True
    enable_lifecycle_tracking: bool = True
    enable_compliance_checking: bool = True

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-MRV-X-002-REFRIGERANTS-FGAS-GENESIS"

    # -- Metrics export ------------------------------------------------------
    enable_metrics: bool = True

    # -- Performance tuning --------------------------------------------------
    pool_size: int = 5
    cache_ttl: int = 300
    rate_limit: int = 1000

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialisation.

        Performs range checks on all numeric fields, enumeration checks on
        string fields (GWP source, timeframe, calculation method, log level),
        and normalisation of values (e.g. log_level to uppercase).

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

        # -- GWP timeframe ---------------------------------------------------
        normalised_tf = self.default_gwp_timeframe.lower()
        if normalised_tf not in _VALID_GWP_TIMEFRAMES:
            errors.append(
                f"default_gwp_timeframe must be one of "
                f"{sorted(_VALID_GWP_TIMEFRAMES)}, "
                f"got '{self.default_gwp_timeframe}'"
            )
        else:
            self.default_gwp_timeframe = normalised_tf

        # -- Calculation method ----------------------------------------------
        normalised_method = self.default_calculation_method.lower()
        if normalised_method not in _VALID_CALCULATION_METHODS:
            errors.append(
                f"default_calculation_method must be one of "
                f"{sorted(_VALID_CALCULATION_METHODS)}, "
                f"got '{self.default_calculation_method}'"
            )
        else:
            self.default_calculation_method = normalised_method

        # -- Capacity limits -------------------------------------------------
        for field_name, value, upper in [
            ("max_refrigerants", self.max_refrigerants, 10_000_000),
            ("max_equipment", self.max_equipment, 10_000_000),
            ("max_calculations", self.max_calculations, 100_000_000),
            ("max_blends", self.max_blends, 1_000_000),
            ("max_service_events", self.max_service_events, 50_000_000),
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
        if self.default_uncertainty_iterations <= 0:
            errors.append(
                f"default_uncertainty_iterations must be > 0, "
                f"got {self.default_uncertainty_iterations}"
            )
        if self.default_uncertainty_iterations > 1_000_000:
            errors.append(
                f"default_uncertainty_iterations must be <= 1000000, "
                f"got {self.default_uncertainty_iterations}"
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

        # -- Phase-down baseline year ----------------------------------------
        if self.phase_down_baseline_year < 1990:
            errors.append(
                f"phase_down_baseline_year must be >= 1990, "
                f"got {self.phase_down_baseline_year}"
            )
        if self.phase_down_baseline_year > 2100:
            errors.append(
                f"phase_down_baseline_year must be <= 2100, "
                f"got {self.phase_down_baseline_year}"
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
                "RefrigerantsFGasConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "RefrigerantsFGasConfig validated successfully: "
            "gwp_source=%s, gwp_timeframe=%s, method=%s, "
            "max_refrigerants=%d, max_equipment=%d, "
            "uncertainty_iterations=%d, confidence_levels=%s, "
            "blend_decomposition=%s, lifecycle_tracking=%s, "
            "compliance_checking=%s, provenance=%s, metrics=%s",
            self.default_gwp_source,
            self.default_gwp_timeframe,
            self.default_calculation_method,
            self.max_refrigerants,
            self.max_equipment,
            self.default_uncertainty_iterations,
            self.confidence_levels,
            self.enable_blend_decomposition,
            self.enable_lifecycle_tracking,
            self.enable_compliance_checking,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> RefrigerantsFGasConfig:
        """Build a RefrigerantsFGasConfig from environment variables.

        Every field can be overridden via
        ``GL_REFRIGERANTS_FGAS_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated RefrigerantsFGasConfig instance, validated via
            ``__post_init__``.

        Example:
            >>> import os
            >>> os.environ["GL_REFRIGERANTS_FGAS_DEFAULT_GWP_SOURCE"] = "AR5"
            >>> cfg = RefrigerantsFGasConfig.from_env()
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
            # GWP methodology defaults
            default_gwp_source=_str(
                "DEFAULT_GWP_SOURCE",
                cls.default_gwp_source,
            ),
            default_gwp_timeframe=_str(
                "DEFAULT_GWP_TIMEFRAME",
                cls.default_gwp_timeframe,
            ),
            # Calculation method
            default_calculation_method=_str(
                "DEFAULT_CALCULATION_METHOD",
                cls.default_calculation_method,
            ),
            # Capacity limits
            max_refrigerants=_int(
                "MAX_REFRIGERANTS",
                cls.max_refrigerants,
            ),
            max_equipment=_int("MAX_EQUIPMENT", cls.max_equipment),
            max_calculations=_int(
                "MAX_CALCULATIONS",
                cls.max_calculations,
            ),
            max_blends=_int("MAX_BLENDS", cls.max_blends),
            max_service_events=_int(
                "MAX_SERVICE_EVENTS",
                cls.max_service_events,
            ),
            # Monte Carlo uncertainty analysis
            default_uncertainty_iterations=_int(
                "DEFAULT_UNCERTAINTY_ITERATIONS",
                cls.default_uncertainty_iterations,
            ),
            confidence_levels=_str(
                "CONFIDENCE_LEVELS",
                cls.confidence_levels,
            ),
            # Phase-down tracking
            phase_down_baseline_year=_int(
                "PHASE_DOWN_BASELINE_YEAR",
                cls.phase_down_baseline_year,
            ),
            # Feature toggles
            enable_blend_decomposition=_bool(
                "ENABLE_BLEND_DECOMPOSITION",
                cls.enable_blend_decomposition,
            ),
            enable_lifecycle_tracking=_bool(
                "ENABLE_LIFECYCLE_TRACKING",
                cls.enable_lifecycle_tracking,
            ),
            enable_compliance_checking=_bool(
                "ENABLE_COMPLIANCE_CHECKING",
                cls.enable_compliance_checking,
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
            "RefrigerantsFGasConfig loaded: "
            "gwp_source=%s, gwp_timeframe=%s, method=%s, "
            "max_refrigerants=%d, max_equipment=%d, "
            "max_calculations=%d, max_blends=%d, "
            "max_service_events=%d, "
            "uncertainty_iterations=%d, confidence_levels=%s, "
            "phase_down_baseline_year=%d, "
            "blend_decomposition=%s, lifecycle_tracking=%s, "
            "compliance_checking=%s, provenance=%s, "
            "pool=%d, cache_ttl=%ds, rate_limit=%d/min, "
            "metrics=%s",
            config.default_gwp_source,
            config.default_gwp_timeframe,
            config.default_calculation_method,
            config.max_refrigerants,
            config.max_equipment,
            config.max_calculations,
            config.max_blends,
            config.max_service_events,
            config.default_uncertainty_iterations,
            config.confidence_levels,
            config.phase_down_baseline_year,
            config.enable_blend_decomposition,
            config.enable_lifecycle_tracking,
            config.enable_compliance_checking,
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
            >>> cfg = RefrigerantsFGasConfig()
            >>> d = cfg.to_dict()
            >>> d["default_gwp_source"]
            'AR6'
            >>> d["database_url"]  # redacted
            '***'
        """
        return {
            # -- Connections (redacted) --------------------------------------
            "database_url": "***" if self.database_url else "",
            "redis_url": "***" if self.redis_url else "",
            # -- Logging -----------------------------------------------------
            "log_level": self.log_level,
            # -- GWP methodology defaults -----------------------------------
            "default_gwp_source": self.default_gwp_source,
            "default_gwp_timeframe": self.default_gwp_timeframe,
            # -- Calculation method ------------------------------------------
            "default_calculation_method": self.default_calculation_method,
            # -- Capacity limits ---------------------------------------------
            "max_refrigerants": self.max_refrigerants,
            "max_equipment": self.max_equipment,
            "max_calculations": self.max_calculations,
            "max_blends": self.max_blends,
            "max_service_events": self.max_service_events,
            # -- Monte Carlo uncertainty analysis ----------------------------
            "default_uncertainty_iterations": self.default_uncertainty_iterations,
            "confidence_levels": self.confidence_levels,
            # -- Phase-down tracking -----------------------------------------
            "phase_down_baseline_year": self.phase_down_baseline_year,
            # -- Feature toggles ---------------------------------------------
            "enable_blend_decomposition": self.enable_blend_decomposition,
            "enable_lifecycle_tracking": self.enable_lifecycle_tracking,
            "enable_compliance_checking": self.enable_compliance_checking,
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
        return f"RefrigerantsFGasConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[RefrigerantsFGasConfig] = None
_config_lock = threading.Lock()


def get_config() -> RefrigerantsFGasConfig:
    """Return the singleton RefrigerantsFGasConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_REFRIGERANTS_FGAS_*`` environment variables via
    :meth:`RefrigerantsFGasConfig.from_env`.

    Returns:
        RefrigerantsFGasConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.default_gwp_source
        'AR6'
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = RefrigerantsFGasConfig.from_env()
    return _config_instance


def set_config(config: RefrigerantsFGasConfig) -> None:
    """Replace the singleton RefrigerantsFGasConfig.

    Primarily intended for testing and dependency injection scenarios
    where a custom configuration must be supplied without relying on
    environment variables.

    Args:
        config: New :class:`RefrigerantsFGasConfig` to install as the
            singleton.

    Example:
        >>> cfg = RefrigerantsFGasConfig(default_gwp_source="AR5")
        >>> set_config(cfg)
        >>> assert get_config().default_gwp_source == "AR5"
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "RefrigerantsFGasConfig replaced programmatically: "
        "gwp_source=%s, gwp_timeframe=%s, method=%s, "
        "max_refrigerants=%d, uncertainty_iterations=%d, "
        "blend_decomposition=%s, lifecycle_tracking=%s, "
        "compliance_checking=%s",
        config.default_gwp_source,
        config.default_gwp_timeframe,
        config.default_calculation_method,
        config.max_refrigerants,
        config.default_uncertainty_iterations,
        config.enable_blend_decomposition,
        config.enable_lifecycle_tracking,
        config.enable_compliance_checking,
    )


def reset_config() -> None:
    """Reset the singleton RefrigerantsFGasConfig to ``None``.

    The next call to :func:`get_config` will re-read environment variables
    and construct a fresh instance. Intended for test teardown to prevent
    state leakage between test cases.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads GL_REFRIGERANTS_FGAS_* env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("RefrigerantsFGasConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "RefrigerantsFGasConfig",
    "get_config",
    "set_config",
    "reset_config",
]
