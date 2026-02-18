# -*- coding: utf-8 -*-
"""
Fugitive Emissions Agent Configuration - AGENT-MRV-005

Centralized configuration for the Fugitive Emissions Agent SDK covering:
- Database, cache, and connection defaults
- GHG Protocol methodology defaults (GWP source, calculation method)
- Decimal precision for emission calculations
- Capacity limits (components, surveys, batches)
- Monte Carlo uncertainty analysis parameters
- Feature toggles (LDAR tracking, component tracking, coal mine methane,
  wastewater, tank losses, pneumatic devices, compliance checking,
  uncertainty, provenance, metrics)
- API configuration (prefix, pagination)
- Background task and worker thread settings

All settings can be overridden via environment variables with the
``GL_FUGITIVE_EMISSIONS_`` prefix (e.g.
``GL_FUGITIVE_EMISSIONS_DEFAULT_GWP_SOURCE``,
``GL_FUGITIVE_EMISSIONS_MAX_BATCH_SIZE``).

Environment Variable Reference (GL_FUGITIVE_EMISSIONS_ prefix):
    GL_FUGITIVE_EMISSIONS_ENABLED                      - Enable/disable agent
    GL_FUGITIVE_EMISSIONS_DATABASE_URL                 - PostgreSQL connection URL
    GL_FUGITIVE_EMISSIONS_REDIS_URL                    - Redis connection URL
    GL_FUGITIVE_EMISSIONS_MAX_BATCH_SIZE               - Maximum records per batch
    GL_FUGITIVE_EMISSIONS_DEFAULT_GWP_SOURCE           - Default GWP source (AR4/AR5/AR6/AR6_20YR)
    GL_FUGITIVE_EMISSIONS_DEFAULT_CALCULATION_METHOD   - Default method
    GL_FUGITIVE_EMISSIONS_DEFAULT_EMISSION_FACTOR_SOURCE - Default EF source
    GL_FUGITIVE_EMISSIONS_DECIMAL_PRECISION            - Decimal places for calculations
    GL_FUGITIVE_EMISSIONS_MONTE_CARLO_ITERATIONS       - Monte Carlo simulation iterations
    GL_FUGITIVE_EMISSIONS_MONTE_CARLO_SEED             - Random seed for reproducibility
    GL_FUGITIVE_EMISSIONS_CONFIDENCE_LEVELS            - Comma-separated confidence levels
    GL_FUGITIVE_EMISSIONS_ENABLE_LDAR_TRACKING         - Enable LDAR survey tracking
    GL_FUGITIVE_EMISSIONS_ENABLE_COMPONENT_TRACKING    - Enable component-level tracking
    GL_FUGITIVE_EMISSIONS_ENABLE_COAL_MINE_METHANE     - Enable coal mine methane
    GL_FUGITIVE_EMISSIONS_ENABLE_WASTEWATER            - Enable wastewater emissions
    GL_FUGITIVE_EMISSIONS_ENABLE_TANK_LOSSES           - Enable tank storage losses
    GL_FUGITIVE_EMISSIONS_ENABLE_PNEUMATIC_DEVICES     - Enable pneumatic devices
    GL_FUGITIVE_EMISSIONS_ENABLE_COMPLIANCE_CHECKING   - Enable compliance checking
    GL_FUGITIVE_EMISSIONS_ENABLE_UNCERTAINTY           - Enable uncertainty quantification
    GL_FUGITIVE_EMISSIONS_ENABLE_PROVENANCE            - Enable SHA-256 provenance chain
    GL_FUGITIVE_EMISSIONS_ENABLE_METRICS               - Enable Prometheus metrics export
    GL_FUGITIVE_EMISSIONS_MAX_COMPONENTS               - Maximum component registrations
    GL_FUGITIVE_EMISSIONS_MAX_SURVEYS                  - Maximum LDAR survey records
    GL_FUGITIVE_EMISSIONS_LDAR_LEAK_THRESHOLD_PPM      - LDAR leak threshold in ppm
    GL_FUGITIVE_EMISSIONS_CACHE_TTL_SECONDS            - Cache time-to-live in seconds
    GL_FUGITIVE_EMISSIONS_API_PREFIX                   - REST API route prefix
    GL_FUGITIVE_EMISSIONS_API_MAX_PAGE_SIZE            - Maximum API page size
    GL_FUGITIVE_EMISSIONS_API_DEFAULT_PAGE_SIZE        - Default API page size
    GL_FUGITIVE_EMISSIONS_LOG_LEVEL                    - Logging level
    GL_FUGITIVE_EMISSIONS_WORKER_THREADS               - Worker thread pool size
    GL_FUGITIVE_EMISSIONS_ENABLE_BACKGROUND_TASKS      - Enable background task processing
    GL_FUGITIVE_EMISSIONS_HEALTH_CHECK_INTERVAL        - Health check interval seconds
    GL_FUGITIVE_EMISSIONS_GENESIS_HASH                 - Genesis anchor for provenance chain

Example:
    >>> from greenlang.fugitive_emissions.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_gwp_source, cfg.default_calculation_method)
    AR6 AVERAGE_EMISSION_FACTOR

    >>> # Override for testing
    >>> from greenlang.fugitive_emissions.config import set_config, reset_config
    >>> from greenlang.fugitive_emissions.config import FugitiveEmissionsConfig
    >>> set_config(FugitiveEmissionsConfig(default_calculation_method="DIRECT_MEASUREMENT"))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
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

_ENV_PREFIX = "GL_FUGITIVE_EMISSIONS_"

# ---------------------------------------------------------------------------
# Valid enumeration values for configuration validation
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

_VALID_GWP_SOURCES = frozenset({"AR4", "AR5", "AR6", "AR6_20YR"})

_VALID_CALCULATION_METHODS = frozenset({
    "AVERAGE_EMISSION_FACTOR",
    "SCREENING_RANGES",
    "CORRELATION_EQUATION",
    "ENGINEERING_ESTIMATE",
    "DIRECT_MEASUREMENT",
})

_VALID_EF_SOURCES = frozenset({
    "EPA",
    "IPCC",
    "DEFRA",
    "EU_ETS",
    "API",
    "CUSTOM",
})


# ---------------------------------------------------------------------------
# FugitiveEmissionsConfig
# ---------------------------------------------------------------------------


@dataclass
class FugitiveEmissionsConfig:
    """Complete configuration for the GreenLang Fugitive Emissions Agent SDK.

    Attributes are grouped by concern: feature flag, connections, methodology
    defaults, calculation precision, capacity limits, Monte Carlo parameters,
    feature toggles, API settings, performance tuning, and provenance.

    All attributes can be overridden via environment variables using the
    ``GL_FUGITIVE_EMISSIONS_`` prefix (e.g.
    ``GL_FUGITIVE_EMISSIONS_DEFAULT_CALCULATION_METHOD=DIRECT_MEASUREMENT``).

    Attributes:
        enabled: Master switch to enable/disable the fugitive emissions agent.
            When False, the agent will not process any requests.
        database_url: PostgreSQL connection URL for persistent storage of
            component data, emission factors, and calculation results.
        redis_url: Redis connection URL for caching emission factor lookups,
            source type metadata, and distributed locks.
        max_batch_size: Maximum number of fugitive emission records that
            can be processed in a single batch calculation request.
        default_gwp_source: Default IPCC Assessment Report edition for
            Global Warming Potential values. Valid: AR4, AR5, AR6, AR6_20YR.
        default_calculation_method: Default calculation methodology for
            fugitive emissions. AVERAGE_EMISSION_FACTOR uses published
            EPA component-level average factors. SCREENING_RANGES uses
            screening value ranges based on detected concentration.
            CORRELATION_EQUATION applies leak rate correlation equations.
            ENGINEERING_ESTIMATE uses engineering calculations for specific
            equipment. DIRECT_MEASUREMENT uses Hi-Flow sampler or bagging
            data.
        default_emission_factor_source: Default authority for emission
            factors. Valid: EPA, IPCC, DEFRA, EU_ETS, API, CUSTOM.
        decimal_precision: Number of decimal places to retain in emission
            calculations for intermediate and final results.
        monte_carlo_iterations: Number of Monte Carlo simulation iterations
            for uncertainty quantification. Higher values yield more
            precise confidence intervals at the cost of computation time.
        monte_carlo_seed: Random seed for Monte Carlo reproducibility.
            Set to 0 for non-deterministic runs.
        confidence_levels: Comma-separated confidence level percentages for
            uncertainty analysis output (e.g. "90,95,99").
        enable_ldar_tracking: When True, Leak Detection and Repair (LDAR)
            survey data is tracked including OGI, Method 21, AVO, and
            Hi-Flow survey types.
        enable_component_tracking: When True, individual equipment
            component-level tracking is enabled for valves, pumps,
            compressors, connectors, flanges, and other components.
        enable_coal_mine_methane: When True, coal mine methane emissions
            calculation is available for underground, surface, and
            post-mining activities.
        enable_wastewater: When True, wastewater treatment plant
            emissions calculation is available for industrial and
            municipal wastewater facilities.
        enable_tank_losses: When True, storage tank losses calculation
            is available for fixed-roof, floating-roof, and pressurized
            tank types.
        enable_pneumatic_devices: When True, pneumatic device emissions
            calculation is available for high-bleed, low-bleed, and
            intermittent-bleed pneumatic controllers.
        enable_compliance_checking: When True, calculation results are
            automatically checked against applicable regulatory frameworks.
        enable_uncertainty: When True, Monte Carlo or analytical uncertainty
            quantification is available for fugitive emission calculations.
        enable_provenance: Compute and store SHA-256 provenance hashes for
            all component registrations, emission factor selections,
            calculation steps, and batch operations.
        enable_metrics: When True, Prometheus metrics are exported under
            the ``gl_fe_`` prefix.
        max_components: Maximum number of equipment component registrations
            allowed in the system simultaneously.
        max_surveys: Maximum number of LDAR survey records that can be
            tracked per facility.
        ldar_leak_threshold_ppm: Concentration threshold in parts per
            million (ppm) above which a component reading is classified
            as a leak per EPA Method 21 / LDAR program requirements.
        cache_ttl_seconds: TTL (seconds) for cached emission factor and
            source type lookups in Redis.
        api_prefix: URL prefix for the REST API endpoints.
        api_max_page_size: Maximum allowed page size for paginated API
            responses.
        api_default_page_size: Default page size for paginated API
            responses when not specified by the client.
        log_level: Logging verbosity level. Accepts DEBUG, INFO, WARNING,
            ERROR, or CRITICAL.
        worker_threads: Number of worker threads in the thread pool for
            parallel batch processing operations.
        enable_background_tasks: When True, long-running operations such
            as large batch calculations are processed asynchronously via
            a background task queue.
        health_check_interval: Interval in seconds between periodic health
            checks of database and cache connections.
        genesis_hash: Anchor string used as the root of every provenance
            chain. Uniquely identifies the Fugitive Emissions agent.
    """

    # -- Feature flag --------------------------------------------------------
    enabled: bool = True

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""

    # -- Capacity limits -----------------------------------------------------
    max_batch_size: int = 500

    # -- GHG methodology defaults --------------------------------------------
    default_gwp_source: str = "AR6"
    default_calculation_method: str = "AVERAGE_EMISSION_FACTOR"
    default_emission_factor_source: str = "EPA"

    # -- Calculation precision -----------------------------------------------
    decimal_precision: int = 8

    # -- Monte Carlo uncertainty analysis ------------------------------------
    monte_carlo_iterations: int = 5_000
    monte_carlo_seed: int = 42
    confidence_levels: str = "90,95,99"

    # -- Feature toggles -----------------------------------------------------
    enable_ldar_tracking: bool = True
    enable_component_tracking: bool = True
    enable_coal_mine_methane: bool = True
    enable_wastewater: bool = True
    enable_tank_losses: bool = True
    enable_pneumatic_devices: bool = True
    enable_compliance_checking: bool = True
    enable_uncertainty: bool = True
    enable_provenance: bool = True
    enable_metrics: bool = True

    # -- Component and survey capacity limits --------------------------------
    max_components: int = 5_000
    max_surveys: int = 1_000
    ldar_leak_threshold_ppm: int = 10_000

    # -- Cache ---------------------------------------------------------------
    cache_ttl_seconds: int = 3600

    # -- API settings --------------------------------------------------------
    api_prefix: str = "/api/v1/fugitive-emissions"
    api_max_page_size: int = 100
    api_default_page_size: int = 20

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Performance tuning --------------------------------------------------
    worker_threads: int = 4
    enable_background_tasks: bool = True
    health_check_interval: int = 30

    # -- Provenance tracking -------------------------------------------------
    genesis_hash: str = "GL-MRV-X-005-FUGITIVE-EMISSIONS-GENESIS"

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialisation.

        Performs range checks on all numeric fields, enumeration checks on
        string fields (GWP source, method, EF source, log level), and
        normalisation of values (e.g. log_level to uppercase).

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

        # -- Emission factor source ------------------------------------------
        normalised_efs = self.default_emission_factor_source.upper()
        if normalised_efs not in _VALID_EF_SOURCES:
            errors.append(
                f"default_emission_factor_source must be one of "
                f"{sorted(_VALID_EF_SOURCES)}, "
                f"got '{self.default_emission_factor_source}'"
            )
        else:
            self.default_emission_factor_source = normalised_efs

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
            ("max_batch_size", self.max_batch_size, 100_000),
            ("max_components", self.max_components, 100_000),
            ("max_surveys", self.max_surveys, 50_000),
        ]:
            if value <= 0:
                errors.append(
                    f"{field_name} must be > 0, got {value}"
                )
            if value > upper:
                errors.append(
                    f"{field_name} must be <= {upper}, got {value}"
                )

        # -- LDAR leak threshold ---------------------------------------------
        if self.ldar_leak_threshold_ppm <= 0:
            errors.append(
                f"ldar_leak_threshold_ppm must be > 0, "
                f"got {self.ldar_leak_threshold_ppm}"
            )
        if self.ldar_leak_threshold_ppm > 100_000:
            errors.append(
                f"ldar_leak_threshold_ppm must be <= 100000, "
                f"got {self.ldar_leak_threshold_ppm}"
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

        # -- Cache TTL -------------------------------------------------------
        if self.cache_ttl_seconds <= 0:
            errors.append(
                f"cache_ttl_seconds must be > 0, "
                f"got {self.cache_ttl_seconds}"
            )

        # -- API settings ----------------------------------------------------
        if self.api_max_page_size <= 0:
            errors.append(
                f"api_max_page_size must be > 0, "
                f"got {self.api_max_page_size}"
            )
        if self.api_default_page_size <= 0:
            errors.append(
                f"api_default_page_size must be > 0, "
                f"got {self.api_default_page_size}"
            )
        if self.api_default_page_size > self.api_max_page_size:
            errors.append(
                f"api_default_page_size ({self.api_default_page_size}) "
                f"must be <= api_max_page_size ({self.api_max_page_size})"
            )

        # -- Worker threads --------------------------------------------------
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

        # -- Health check interval -------------------------------------------
        if self.health_check_interval <= 0:
            errors.append(
                f"health_check_interval must be > 0, "
                f"got {self.health_check_interval}"
            )

        # -- Provenance ------------------------------------------------------
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        if errors:
            raise ValueError(
                "FugitiveEmissionsConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "FugitiveEmissionsConfig validated successfully: "
            "enabled=%s, gwp_source=%s, method=%s, "
            "ef_source=%s, decimal_precision=%d, "
            "max_batch_size=%d, monte_carlo_iterations=%d, "
            "confidence_levels=%s, ldar=%s, "
            "component=%s, coal_mine=%s, "
            "wastewater=%s, tank_losses=%s, "
            "pneumatic=%s, compliance=%s, "
            "uncertainty=%s, provenance=%s, metrics=%s",
            self.enabled,
            self.default_gwp_source,
            self.default_calculation_method,
            self.default_emission_factor_source,
            self.decimal_precision,
            self.max_batch_size,
            self.monte_carlo_iterations,
            self.confidence_levels,
            self.enable_ldar_tracking,
            self.enable_component_tracking,
            self.enable_coal_mine_methane,
            self.enable_wastewater,
            self.enable_tank_losses,
            self.enable_pneumatic_devices,
            self.enable_compliance_checking,
            self.enable_uncertainty,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> FugitiveEmissionsConfig:
        """Build a FugitiveEmissionsConfig from environment variables.

        Every field can be overridden via
        ``GL_FUGITIVE_EMISSIONS_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated FugitiveEmissionsConfig instance, validated via
            ``__post_init__``.

        Example:
            >>> import os
            >>> os.environ["GL_FUGITIVE_EMISSIONS_DEFAULT_CALCULATION_METHOD"] = "DIRECT_MEASUREMENT"
            >>> cfg = FugitiveEmissionsConfig.from_env()
            >>> cfg.default_calculation_method
            'DIRECT_MEASUREMENT'
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

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val.strip()

        config = cls(
            # Feature flag
            enabled=_bool("ENABLED", cls.enabled),
            # Connections
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            # Capacity limits
            max_batch_size=_int("MAX_BATCH_SIZE", cls.max_batch_size),
            # GHG methodology defaults
            default_gwp_source=_str(
                "DEFAULT_GWP_SOURCE",
                cls.default_gwp_source,
            ),
            default_calculation_method=_str(
                "DEFAULT_CALCULATION_METHOD",
                cls.default_calculation_method,
            ),
            default_emission_factor_source=_str(
                "DEFAULT_EMISSION_FACTOR_SOURCE",
                cls.default_emission_factor_source,
            ),
            # Calculation precision
            decimal_precision=_int(
                "DECIMAL_PRECISION",
                cls.decimal_precision,
            ),
            # Monte Carlo uncertainty analysis
            monte_carlo_iterations=_int(
                "MONTE_CARLO_ITERATIONS",
                cls.monte_carlo_iterations,
            ),
            monte_carlo_seed=_int(
                "MONTE_CARLO_SEED",
                cls.monte_carlo_seed,
            ),
            confidence_levels=_str(
                "CONFIDENCE_LEVELS",
                cls.confidence_levels,
            ),
            # Feature toggles
            enable_ldar_tracking=_bool(
                "ENABLE_LDAR_TRACKING",
                cls.enable_ldar_tracking,
            ),
            enable_component_tracking=_bool(
                "ENABLE_COMPONENT_TRACKING",
                cls.enable_component_tracking,
            ),
            enable_coal_mine_methane=_bool(
                "ENABLE_COAL_MINE_METHANE",
                cls.enable_coal_mine_methane,
            ),
            enable_wastewater=_bool(
                "ENABLE_WASTEWATER",
                cls.enable_wastewater,
            ),
            enable_tank_losses=_bool(
                "ENABLE_TANK_LOSSES",
                cls.enable_tank_losses,
            ),
            enable_pneumatic_devices=_bool(
                "ENABLE_PNEUMATIC_DEVICES",
                cls.enable_pneumatic_devices,
            ),
            enable_compliance_checking=_bool(
                "ENABLE_COMPLIANCE_CHECKING",
                cls.enable_compliance_checking,
            ),
            enable_uncertainty=_bool(
                "ENABLE_UNCERTAINTY",
                cls.enable_uncertainty,
            ),
            enable_provenance=_bool(
                "ENABLE_PROVENANCE",
                cls.enable_provenance,
            ),
            enable_metrics=_bool(
                "ENABLE_METRICS",
                cls.enable_metrics,
            ),
            # Component and survey capacity limits
            max_components=_int(
                "MAX_COMPONENTS",
                cls.max_components,
            ),
            max_surveys=_int(
                "MAX_SURVEYS",
                cls.max_surveys,
            ),
            ldar_leak_threshold_ppm=_int(
                "LDAR_LEAK_THRESHOLD_PPM",
                cls.ldar_leak_threshold_ppm,
            ),
            # Cache
            cache_ttl_seconds=_int(
                "CACHE_TTL_SECONDS",
                cls.cache_ttl_seconds,
            ),
            # API settings
            api_prefix=_str("API_PREFIX", cls.api_prefix),
            api_max_page_size=_int(
                "API_MAX_PAGE_SIZE",
                cls.api_max_page_size,
            ),
            api_default_page_size=_int(
                "API_DEFAULT_PAGE_SIZE",
                cls.api_default_page_size,
            ),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
            # Performance tuning
            worker_threads=_int(
                "WORKER_THREADS",
                cls.worker_threads,
            ),
            enable_background_tasks=_bool(
                "ENABLE_BACKGROUND_TASKS",
                cls.enable_background_tasks,
            ),
            health_check_interval=_int(
                "HEALTH_CHECK_INTERVAL",
                cls.health_check_interval,
            ),
            # Provenance tracking
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
        )

        logger.info(
            "FugitiveEmissionsConfig loaded: "
            "enabled=%s, gwp_source=%s, method=%s, "
            "ef_source=%s, decimal_precision=%d, "
            "max_batch_size=%d, max_components=%d, "
            "max_surveys=%d, ldar_threshold_ppm=%d, "
            "monte_carlo_iterations=%d, confidence_levels=%s, "
            "ldar=%s, component=%s, coal_mine=%s, "
            "wastewater=%s, tank_losses=%s, pneumatic=%s, "
            "compliance=%s, uncertainty=%s, "
            "provenance=%s, metrics=%s, "
            "cache_ttl=%ds, worker_threads=%d, "
            "background_tasks=%s, health_check=%ds",
            config.enabled,
            config.default_gwp_source,
            config.default_calculation_method,
            config.default_emission_factor_source,
            config.decimal_precision,
            config.max_batch_size,
            config.max_components,
            config.max_surveys,
            config.ldar_leak_threshold_ppm,
            config.monte_carlo_iterations,
            config.confidence_levels,
            config.enable_ldar_tracking,
            config.enable_component_tracking,
            config.enable_coal_mine_methane,
            config.enable_wastewater,
            config.enable_tank_losses,
            config.enable_pneumatic_devices,
            config.enable_compliance_checking,
            config.enable_uncertainty,
            config.enable_provenance,
            config.enable_metrics,
            config.cache_ttl_seconds,
            config.worker_threads,
            config.enable_background_tasks,
            config.health_check_interval,
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
            >>> cfg = FugitiveEmissionsConfig()
            >>> d = cfg.to_dict()
            >>> d["default_calculation_method"]
            'AVERAGE_EMISSION_FACTOR'
            >>> d["database_url"]  # redacted
            '***'
        """
        return {
            # -- Feature flag -----------------------------------------------
            "enabled": self.enabled,
            # -- Connections (redacted) -------------------------------------
            "database_url": "***" if self.database_url else "",
            "redis_url": "***" if self.redis_url else "",
            # -- Capacity limits --------------------------------------------
            "max_batch_size": self.max_batch_size,
            # -- GHG methodology defaults -----------------------------------
            "default_gwp_source": self.default_gwp_source,
            "default_calculation_method": self.default_calculation_method,
            "default_emission_factor_source": self.default_emission_factor_source,
            # -- Calculation precision --------------------------------------
            "decimal_precision": self.decimal_precision,
            # -- Monte Carlo uncertainty analysis ---------------------------
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "monte_carlo_seed": self.monte_carlo_seed,
            "confidence_levels": self.confidence_levels,
            # -- Feature toggles --------------------------------------------
            "enable_ldar_tracking": self.enable_ldar_tracking,
            "enable_component_tracking": self.enable_component_tracking,
            "enable_coal_mine_methane": self.enable_coal_mine_methane,
            "enable_wastewater": self.enable_wastewater,
            "enable_tank_losses": self.enable_tank_losses,
            "enable_pneumatic_devices": self.enable_pneumatic_devices,
            "enable_compliance_checking": self.enable_compliance_checking,
            "enable_uncertainty": self.enable_uncertainty,
            "enable_provenance": self.enable_provenance,
            "enable_metrics": self.enable_metrics,
            # -- Component and survey capacity limits -----------------------
            "max_components": self.max_components,
            "max_surveys": self.max_surveys,
            "ldar_leak_threshold_ppm": self.ldar_leak_threshold_ppm,
            # -- Cache ------------------------------------------------------
            "cache_ttl_seconds": self.cache_ttl_seconds,
            # -- API settings -----------------------------------------------
            "api_prefix": self.api_prefix,
            "api_max_page_size": self.api_max_page_size,
            "api_default_page_size": self.api_default_page_size,
            # -- Logging ----------------------------------------------------
            "log_level": self.log_level,
            # -- Performance tuning -----------------------------------------
            "worker_threads": self.worker_threads,
            "enable_background_tasks": self.enable_background_tasks,
            "health_check_interval": self.health_check_interval,
            # -- Provenance tracking ----------------------------------------
            "genesis_hash": self.genesis_hash,
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
        return f"FugitiveEmissionsConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[FugitiveEmissionsConfig] = None
_config_lock = threading.Lock()


def get_config() -> FugitiveEmissionsConfig:
    """Return the singleton FugitiveEmissionsConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_FUGITIVE_EMISSIONS_*`` environment variables via
    :meth:`FugitiveEmissionsConfig.from_env`.

    Returns:
        FugitiveEmissionsConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.default_calculation_method
        'AVERAGE_EMISSION_FACTOR'
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = FugitiveEmissionsConfig.from_env()
    return _config_instance


def set_config(config: FugitiveEmissionsConfig) -> None:
    """Replace the singleton FugitiveEmissionsConfig.

    Primarily intended for testing and dependency injection scenarios
    where a custom configuration must be supplied without relying on
    environment variables.

    Args:
        config: New :class:`FugitiveEmissionsConfig` to install as the
            singleton.

    Example:
        >>> cfg = FugitiveEmissionsConfig(default_calculation_method="DIRECT_MEASUREMENT")
        >>> set_config(cfg)
        >>> assert get_config().default_calculation_method == "DIRECT_MEASUREMENT"
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "FugitiveEmissionsConfig replaced programmatically: "
        "enabled=%s, gwp_source=%s, method=%s, "
        "ef_source=%s, max_batch_size=%d, "
        "monte_carlo_iterations=%d",
        config.enabled,
        config.default_gwp_source,
        config.default_calculation_method,
        config.default_emission_factor_source,
        config.max_batch_size,
        config.monte_carlo_iterations,
    )


def reset_config() -> None:
    """Reset the singleton FugitiveEmissionsConfig to ``None``.

    The next call to :func:`get_config` will re-read environment variables
    and construct a fresh instance. Intended for test teardown to prevent
    state leakage between test cases.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads GL_FUGITIVE_EMISSIONS_* env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("FugitiveEmissionsConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "FugitiveEmissionsConfig",
    "get_config",
    "set_config",
    "reset_config",
]
