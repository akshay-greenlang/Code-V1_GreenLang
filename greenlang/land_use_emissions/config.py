# -*- coding: utf-8 -*-
"""
Land Use Emissions Agent Configuration - AGENT-MRV-006

Centralized configuration for the Land Use Emissions Agent SDK covering:
- Database, cache, and connection defaults
- IPCC methodology defaults (GWP source, calculation tier, method)
- Carbon stock calculation parameters (carbon fraction, transition years)
- Soil organic carbon assessment defaults (depth, method)
- Decimal precision for emission calculations
- Capacity limits (parcels, transitions, batches)
- Monte Carlo uncertainty analysis parameters
- Peatland emission settings
- Fire emission settings
- Feature toggles (SOC assessment, peatland, fire, N2O, compliance
  checking, uncertainty, provenance, metrics)
- API configuration (prefix, pagination)
- Background task and worker thread settings

All settings can be overridden via environment variables with the
``GL_LAND_USE_`` prefix (e.g.
``GL_LAND_USE_DEFAULT_GWP_SOURCE``,
``GL_LAND_USE_MAX_BATCH_SIZE``).

Environment Variable Reference (GL_LAND_USE_ prefix):
    GL_LAND_USE_ENABLED                       - Enable/disable agent
    GL_LAND_USE_DATABASE_URL                  - PostgreSQL connection URL
    GL_LAND_USE_REDIS_URL                     - Redis connection URL
    GL_LAND_USE_MAX_BATCH_SIZE                - Maximum records per batch
    GL_LAND_USE_DEFAULT_GWP_SOURCE            - Default GWP source (AR4/AR5/AR6)
    GL_LAND_USE_DEFAULT_TIER                  - Default IPCC calculation tier (1/2/3)
    GL_LAND_USE_DEFAULT_METHOD                - Default method (STOCK_DIFFERENCE/GAIN_LOSS)
    GL_LAND_USE_DEFAULT_EMISSION_FACTOR_SOURCE - Default EF source
    GL_LAND_USE_DECIMAL_PRECISION             - Decimal places for calculations
    GL_LAND_USE_DEFAULT_TRANSITION_YEARS      - Default transition period in years
    GL_LAND_USE_CARBON_FRACTION               - Default carbon fraction of dry matter
    GL_LAND_USE_SOC_DEPTH_CM                  - Default SOC assessment depth in cm
    GL_LAND_USE_DEFAULT_SOC_METHOD            - Default SOC method (tier1/tier2/tier3)
    GL_LAND_USE_MONTE_CARLO_ITERATIONS        - Monte Carlo simulation iterations
    GL_LAND_USE_MONTE_CARLO_SEED              - Random seed for reproducibility
    GL_LAND_USE_CONFIDENCE_LEVELS             - Comma-separated confidence levels
    GL_LAND_USE_ENABLE_SOC_ASSESSMENT         - Enable SOC assessment
    GL_LAND_USE_ENABLE_PEATLAND               - Enable peatland emission tracking
    GL_LAND_USE_ENABLE_FIRE_EMISSIONS         - Enable fire/disturbance emissions
    GL_LAND_USE_ENABLE_N2O_SOIL               - Enable soil N2O emissions
    GL_LAND_USE_ENABLE_COMPLIANCE_CHECKING    - Enable compliance checking
    GL_LAND_USE_ENABLE_UNCERTAINTY            - Enable uncertainty quantification
    GL_LAND_USE_ENABLE_PROVENANCE             - Enable SHA-256 provenance chain
    GL_LAND_USE_ENABLE_METRICS                - Enable Prometheus metrics export
    GL_LAND_USE_MAX_PARCELS                   - Maximum land parcel registrations
    GL_LAND_USE_MAX_TRANSITIONS               - Maximum transition records
    GL_LAND_USE_PEATLAND_DEFAULT_DOC_FACTOR   - Default dissolved organic carbon factor
    GL_LAND_USE_FIRE_COMBUSTION_EFFICIENCY    - Default fire combustion efficiency
    GL_LAND_USE_CACHE_TTL_SECONDS             - Cache time-to-live in seconds
    GL_LAND_USE_API_PREFIX                    - REST API route prefix
    GL_LAND_USE_API_MAX_PAGE_SIZE             - Maximum API page size
    GL_LAND_USE_API_DEFAULT_PAGE_SIZE         - Default API page size
    GL_LAND_USE_LOG_LEVEL                     - Logging level
    GL_LAND_USE_WORKER_THREADS                - Worker thread pool size
    GL_LAND_USE_ENABLE_BACKGROUND_TASKS       - Enable background task processing
    GL_LAND_USE_HEALTH_CHECK_INTERVAL         - Health check interval seconds
    GL_LAND_USE_GENESIS_HASH                  - Genesis anchor for provenance chain
    GL_LAND_USE_ENABLE_AUTH                    - Enable authentication middleware
    GL_LAND_USE_ENABLE_TRACING                - Enable OpenTelemetry tracing

Example:
    >>> from greenlang.land_use_emissions.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_gwp_source, cfg.default_method)
    AR6 STOCK_DIFFERENCE

    >>> # Override for testing
    >>> from greenlang.land_use_emissions.config import set_config, reset_config
    >>> from greenlang.land_use_emissions.config import LandUseConfig
    >>> set_config(LandUseConfig(default_tier=2))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Land Use Emissions (GL-MRV-SCOPE1-006)
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

_ENV_PREFIX = "GL_LAND_USE_"

# ---------------------------------------------------------------------------
# Valid enumeration values for configuration validation
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

_VALID_GWP_SOURCES = frozenset({"AR4", "AR5", "AR6"})

_VALID_METHODS = frozenset({
    "STOCK_DIFFERENCE",
    "GAIN_LOSS",
})

_VALID_TIERS = frozenset({1, 2, 3})

_VALID_EF_SOURCES = frozenset({
    "IPCC_2006",
    "IPCC_2019",
    "IPCC_WETLANDS_2013",
    "NATIONAL_INVENTORY",
    "LITERATURE",
    "CUSTOM",
})

_VALID_SOC_METHODS = frozenset({
    "tier1",
    "tier2",
    "tier3",
})


# ---------------------------------------------------------------------------
# LandUseConfig
# ---------------------------------------------------------------------------


@dataclass
class LandUseConfig:
    """Complete configuration for the GreenLang Land Use Emissions Agent SDK.

    Attributes are grouped by concern: feature flag, connections, methodology
    defaults, carbon stock parameters, SOC assessment, calculation precision,
    capacity limits, Monte Carlo parameters, peatland settings, fire settings,
    feature toggles, API settings, performance tuning, and provenance.

    All attributes can be overridden via environment variables using the
    ``GL_LAND_USE_`` prefix (e.g.
    ``GL_LAND_USE_DEFAULT_METHOD=GAIN_LOSS``).

    Attributes:
        enabled: Master switch to enable/disable the land use emissions agent.
            When False, the agent will not process any requests.
        database_url: PostgreSQL connection URL for persistent storage of
            parcel data, carbon stocks, and calculation results.
        redis_url: Redis connection URL for caching emission factor lookups,
            carbon stock defaults, and distributed locks.
        max_batch_size: Maximum number of land use emission records that
            can be processed in a single batch calculation request.
        default_gwp_source: Default IPCC Assessment Report edition for
            Global Warming Potential values. Valid: AR4, AR5, AR6.
        default_tier: Default IPCC calculation tier (1, 2, or 3). Tier 1
            uses global defaults, Tier 2 uses country-specific data, and
            Tier 3 uses spatially explicit models.
        default_method: Default calculation methodology for carbon stock
            changes. STOCK_DIFFERENCE compares carbon stocks at two points
            in time. GAIN_LOSS tracks annual carbon gains and losses.
        default_emission_factor_source: Default authority for emission
            factors. Valid: IPCC_2006, IPCC_2019, IPCC_WETLANDS_2013,
            NATIONAL_INVENTORY, LITERATURE, CUSTOM.
        decimal_precision: Number of decimal places to retain in emission
            calculations for intermediate and final results.
        default_transition_years: Default number of years over which
            carbon stock changes from land-use conversion are distributed.
            IPCC default is 20 years for above-ground biomass.
        carbon_fraction: Default carbon fraction of dry matter (CF).
            IPCC default is 0.47 for above-ground biomass.
        soc_depth_cm: Default soil depth in centimeters for SOC assessment.
            IPCC standard reference depth is 30 cm.
        default_soc_method: Default SOC assessment method (tier1, tier2,
            tier3). Tier 1 uses IPCC reference stocks with land use,
            management, and input factors.
        monte_carlo_iterations: Number of Monte Carlo simulation iterations
            for uncertainty quantification. Higher values yield more
            precise confidence intervals at the cost of computation time.
        monte_carlo_seed: Random seed for Monte Carlo reproducibility.
            Set to 0 for non-deterministic runs.
        confidence_levels: Comma-separated confidence level percentages for
            uncertainty analysis output (e.g. "90,95,99").
        enable_soc_assessment: When True, soil organic carbon assessment
            is available using IPCC Tier 1 reference stocks and factors.
        enable_peatland: When True, peatland emission estimation is
            available for natural, drained, rewetted, and extracted
            peatlands following IPCC Wetlands Supplement 2013.
        enable_fire_emissions: When True, fire and disturbance emission
            calculations are available including biomass burning CO2, CH4,
            N2O, and CO emissions.
        enable_n2o_soil: When True, direct soil N2O emissions from
            land management are calculated using IPCC EF1 default.
        enable_compliance_checking: When True, calculation results are
            automatically checked against applicable regulatory frameworks.
        enable_uncertainty: When True, Monte Carlo or analytical uncertainty
            quantification is available for land use emission calculations.
        enable_provenance: Compute and store SHA-256 provenance hashes for
            all parcel registrations, carbon stock snapshots, transitions,
            calculation steps, and batch operations.
        enable_metrics: When True, Prometheus metrics are exported under
            the ``gl_lu_`` prefix.
        max_parcels: Maximum number of land parcel registrations allowed
            in the system simultaneously.
        max_transitions: Maximum number of land-use transition records
            that can be tracked per tenant.
        peatland_default_doc_factor: Default dissolved organic carbon
            export factor for peatlands (fraction of carbon loss as DOC).
        fire_combustion_efficiency: Default combustion efficiency for
            fire emission calculations (fraction of fuel consumed).
        cache_ttl_seconds: TTL (seconds) for cached emission factor and
            carbon stock lookups in Redis.
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
            chain. Uniquely identifies the Land Use Emissions agent.
        enable_auth: When True, authentication middleware is enabled for
            API endpoints.
        enable_tracing: When True, OpenTelemetry tracing is enabled for
            distributed trace propagation.
    """

    # -- Feature flag --------------------------------------------------------
    enabled: bool = True

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""

    # -- Capacity limits -----------------------------------------------------
    max_batch_size: int = 10_000

    # -- IPCC methodology defaults -------------------------------------------
    default_gwp_source: str = "AR6"
    default_tier: int = 1
    default_method: str = "STOCK_DIFFERENCE"
    default_emission_factor_source: str = "IPCC_2006"

    # -- Calculation precision -----------------------------------------------
    decimal_precision: int = 8

    # -- Carbon stock calculation defaults -----------------------------------
    default_transition_years: int = 20
    carbon_fraction: float = 0.47

    # -- SOC assessment defaults ---------------------------------------------
    soc_depth_cm: int = 30
    default_soc_method: str = "tier1"

    # -- Monte Carlo uncertainty analysis ------------------------------------
    monte_carlo_iterations: int = 5_000
    monte_carlo_seed: int = 42
    confidence_levels: str = "90,95,99"

    # -- Feature toggles -----------------------------------------------------
    enable_soc_assessment: bool = True
    enable_peatland: bool = True
    enable_fire_emissions: bool = True
    enable_n2o_soil: bool = True
    enable_compliance_checking: bool = True
    enable_uncertainty: bool = True
    enable_provenance: bool = True
    enable_metrics: bool = True

    # -- Parcel and transition capacity limits --------------------------------
    max_parcels: int = 50_000
    max_transitions: int = 100_000

    # -- Peatland settings ---------------------------------------------------
    peatland_default_doc_factor: float = 0.12
    fire_combustion_efficiency: float = 0.45

    # -- Cache ---------------------------------------------------------------
    cache_ttl_seconds: int = 3600

    # -- API settings --------------------------------------------------------
    api_prefix: str = "/api/v1/land-use-emissions"
    api_max_page_size: int = 100
    api_default_page_size: int = 20

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Performance tuning --------------------------------------------------
    worker_threads: int = 4
    enable_background_tasks: bool = True
    health_check_interval: int = 30

    # -- Provenance tracking -------------------------------------------------
    genesis_hash: str = "GL-MRV-X-006-LAND-USE-EMISSIONS-GENESIS"

    # -- Auth and tracing ----------------------------------------------------
    enable_auth: bool = True
    enable_tracing: bool = True

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialisation.

        Performs range checks on all numeric fields, enumeration checks on
        string fields (GWP source, method, tier, EF source, SOC method,
        log level), and normalisation of values (e.g. log_level to
        uppercase).

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

        # -- Calculation tier ------------------------------------------------
        if self.default_tier not in _VALID_TIERS:
            errors.append(
                f"default_tier must be one of {sorted(_VALID_TIERS)}, "
                f"got {self.default_tier}"
            )

        # -- Calculation method ----------------------------------------------
        normalised_method = self.default_method.upper()
        if normalised_method not in _VALID_METHODS:
            errors.append(
                f"default_method must be one of "
                f"{sorted(_VALID_METHODS)}, "
                f"got '{self.default_method}'"
            )
        else:
            self.default_method = normalised_method

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

        # -- SOC method ------------------------------------------------------
        normalised_soc = self.default_soc_method.lower()
        if normalised_soc not in _VALID_SOC_METHODS:
            errors.append(
                f"default_soc_method must be one of "
                f"{sorted(_VALID_SOC_METHODS)}, "
                f"got '{self.default_soc_method}'"
            )
        else:
            self.default_soc_method = normalised_soc

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

        # -- Transition years ------------------------------------------------
        if self.default_transition_years <= 0:
            errors.append(
                f"default_transition_years must be > 0, "
                f"got {self.default_transition_years}"
            )
        if self.default_transition_years > 100:
            errors.append(
                f"default_transition_years must be <= 100, "
                f"got {self.default_transition_years}"
            )

        # -- Carbon fraction -------------------------------------------------
        if self.carbon_fraction <= 0.0:
            errors.append(
                f"carbon_fraction must be > 0.0, "
                f"got {self.carbon_fraction}"
            )
        if self.carbon_fraction > 1.0:
            errors.append(
                f"carbon_fraction must be <= 1.0, "
                f"got {self.carbon_fraction}"
            )

        # -- SOC depth -------------------------------------------------------
        if self.soc_depth_cm <= 0:
            errors.append(
                f"soc_depth_cm must be > 0, "
                f"got {self.soc_depth_cm}"
            )
        if self.soc_depth_cm > 300:
            errors.append(
                f"soc_depth_cm must be <= 300, "
                f"got {self.soc_depth_cm}"
            )

        # -- Capacity limits -------------------------------------------------
        for field_name, value, upper in [
            ("max_batch_size", self.max_batch_size, 100_000),
            ("max_parcels", self.max_parcels, 500_000),
            ("max_transitions", self.max_transitions, 1_000_000),
        ]:
            if value <= 0:
                errors.append(
                    f"{field_name} must be > 0, got {value}"
                )
            if value > upper:
                errors.append(
                    f"{field_name} must be <= {upper}, got {value}"
                )

        # -- Peatland DOC factor ---------------------------------------------
        if self.peatland_default_doc_factor < 0.0:
            errors.append(
                f"peatland_default_doc_factor must be >= 0.0, "
                f"got {self.peatland_default_doc_factor}"
            )
        if self.peatland_default_doc_factor > 1.0:
            errors.append(
                f"peatland_default_doc_factor must be <= 1.0, "
                f"got {self.peatland_default_doc_factor}"
            )

        # -- Fire combustion efficiency --------------------------------------
        if self.fire_combustion_efficiency < 0.0:
            errors.append(
                f"fire_combustion_efficiency must be >= 0.0, "
                f"got {self.fire_combustion_efficiency}"
            )
        if self.fire_combustion_efficiency > 1.0:
            errors.append(
                f"fire_combustion_efficiency must be <= 1.0, "
                f"got {self.fire_combustion_efficiency}"
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
                "LandUseConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "LandUseConfig validated successfully: "
            "enabled=%s, gwp_source=%s, tier=%d, method=%s, "
            "ef_source=%s, decimal_precision=%d, "
            "transition_years=%d, carbon_fraction=%.2f, "
            "soc_depth_cm=%d, soc_method=%s, "
            "max_batch_size=%d, monte_carlo_iterations=%d, "
            "confidence_levels=%s, soc=%s, peatland=%s, "
            "fire=%s, n2o_soil=%s, compliance=%s, "
            "uncertainty=%s, provenance=%s, metrics=%s",
            self.enabled,
            self.default_gwp_source,
            self.default_tier,
            self.default_method,
            self.default_emission_factor_source,
            self.decimal_precision,
            self.default_transition_years,
            self.carbon_fraction,
            self.soc_depth_cm,
            self.default_soc_method,
            self.max_batch_size,
            self.monte_carlo_iterations,
            self.confidence_levels,
            self.enable_soc_assessment,
            self.enable_peatland,
            self.enable_fire_emissions,
            self.enable_n2o_soil,
            self.enable_compliance_checking,
            self.enable_uncertainty,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> LandUseConfig:
        """Build a LandUseConfig from environment variables.

        Every field can be overridden via
        ``GL_LAND_USE_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated LandUseConfig instance, validated via
            ``__post_init__``.

        Example:
            >>> import os
            >>> os.environ["GL_LAND_USE_DEFAULT_METHOD"] = "GAIN_LOSS"
            >>> cfg = LandUseConfig.from_env()
            >>> cfg.default_method
            'GAIN_LOSS'
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
            # Feature flag
            enabled=_bool("ENABLED", cls.enabled),
            # Connections
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            # Capacity limits
            max_batch_size=_int("MAX_BATCH_SIZE", cls.max_batch_size),
            # IPCC methodology defaults
            default_gwp_source=_str(
                "DEFAULT_GWP_SOURCE",
                cls.default_gwp_source,
            ),
            default_tier=_int(
                "DEFAULT_TIER",
                cls.default_tier,
            ),
            default_method=_str(
                "DEFAULT_METHOD",
                cls.default_method,
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
            # Carbon stock defaults
            default_transition_years=_int(
                "DEFAULT_TRANSITION_YEARS",
                cls.default_transition_years,
            ),
            carbon_fraction=_float(
                "CARBON_FRACTION",
                cls.carbon_fraction,
            ),
            # SOC assessment defaults
            soc_depth_cm=_int(
                "SOC_DEPTH_CM",
                cls.soc_depth_cm,
            ),
            default_soc_method=_str(
                "DEFAULT_SOC_METHOD",
                cls.default_soc_method,
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
            enable_soc_assessment=_bool(
                "ENABLE_SOC_ASSESSMENT",
                cls.enable_soc_assessment,
            ),
            enable_peatland=_bool(
                "ENABLE_PEATLAND",
                cls.enable_peatland,
            ),
            enable_fire_emissions=_bool(
                "ENABLE_FIRE_EMISSIONS",
                cls.enable_fire_emissions,
            ),
            enable_n2o_soil=_bool(
                "ENABLE_N2O_SOIL",
                cls.enable_n2o_soil,
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
            # Parcel and transition capacity limits
            max_parcels=_int(
                "MAX_PARCELS",
                cls.max_parcels,
            ),
            max_transitions=_int(
                "MAX_TRANSITIONS",
                cls.max_transitions,
            ),
            # Peatland and fire settings
            peatland_default_doc_factor=_float(
                "PEATLAND_DEFAULT_DOC_FACTOR",
                cls.peatland_default_doc_factor,
            ),
            fire_combustion_efficiency=_float(
                "FIRE_COMBUSTION_EFFICIENCY",
                cls.fire_combustion_efficiency,
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
            # Auth and tracing
            enable_auth=_bool("ENABLE_AUTH", cls.enable_auth),
            enable_tracing=_bool("ENABLE_TRACING", cls.enable_tracing),
        )

        logger.info(
            "LandUseConfig loaded: "
            "enabled=%s, gwp_source=%s, tier=%d, method=%s, "
            "ef_source=%s, decimal_precision=%d, "
            "transition_years=%d, carbon_fraction=%.2f, "
            "soc_depth=%dcm, soc_method=%s, "
            "max_batch_size=%d, max_parcels=%d, "
            "max_transitions=%d, "
            "monte_carlo_iterations=%d, confidence_levels=%s, "
            "soc=%s, peatland=%s, fire=%s, n2o_soil=%s, "
            "compliance=%s, uncertainty=%s, "
            "provenance=%s, metrics=%s, "
            "cache_ttl=%ds, worker_threads=%d, "
            "background_tasks=%s, health_check=%ds, "
            "auth=%s, tracing=%s",
            config.enabled,
            config.default_gwp_source,
            config.default_tier,
            config.default_method,
            config.default_emission_factor_source,
            config.decimal_precision,
            config.default_transition_years,
            config.carbon_fraction,
            config.soc_depth_cm,
            config.default_soc_method,
            config.max_batch_size,
            config.max_parcels,
            config.max_transitions,
            config.monte_carlo_iterations,
            config.confidence_levels,
            config.enable_soc_assessment,
            config.enable_peatland,
            config.enable_fire_emissions,
            config.enable_n2o_soil,
            config.enable_compliance_checking,
            config.enable_uncertainty,
            config.enable_provenance,
            config.enable_metrics,
            config.cache_ttl_seconds,
            config.worker_threads,
            config.enable_background_tasks,
            config.health_check_interval,
            config.enable_auth,
            config.enable_tracing,
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
            >>> cfg = LandUseConfig()
            >>> d = cfg.to_dict()
            >>> d["default_method"]
            'STOCK_DIFFERENCE'
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
            # -- IPCC methodology defaults ----------------------------------
            "default_gwp_source": self.default_gwp_source,
            "default_tier": self.default_tier,
            "default_method": self.default_method,
            "default_emission_factor_source": self.default_emission_factor_source,
            # -- Calculation precision --------------------------------------
            "decimal_precision": self.decimal_precision,
            # -- Carbon stock defaults --------------------------------------
            "default_transition_years": self.default_transition_years,
            "carbon_fraction": self.carbon_fraction,
            # -- SOC assessment defaults ------------------------------------
            "soc_depth_cm": self.soc_depth_cm,
            "default_soc_method": self.default_soc_method,
            # -- Monte Carlo uncertainty analysis ---------------------------
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "monte_carlo_seed": self.monte_carlo_seed,
            "confidence_levels": self.confidence_levels,
            # -- Feature toggles --------------------------------------------
            "enable_soc_assessment": self.enable_soc_assessment,
            "enable_peatland": self.enable_peatland,
            "enable_fire_emissions": self.enable_fire_emissions,
            "enable_n2o_soil": self.enable_n2o_soil,
            "enable_compliance_checking": self.enable_compliance_checking,
            "enable_uncertainty": self.enable_uncertainty,
            "enable_provenance": self.enable_provenance,
            "enable_metrics": self.enable_metrics,
            # -- Parcel and transition capacity limits ----------------------
            "max_parcels": self.max_parcels,
            "max_transitions": self.max_transitions,
            # -- Peatland and fire settings ---------------------------------
            "peatland_default_doc_factor": self.peatland_default_doc_factor,
            "fire_combustion_efficiency": self.fire_combustion_efficiency,
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
            # -- Auth and tracing -------------------------------------------
            "enable_auth": self.enable_auth,
            "enable_tracing": self.enable_tracing,
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
        return f"LandUseConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------


class _LandUseConfigHolder:
    """Thread-safe singleton holder for LandUseConfig.

    Uses double-checked locking with a threading.Lock to ensure
    exactly one LandUseConfig instance is created from environment
    variables. Subsequent calls to ``get()`` return the cached
    instance without lock contention.

    Attributes:
        _instance: Cached LandUseConfig or None.
        _lock: Threading lock for initialization.
    """

    _instance: Optional[LandUseConfig] = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get(cls) -> LandUseConfig:
        """Return the singleton LandUseConfig, creating from env if needed.

        Uses double-checked locking for thread safety with minimal
        contention on the hot path.

        Returns:
            LandUseConfig singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = LandUseConfig.from_env()
        return cls._instance

    @classmethod
    def set(cls, config: LandUseConfig) -> None:
        """Replace the singleton LandUseConfig.

        Args:
            config: New LandUseConfig to install as the singleton.
        """
        with cls._lock:
            cls._instance = config

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton to None for test teardown."""
        with cls._lock:
            cls._instance = None


def get_config() -> LandUseConfig:
    """Return the singleton LandUseConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_LAND_USE_*`` environment variables via
    :meth:`LandUseConfig.from_env`.

    Returns:
        LandUseConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.default_method
        'STOCK_DIFFERENCE'
    """
    return _LandUseConfigHolder.get()


def set_config(config: LandUseConfig) -> None:
    """Replace the singleton LandUseConfig.

    Primarily intended for testing and dependency injection scenarios
    where a custom configuration must be supplied without relying on
    environment variables.

    Args:
        config: New :class:`LandUseConfig` to install as the singleton.

    Example:
        >>> cfg = LandUseConfig(default_method="GAIN_LOSS")
        >>> set_config(cfg)
        >>> assert get_config().default_method == "GAIN_LOSS"
    """
    _LandUseConfigHolder.set(config)
    logger.info(
        "LandUseConfig replaced programmatically: "
        "enabled=%s, gwp_source=%s, tier=%d, method=%s, "
        "ef_source=%s, max_batch_size=%d, "
        "monte_carlo_iterations=%d",
        config.enabled,
        config.default_gwp_source,
        config.default_tier,
        config.default_method,
        config.default_emission_factor_source,
        config.max_batch_size,
        config.monte_carlo_iterations,
    )


def reset_config() -> None:
    """Reset the singleton LandUseConfig to ``None``.

    The next call to :func:`get_config` will re-read environment variables
    and construct a fresh instance. Intended for test teardown to prevent
    state leakage between test cases.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads GL_LAND_USE_* env vars
    """
    _LandUseConfigHolder.reset()
    logger.debug("LandUseConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "LandUseConfig",
    "get_config",
    "set_config",
    "reset_config",
]
