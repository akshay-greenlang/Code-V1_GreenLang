# -*- coding: utf-8 -*-
"""
Process Emissions Agent Configuration - AGENT-MRV-004

Centralized configuration for the Process Emissions Agent SDK covering:
- Database, cache, and connection defaults
- GHG Protocol methodology defaults (GWP source, calculation tier, method)
- Decimal precision for emission calculations
- Capacity limits (material inputs, process units, abatement records)
- Monte Carlo uncertainty analysis parameters
- Feature toggles (mass balance, abatement tracking, by-product credits,
  compliance checking, uncertainty, provenance, metrics)
- API configuration (prefix, pagination)
- Background task and worker thread settings

All settings can be overridden via environment variables with the
``GL_PROCESS_EMISSIONS_`` prefix (e.g. ``GL_PROCESS_EMISSIONS_DEFAULT_GWP_SOURCE``,
``GL_PROCESS_EMISSIONS_MAX_BATCH_SIZE``).

Environment Variable Reference (GL_PROCESS_EMISSIONS_ prefix):
    GL_PROCESS_EMISSIONS_ENABLED                    - Enable/disable agent
    GL_PROCESS_EMISSIONS_DATABASE_URL               - PostgreSQL connection URL
    GL_PROCESS_EMISSIONS_REDIS_URL                  - Redis connection URL
    GL_PROCESS_EMISSIONS_MAX_BATCH_SIZE             - Maximum records per batch
    GL_PROCESS_EMISSIONS_DEFAULT_GWP_SOURCE         - Default GWP source (AR4/AR5/AR6/AR6_20YR)
    GL_PROCESS_EMISSIONS_DEFAULT_CALCULATION_TIER   - Default tier (TIER_1/TIER_2/TIER_3)
    GL_PROCESS_EMISSIONS_DEFAULT_CALCULATION_METHOD - Default method (EMISSION_FACTOR/MASS_BALANCE/etc.)
    GL_PROCESS_EMISSIONS_DEFAULT_EMISSION_FACTOR_SOURCE - Default EF source (EPA/IPCC/DEFRA/EU_ETS/CUSTOM)
    GL_PROCESS_EMISSIONS_DECIMAL_PRECISION          - Decimal places for calculations
    GL_PROCESS_EMISSIONS_MONTE_CARLO_ITERATIONS     - Monte Carlo simulation iterations
    GL_PROCESS_EMISSIONS_MONTE_CARLO_SEED           - Random seed for reproducibility
    GL_PROCESS_EMISSIONS_CONFIDENCE_LEVELS          - Comma-separated confidence levels
    GL_PROCESS_EMISSIONS_ENABLE_MASS_BALANCE        - Enable mass balance method
    GL_PROCESS_EMISSIONS_ENABLE_ABATEMENT_TRACKING  - Enable abatement tracking
    GL_PROCESS_EMISSIONS_ENABLE_BY_PRODUCT_CREDITS  - Enable by-product credits
    GL_PROCESS_EMISSIONS_ENABLE_COMPLIANCE_CHECKING - Enable compliance checking
    GL_PROCESS_EMISSIONS_ENABLE_UNCERTAINTY         - Enable uncertainty quantification
    GL_PROCESS_EMISSIONS_ENABLE_PROVENANCE          - Enable SHA-256 provenance chain
    GL_PROCESS_EMISSIONS_ENABLE_METRICS             - Enable Prometheus metrics export
    GL_PROCESS_EMISSIONS_MAX_MATERIAL_INPUTS        - Maximum material inputs per calc
    GL_PROCESS_EMISSIONS_MAX_PROCESS_UNITS          - Maximum process unit registrations
    GL_PROCESS_EMISSIONS_MAX_ABATEMENT_RECORDS      - Maximum abatement records
    GL_PROCESS_EMISSIONS_CACHE_TTL_SECONDS          - Cache time-to-live in seconds
    GL_PROCESS_EMISSIONS_API_PREFIX                 - REST API route prefix
    GL_PROCESS_EMISSIONS_API_MAX_PAGE_SIZE          - Maximum API page size
    GL_PROCESS_EMISSIONS_API_DEFAULT_PAGE_SIZE      - Default API page size
    GL_PROCESS_EMISSIONS_LOG_LEVEL                  - Logging level
    GL_PROCESS_EMISSIONS_WORKER_THREADS             - Worker thread pool size
    GL_PROCESS_EMISSIONS_ENABLE_BACKGROUND_TASKS    - Enable background task processing
    GL_PROCESS_EMISSIONS_HEALTH_CHECK_INTERVAL      - Health check interval seconds
    GL_PROCESS_EMISSIONS_GENESIS_HASH               - Genesis anchor for provenance chain

Example:
    >>> from greenlang.process_emissions.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_gwp_source, cfg.default_calculation_tier)
    AR6 TIER_1

    >>> # Override for testing
    >>> from greenlang.process_emissions.config import set_config, reset_config
    >>> from greenlang.process_emissions.config import ProcessEmissionsConfig
    >>> set_config(ProcessEmissionsConfig(default_calculation_tier="TIER_3"))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-004 Process Emissions (GL-MRV-SCOPE1-004)
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

_ENV_PREFIX = "GL_PROCESS_EMISSIONS_"

# ---------------------------------------------------------------------------
# Valid enumeration values for configuration validation
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

_VALID_GWP_SOURCES = frozenset({"AR4", "AR5", "AR6", "AR6_20YR"})

_VALID_CALCULATION_TIERS = frozenset({"TIER_1", "TIER_2", "TIER_3"})

_VALID_CALCULATION_METHODS = frozenset({
    "EMISSION_FACTOR",
    "MASS_BALANCE",
    "STOICHIOMETRIC",
    "DIRECT_MEASUREMENT",
})

_VALID_EF_SOURCES = frozenset({
    "EPA",
    "IPCC",
    "DEFRA",
    "EU_ETS",
    "CUSTOM",
})


# ---------------------------------------------------------------------------
# ProcessEmissionsConfig
# ---------------------------------------------------------------------------


@dataclass
class ProcessEmissionsConfig:
    """Complete configuration for the GreenLang Process Emissions Agent SDK.

    Attributes are grouped by concern: feature flag, connections, methodology
    defaults, calculation precision, capacity limits, Monte Carlo parameters,
    feature toggles, API settings, performance tuning, and provenance.

    All attributes can be overridden via environment variables using the
    ``GL_PROCESS_EMISSIONS_`` prefix (e.g.
    ``GL_PROCESS_EMISSIONS_DEFAULT_CALCULATION_TIER=TIER_3``).

    Attributes:
        enabled: Master switch to enable/disable the process emissions agent.
            When False, the agent will not process any requests.
        database_url: PostgreSQL connection URL for persistent storage of
            process data, emission factors, and calculation results.
        redis_url: Redis connection URL for caching emission factor lookups,
            process type metadata, and distributed locks.
        max_batch_size: Maximum number of process emission records that
            can be processed in a single batch calculation request.
        default_gwp_source: Default IPCC Assessment Report edition for
            Global Warming Potential values. Valid: AR4, AR5, AR6, AR6_20YR.
        default_calculation_tier: Default IPCC calculation tier.
            TIER_1 uses default factors; TIER_2 uses country/process-specific
            factors; TIER_3 uses facility-specific measurements.
        default_calculation_method: Default calculation methodology.
            EMISSION_FACTOR applies published factors to activity data.
            MASS_BALANCE tracks carbon across inputs and outputs.
            STOICHIOMETRIC uses chemical reaction equations.
            DIRECT_MEASUREMENT uses CEMS or stack test data.
        default_emission_factor_source: Default authority for emission
            factors. Valid: EPA, IPCC, DEFRA, EU_ETS, CUSTOM.
        decimal_precision: Number of decimal places to retain in emission
            calculations for intermediate and final results.
        monte_carlo_iterations: Number of Monte Carlo simulation iterations
            for uncertainty quantification. Higher values yield more
            precise confidence intervals at the cost of computation time.
        monte_carlo_seed: Random seed for Monte Carlo reproducibility.
            Set to 0 for non-deterministic runs.
        confidence_levels: Comma-separated confidence level percentages for
            uncertainty analysis output (e.g. "90,95,99").
        enable_mass_balance: When True, the mass balance calculation method
            is available for processes where input/output carbon tracking
            is feasible (e.g. iron/steel, chemical processes).
        enable_abatement_tracking: When True, emission abatement measures
            (catalytic reduction, carbon capture, etc.) are tracked and
            applied as reductions to gross process emissions.
        enable_by_product_credits: When True, by-product emission credits
            are calculated for co-products with avoided emissions.
        enable_compliance_checking: When True, calculation results are
            automatically checked against applicable regulatory frameworks.
        enable_uncertainty: When True, Monte Carlo or analytical uncertainty
            quantification is available for process emission calculations.
        enable_provenance: Compute and store SHA-256 provenance hashes for
            all process registrations, emission factor selections, calculation
            steps, and batch operations.
        enable_metrics: When True, Prometheus metrics are exported under
            the ``gl_pe_`` prefix.
        max_material_inputs: Maximum number of raw material inputs allowed
            per single calculation request.
        max_process_units: Maximum number of process unit registrations
            allowed in the system simultaneously.
        max_abatement_records: Maximum number of abatement records that
            can be tracked per process unit or facility.
        cache_ttl_seconds: TTL (seconds) for cached emission factor and
            process type lookups in Redis.
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
            chain. Uniquely identifies the Process Emissions agent.
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
    default_calculation_tier: str = "TIER_1"
    default_calculation_method: str = "EMISSION_FACTOR"
    default_emission_factor_source: str = "EPA"

    # -- Calculation precision -----------------------------------------------
    decimal_precision: int = 8

    # -- Monte Carlo uncertainty analysis ------------------------------------
    monte_carlo_iterations: int = 5_000
    monte_carlo_seed: int = 42
    confidence_levels: str = "90,95,99"

    # -- Feature toggles -----------------------------------------------------
    enable_mass_balance: bool = True
    enable_abatement_tracking: bool = True
    enable_by_product_credits: bool = True
    enable_compliance_checking: bool = True
    enable_uncertainty: bool = True
    enable_provenance: bool = True
    enable_metrics: bool = True

    # -- Process capacity limits ---------------------------------------------
    max_material_inputs: int = 50
    max_process_units: int = 200
    max_abatement_records: int = 100

    # -- Cache ---------------------------------------------------------------
    cache_ttl_seconds: int = 3600

    # -- API settings --------------------------------------------------------
    api_prefix: str = "/api/v1/process-emissions"
    api_max_page_size: int = 100
    api_default_page_size: int = 20

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Performance tuning --------------------------------------------------
    worker_threads: int = 4
    enable_background_tasks: bool = True
    health_check_interval: int = 30

    # -- Provenance tracking -------------------------------------------------
    genesis_hash: str = "GL-MRV-X-004-PROCESS-EMISSIONS-GENESIS"

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialisation.

        Performs range checks on all numeric fields, enumeration checks on
        string fields (GWP source, tier, method, EF source, log level), and
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

        # -- Calculation tier ------------------------------------------------
        normalised_tier = self.default_calculation_tier.upper()
        if normalised_tier not in _VALID_CALCULATION_TIERS:
            errors.append(
                f"default_calculation_tier must be one of "
                f"{sorted(_VALID_CALCULATION_TIERS)}, "
                f"got '{self.default_calculation_tier}'"
            )
        else:
            self.default_calculation_tier = normalised_tier

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
            ("max_material_inputs", self.max_material_inputs, 1_000),
            ("max_process_units", self.max_process_units, 50_000),
            ("max_abatement_records", self.max_abatement_records, 10_000),
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
                "ProcessEmissionsConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "ProcessEmissionsConfig validated successfully: "
            "enabled=%s, gwp_source=%s, tier=%s, method=%s, "
            "ef_source=%s, decimal_precision=%d, "
            "max_batch_size=%d, monte_carlo_iterations=%d, "
            "confidence_levels=%s, mass_balance=%s, "
            "abatement=%s, by_product_credits=%s, "
            "compliance=%s, uncertainty=%s, "
            "provenance=%s, metrics=%s",
            self.enabled,
            self.default_gwp_source,
            self.default_calculation_tier,
            self.default_calculation_method,
            self.default_emission_factor_source,
            self.decimal_precision,
            self.max_batch_size,
            self.monte_carlo_iterations,
            self.confidence_levels,
            self.enable_mass_balance,
            self.enable_abatement_tracking,
            self.enable_by_product_credits,
            self.enable_compliance_checking,
            self.enable_uncertainty,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> ProcessEmissionsConfig:
        """Build a ProcessEmissionsConfig from environment variables.

        Every field can be overridden via
        ``GL_PROCESS_EMISSIONS_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated ProcessEmissionsConfig instance, validated via
            ``__post_init__``.

        Example:
            >>> import os
            >>> os.environ["GL_PROCESS_EMISSIONS_DEFAULT_CALCULATION_TIER"] = "TIER_3"
            >>> cfg = ProcessEmissionsConfig.from_env()
            >>> cfg.default_calculation_tier
            'TIER_3'
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
            default_calculation_tier=_str(
                "DEFAULT_CALCULATION_TIER",
                cls.default_calculation_tier,
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
            enable_mass_balance=_bool(
                "ENABLE_MASS_BALANCE",
                cls.enable_mass_balance,
            ),
            enable_abatement_tracking=_bool(
                "ENABLE_ABATEMENT_TRACKING",
                cls.enable_abatement_tracking,
            ),
            enable_by_product_credits=_bool(
                "ENABLE_BY_PRODUCT_CREDITS",
                cls.enable_by_product_credits,
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
            # Process capacity limits
            max_material_inputs=_int(
                "MAX_MATERIAL_INPUTS",
                cls.max_material_inputs,
            ),
            max_process_units=_int(
                "MAX_PROCESS_UNITS",
                cls.max_process_units,
            ),
            max_abatement_records=_int(
                "MAX_ABATEMENT_RECORDS",
                cls.max_abatement_records,
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
            "ProcessEmissionsConfig loaded: "
            "enabled=%s, gwp_source=%s, tier=%s, method=%s, "
            "ef_source=%s, decimal_precision=%d, "
            "max_batch_size=%d, max_material_inputs=%d, "
            "max_process_units=%d, max_abatement_records=%d, "
            "monte_carlo_iterations=%d, confidence_levels=%s, "
            "mass_balance=%s, abatement=%s, by_product_credits=%s, "
            "compliance=%s, uncertainty=%s, "
            "provenance=%s, metrics=%s, "
            "cache_ttl=%ds, worker_threads=%d, "
            "background_tasks=%s, health_check=%ds",
            config.enabled,
            config.default_gwp_source,
            config.default_calculation_tier,
            config.default_calculation_method,
            config.default_emission_factor_source,
            config.decimal_precision,
            config.max_batch_size,
            config.max_material_inputs,
            config.max_process_units,
            config.max_abatement_records,
            config.monte_carlo_iterations,
            config.confidence_levels,
            config.enable_mass_balance,
            config.enable_abatement_tracking,
            config.enable_by_product_credits,
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
            >>> cfg = ProcessEmissionsConfig()
            >>> d = cfg.to_dict()
            >>> d["default_calculation_tier"]
            'TIER_1'
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
            "default_calculation_tier": self.default_calculation_tier,
            "default_calculation_method": self.default_calculation_method,
            "default_emission_factor_source": self.default_emission_factor_source,
            # -- Calculation precision --------------------------------------
            "decimal_precision": self.decimal_precision,
            # -- Monte Carlo uncertainty analysis ---------------------------
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "monte_carlo_seed": self.monte_carlo_seed,
            "confidence_levels": self.confidence_levels,
            # -- Feature toggles --------------------------------------------
            "enable_mass_balance": self.enable_mass_balance,
            "enable_abatement_tracking": self.enable_abatement_tracking,
            "enable_by_product_credits": self.enable_by_product_credits,
            "enable_compliance_checking": self.enable_compliance_checking,
            "enable_uncertainty": self.enable_uncertainty,
            "enable_provenance": self.enable_provenance,
            "enable_metrics": self.enable_metrics,
            # -- Process capacity limits ------------------------------------
            "max_material_inputs": self.max_material_inputs,
            "max_process_units": self.max_process_units,
            "max_abatement_records": self.max_abatement_records,
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
        return f"ProcessEmissionsConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[ProcessEmissionsConfig] = None
_config_lock = threading.Lock()


def get_config() -> ProcessEmissionsConfig:
    """Return the singleton ProcessEmissionsConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_PROCESS_EMISSIONS_*`` environment variables via
    :meth:`ProcessEmissionsConfig.from_env`.

    Returns:
        ProcessEmissionsConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.default_calculation_tier
        'TIER_1'
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ProcessEmissionsConfig.from_env()
    return _config_instance


def set_config(config: ProcessEmissionsConfig) -> None:
    """Replace the singleton ProcessEmissionsConfig.

    Primarily intended for testing and dependency injection scenarios
    where a custom configuration must be supplied without relying on
    environment variables.

    Args:
        config: New :class:`ProcessEmissionsConfig` to install as the
            singleton.

    Example:
        >>> cfg = ProcessEmissionsConfig(default_calculation_tier="TIER_3")
        >>> set_config(cfg)
        >>> assert get_config().default_calculation_tier == "TIER_3"
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "ProcessEmissionsConfig replaced programmatically: "
        "enabled=%s, gwp_source=%s, tier=%s, method=%s, "
        "ef_source=%s, max_batch_size=%d, "
        "monte_carlo_iterations=%d",
        config.enabled,
        config.default_gwp_source,
        config.default_calculation_tier,
        config.default_calculation_method,
        config.default_emission_factor_source,
        config.max_batch_size,
        config.monte_carlo_iterations,
    )


def reset_config() -> None:
    """Reset the singleton ProcessEmissionsConfig to ``None``.

    The next call to :func:`get_config` will re-read environment variables
    and construct a fresh instance. Intended for test teardown to prevent
    state leakage between test cases.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads GL_PROCESS_EMISSIONS_* env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("ProcessEmissionsConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "ProcessEmissionsConfig",
    "get_config",
    "set_config",
    "reset_config",
]
