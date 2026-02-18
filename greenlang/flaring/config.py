# -*- coding: utf-8 -*-
"""
Flaring Agent Configuration - AGENT-MRV-006

Centralized configuration for the Flaring Agent SDK covering:
- Database, cache, and connection defaults
- GHG Protocol methodology defaults (GWP source, calculation method)
- Default combustion efficiency by flare type (98% elevated, 99% enclosed)
- Gas composition defaults and component heating values
- Pilot gas defaults (1.0 MMBTU/hr per tip)
- Purge gas defaults (100 scfh nitrogen)
- Standard conditions: EPA (60F/14.696 psia) and ISO (15C/101.325 kPa)
- Default heating values: CH4=1012, C2H6=1773, C3H8=2524 BTU/scf
- OGMP 2.0 reporting level defaults
- Default flare tip velocity limits (Mach 0.2-0.5 optimal)
- Decimal precision for emission calculations
- Monte Carlo uncertainty analysis parameters (5000 iterations)
- Capacity limits (flare systems, events, compositions, calculations)
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Connection pool sizing, rate limiting, and cache TTL
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_FLARING_`` prefix (e.g. ``GL_FLARING_DEFAULT_GWP_SOURCE``,
``GL_FLARING_MAX_BATCH_SIZE``).

Environment Variable Reference (GL_FLARING_ prefix):
    GL_FLARING_ENABLED                      - Enable/disable agent
    GL_FLARING_DATABASE_URL                  - PostgreSQL connection URL
    GL_FLARING_REDIS_URL                     - Redis connection URL
    GL_FLARING_LOG_LEVEL                     - Logging level
    GL_FLARING_DEFAULT_GWP_SOURCE            - Default GWP source (AR4/AR5/AR6/AR6_20YR)
    GL_FLARING_DEFAULT_CALCULATION_METHOD    - Default calc method
    GL_FLARING_DEFAULT_EMISSION_FACTOR_SOURCE - Default EF source
    GL_FLARING_DEFAULT_STANDARD_CONDITION    - EPA_60F or ISO_15C
    GL_FLARING_DEFAULT_COMBUSTION_EFFICIENCY - Default CE (0.0-1.0)
    GL_FLARING_DEFAULT_ENCLOSED_CE           - Enclosed flare CE (0.0-1.0)
    GL_FLARING_DEFAULT_OGMP_LEVEL            - Default OGMP reporting level
    GL_FLARING_PILOT_FLOW_RATE_MMBTU_HR     - Pilot gas flow rate
    GL_FLARING_PURGE_FLOW_RATE_SCFH         - Purge gas flow rate
    GL_FLARING_MIN_TIP_VELOCITY_MACH        - Min optimal tip velocity
    GL_FLARING_MAX_TIP_VELOCITY_MACH        - Max optimal tip velocity
    GL_FLARING_MIN_LHV_BTU_SCF              - Min LHV for stable combustion
    GL_FLARING_DECIMAL_PRECISION             - Decimal places for calculations
    GL_FLARING_MAX_BATCH_SIZE                - Maximum records per batch
    GL_FLARING_MAX_FLARE_SYSTEMS             - Maximum flare system registrations
    GL_FLARING_MAX_FLARING_EVENTS            - Maximum flaring event records
    GL_FLARING_MAX_GAS_COMPOSITIONS          - Maximum gas composition records
    GL_FLARING_MAX_CALCULATIONS              - Maximum stored calculation results
    GL_FLARING_MONTE_CARLO_ITERATIONS        - Monte Carlo simulation iterations
    GL_FLARING_MONTE_CARLO_SEED              - Random seed for reproducibility
    GL_FLARING_CONFIDENCE_LEVELS             - Comma-separated confidence levels
    GL_FLARING_ENABLE_PILOT_PURGE_ACCOUNTING - Enable pilot/purge tracking
    GL_FLARING_ENABLE_BLACK_CARBON_TRACKING  - Enable black carbon tracking
    GL_FLARING_ENABLE_ZRF_TRACKING           - Enable Zero Routine Flaring
    GL_FLARING_ENABLE_COMPLIANCE_CHECKING    - Enable compliance checking
    GL_FLARING_ENABLE_UNCERTAINTY            - Enable uncertainty quantification
    GL_FLARING_ENABLE_PROVENANCE             - Enable SHA-256 provenance chain
    GL_FLARING_ENABLE_METRICS                - Enable Prometheus metrics export
    GL_FLARING_GENESIS_HASH                  - Genesis anchor for provenance
    GL_FLARING_POOL_SIZE                     - Database connection pool size
    GL_FLARING_CACHE_TTL                     - Cache time-to-live in seconds
    GL_FLARING_RATE_LIMIT                    - Max API requests per minute
    GL_FLARING_API_PREFIX                    - REST API route prefix
    GL_FLARING_API_MAX_PAGE_SIZE             - Maximum API page size
    GL_FLARING_API_DEFAULT_PAGE_SIZE         - Default API page size
    GL_FLARING_WORKER_THREADS                - Worker thread pool size
    GL_FLARING_HEALTH_CHECK_INTERVAL         - Health check interval seconds

Example:
    >>> from greenlang.flaring.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_gwp_source, cfg.default_combustion_efficiency)
    AR6 0.98

    >>> # Override for testing
    >>> from greenlang.flaring.config import set_config, reset_config
    >>> from greenlang.flaring.config import FlaringConfig
    >>> set_config(FlaringConfig(default_combustion_efficiency=0.96))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Flaring Agent (GL-MRV-SCOPE1-006)
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

_ENV_PREFIX = "GL_FLARING_"

# ---------------------------------------------------------------------------
# Valid enumeration values for configuration validation
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

_VALID_GWP_SOURCES = frozenset({"AR4", "AR5", "AR6", "AR6_20YR"})

_VALID_CALCULATION_METHODS = frozenset({
    "GAS_COMPOSITION",
    "DEFAULT_EMISSION_FACTOR",
    "ENGINEERING_ESTIMATE",
    "DIRECT_MEASUREMENT",
})

_VALID_EF_SOURCES = frozenset({
    "EPA", "IPCC", "DEFRA", "EU_ETS", "API", "CUSTOM",
})

_VALID_STANDARD_CONDITIONS = frozenset({"EPA_60F", "ISO_15C"})

_VALID_OGMP_LEVELS = frozenset({
    "LEVEL_1", "LEVEL_2", "LEVEL_3", "LEVEL_4", "LEVEL_5",
})


# ---------------------------------------------------------------------------
# Standard condition reference values
# ---------------------------------------------------------------------------

#: EPA standard conditions: 60 deg F (15.56 deg C), 14.696 psia
EPA_STANDARD_TEMP_F: float = 60.0
EPA_STANDARD_TEMP_C: float = 15.556
EPA_STANDARD_PRESSURE_PSIA: float = 14.696
EPA_STANDARD_PRESSURE_KPA: float = 101.325

#: ISO standard conditions: 15 deg C, 101.325 kPa
ISO_STANDARD_TEMP_C: float = 15.0
ISO_STANDARD_TEMP_F: float = 59.0
ISO_STANDARD_PRESSURE_KPA: float = 101.325
ISO_STANDARD_PRESSURE_PSIA: float = 14.696

#: Default component Higher Heating Values (BTU/scf) for quick reference
DEFAULT_HHV_CH4_BTU_SCF: float = 1012.0
DEFAULT_HHV_C2H6_BTU_SCF: float = 1773.0
DEFAULT_HHV_C3H8_BTU_SCF: float = 2524.0
DEFAULT_HHV_NC4H10_BTU_SCF: float = 3271.0
DEFAULT_HHV_IC4H10_BTU_SCF: float = 3254.0
DEFAULT_HHV_C5H12_BTU_SCF: float = 4010.0
DEFAULT_HHV_C6PLUS_BTU_SCF: float = 4756.0
DEFAULT_HHV_H2_BTU_SCF: float = 325.0
DEFAULT_HHV_CO_BTU_SCF: float = 321.0
DEFAULT_HHV_C2H4_BTU_SCF: float = 1614.0
DEFAULT_HHV_C3H6_BTU_SCF: float = 2336.0

#: Speed of sound in air at standard conditions (m/s) for Mach conversion
SPEED_OF_SOUND_MS: float = 343.0


# ---------------------------------------------------------------------------
# FlaringConfig
# ---------------------------------------------------------------------------


@dataclass
class FlaringConfig:
    """Complete configuration for the GreenLang Flaring Agent SDK.

    Attributes are grouped by concern: connections, logging, GHG methodology
    defaults, combustion efficiency, pilot/purge gas, flare tip velocity,
    standard conditions, OGMP reporting, calculation precision, capacity
    limits, Monte Carlo parameters, feature toggles, provenance tracking,
    metrics export, API configuration, and performance tuning.

    All attributes can be overridden via environment variables using the
    ``GL_FLARING_`` prefix (e.g. ``GL_FLARING_DEFAULT_COMBUSTION_EFFICIENCY=0.96``).

    Attributes:
        enabled: Master enable flag for the flaring agent.
        database_url: PostgreSQL connection URL.
        redis_url: Redis connection URL.
        log_level: Logging verbosity level.
        default_gwp_source: Default IPCC AR edition for GWP values.
        default_calculation_method: Default calculation methodology.
        default_emission_factor_source: Default EF source authority.
        default_standard_condition: Default T/P standard (EPA_60F/ISO_15C).
        default_combustion_efficiency: Default CE for elevated flares.
        default_enclosed_ce: Default CE for enclosed ground flares.
        default_ogmp_level: Default OGMP 2.0 reporting level.
        pilot_flow_rate_mmbtu_hr: Default pilot gas flow per tip (MMBTU/hr).
        purge_flow_rate_scfh: Default purge gas flow rate (scfh).
        default_num_pilots: Default number of pilot tips per flare.
        min_tip_velocity_mach: Minimum optimal tip velocity (Mach number).
        max_tip_velocity_mach: Maximum optimal tip velocity (Mach number).
        min_lhv_btu_scf: Minimum LHV for stable combustion (BTU/scf).
        wind_speed_ce_threshold_ms: Wind speed above which CE degrades (m/s).
        steam_ratio_optimal_min: Minimum optimal steam-to-gas ratio.
        steam_ratio_optimal_max: Maximum optimal steam-to-gas ratio.
        decimal_precision: Number of decimal places for calculations.
        max_batch_size: Maximum records per batch calculation.
        max_flare_systems: Maximum flare system registrations.
        max_flaring_events: Maximum flaring event records.
        max_gas_compositions: Maximum gas composition records.
        max_calculations: Maximum stored calculation results.
        monte_carlo_iterations: Monte Carlo simulation iterations.
        monte_carlo_seed: Optional random seed for reproducibility.
        confidence_levels: Comma-separated confidence level percentages.
        enable_pilot_purge_accounting: Enable pilot and purge gas tracking.
        enable_black_carbon_tracking: Enable black carbon/soot tracking.
        enable_zrf_tracking: Enable Zero Routine Flaring tracking.
        enable_compliance_checking: Enable multi-framework compliance.
        enable_uncertainty: Enable Monte Carlo uncertainty analysis.
        enable_provenance: Enable SHA-256 provenance chain.
        genesis_hash: Genesis anchor string for provenance chain.
        enable_metrics: Enable Prometheus metrics export.
        pool_size: Database connection pool size.
        cache_ttl: Cache time-to-live in seconds.
        rate_limit: Maximum inbound API requests per minute.
        api_prefix: REST API route prefix.
        api_max_page_size: Maximum API page size.
        api_default_page_size: Default API page size.
        worker_threads: Worker thread pool size.
        health_check_interval: Health check interval in seconds.
    """

    # -- Master enable -------------------------------------------------------
    enabled: bool = True

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- GHG methodology defaults --------------------------------------------
    default_gwp_source: str = "AR6"
    default_calculation_method: str = "DEFAULT_EMISSION_FACTOR"
    default_emission_factor_source: str = "EPA"
    default_standard_condition: str = "EPA_60F"

    # -- Combustion efficiency defaults --------------------------------------
    default_combustion_efficiency: float = 0.98
    default_enclosed_ce: float = 0.99

    # -- OGMP reporting ------------------------------------------------------
    default_ogmp_level: str = "LEVEL_2"

    # -- Pilot and purge gas defaults ----------------------------------------
    pilot_flow_rate_mmbtu_hr: float = 1.0
    purge_flow_rate_scfh: float = 100.0
    default_num_pilots: int = 1

    # -- Flare tip velocity limits -------------------------------------------
    min_tip_velocity_mach: float = 0.2
    max_tip_velocity_mach: float = 0.5

    # -- LHV stability threshold ---------------------------------------------
    min_lhv_btu_scf: float = 200.0

    # -- Wind and assist parameters ------------------------------------------
    wind_speed_ce_threshold_ms: float = 10.0
    steam_ratio_optimal_min: float = 0.3
    steam_ratio_optimal_max: float = 0.5

    # -- Calculation precision -----------------------------------------------
    decimal_precision: int = 8

    # -- Capacity limits -----------------------------------------------------
    max_batch_size: int = 10_000
    max_flare_systems: int = 10_000
    max_flaring_events: int = 1_000_000
    max_gas_compositions: int = 100_000
    max_calculations: int = 1_000_000

    # -- Monte Carlo uncertainty analysis ------------------------------------
    monte_carlo_iterations: int = 5_000
    monte_carlo_seed: int = -1  # -1 means no fixed seed
    confidence_levels: str = "90,95,99"

    # -- Feature toggles -----------------------------------------------------
    enable_pilot_purge_accounting: bool = True
    enable_black_carbon_tracking: bool = True
    enable_zrf_tracking: bool = True
    enable_compliance_checking: bool = True
    enable_uncertainty: bool = True

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-MRV-X-006-FLARING-GENESIS"

    # -- Metrics export ------------------------------------------------------
    enable_metrics: bool = True

    # -- Performance tuning --------------------------------------------------
    pool_size: int = 10
    cache_ttl: int = 3600
    rate_limit: int = 1000

    # -- API configuration ---------------------------------------------------
    api_prefix: str = "/api/v1/flaring"
    api_max_page_size: int = 1000
    api_default_page_size: int = 50

    # -- Background tasks ----------------------------------------------------
    worker_threads: int = 4
    health_check_interval: int = 30

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialisation.

        Performs range checks on all numeric fields, enumeration checks
        on string fields, and normalisation of values. Collects all
        validation errors before raising a single ValueError.

        Raises:
            ValueError: If any configuration value is outside its valid
                range or violates a constraint.
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
        normalised_ef = self.default_emission_factor_source.upper()
        if normalised_ef not in _VALID_EF_SOURCES:
            errors.append(
                f"default_emission_factor_source must be one of "
                f"{sorted(_VALID_EF_SOURCES)}, "
                f"got '{self.default_emission_factor_source}'"
            )
        else:
            self.default_emission_factor_source = normalised_ef

        # -- Standard condition ----------------------------------------------
        normalised_sc = self.default_standard_condition.upper()
        if normalised_sc not in _VALID_STANDARD_CONDITIONS:
            errors.append(
                f"default_standard_condition must be one of "
                f"{sorted(_VALID_STANDARD_CONDITIONS)}, "
                f"got '{self.default_standard_condition}'"
            )
        else:
            self.default_standard_condition = normalised_sc

        # -- OGMP level ------------------------------------------------------
        normalised_ogmp = self.default_ogmp_level.upper()
        if normalised_ogmp not in _VALID_OGMP_LEVELS:
            errors.append(
                f"default_ogmp_level must be one of "
                f"{sorted(_VALID_OGMP_LEVELS)}, "
                f"got '{self.default_ogmp_level}'"
            )
        else:
            self.default_ogmp_level = normalised_ogmp

        # -- Combustion efficiency -------------------------------------------
        if not (0.0 <= self.default_combustion_efficiency <= 1.0):
            errors.append(
                f"default_combustion_efficiency must be in [0.0, 1.0], "
                f"got {self.default_combustion_efficiency}"
            )
        if not (0.0 <= self.default_enclosed_ce <= 1.0):
            errors.append(
                f"default_enclosed_ce must be in [0.0, 1.0], "
                f"got {self.default_enclosed_ce}"
            )

        # -- Pilot and purge -------------------------------------------------
        if self.pilot_flow_rate_mmbtu_hr < 0:
            errors.append(
                f"pilot_flow_rate_mmbtu_hr must be >= 0, "
                f"got {self.pilot_flow_rate_mmbtu_hr}"
            )
        if self.purge_flow_rate_scfh < 0:
            errors.append(
                f"purge_flow_rate_scfh must be >= 0, "
                f"got {self.purge_flow_rate_scfh}"
            )
        if self.default_num_pilots < 0:
            errors.append(
                f"default_num_pilots must be >= 0, "
                f"got {self.default_num_pilots}"
            )

        # -- Tip velocity ----------------------------------------------------
        if self.min_tip_velocity_mach < 0:
            errors.append(
                f"min_tip_velocity_mach must be >= 0, "
                f"got {self.min_tip_velocity_mach}"
            )
        if self.max_tip_velocity_mach < self.min_tip_velocity_mach:
            errors.append(
                f"max_tip_velocity_mach ({self.max_tip_velocity_mach}) must "
                f"be >= min_tip_velocity_mach ({self.min_tip_velocity_mach})"
            )

        # -- LHV threshold ---------------------------------------------------
        if self.min_lhv_btu_scf < 0:
            errors.append(
                f"min_lhv_btu_scf must be >= 0, "
                f"got {self.min_lhv_btu_scf}"
            )

        # -- Wind and assist -------------------------------------------------
        if self.wind_speed_ce_threshold_ms < 0:
            errors.append(
                f"wind_speed_ce_threshold_ms must be >= 0, "
                f"got {self.wind_speed_ce_threshold_ms}"
            )
        if self.steam_ratio_optimal_min < 0:
            errors.append(
                f"steam_ratio_optimal_min must be >= 0, "
                f"got {self.steam_ratio_optimal_min}"
            )
        if self.steam_ratio_optimal_max < self.steam_ratio_optimal_min:
            errors.append(
                f"steam_ratio_optimal_max ({self.steam_ratio_optimal_max}) "
                f"must be >= steam_ratio_optimal_min "
                f"({self.steam_ratio_optimal_min})"
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
            ("max_flare_systems", self.max_flare_systems, 1_000_000),
            ("max_flaring_events", self.max_flaring_events, 100_000_000),
            ("max_gas_compositions", self.max_gas_compositions, 10_000_000),
            ("max_calculations", self.max_calculations, 100_000_000),
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

        # -- API configuration -----------------------------------------------
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
                f"worker_threads must be > 0, got {self.worker_threads}"
            )

        # -- Health check interval -------------------------------------------
        if self.health_check_interval <= 0:
            errors.append(
                f"health_check_interval must be > 0, "
                f"got {self.health_check_interval}"
            )

        if errors:
            raise ValueError(
                "FlaringConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "FlaringConfig validated successfully: "
            "gwp_source=%s, method=%s, ef_source=%s, "
            "standard_condition=%s, default_ce=%.3f, enclosed_ce=%.3f, "
            "ogmp_level=%s, decimal_precision=%d, max_batch_size=%d, "
            "monte_carlo_iterations=%d, pilot_purge=%s, "
            "black_carbon=%s, zrf=%s, provenance=%s, metrics=%s",
            self.default_gwp_source,
            self.default_calculation_method,
            self.default_emission_factor_source,
            self.default_standard_condition,
            self.default_combustion_efficiency,
            self.default_enclosed_ce,
            self.default_ogmp_level,
            self.decimal_precision,
            self.max_batch_size,
            self.monte_carlo_iterations,
            self.enable_pilot_purge_accounting,
            self.enable_black_carbon_tracking,
            self.enable_zrf_tracking,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> FlaringConfig:
        """Build a FlaringConfig from environment variables.

        Every field can be overridden via ``GL_FLARING_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log.

        Returns:
            Populated FlaringConfig instance, validated via ``__post_init__``.

        Example:
            >>> import os
            >>> os.environ["GL_FLARING_DEFAULT_GWP_SOURCE"] = "AR5"
            >>> cfg = FlaringConfig.from_env()
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
                    prefix, name, val, default,
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
                    prefix, name, val, default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val.strip()

        config = cls(
            # Master enable
            enabled=_bool("ENABLED", cls.enabled),
            # Connections
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
            # GHG methodology defaults
            default_gwp_source=_str(
                "DEFAULT_GWP_SOURCE", cls.default_gwp_source,
            ),
            default_calculation_method=_str(
                "DEFAULT_CALCULATION_METHOD",
                cls.default_calculation_method,
            ),
            default_emission_factor_source=_str(
                "DEFAULT_EMISSION_FACTOR_SOURCE",
                cls.default_emission_factor_source,
            ),
            default_standard_condition=_str(
                "DEFAULT_STANDARD_CONDITION",
                cls.default_standard_condition,
            ),
            # Combustion efficiency defaults
            default_combustion_efficiency=_float(
                "DEFAULT_COMBUSTION_EFFICIENCY",
                cls.default_combustion_efficiency,
            ),
            default_enclosed_ce=_float(
                "DEFAULT_ENCLOSED_CE", cls.default_enclosed_ce,
            ),
            # OGMP reporting
            default_ogmp_level=_str(
                "DEFAULT_OGMP_LEVEL", cls.default_ogmp_level,
            ),
            # Pilot and purge gas defaults
            pilot_flow_rate_mmbtu_hr=_float(
                "PILOT_FLOW_RATE_MMBTU_HR",
                cls.pilot_flow_rate_mmbtu_hr,
            ),
            purge_flow_rate_scfh=_float(
                "PURGE_FLOW_RATE_SCFH", cls.purge_flow_rate_scfh,
            ),
            default_num_pilots=_int(
                "DEFAULT_NUM_PILOTS", cls.default_num_pilots,
            ),
            # Tip velocity limits
            min_tip_velocity_mach=_float(
                "MIN_TIP_VELOCITY_MACH", cls.min_tip_velocity_mach,
            ),
            max_tip_velocity_mach=_float(
                "MAX_TIP_VELOCITY_MACH", cls.max_tip_velocity_mach,
            ),
            # LHV stability threshold
            min_lhv_btu_scf=_float(
                "MIN_LHV_BTU_SCF", cls.min_lhv_btu_scf,
            ),
            # Wind and assist parameters
            wind_speed_ce_threshold_ms=_float(
                "WIND_SPEED_CE_THRESHOLD_MS",
                cls.wind_speed_ce_threshold_ms,
            ),
            steam_ratio_optimal_min=_float(
                "STEAM_RATIO_OPTIMAL_MIN",
                cls.steam_ratio_optimal_min,
            ),
            steam_ratio_optimal_max=_float(
                "STEAM_RATIO_OPTIMAL_MAX",
                cls.steam_ratio_optimal_max,
            ),
            # Calculation precision
            decimal_precision=_int(
                "DECIMAL_PRECISION", cls.decimal_precision,
            ),
            # Capacity limits
            max_batch_size=_int(
                "MAX_BATCH_SIZE", cls.max_batch_size,
            ),
            max_flare_systems=_int(
                "MAX_FLARE_SYSTEMS", cls.max_flare_systems,
            ),
            max_flaring_events=_int(
                "MAX_FLARING_EVENTS", cls.max_flaring_events,
            ),
            max_gas_compositions=_int(
                "MAX_GAS_COMPOSITIONS", cls.max_gas_compositions,
            ),
            max_calculations=_int(
                "MAX_CALCULATIONS", cls.max_calculations,
            ),
            # Monte Carlo uncertainty analysis
            monte_carlo_iterations=_int(
                "MONTE_CARLO_ITERATIONS",
                cls.monte_carlo_iterations,
            ),
            monte_carlo_seed=_int(
                "MONTE_CARLO_SEED", cls.monte_carlo_seed,
            ),
            confidence_levels=_str(
                "CONFIDENCE_LEVELS", cls.confidence_levels,
            ),
            # Feature toggles
            enable_pilot_purge_accounting=_bool(
                "ENABLE_PILOT_PURGE_ACCOUNTING",
                cls.enable_pilot_purge_accounting,
            ),
            enable_black_carbon_tracking=_bool(
                "ENABLE_BLACK_CARBON_TRACKING",
                cls.enable_black_carbon_tracking,
            ),
            enable_zrf_tracking=_bool(
                "ENABLE_ZRF_TRACKING", cls.enable_zrf_tracking,
            ),
            enable_compliance_checking=_bool(
                "ENABLE_COMPLIANCE_CHECKING",
                cls.enable_compliance_checking,
            ),
            enable_uncertainty=_bool(
                "ENABLE_UNCERTAINTY", cls.enable_uncertainty,
            ),
            # Provenance tracking
            enable_provenance=_bool(
                "ENABLE_PROVENANCE", cls.enable_provenance,
            ),
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
            # Metrics export
            enable_metrics=_bool(
                "ENABLE_METRICS", cls.enable_metrics,
            ),
            # Performance tuning
            pool_size=_int("POOL_SIZE", cls.pool_size),
            cache_ttl=_int("CACHE_TTL", cls.cache_ttl),
            rate_limit=_int("RATE_LIMIT", cls.rate_limit),
            # API configuration
            api_prefix=_str("API_PREFIX", cls.api_prefix),
            api_max_page_size=_int(
                "API_MAX_PAGE_SIZE", cls.api_max_page_size,
            ),
            api_default_page_size=_int(
                "API_DEFAULT_PAGE_SIZE", cls.api_default_page_size,
            ),
            # Background tasks
            worker_threads=_int(
                "WORKER_THREADS", cls.worker_threads,
            ),
            health_check_interval=_int(
                "HEALTH_CHECK_INTERVAL", cls.health_check_interval,
            ),
        )

        logger.info(
            "FlaringConfig loaded: gwp_source=%s, method=%s, "
            "ef_source=%s, standard=%s, default_ce=%.3f, "
            "enclosed_ce=%.3f, ogmp=%s, pilot=%.2f MMBTU/hr, "
            "purge=%.1f scfh, precision=%d, batch=%d, mc=%d, "
            "pilot_purge=%s, black_carbon=%s, zrf=%s, "
            "provenance=%s, metrics=%s, pool=%d, cache=%ds, "
            "rate=%d/min",
            config.default_gwp_source,
            config.default_calculation_method,
            config.default_emission_factor_source,
            config.default_standard_condition,
            config.default_combustion_efficiency,
            config.default_enclosed_ce,
            config.default_ogmp_level,
            config.pilot_flow_rate_mmbtu_hr,
            config.purge_flow_rate_scfh,
            config.decimal_precision,
            config.max_batch_size,
            config.monte_carlo_iterations,
            config.enable_pilot_purge_accounting,
            config.enable_black_carbon_tracking,
            config.enable_zrf_tracking,
            config.enable_provenance,
            config.enable_metrics,
            config.pool_size,
            config.cache_ttl,
            config.rate_limit,
        )
        return config

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a plain Python dictionary.

        Sensitive connection strings (database_url, redis_url) are
        redacted to prevent accidental credential leakage.

        Returns:
            Dictionary representation with sensitive fields redacted.
        """
        return {
            # -- Master enable -----------------------------------------------
            "enabled": self.enabled,
            # -- Connections (redacted) --------------------------------------
            "database_url": "***" if self.database_url else "",
            "redis_url": "***" if self.redis_url else "",
            # -- Logging -----------------------------------------------------
            "log_level": self.log_level,
            # -- GHG methodology defaults ------------------------------------
            "default_gwp_source": self.default_gwp_source,
            "default_calculation_method": self.default_calculation_method,
            "default_emission_factor_source": self.default_emission_factor_source,
            "default_standard_condition": self.default_standard_condition,
            # -- Combustion efficiency ---------------------------------------
            "default_combustion_efficiency": self.default_combustion_efficiency,
            "default_enclosed_ce": self.default_enclosed_ce,
            # -- OGMP reporting ----------------------------------------------
            "default_ogmp_level": self.default_ogmp_level,
            # -- Pilot and purge gas -----------------------------------------
            "pilot_flow_rate_mmbtu_hr": self.pilot_flow_rate_mmbtu_hr,
            "purge_flow_rate_scfh": self.purge_flow_rate_scfh,
            "default_num_pilots": self.default_num_pilots,
            # -- Tip velocity ------------------------------------------------
            "min_tip_velocity_mach": self.min_tip_velocity_mach,
            "max_tip_velocity_mach": self.max_tip_velocity_mach,
            # -- LHV stability -----------------------------------------------
            "min_lhv_btu_scf": self.min_lhv_btu_scf,
            # -- Wind and assist ---------------------------------------------
            "wind_speed_ce_threshold_ms": self.wind_speed_ce_threshold_ms,
            "steam_ratio_optimal_min": self.steam_ratio_optimal_min,
            "steam_ratio_optimal_max": self.steam_ratio_optimal_max,
            # -- Calculation precision ---------------------------------------
            "decimal_precision": self.decimal_precision,
            # -- Capacity limits ---------------------------------------------
            "max_batch_size": self.max_batch_size,
            "max_flare_systems": self.max_flare_systems,
            "max_flaring_events": self.max_flaring_events,
            "max_gas_compositions": self.max_gas_compositions,
            "max_calculations": self.max_calculations,
            # -- Monte Carlo uncertainty analysis ----------------------------
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "monte_carlo_seed": self.monte_carlo_seed,
            "confidence_levels": self.confidence_levels,
            # -- Feature toggles ---------------------------------------------
            "enable_pilot_purge_accounting": self.enable_pilot_purge_accounting,
            "enable_black_carbon_tracking": self.enable_black_carbon_tracking,
            "enable_zrf_tracking": self.enable_zrf_tracking,
            "enable_compliance_checking": self.enable_compliance_checking,
            "enable_uncertainty": self.enable_uncertainty,
            # -- Provenance tracking -----------------------------------------
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # -- Metrics export ----------------------------------------------
            "enable_metrics": self.enable_metrics,
            # -- Performance tuning ------------------------------------------
            "pool_size": self.pool_size,
            "cache_ttl": self.cache_ttl,
            "rate_limit": self.rate_limit,
            # -- API configuration -------------------------------------------
            "api_prefix": self.api_prefix,
            "api_max_page_size": self.api_max_page_size,
            "api_default_page_size": self.api_default_page_size,
            # -- Background tasks --------------------------------------------
            "worker_threads": self.worker_threads,
            "health_check_interval": self.health_check_interval,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Returns:
            String representation of the configuration.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"FlaringConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[FlaringConfig] = None
_config_lock = threading.Lock()


def get_config() -> FlaringConfig:
    """Return the singleton FlaringConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path.

    Returns:
        FlaringConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.default_combustion_efficiency
        0.98
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = FlaringConfig.from_env()
    return _config_instance


def set_config(config: FlaringConfig) -> None:
    """Replace the singleton FlaringConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New FlaringConfig to install as the singleton.

    Example:
        >>> cfg = FlaringConfig(default_combustion_efficiency=0.96)
        >>> set_config(cfg)
        >>> assert get_config().default_combustion_efficiency == 0.96
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "FlaringConfig replaced programmatically: "
        "gwp_source=%s, method=%s, default_ce=%.3f, "
        "enclosed_ce=%.3f, ogmp=%s, batch=%d, mc=%d",
        config.default_gwp_source,
        config.default_calculation_method,
        config.default_combustion_efficiency,
        config.default_enclosed_ce,
        config.default_ogmp_level,
        config.max_batch_size,
        config.monte_carlo_iterations,
    )


def reset_config() -> None:
    """Reset the singleton FlaringConfig to None.

    The next call to get_config() will re-read environment variables
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads GL_FLARING_* env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("FlaringConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "FlaringConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Standard condition constants
    "EPA_STANDARD_TEMP_F",
    "EPA_STANDARD_TEMP_C",
    "EPA_STANDARD_PRESSURE_PSIA",
    "EPA_STANDARD_PRESSURE_KPA",
    "ISO_STANDARD_TEMP_C",
    "ISO_STANDARD_TEMP_F",
    "ISO_STANDARD_PRESSURE_KPA",
    "ISO_STANDARD_PRESSURE_PSIA",
    # Default HHV constants
    "DEFAULT_HHV_CH4_BTU_SCF",
    "DEFAULT_HHV_C2H6_BTU_SCF",
    "DEFAULT_HHV_C3H8_BTU_SCF",
    "DEFAULT_HHV_NC4H10_BTU_SCF",
    "DEFAULT_HHV_IC4H10_BTU_SCF",
    "DEFAULT_HHV_C5H12_BTU_SCF",
    "DEFAULT_HHV_C6PLUS_BTU_SCF",
    "DEFAULT_HHV_H2_BTU_SCF",
    "DEFAULT_HHV_CO_BTU_SCF",
    "DEFAULT_HHV_C2H4_BTU_SCF",
    "DEFAULT_HHV_C3H6_BTU_SCF",
    # Speed of sound
    "SPEED_OF_SOUND_MS",
]
