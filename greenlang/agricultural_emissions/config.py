# -*- coding: utf-8 -*-
"""
Agricultural Emissions Agent Configuration - AGENT-MRV-008

Centralized configuration for the Agricultural Emissions Agent SDK covering:
- Database, cache, and connection defaults
- IPCC methodology defaults (GWP source, calculation method, EF source)
- Enteric fermentation parameters (Ym%, DE%, CFi, activity coefficients)
- Manure management parameters (MCF, Bo, VS, temperature)
- Agricultural soils parameters (EF1-EF5, FRAC_GASF/GASM/LEACH)
- Indirect N2O emission parameters (EF4, EF5)
- Liming and urea application parameters (limestone, dolomite, urea EFs)
- Rice cultivation parameters (baseline EF, cultivation days, water regime)
- Field burning parameters (combustion factor, burn fraction)
- Fossil vs biogenic CH4 separation control
- Decimal precision for emission calculations
- Capacity limits (farms, livestock records, batches)
- Monte Carlo uncertainty analysis parameters
- Feature toggles (enteric, manure, soils, rice, field burning,
  compliance checking, uncertainty, provenance, metrics)
- API configuration (prefix, pagination)
- Background task and worker thread settings

All settings can be overridden via environment variables with the
``GL_AGRICULTURAL_`` prefix (e.g.
``GL_AGRICULTURAL_DEFAULT_GWP_SOURCE``,
``GL_AGRICULTURAL_MAX_BATCH_SIZE``).

Environment Variable Reference (GL_AGRICULTURAL_ prefix):
    GL_AGRICULTURAL_ENABLED                       - Enable/disable agent
    GL_AGRICULTURAL_DATABASE_URL                  - PostgreSQL connection URL
    GL_AGRICULTURAL_REDIS_URL                     - Redis connection URL
    GL_AGRICULTURAL_MAX_BATCH_SIZE                - Maximum records per batch
    GL_AGRICULTURAL_DEFAULT_GWP_SOURCE            - Default GWP source (AR4/AR5/AR6/AR6_20YR)
    GL_AGRICULTURAL_DEFAULT_CALCULATION_METHOD    - Default method (IPCC_TIER_1/IPCC_TIER_2/
                                                     IPCC_TIER_3/MASS_BALANCE/DIRECT_MEASUREMENT/
                                                     SPEND_BASED)
    GL_AGRICULTURAL_DEFAULT_EMISSION_FACTOR_SOURCE - Default EF source
    GL_AGRICULTURAL_DECIMAL_PRECISION             - Decimal places for calculations
    GL_AGRICULTURAL_DEFAULT_CLIMATE_ZONE          - Default IPCC climate zone
    GL_AGRICULTURAL_DEFAULT_YM_PCT                - Enteric methane conversion factor (Ym%)
    GL_AGRICULTURAL_DEFAULT_DE_PCT                - Feed digestibility (DE%)
    GL_AGRICULTURAL_DEFAULT_CFI_DAIRY             - Cattle feed intake (dairy)
    GL_AGRICULTURAL_DEFAULT_CFI_NON_DAIRY         - Cattle feed intake (non-dairy)
    GL_AGRICULTURAL_DEFAULT_ACTIVITY_COEFFICIENT  - Activity coefficient (Ca)
    GL_AGRICULTURAL_DEFAULT_PREGNANCY_FACTOR      - Pregnancy factor (NEp/NE)
    GL_AGRICULTURAL_DEFAULT_MANURE_MCF            - Manure methane correction factor
    GL_AGRICULTURAL_DEFAULT_MANURE_BO_DAIRY       - Manure max CH4 capacity (dairy)
    GL_AGRICULTURAL_DEFAULT_MANURE_BO_SWINE       - Manure max CH4 capacity (swine)
    GL_AGRICULTURAL_DEFAULT_VS_DAIRY              - Volatile solids (dairy, kg/head/day)
    GL_AGRICULTURAL_DEFAULT_TEMPERATURE_C         - Annual average temperature
    GL_AGRICULTURAL_DEFAULT_EF1                   - Direct N2O EF (kg N2O-N/kg N)
    GL_AGRICULTURAL_DEFAULT_EF2_CG                - Organic soils cropland/grassland EF
    GL_AGRICULTURAL_DEFAULT_EF2_F                 - Organic soils forest land EF
    GL_AGRICULTURAL_DEFAULT_EF3_PRP_CATTLE        - Pasture/range/paddock EF (cattle)
    GL_AGRICULTURAL_DEFAULT_EF3_PRP_OTHER         - Pasture/range/paddock EF (other)
    GL_AGRICULTURAL_DEFAULT_FRAC_GASF             - Fraction synthetic N volatilised
    GL_AGRICULTURAL_DEFAULT_FRAC_GASM             - Fraction organic N volatilised
    GL_AGRICULTURAL_DEFAULT_FRAC_LEACH            - Fraction N leached/runoff
    GL_AGRICULTURAL_DEFAULT_EF4                   - Indirect N2O from atmospheric deposition
    GL_AGRICULTURAL_DEFAULT_EF5                   - Indirect N2O from leaching/runoff
    GL_AGRICULTURAL_DEFAULT_LIMESTONE_EF          - Limestone CO2 emission factor
    GL_AGRICULTURAL_DEFAULT_DOLOMITE_EF           - Dolomite CO2 emission factor
    GL_AGRICULTURAL_DEFAULT_UREA_EF               - Urea CO2 emission factor
    GL_AGRICULTURAL_DEFAULT_RICE_BASELINE_EF      - Rice baseline CH4 EF (kg/ha/day)
    GL_AGRICULTURAL_DEFAULT_RICE_CULTIVATION_DAYS - Rice cultivation period (days)
    GL_AGRICULTURAL_DEFAULT_WATER_REGIME          - Rice water management regime
    GL_AGRICULTURAL_DEFAULT_COMBUSTION_FACTOR     - Field burning combustion factor
    GL_AGRICULTURAL_DEFAULT_BURN_FRACTION         - Fraction of residues burned
    GL_AGRICULTURAL_MONTE_CARLO_ITERATIONS        - Monte Carlo simulation iterations
    GL_AGRICULTURAL_MONTE_CARLO_SEED              - Random seed for reproducibility
    GL_AGRICULTURAL_CONFIDENCE_LEVELS             - Comma-separated confidence levels
    GL_AGRICULTURAL_ENABLE_ENTERIC                - Enable enteric fermentation engine
    GL_AGRICULTURAL_ENABLE_MANURE                 - Enable manure management engine
    GL_AGRICULTURAL_ENABLE_SOILS                  - Enable agricultural soils engine
    GL_AGRICULTURAL_ENABLE_RICE                   - Enable rice cultivation engine
    GL_AGRICULTURAL_ENABLE_FIELD_BURNING          - Enable field burning engine
    GL_AGRICULTURAL_ENABLE_COMPLIANCE_CHECKING    - Enable compliance checking
    GL_AGRICULTURAL_ENABLE_UNCERTAINTY            - Enable uncertainty quantification
    GL_AGRICULTURAL_ENABLE_PROVENANCE             - Enable SHA-256 provenance chain
    GL_AGRICULTURAL_ENABLE_METRICS                - Enable Prometheus metrics export
    GL_AGRICULTURAL_SEPARATE_BIOGENIC_CH4         - Separate biogenic/fossil CH4
    GL_AGRICULTURAL_MAX_FARMS                     - Maximum farm registrations
    GL_AGRICULTURAL_MAX_LIVESTOCK_RECORDS         - Maximum livestock records
    GL_AGRICULTURAL_CACHE_TTL_SECONDS             - Cache time-to-live in seconds
    GL_AGRICULTURAL_API_PREFIX                    - REST API route prefix
    GL_AGRICULTURAL_API_MAX_PAGE_SIZE             - Maximum API page size
    GL_AGRICULTURAL_API_DEFAULT_PAGE_SIZE         - Default API page size
    GL_AGRICULTURAL_LOG_LEVEL                     - Logging level
    GL_AGRICULTURAL_WORKER_THREADS                - Worker thread pool size
    GL_AGRICULTURAL_ENABLE_BACKGROUND_TASKS       - Enable background task processing
    GL_AGRICULTURAL_HEALTH_CHECK_INTERVAL         - Health check interval seconds
    GL_AGRICULTURAL_GENESIS_HASH                  - Genesis anchor for provenance chain
    GL_AGRICULTURAL_ENABLE_AUTH                    - Enable authentication middleware
    GL_AGRICULTURAL_ENABLE_TRACING                - Enable OpenTelemetry tracing

Example:
    >>> from greenlang.agricultural_emissions.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_gwp_source, cfg.default_calculation_method)
    AR6 IPCC_TIER_1

    >>> # Override for testing
    >>> from greenlang.agricultural_emissions.config import set_config, reset_config
    >>> from greenlang.agricultural_emissions.config import AgriculturalEmissionsConfig
    >>> set_config(AgriculturalEmissionsConfig(default_calculation_method="IPCC_TIER_2"))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-008 Agricultural Emissions (GL-MRV-SCOPE1-008)
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

_ENV_PREFIX = "GL_AGRICULTURAL_"

# ---------------------------------------------------------------------------
# Valid enumeration values for configuration validation
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

_VALID_GWP_SOURCES = frozenset({"AR4", "AR5", "AR6", "AR6_20YR"})

_VALID_CALCULATION_METHODS = frozenset({
    "IPCC_TIER_1",
    "IPCC_TIER_2",
    "IPCC_TIER_3",
    "MASS_BALANCE",
    "DIRECT_MEASUREMENT",
    "SPEND_BASED",
})

_VALID_EF_SOURCES = frozenset({
    "IPCC_2006",
    "IPCC_2019",
    "EPA_AP42",
    "DEFRA",
    "ECOINVENT",
    "NATIONAL_INVENTORY",
    "CUSTOM",
})

_VALID_CLIMATE_ZONES = frozenset({
    "TROPICAL_WET",
    "TROPICAL_DRY",
    "WARM_TEMPERATE_WET",
    "WARM_TEMPERATE_DRY",
    "COOL_TEMPERATE_WET",
    "COOL_TEMPERATE_DRY",
    "BOREAL_WET",
    "BOREAL_DRY",
})

_VALID_WATER_REGIMES = frozenset({
    "continuously_flooded",
    "intermittently_flooded_single",
    "intermittently_flooded_multiple",
    "rainfed_regular",
    "rainfed_drought_prone",
    "deep_water",
    "upland",
})


# ---------------------------------------------------------------------------
# AgriculturalEmissionsConfig
# ---------------------------------------------------------------------------


@dataclass
class AgriculturalEmissionsConfig:
    """Complete configuration for the GreenLang Agricultural Emissions Agent SDK.

    Attributes are grouped by concern: feature flag, connections, methodology
    defaults, enteric fermentation parameters, manure management parameters,
    agricultural soils parameters, indirect N2O parameters, liming and urea
    parameters, rice cultivation parameters, field burning parameters,
    biogenic CH4 separation, calculation precision, capacity limits,
    Monte Carlo parameters, feature toggles, API settings, performance
    tuning, and provenance.

    All attributes can be overridden via environment variables using the
    ``GL_AGRICULTURAL_`` prefix (e.g.
    ``GL_AGRICULTURAL_DEFAULT_CALCULATION_METHOD=IPCC_TIER_2``).

    Attributes:
        enabled: Master switch to enable/disable the agricultural emissions
            agent. When False, the agent will not process any requests.
        database_url: PostgreSQL connection URL for persistent storage of
            farm data, livestock records, crop data, and calculation
            results.
        redis_url: Redis connection URL for caching emission factor lookups,
            IPCC default tables, and distributed locks.
        max_batch_size: Maximum number of agricultural emission records
            that can be processed in a single batch calculation request.
        default_gwp_source: Default IPCC Assessment Report edition for
            Global Warming Potential values. Valid: AR4, AR5, AR6, AR6_20YR.
            AR6 provides updated CH4 GWP (27.9 fossil, 27.2 biogenic for
            100-year). AR6_20YR uses 20-year time horizon values.
        default_calculation_method: Default calculation methodology for
            agricultural emissions. Valid: IPCC_TIER_1 (default EFs),
            IPCC_TIER_2 (country-specific data), IPCC_TIER_3 (detailed
            modelling/measurements), MASS_BALANCE (N balance approach),
            DIRECT_MEASUREMENT (field measurements), SPEND_BASED (economic
            input-output EFs from DEFRA/EPA).
        default_emission_factor_source: Default authority for emission
            factors. Valid: IPCC_2006, IPCC_2019, EPA_AP42, DEFRA,
            ECOINVENT, NATIONAL_INVENTORY, CUSTOM.
        decimal_precision: Number of decimal places to retain in emission
            calculations for intermediate and final results.
        default_climate_zone: Default IPCC climate zone for selecting
            region-specific emission factors and parameters. Valid:
            TROPICAL_WET, TROPICAL_DRY, WARM_TEMPERATE_WET,
            WARM_TEMPERATE_DRY, COOL_TEMPERATE_WET, COOL_TEMPERATE_DRY,
            BOREAL_WET, BOREAL_DRY.
        default_ym_pct: Default methane conversion factor (Ym%) representing
            the fraction of gross energy intake converted to CH4 by enteric
            fermentation. IPCC Tier 1 default is 6.5% for cattle. Range
            typically 2-12% depending on diet quality and animal type.
        default_de_pct: Default feed digestibility (DE%) as a percentage
            of gross energy. IPCC default 65% for cattle on mixed diets.
            Higher values (70-80%) for concentrates, lower (45-55%) for
            crop residues and straw.
        default_cfi_dairy: Default cattle feed intake coefficient for
            dairy cattle (dimensionless). IPCC default 0.386 used in
            Tier 2 net energy calculations.
        default_cfi_non_dairy: Default cattle feed intake coefficient for
            non-dairy cattle (dimensionless). IPCC default 0.322.
        default_activity_coefficient: Default activity coefficient (Ca)
            for net energy calculations. 0.0 for confined animals, 0.17
            for animals grazing large pastures, 0.36 for hilly terrain.
            Default 0.0 for confined feeding operations.
        default_pregnancy_factor: Default pregnancy factor representing the
            fraction of NE_maintenance required for pregnancy (NEp/NE).
            IPCC default 0.10 (10%) for cattle.
        default_manure_mcf: Default methane correction factor for manure
            management systems. Varies by system: 0.0 for daily spread,
            0.10 for solid storage, 0.39 for liquid/slurry without crust,
            0.66 for anaerobic lagoon. Default 0.10 for solid storage.
        default_manure_bo_dairy: Default maximum methane producing capacity
            (Bo) for dairy cattle manure (m3 CH4/kg VS). IPCC default 0.24.
        default_manure_bo_swine: Default maximum methane producing capacity
            (Bo) for swine manure (m3 CH4/kg VS). IPCC default 0.48.
        default_vs_dairy: Default volatile solids excretion rate for dairy
            cattle (kg VS/head/day). IPCC default varies by region;
            5.4 for North American dairy cattle.
        default_temperature_c: Default annual average temperature (Celsius)
            for temperature-dependent MCF calculations. Affects methane
            generation from manure storage and anaerobic decomposition.
        default_ef1: Default IPCC emission factor EF1 for direct N2O
            emissions from managed soils (kg N2O-N/kg N input). IPCC 2019
            default is 0.01 (1%).
        default_ef2_cg: Default IPCC emission factor EF2 for N2O emissions
            from drained/managed organic soils in cropland and grassland
            (kg N2O-N/ha/yr). IPCC 2019 default 8.0.
        default_ef2_f: Default IPCC emission factor EF2 for N2O emissions
            from drained/managed organic soils in forest land
            (kg N2O-N/ha/yr). IPCC 2019 default 2.5.
        default_ef3_prp_cattle: Default IPCC emission factor EF3_PRP for
            N2O emissions from cattle dung and urine deposited on pasture,
            range, and paddock (kg N2O-N/kg N). IPCC 2019 default 0.02.
        default_ef3_prp_other: Default IPCC emission factor EF3_PRP for
            N2O emissions from non-cattle livestock dung and urine deposited
            on pasture, range, and paddock (kg N2O-N/kg N). IPCC 2019
            default 0.01.
        default_frac_gasf: Default fraction of synthetic fertiliser N that
            volatilises as NH3 and NOx (FRAC_GASF). IPCC default 0.10.
        default_frac_gasm: Default fraction of organic N (manure, compost,
            sewage sludge) that volatilises as NH3 and NOx (FRAC_GASM).
            IPCC default 0.20.
        default_frac_leach: Default fraction of all N additions that is
            leached or runs off in regions where leaching occurs
            (FRAC_LEACH). IPCC default 0.30.
        default_ef4: Default IPCC emission factor EF4 for indirect N2O
            from atmospheric deposition of N volatilised from managed soils
            (kg N2O-N/kg NH3-N + NOx-N). IPCC 2019 default 0.01.
        default_ef5: Default IPCC emission factor EF5 for indirect N2O
            from N leaching/runoff from managed soils (kg N2O-N/kg N
            leached). IPCC 2019 default 0.0075.
        default_limestone_ef: Default CO2 emission factor for limestone
            (CaCO3) application to soils. IPCC default 0.12 (12% of
            limestone mass is emitted as CO2).
        default_dolomite_ef: Default CO2 emission factor for dolomite
            (CaMg(CO3)2) application to soils. IPCC default 0.13 (13% of
            dolomite mass is emitted as CO2).
        default_urea_ef: Default CO2 emission factor for urea (CO(NH2)2)
            application to soils. IPCC default 0.20 (20% of urea mass
            is emitted as CO2).
        default_rice_baseline_ef: Default baseline CH4 emission factor for
            rice cultivation under continuously flooded conditions without
            organic amendments (kg CH4/ha/day). IPCC default 1.30.
        default_rice_cultivation_days: Default number of days in the rice
            cultivation period. Typical range 90-150 days depending on
            variety and region. Default 120 days.
        default_water_regime: Default water management regime for rice
            paddies. Determines the scaling factor applied to the baseline
            EF. Valid: continuously_flooded, intermittently_flooded_single,
            intermittently_flooded_multiple, rainfed_regular,
            rainfed_drought_prone, deep_water, upland.
        default_combustion_factor: Default combustion factor representing
            the fraction of crop residue biomass that is actually oxidised
            during field burning. IPCC default 0.80 for most crop types.
        default_burn_fraction: Default fraction of total crop residues
            that are burned in the field. Varies widely by region and
            practice. Default 0.25 as a conservative global estimate.
        monte_carlo_iterations: Number of Monte Carlo simulation iterations
            for uncertainty quantification. Higher values yield more
            precise confidence intervals at the cost of computation time.
        monte_carlo_seed: Random seed for Monte Carlo reproducibility.
            Set to 0 for non-deterministic runs.
        confidence_levels: Comma-separated confidence level percentages for
            uncertainty analysis output (e.g. "90,95,99").
        enable_enteric: When True, the enteric fermentation engine is
            available for CH4 emission calculations from livestock
            digestive processes (ruminants and non-ruminants).
        enable_manure: When True, the manure management engine is available
            for CH4 and N2O emission calculations from livestock manure
            storage, treatment, and application systems.
        enable_soils: When True, the agricultural soils engine is available
            for direct and indirect N2O calculations from managed soils,
            including synthetic fertiliser, organic amendments, crop
            residues, and soil organic matter mineralisation.
        enable_rice: When True, the rice cultivation engine is available
            for CH4 emission calculations from flooded rice paddies under
            various water management regimes.
        enable_field_burning: When True, the field burning engine is
            available for CH4, N2O, CO, and NOx calculations from burning
            of crop residues in agricultural fields.
        enable_compliance_checking: When True, calculation results are
            automatically checked against applicable regulatory frameworks
            (IPCC, GHG Protocol, CSRD, UNFCCC, National Inventory).
        enable_uncertainty: When True, Monte Carlo or analytical uncertainty
            quantification is available for agricultural emission
            calculations.
        enable_provenance: Compute and store SHA-256 provenance hashes for
            all farm registrations, livestock records, crop data, calculation
            steps, and batch operations.
        enable_metrics: When True, Prometheus metrics are exported under
            the ``gl_ag_`` prefix.
        separate_biogenic_ch4: When True, biogenic and fossil CH4 are
            tracked and reported separately. Relevant for AR6 GWP
            differences (27.2 biogenic vs 29.8 fossil for 100-year) and
            UNFCCC reporting where agricultural CH4 is reported as
            biogenic.
        max_farms: Maximum number of farm registrations allowed in the
            system simultaneously.
        max_livestock_records: Maximum number of livestock population
            records that can be tracked per tenant.
        cache_ttl_seconds: TTL (seconds) for cached emission factor and
            IPCC default table lookups in Redis.
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
            as large batch calculations and multi-year projections are
            processed asynchronously via a background task queue.
        health_check_interval: Interval in seconds between periodic health
            checks of database and cache connections.
        genesis_hash: Anchor string used as the root of every provenance
            chain. Uniquely identifies the Agricultural Emissions agent.
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
    default_calculation_method: str = "IPCC_TIER_1"
    default_emission_factor_source: str = "IPCC_2019"

    # -- Calculation precision -----------------------------------------------
    decimal_precision: int = 8

    # -- IPCC climate zone ---------------------------------------------------
    default_climate_zone: str = "COOL_TEMPERATE_WET"

    # -- Enteric fermentation parameters -------------------------------------
    default_ym_pct: float = 6.5
    default_de_pct: float = 65.0
    default_cfi_dairy: float = 0.386
    default_cfi_non_dairy: float = 0.322
    default_activity_coefficient: float = 0.0
    default_pregnancy_factor: float = 0.10

    # -- Manure management parameters ----------------------------------------
    default_manure_mcf: float = 0.10
    default_manure_bo_dairy: float = 0.24
    default_manure_bo_swine: float = 0.48
    default_vs_dairy: float = 5.4
    default_temperature_c: float = 15.0

    # -- Agricultural soils parameters ---------------------------------------
    default_ef1: float = 0.01
    default_ef2_cg: float = 8.0
    default_ef2_f: float = 2.5
    default_ef3_prp_cattle: float = 0.02
    default_ef3_prp_other: float = 0.01
    default_frac_gasf: float = 0.10
    default_frac_gasm: float = 0.20
    default_frac_leach: float = 0.30

    # -- Indirect N2O parameters ---------------------------------------------
    default_ef4: float = 0.01
    default_ef5: float = 0.0075

    # -- Liming and urea parameters ------------------------------------------
    default_limestone_ef: float = 0.12
    default_dolomite_ef: float = 0.13
    default_urea_ef: float = 0.20

    # -- Rice cultivation parameters -----------------------------------------
    default_rice_baseline_ef: float = 1.30
    default_rice_cultivation_days: int = 120
    default_water_regime: str = "continuously_flooded"

    # -- Field burning parameters --------------------------------------------
    default_combustion_factor: float = 0.80
    default_burn_fraction: float = 0.25

    # -- Monte Carlo uncertainty analysis ------------------------------------
    monte_carlo_iterations: int = 5_000
    monte_carlo_seed: int = 42
    confidence_levels: str = "90,95,99"

    # -- Feature toggles -----------------------------------------------------
    enable_enteric: bool = True
    enable_manure: bool = True
    enable_soils: bool = True
    enable_rice: bool = True
    enable_field_burning: bool = True
    enable_compliance_checking: bool = True
    enable_uncertainty: bool = True
    enable_provenance: bool = True
    enable_metrics: bool = True
    separate_biogenic_ch4: bool = True

    # -- Farm and livestock capacity limits ----------------------------------
    max_farms: int = 10_000
    max_livestock_records: int = 100_000

    # -- Cache ---------------------------------------------------------------
    cache_ttl_seconds: int = 3600

    # -- API settings --------------------------------------------------------
    api_prefix: str = "/api/v1/agricultural-emissions"
    api_max_page_size: int = 100
    api_default_page_size: int = 20

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Performance tuning --------------------------------------------------
    worker_threads: int = 4
    enable_background_tasks: bool = True
    health_check_interval: int = 30

    # -- Provenance tracking -------------------------------------------------
    genesis_hash: str = "GL-MRV-X-008-AGRICULTURAL-EMISSIONS-GENESIS"

    # -- Auth and tracing ----------------------------------------------------
    enable_auth: bool = True
    enable_tracing: bool = True

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialisation.

        Performs range checks on all numeric fields, enumeration checks on
        string fields (GWP source, calculation method, EF source, climate
        zone, water regime, log level), and normalisation of values (e.g.
        log_level to uppercase, climate_zone to uppercase, water_regime
        to lowercase).

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

        # -- Climate zone ----------------------------------------------------
        normalised_cz = self.default_climate_zone.upper()
        if normalised_cz not in _VALID_CLIMATE_ZONES:
            errors.append(
                f"default_climate_zone must be one of "
                f"{sorted(_VALID_CLIMATE_ZONES)}, "
                f"got '{self.default_climate_zone}'"
            )
        else:
            self.default_climate_zone = normalised_cz

        # -- Water regime ----------------------------------------------------
        normalised_wr = self.default_water_regime.lower()
        if normalised_wr not in _VALID_WATER_REGIMES:
            errors.append(
                f"default_water_regime must be one of "
                f"{sorted(_VALID_WATER_REGIMES)}, "
                f"got '{self.default_water_regime}'"
            )
        else:
            self.default_water_regime = normalised_wr

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

        # -- Enteric fermentation parameters ---------------------------------
        if self.default_ym_pct < 0.0:
            errors.append(
                f"default_ym_pct must be >= 0.0, "
                f"got {self.default_ym_pct}"
            )
        if self.default_ym_pct > 25.0:
            errors.append(
                f"default_ym_pct must be <= 25.0, "
                f"got {self.default_ym_pct}"
            )

        if self.default_de_pct < 0.0:
            errors.append(
                f"default_de_pct must be >= 0.0, "
                f"got {self.default_de_pct}"
            )
        if self.default_de_pct > 100.0:
            errors.append(
                f"default_de_pct must be <= 100.0, "
                f"got {self.default_de_pct}"
            )

        if self.default_cfi_dairy < 0.0:
            errors.append(
                f"default_cfi_dairy must be >= 0.0, "
                f"got {self.default_cfi_dairy}"
            )
        if self.default_cfi_dairy > 2.0:
            errors.append(
                f"default_cfi_dairy must be <= 2.0, "
                f"got {self.default_cfi_dairy}"
            )

        if self.default_cfi_non_dairy < 0.0:
            errors.append(
                f"default_cfi_non_dairy must be >= 0.0, "
                f"got {self.default_cfi_non_dairy}"
            )
        if self.default_cfi_non_dairy > 2.0:
            errors.append(
                f"default_cfi_non_dairy must be <= 2.0, "
                f"got {self.default_cfi_non_dairy}"
            )

        if self.default_activity_coefficient < 0.0:
            errors.append(
                f"default_activity_coefficient must be >= 0.0, "
                f"got {self.default_activity_coefficient}"
            )
        if self.default_activity_coefficient > 1.0:
            errors.append(
                f"default_activity_coefficient must be <= 1.0, "
                f"got {self.default_activity_coefficient}"
            )

        if self.default_pregnancy_factor < 0.0:
            errors.append(
                f"default_pregnancy_factor must be >= 0.0, "
                f"got {self.default_pregnancy_factor}"
            )
        if self.default_pregnancy_factor > 1.0:
            errors.append(
                f"default_pregnancy_factor must be <= 1.0, "
                f"got {self.default_pregnancy_factor}"
            )

        # -- Manure management parameters ------------------------------------
        if self.default_manure_mcf < 0.0:
            errors.append(
                f"default_manure_mcf must be >= 0.0, "
                f"got {self.default_manure_mcf}"
            )
        if self.default_manure_mcf > 1.0:
            errors.append(
                f"default_manure_mcf must be <= 1.0, "
                f"got {self.default_manure_mcf}"
            )

        if self.default_manure_bo_dairy <= 0.0:
            errors.append(
                f"default_manure_bo_dairy must be > 0.0, "
                f"got {self.default_manure_bo_dairy}"
            )
        if self.default_manure_bo_dairy > 1.0:
            errors.append(
                f"default_manure_bo_dairy must be <= 1.0, "
                f"got {self.default_manure_bo_dairy}"
            )

        if self.default_manure_bo_swine <= 0.0:
            errors.append(
                f"default_manure_bo_swine must be > 0.0, "
                f"got {self.default_manure_bo_swine}"
            )
        if self.default_manure_bo_swine > 1.0:
            errors.append(
                f"default_manure_bo_swine must be <= 1.0, "
                f"got {self.default_manure_bo_swine}"
            )

        if self.default_vs_dairy <= 0.0:
            errors.append(
                f"default_vs_dairy must be > 0.0, "
                f"got {self.default_vs_dairy}"
            )
        if self.default_vs_dairy > 50.0:
            errors.append(
                f"default_vs_dairy must be <= 50.0, "
                f"got {self.default_vs_dairy}"
            )

        if self.default_temperature_c < -60.0:
            errors.append(
                f"default_temperature_c must be >= -60.0, "
                f"got {self.default_temperature_c}"
            )
        if self.default_temperature_c > 60.0:
            errors.append(
                f"default_temperature_c must be <= 60.0, "
                f"got {self.default_temperature_c}"
            )

        # -- Agricultural soils parameters -----------------------------------
        for field_name, value in [
            ("default_ef1", self.default_ef1),
            ("default_ef3_prp_cattle", self.default_ef3_prp_cattle),
            ("default_ef3_prp_other", self.default_ef3_prp_other),
            ("default_frac_gasf", self.default_frac_gasf),
            ("default_frac_gasm", self.default_frac_gasm),
            ("default_frac_leach", self.default_frac_leach),
        ]:
            if value < 0.0:
                errors.append(
                    f"{field_name} must be >= 0.0, got {value}"
                )
            if value > 1.0:
                errors.append(
                    f"{field_name} must be <= 1.0, got {value}"
                )

        if self.default_ef2_cg < 0.0:
            errors.append(
                f"default_ef2_cg must be >= 0.0, "
                f"got {self.default_ef2_cg}"
            )
        if self.default_ef2_cg > 50.0:
            errors.append(
                f"default_ef2_cg must be <= 50.0, "
                f"got {self.default_ef2_cg}"
            )

        if self.default_ef2_f < 0.0:
            errors.append(
                f"default_ef2_f must be >= 0.0, "
                f"got {self.default_ef2_f}"
            )
        if self.default_ef2_f > 50.0:
            errors.append(
                f"default_ef2_f must be <= 50.0, "
                f"got {self.default_ef2_f}"
            )

        # -- Indirect N2O parameters -----------------------------------------
        for field_name, value in [
            ("default_ef4", self.default_ef4),
            ("default_ef5", self.default_ef5),
        ]:
            if value < 0.0:
                errors.append(
                    f"{field_name} must be >= 0.0, got {value}"
                )
            if value > 1.0:
                errors.append(
                    f"{field_name} must be <= 1.0, got {value}"
                )

        # -- Liming and urea parameters --------------------------------------
        for field_name, value in [
            ("default_limestone_ef", self.default_limestone_ef),
            ("default_dolomite_ef", self.default_dolomite_ef),
            ("default_urea_ef", self.default_urea_ef),
        ]:
            if value < 0.0:
                errors.append(
                    f"{field_name} must be >= 0.0, got {value}"
                )
            if value > 1.0:
                errors.append(
                    f"{field_name} must be <= 1.0, got {value}"
                )

        # -- Rice cultivation parameters -------------------------------------
        if self.default_rice_baseline_ef < 0.0:
            errors.append(
                f"default_rice_baseline_ef must be >= 0.0, "
                f"got {self.default_rice_baseline_ef}"
            )
        if self.default_rice_baseline_ef > 20.0:
            errors.append(
                f"default_rice_baseline_ef must be <= 20.0, "
                f"got {self.default_rice_baseline_ef}"
            )

        if self.default_rice_cultivation_days <= 0:
            errors.append(
                f"default_rice_cultivation_days must be > 0, "
                f"got {self.default_rice_cultivation_days}"
            )
        if self.default_rice_cultivation_days > 365:
            errors.append(
                f"default_rice_cultivation_days must be <= 365, "
                f"got {self.default_rice_cultivation_days}"
            )

        # -- Field burning parameters ----------------------------------------
        for field_name, value in [
            ("default_combustion_factor", self.default_combustion_factor),
            ("default_burn_fraction", self.default_burn_fraction),
        ]:
            if value < 0.0:
                errors.append(
                    f"{field_name} must be >= 0.0, got {value}"
                )
            if value > 1.0:
                errors.append(
                    f"{field_name} must be <= 1.0, got {value}"
                )

        # -- Capacity limits -------------------------------------------------
        for field_name, value, upper in [
            ("max_batch_size", self.max_batch_size, 100_000),
            ("max_farms", self.max_farms, 100_000),
            ("max_livestock_records", self.max_livestock_records, 1_000_000),
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
                "AgriculturalEmissionsConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "AgriculturalEmissionsConfig validated successfully: "
            "enabled=%s, gwp_source=%s, method=%s, "
            "ef_source=%s, decimal_precision=%d, "
            "climate_zone=%s, "
            "ym_pct=%.1f, de_pct=%.1f, cfi_dairy=%.3f, "
            "cfi_non_dairy=%.3f, activity_coeff=%.2f, "
            "pregnancy_factor=%.2f, "
            "manure_mcf=%.2f, bo_dairy=%.2f, "
            "bo_swine=%.2f, vs_dairy=%.1f, "
            "temperature_c=%.1f, "
            "ef1=%.4f, ef2_cg=%.1f, ef2_f=%.1f, "
            "ef3_prp_cattle=%.4f, ef3_prp_other=%.4f, "
            "frac_gasf=%.2f, frac_gasm=%.2f, "
            "frac_leach=%.2f, "
            "ef4=%.4f, ef5=%.4f, "
            "limestone_ef=%.2f, dolomite_ef=%.2f, "
            "urea_ef=%.2f, "
            "rice_baseline_ef=%.2f, rice_days=%d, "
            "water_regime=%s, "
            "combustion_factor=%.2f, burn_fraction=%.2f, "
            "max_batch_size=%d, monte_carlo_iterations=%d, "
            "confidence_levels=%s, "
            "enteric=%s, manure=%s, soils=%s, "
            "rice=%s, field_burning=%s, "
            "separate_biogenic_ch4=%s, "
            "compliance=%s, uncertainty=%s, "
            "provenance=%s, metrics=%s",
            self.enabled,
            self.default_gwp_source,
            self.default_calculation_method,
            self.default_emission_factor_source,
            self.decimal_precision,
            self.default_climate_zone,
            self.default_ym_pct,
            self.default_de_pct,
            self.default_cfi_dairy,
            self.default_cfi_non_dairy,
            self.default_activity_coefficient,
            self.default_pregnancy_factor,
            self.default_manure_mcf,
            self.default_manure_bo_dairy,
            self.default_manure_bo_swine,
            self.default_vs_dairy,
            self.default_temperature_c,
            self.default_ef1,
            self.default_ef2_cg,
            self.default_ef2_f,
            self.default_ef3_prp_cattle,
            self.default_ef3_prp_other,
            self.default_frac_gasf,
            self.default_frac_gasm,
            self.default_frac_leach,
            self.default_ef4,
            self.default_ef5,
            self.default_limestone_ef,
            self.default_dolomite_ef,
            self.default_urea_ef,
            self.default_rice_baseline_ef,
            self.default_rice_cultivation_days,
            self.default_water_regime,
            self.default_combustion_factor,
            self.default_burn_fraction,
            self.max_batch_size,
            self.monte_carlo_iterations,
            self.confidence_levels,
            self.enable_enteric,
            self.enable_manure,
            self.enable_soils,
            self.enable_rice,
            self.enable_field_burning,
            self.separate_biogenic_ch4,
            self.enable_compliance_checking,
            self.enable_uncertainty,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> AgriculturalEmissionsConfig:
        """Build an AgriculturalEmissionsConfig from environment variables.

        Every field can be overridden via
        ``GL_AGRICULTURAL_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated AgriculturalEmissionsConfig instance, validated via
            ``__post_init__``.

        Example:
            >>> import os
            >>> os.environ["GL_AGRICULTURAL_DEFAULT_CALCULATION_METHOD"] = "IPCC_TIER_2"
            >>> cfg = AgriculturalEmissionsConfig.from_env()
            >>> cfg.default_calculation_method
            'IPCC_TIER_2'
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
            # Climate zone
            default_climate_zone=_str(
                "DEFAULT_CLIMATE_ZONE",
                cls.default_climate_zone,
            ),
            # Enteric fermentation parameters
            default_ym_pct=_float(
                "DEFAULT_YM_PCT",
                cls.default_ym_pct,
            ),
            default_de_pct=_float(
                "DEFAULT_DE_PCT",
                cls.default_de_pct,
            ),
            default_cfi_dairy=_float(
                "DEFAULT_CFI_DAIRY",
                cls.default_cfi_dairy,
            ),
            default_cfi_non_dairy=_float(
                "DEFAULT_CFI_NON_DAIRY",
                cls.default_cfi_non_dairy,
            ),
            default_activity_coefficient=_float(
                "DEFAULT_ACTIVITY_COEFFICIENT",
                cls.default_activity_coefficient,
            ),
            default_pregnancy_factor=_float(
                "DEFAULT_PREGNANCY_FACTOR",
                cls.default_pregnancy_factor,
            ),
            # Manure management parameters
            default_manure_mcf=_float(
                "DEFAULT_MANURE_MCF",
                cls.default_manure_mcf,
            ),
            default_manure_bo_dairy=_float(
                "DEFAULT_MANURE_BO_DAIRY",
                cls.default_manure_bo_dairy,
            ),
            default_manure_bo_swine=_float(
                "DEFAULT_MANURE_BO_SWINE",
                cls.default_manure_bo_swine,
            ),
            default_vs_dairy=_float(
                "DEFAULT_VS_DAIRY",
                cls.default_vs_dairy,
            ),
            default_temperature_c=_float(
                "DEFAULT_TEMPERATURE_C",
                cls.default_temperature_c,
            ),
            # Agricultural soils parameters
            default_ef1=_float(
                "DEFAULT_EF1",
                cls.default_ef1,
            ),
            default_ef2_cg=_float(
                "DEFAULT_EF2_CG",
                cls.default_ef2_cg,
            ),
            default_ef2_f=_float(
                "DEFAULT_EF2_F",
                cls.default_ef2_f,
            ),
            default_ef3_prp_cattle=_float(
                "DEFAULT_EF3_PRP_CATTLE",
                cls.default_ef3_prp_cattle,
            ),
            default_ef3_prp_other=_float(
                "DEFAULT_EF3_PRP_OTHER",
                cls.default_ef3_prp_other,
            ),
            default_frac_gasf=_float(
                "DEFAULT_FRAC_GASF",
                cls.default_frac_gasf,
            ),
            default_frac_gasm=_float(
                "DEFAULT_FRAC_GASM",
                cls.default_frac_gasm,
            ),
            default_frac_leach=_float(
                "DEFAULT_FRAC_LEACH",
                cls.default_frac_leach,
            ),
            # Indirect N2O parameters
            default_ef4=_float(
                "DEFAULT_EF4",
                cls.default_ef4,
            ),
            default_ef5=_float(
                "DEFAULT_EF5",
                cls.default_ef5,
            ),
            # Liming and urea parameters
            default_limestone_ef=_float(
                "DEFAULT_LIMESTONE_EF",
                cls.default_limestone_ef,
            ),
            default_dolomite_ef=_float(
                "DEFAULT_DOLOMITE_EF",
                cls.default_dolomite_ef,
            ),
            default_urea_ef=_float(
                "DEFAULT_UREA_EF",
                cls.default_urea_ef,
            ),
            # Rice cultivation parameters
            default_rice_baseline_ef=_float(
                "DEFAULT_RICE_BASELINE_EF",
                cls.default_rice_baseline_ef,
            ),
            default_rice_cultivation_days=_int(
                "DEFAULT_RICE_CULTIVATION_DAYS",
                cls.default_rice_cultivation_days,
            ),
            default_water_regime=_str(
                "DEFAULT_WATER_REGIME",
                cls.default_water_regime,
            ),
            # Field burning parameters
            default_combustion_factor=_float(
                "DEFAULT_COMBUSTION_FACTOR",
                cls.default_combustion_factor,
            ),
            default_burn_fraction=_float(
                "DEFAULT_BURN_FRACTION",
                cls.default_burn_fraction,
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
            enable_enteric=_bool(
                "ENABLE_ENTERIC",
                cls.enable_enteric,
            ),
            enable_manure=_bool(
                "ENABLE_MANURE",
                cls.enable_manure,
            ),
            enable_soils=_bool(
                "ENABLE_SOILS",
                cls.enable_soils,
            ),
            enable_rice=_bool(
                "ENABLE_RICE",
                cls.enable_rice,
            ),
            enable_field_burning=_bool(
                "ENABLE_FIELD_BURNING",
                cls.enable_field_burning,
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
            separate_biogenic_ch4=_bool(
                "SEPARATE_BIOGENIC_CH4",
                cls.separate_biogenic_ch4,
            ),
            # Farm and livestock capacity limits
            max_farms=_int(
                "MAX_FARMS",
                cls.max_farms,
            ),
            max_livestock_records=_int(
                "MAX_LIVESTOCK_RECORDS",
                cls.max_livestock_records,
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
            "AgriculturalEmissionsConfig loaded: "
            "enabled=%s, gwp_source=%s, method=%s, "
            "ef_source=%s, decimal_precision=%d, "
            "climate_zone=%s, "
            "ym_pct=%.1f, de_pct=%.1f, cfi_dairy=%.3f, "
            "cfi_non_dairy=%.3f, activity_coeff=%.2f, "
            "pregnancy_factor=%.2f, "
            "manure_mcf=%.2f, bo_dairy=%.2f, "
            "bo_swine=%.2f, vs_dairy=%.1f, "
            "temperature_c=%.1f, "
            "ef1=%.4f, ef2_cg=%.1f, ef2_f=%.1f, "
            "ef3_prp_cattle=%.4f, ef3_prp_other=%.4f, "
            "frac_gasf=%.2f, frac_gasm=%.2f, "
            "frac_leach=%.2f, "
            "ef4=%.4f, ef5=%.4f, "
            "limestone_ef=%.2f, dolomite_ef=%.2f, "
            "urea_ef=%.2f, "
            "rice_baseline_ef=%.2f, rice_days=%d, "
            "water_regime=%s, "
            "combustion_factor=%.2f, burn_fraction=%.2f, "
            "max_batch_size=%d, max_farms=%d, "
            "max_livestock_records=%d, "
            "monte_carlo_iterations=%d, confidence_levels=%s, "
            "enteric=%s, manure=%s, soils=%s, "
            "rice=%s, field_burning=%s, "
            "separate_biogenic_ch4=%s, "
            "compliance=%s, uncertainty=%s, "
            "provenance=%s, metrics=%s, "
            "cache_ttl=%ds, worker_threads=%d, "
            "background_tasks=%s, health_check=%ds, "
            "auth=%s, tracing=%s",
            config.enabled,
            config.default_gwp_source,
            config.default_calculation_method,
            config.default_emission_factor_source,
            config.decimal_precision,
            config.default_climate_zone,
            config.default_ym_pct,
            config.default_de_pct,
            config.default_cfi_dairy,
            config.default_cfi_non_dairy,
            config.default_activity_coefficient,
            config.default_pregnancy_factor,
            config.default_manure_mcf,
            config.default_manure_bo_dairy,
            config.default_manure_bo_swine,
            config.default_vs_dairy,
            config.default_temperature_c,
            config.default_ef1,
            config.default_ef2_cg,
            config.default_ef2_f,
            config.default_ef3_prp_cattle,
            config.default_ef3_prp_other,
            config.default_frac_gasf,
            config.default_frac_gasm,
            config.default_frac_leach,
            config.default_ef4,
            config.default_ef5,
            config.default_limestone_ef,
            config.default_dolomite_ef,
            config.default_urea_ef,
            config.default_rice_baseline_ef,
            config.default_rice_cultivation_days,
            config.default_water_regime,
            config.default_combustion_factor,
            config.default_burn_fraction,
            config.max_batch_size,
            config.max_farms,
            config.max_livestock_records,
            config.monte_carlo_iterations,
            config.confidence_levels,
            config.enable_enteric,
            config.enable_manure,
            config.enable_soils,
            config.enable_rice,
            config.enable_field_burning,
            config.separate_biogenic_ch4,
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
            >>> cfg = AgriculturalEmissionsConfig()
            >>> d = cfg.to_dict()
            >>> d["default_calculation_method"]
            'IPCC_TIER_1'
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
            "default_calculation_method": self.default_calculation_method,
            "default_emission_factor_source": self.default_emission_factor_source,
            # -- Calculation precision --------------------------------------
            "decimal_precision": self.decimal_precision,
            # -- Climate zone -----------------------------------------------
            "default_climate_zone": self.default_climate_zone,
            # -- Enteric fermentation parameters ----------------------------
            "default_ym_pct": self.default_ym_pct,
            "default_de_pct": self.default_de_pct,
            "default_cfi_dairy": self.default_cfi_dairy,
            "default_cfi_non_dairy": self.default_cfi_non_dairy,
            "default_activity_coefficient": self.default_activity_coefficient,
            "default_pregnancy_factor": self.default_pregnancy_factor,
            # -- Manure management parameters -------------------------------
            "default_manure_mcf": self.default_manure_mcf,
            "default_manure_bo_dairy": self.default_manure_bo_dairy,
            "default_manure_bo_swine": self.default_manure_bo_swine,
            "default_vs_dairy": self.default_vs_dairy,
            "default_temperature_c": self.default_temperature_c,
            # -- Agricultural soils parameters ------------------------------
            "default_ef1": self.default_ef1,
            "default_ef2_cg": self.default_ef2_cg,
            "default_ef2_f": self.default_ef2_f,
            "default_ef3_prp_cattle": self.default_ef3_prp_cattle,
            "default_ef3_prp_other": self.default_ef3_prp_other,
            "default_frac_gasf": self.default_frac_gasf,
            "default_frac_gasm": self.default_frac_gasm,
            "default_frac_leach": self.default_frac_leach,
            # -- Indirect N2O parameters ------------------------------------
            "default_ef4": self.default_ef4,
            "default_ef5": self.default_ef5,
            # -- Liming and urea parameters ---------------------------------
            "default_limestone_ef": self.default_limestone_ef,
            "default_dolomite_ef": self.default_dolomite_ef,
            "default_urea_ef": self.default_urea_ef,
            # -- Rice cultivation parameters --------------------------------
            "default_rice_baseline_ef": self.default_rice_baseline_ef,
            "default_rice_cultivation_days": self.default_rice_cultivation_days,
            "default_water_regime": self.default_water_regime,
            # -- Field burning parameters -----------------------------------
            "default_combustion_factor": self.default_combustion_factor,
            "default_burn_fraction": self.default_burn_fraction,
            # -- Monte Carlo uncertainty analysis ---------------------------
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "monte_carlo_seed": self.monte_carlo_seed,
            "confidence_levels": self.confidence_levels,
            # -- Feature toggles --------------------------------------------
            "enable_enteric": self.enable_enteric,
            "enable_manure": self.enable_manure,
            "enable_soils": self.enable_soils,
            "enable_rice": self.enable_rice,
            "enable_field_burning": self.enable_field_burning,
            "enable_compliance_checking": self.enable_compliance_checking,
            "enable_uncertainty": self.enable_uncertainty,
            "enable_provenance": self.enable_provenance,
            "enable_metrics": self.enable_metrics,
            "separate_biogenic_ch4": self.separate_biogenic_ch4,
            # -- Farm and livestock capacity limits -------------------------
            "max_farms": self.max_farms,
            "max_livestock_records": self.max_livestock_records,
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
        return f"AgriculturalEmissionsConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------


class _AgriculturalEmissionsConfigHolder:
    """Thread-safe singleton holder for AgriculturalEmissionsConfig.

    Uses double-checked locking with a threading.Lock to ensure
    exactly one AgriculturalEmissionsConfig instance is created from
    environment variables. Subsequent calls to ``get()`` return the
    cached instance without lock contention.

    Attributes:
        _instance: Cached AgriculturalEmissionsConfig or None.
        _lock: Threading lock for initialization.
    """

    _instance: Optional[AgriculturalEmissionsConfig] = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get(cls) -> AgriculturalEmissionsConfig:
        """Return the singleton AgriculturalEmissionsConfig, creating from env if needed.

        Uses double-checked locking for thread safety with minimal
        contention on the hot path.

        Returns:
            AgriculturalEmissionsConfig singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = AgriculturalEmissionsConfig.from_env()
        return cls._instance

    @classmethod
    def set(cls, config: AgriculturalEmissionsConfig) -> None:
        """Replace the singleton AgriculturalEmissionsConfig.

        Args:
            config: New AgriculturalEmissionsConfig to install as the singleton.
        """
        with cls._lock:
            cls._instance = config

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton to None for test teardown."""
        with cls._lock:
            cls._instance = None


def get_config() -> AgriculturalEmissionsConfig:
    """Return the singleton AgriculturalEmissionsConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_AGRICULTURAL_*`` environment variables via
    :meth:`AgriculturalEmissionsConfig.from_env`.

    Returns:
        AgriculturalEmissionsConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.default_calculation_method
        'IPCC_TIER_1'
    """
    return _AgriculturalEmissionsConfigHolder.get()


def set_config(config: AgriculturalEmissionsConfig) -> None:
    """Replace the singleton AgriculturalEmissionsConfig.

    Primarily intended for testing and dependency injection scenarios
    where a custom configuration must be supplied without relying on
    environment variables.

    Args:
        config: New :class:`AgriculturalEmissionsConfig` to install as the singleton.

    Example:
        >>> cfg = AgriculturalEmissionsConfig(default_calculation_method="IPCC_TIER_2")
        >>> set_config(cfg)
        >>> assert get_config().default_calculation_method == "IPCC_TIER_2"
    """
    _AgriculturalEmissionsConfigHolder.set(config)
    logger.info(
        "AgriculturalEmissionsConfig replaced programmatically: "
        "enabled=%s, gwp_source=%s, method=%s, "
        "ef_source=%s, max_batch_size=%d, "
        "monte_carlo_iterations=%d, "
        "enteric=%s, manure=%s, soils=%s, "
        "rice=%s, field_burning=%s, "
        "separate_biogenic_ch4=%s",
        config.enabled,
        config.default_gwp_source,
        config.default_calculation_method,
        config.default_emission_factor_source,
        config.max_batch_size,
        config.monte_carlo_iterations,
        config.enable_enteric,
        config.enable_manure,
        config.enable_soils,
        config.enable_rice,
        config.enable_field_burning,
        config.separate_biogenic_ch4,
    )


def reset_config() -> None:
    """Reset the singleton AgriculturalEmissionsConfig to ``None``.

    The next call to :func:`get_config` will re-read environment variables
    and construct a fresh instance. Intended for test teardown to prevent
    state leakage between test cases.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads GL_AGRICULTURAL_* env vars
    """
    _AgriculturalEmissionsConfigHolder.reset()
    logger.debug("AgriculturalEmissionsConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "AgriculturalEmissionsConfig",
    "get_config",
    "set_config",
    "reset_config",
]
