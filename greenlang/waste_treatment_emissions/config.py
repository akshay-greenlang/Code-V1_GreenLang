# -*- coding: utf-8 -*-
"""
Waste Treatment Emissions Agent Configuration - AGENT-MRV-007

Centralized configuration for the Waste Treatment Emissions Agent SDK covering:
- Database, cache, and connection defaults
- IPCC methodology defaults (GWP source, calculation method, EF source)
- Biological treatment parameters (composting, anaerobic digestion, MBT)
- Thermal treatment parameters (incineration, pyrolysis, gasification)
- Wastewater treatment parameters (BOD/COD-based CH4, N2O)
- Methane recovery parameters (collection, flaring, utilization)
- Energy recovery parameters (WtE efficiency, grid displacement)
- First Order Decay (FOD) model parameters (DOCf, MCF, k, F)
- Fossil vs biogenic CO2 separation control
- Decimal precision for emission calculations
- Capacity limits (facilities, waste streams, batches)
- Monte Carlo uncertainty analysis parameters
- Feature toggles (biological, thermal, wastewater, methane recovery,
  energy recovery, compliance checking, uncertainty, provenance, metrics)
- API configuration (prefix, pagination)
- Background task and worker thread settings

All settings can be overridden via environment variables with the
``GL_WASTE_TREATMENT_`` prefix (e.g.
``GL_WASTE_TREATMENT_DEFAULT_GWP_SOURCE``,
``GL_WASTE_TREATMENT_MAX_BATCH_SIZE``).

Environment Variable Reference (GL_WASTE_TREATMENT_ prefix):
    GL_WASTE_TREATMENT_ENABLED                       - Enable/disable agent
    GL_WASTE_TREATMENT_DATABASE_URL                  - PostgreSQL connection URL
    GL_WASTE_TREATMENT_REDIS_URL                     - Redis connection URL
    GL_WASTE_TREATMENT_MAX_BATCH_SIZE                - Maximum records per batch
    GL_WASTE_TREATMENT_DEFAULT_GWP_SOURCE            - Default GWP source (AR4/AR5/AR6)
    GL_WASTE_TREATMENT_DEFAULT_CALCULATION_METHOD    - Default method (IPCC_TIER_1/IPCC_TIER_2/
                                                       IPCC_TIER_3/MASS_BALANCE/DIRECT_MEASUREMENT/
                                                       FOD/SPEND_BASED)
    GL_WASTE_TREATMENT_DEFAULT_EMISSION_FACTOR_SOURCE - Default EF source
    GL_WASTE_TREATMENT_DECIMAL_PRECISION             - Decimal places for calculations
    GL_WASTE_TREATMENT_DEFAULT_DOC_F                 - Fraction of DOC that decomposes (DOCf)
    GL_WASTE_TREATMENT_DEFAULT_MCF                   - Methane correction factor
    GL_WASTE_TREATMENT_DEFAULT_F_CH4                 - CH4 fraction in landfill gas
    GL_WASTE_TREATMENT_DEFAULT_OXIDATION_FACTOR      - Oxidation factor (OF)
    GL_WASTE_TREATMENT_DEFAULT_COLLECTION_EFFICIENCY - Gas collection efficiency
    GL_WASTE_TREATMENT_DEFAULT_FLARE_EFFICIENCY      - Flare destruction removal efficiency
    GL_WASTE_TREATMENT_DEFAULT_UTILIZATION_EFFICIENCY - Energy utilization conversion efficiency
    GL_WASTE_TREATMENT_DEFAULT_CLIMATE_ZONE          - Default IPCC climate zone
    GL_WASTE_TREATMENT_DEFAULT_DECAY_RATE            - First Order Decay rate constant (1/yr)
    GL_WASTE_TREATMENT_COMPOSTING_CH4_EF             - Composting CH4 emission factor (g/kg)
    GL_WASTE_TREATMENT_COMPOSTING_N2O_EF             - Composting N2O emission factor (g/kg)
    GL_WASTE_TREATMENT_AD_CH4_EF                     - Anaerobic digestion CH4 EF (g/kg)
    GL_WASTE_TREATMENT_DEFAULT_DIGESTION_EFFICIENCY  - AD digestion efficiency
    GL_WASTE_TREATMENT_DEFAULT_BIOGAS_CH4_FRACTION   - CH4 fraction in biogas
    GL_WASTE_TREATMENT_DEFAULT_INCINERATION_OF       - Incineration oxidation factor
    GL_WASTE_TREATMENT_DEFAULT_ENERGY_RECOVERY_EFF   - WtE energy recovery efficiency
    GL_WASTE_TREATMENT_DEFAULT_OPEN_BURNING_OF       - Open burning oxidation factor
    GL_WASTE_TREATMENT_DEFAULT_BOD_CH4_CAPACITY      - BOD max CH4 producing capacity
    GL_WASTE_TREATMENT_DEFAULT_COD_CH4_CAPACITY      - COD max CH4 producing capacity
    GL_WASTE_TREATMENT_DEFAULT_WW_MCF                - Wastewater methane correction factor
    GL_WASTE_TREATMENT_DEFAULT_N2O_EF_PLANT          - Wastewater plant N2O EF
    GL_WASTE_TREATMENT_DEFAULT_N2O_EF_EFFLUENT       - Effluent N2O EF
    GL_WASTE_TREATMENT_MONTE_CARLO_ITERATIONS        - Monte Carlo simulation iterations
    GL_WASTE_TREATMENT_MONTE_CARLO_SEED              - Random seed for reproducibility
    GL_WASTE_TREATMENT_CONFIDENCE_LEVELS             - Comma-separated confidence levels
    GL_WASTE_TREATMENT_ENABLE_BIOLOGICAL             - Enable biological treatment engine
    GL_WASTE_TREATMENT_ENABLE_THERMAL                - Enable thermal treatment engine
    GL_WASTE_TREATMENT_ENABLE_WASTEWATER             - Enable wastewater treatment engine
    GL_WASTE_TREATMENT_ENABLE_METHANE_RECOVERY       - Enable methane recovery tracking
    GL_WASTE_TREATMENT_ENABLE_ENERGY_RECOVERY        - Enable energy recovery credits
    GL_WASTE_TREATMENT_ENABLE_COMPLIANCE_CHECKING    - Enable compliance checking
    GL_WASTE_TREATMENT_ENABLE_UNCERTAINTY            - Enable uncertainty quantification
    GL_WASTE_TREATMENT_ENABLE_PROVENANCE             - Enable SHA-256 provenance chain
    GL_WASTE_TREATMENT_ENABLE_METRICS                - Enable Prometheus metrics export
    GL_WASTE_TREATMENT_SEPARATE_BIOGENIC_CO2         - Separate biogenic/fossil CO2
    GL_WASTE_TREATMENT_MAX_FACILITIES                - Maximum treatment facilities
    GL_WASTE_TREATMENT_MAX_WASTE_STREAMS             - Maximum waste stream records
    GL_WASTE_TREATMENT_CACHE_TTL_SECONDS             - Cache time-to-live in seconds
    GL_WASTE_TREATMENT_API_PREFIX                    - REST API route prefix
    GL_WASTE_TREATMENT_API_MAX_PAGE_SIZE             - Maximum API page size
    GL_WASTE_TREATMENT_API_DEFAULT_PAGE_SIZE         - Default API page size
    GL_WASTE_TREATMENT_LOG_LEVEL                     - Logging level
    GL_WASTE_TREATMENT_WORKER_THREADS                - Worker thread pool size
    GL_WASTE_TREATMENT_ENABLE_BACKGROUND_TASKS       - Enable background task processing
    GL_WASTE_TREATMENT_HEALTH_CHECK_INTERVAL         - Health check interval seconds
    GL_WASTE_TREATMENT_GENESIS_HASH                  - Genesis anchor for provenance chain
    GL_WASTE_TREATMENT_ENABLE_AUTH                    - Enable authentication middleware
    GL_WASTE_TREATMENT_ENABLE_TRACING                - Enable OpenTelemetry tracing

Example:
    >>> from greenlang.waste_treatment_emissions.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_gwp_source, cfg.default_calculation_method)
    AR6 IPCC_TIER_2

    >>> # Override for testing
    >>> from greenlang.waste_treatment_emissions.config import set_config, reset_config
    >>> from greenlang.waste_treatment_emissions.config import WasteTreatmentConfig
    >>> set_config(WasteTreatmentConfig(default_calculation_method="FOD"))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-007 Waste Treatment Emissions (GL-MRV-SCOPE1-007)
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

_ENV_PREFIX = "GL_WASTE_TREATMENT_"

# ---------------------------------------------------------------------------
# Valid enumeration values for configuration validation
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

_VALID_GWP_SOURCES = frozenset({"AR4", "AR5", "AR6"})

_VALID_CALCULATION_METHODS = frozenset({
    "IPCC_TIER_1",
    "IPCC_TIER_2",
    "IPCC_TIER_3",
    "MASS_BALANCE",
    "DIRECT_MEASUREMENT",
    "FOD",
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
    "boreal_dry",
    "boreal_moist",
    "temperate_dry",
    "temperate_moist",
    "temperate",
    "tropical_dry",
    "tropical_moist",
    "tropical_wet",
})


# ---------------------------------------------------------------------------
# WasteTreatmentConfig
# ---------------------------------------------------------------------------


@dataclass
class WasteTreatmentConfig:
    """Complete configuration for the GreenLang Waste Treatment Emissions Agent SDK.

    Attributes are grouped by concern: feature flag, connections, methodology
    defaults, FOD model parameters, biological treatment defaults, thermal
    treatment defaults, wastewater treatment defaults, methane recovery
    defaults, energy recovery settings, fossil/biogenic separation,
    calculation precision, capacity limits, Monte Carlo parameters, feature
    toggles, API settings, performance tuning, and provenance.

    All attributes can be overridden via environment variables using the
    ``GL_WASTE_TREATMENT_`` prefix (e.g.
    ``GL_WASTE_TREATMENT_DEFAULT_CALCULATION_METHOD=FOD``).

    Attributes:
        enabled: Master switch to enable/disable the waste treatment emissions
            agent. When False, the agent will not process any requests.
        database_url: PostgreSQL connection URL for persistent storage of
            facility data, waste streams, treatment events, and calculation
            results.
        redis_url: Redis connection URL for caching emission factor lookups,
            DOC/MCF tables, and distributed locks.
        max_batch_size: Maximum number of waste treatment emission records
            that can be processed in a single batch calculation request.
        default_gwp_source: Default IPCC Assessment Report edition for
            Global Warming Potential values. Valid: AR4, AR5, AR6. AR6
            distinguishes fossil and biogenic CH4 GWP values.
        default_calculation_method: Default calculation methodology for
            waste treatment emissions. Valid: IPCC_TIER_1 (default EFs),
            IPCC_TIER_2 (country-specific data), IPCC_TIER_3 (facility-
            specific/CEMS), MASS_BALANCE (carbon in/out), DIRECT_MEASUREMENT
            (stack monitoring), FOD (first order decay for landfill),
            SPEND_BASED (waste sector EFs from DEFRA/EPA).
        default_emission_factor_source: Default authority for emission
            factors. Valid: IPCC_2006, IPCC_2019, EPA_AP42, DEFRA,
            ECOINVENT, NATIONAL_INVENTORY, CUSTOM.
        decimal_precision: Number of decimal places to retain in emission
            calculations for intermediate and final results.
        default_doc_f: Default fraction of degradable organic carbon (DOC)
            that actually decomposes under anaerobic conditions. IPCC default
            is 0.5 (50%). Used in FOD model and landfill calculations.
        default_mcf: Default methane correction factor. Depends on landfill
            management type: 1.0 for managed anaerobic, 0.5 for managed
            semi-aerobic, 0.8 for unmanaged deep (>5m), 0.4 for unmanaged
            shallow (<5m). Default 1.0 for managed anaerobic.
        default_f_ch4: Default fraction of methane in generated landfill
            gas by volume. IPCC default is 0.5 (50%).
        default_oxidation_factor: Default oxidation factor (OF) representing
            the fraction of CH4 oxidised in the soil cover of a landfill.
            IPCC Tier 1 default is 0.0 (no oxidation) for unmanaged and
            0.1 for managed landfills. Set to 0.1 for conservative default
            with managed sites. Use 1.0 for complete oxidation contexts
            like incineration.
        default_collection_efficiency: Default landfill gas or biogas
            collection system efficiency (fraction captured). Typical range
            0.50-0.95, default 0.75 represents a well-maintained system.
        default_flare_efficiency: Default flare destruction and removal
            efficiency (DRE). Modern enclosed flares achieve 0.995+; open
            flares typically 0.96-0.98. Default 0.98 is conservative.
        default_utilization_efficiency: Default energy utilization conversion
            efficiency for captured methane used in engines/turbines.
            Typical range 0.90-0.98. Default 0.95 for modern gas engines.
        default_climate_zone: Default IPCC climate zone for decay rate
            selection in FOD model. Valid: boreal_dry, boreal_moist,
            temperate_dry, temperate_moist, temperate, tropical_dry,
            tropical_moist, tropical_wet.
        default_decay_rate: Default first order decay rate constant k
            (1/year). IPCC defaults vary by climate and waste type:
            bulk waste k=0.05 (temperate). Set to 0.05 as default.
        composting_ch4_ef: Default composting methane emission factor in
            g CH4 per kg wet waste. IPCC 2019 default: 4.0 for well-managed,
            10.0 for poorly managed. Default 4.0 assumes well-managed.
        composting_n2o_ef: Default composting N2O emission factor in g N2O
            per kg wet waste. IPCC 2019 default: 0.24 for well-managed,
            0.6 for poorly managed. Default 0.24 assumes well-managed.
        ad_ch4_ef: Default anaerobic digestion methane emission factor in
            g CH4 per kg wet waste. IPCC 2019 default: 0.8 with gas
            recovery, 2.0 with venting. Default 0.8 assumes gas recovery.
        default_digestion_efficiency: Default anaerobic digestion process
            efficiency (fraction of VS converted). Range 0.50-0.90.
            Default 0.70 for mesophilic single-stage.
        default_biogas_ch4_fraction: Default methane fraction in biogas
            by volume. Range 0.50-0.70. Default 0.60 for typical AD.
        default_incineration_of: Default oxidation factor for modern
            incineration. 1.0 for mass burn (complete combustion), 0.95
            for batch type. Default 1.0 for modern grate incinerators.
        default_energy_recovery_efficiency: Default waste-to-energy gross
            energy recovery efficiency. Range 0.15-0.35. Default 0.25
            for a typical modern WtE plant.
        default_open_burning_of: Default oxidation factor for open burning
            of waste. IPCC default 0.58 for incomplete combustion.
        default_bod_ch4_capacity: Default maximum CH4 producing capacity
            based on BOD (kg CH4/kg BOD). IPCC default 0.6.
        default_cod_ch4_capacity: Default maximum CH4 producing capacity
            based on COD (kg CH4/kg COD). IPCC default 0.25.
        default_ww_mcf: Default methane correction factor for wastewater
            treatment system. Varies by system type (0.0 for well-managed
            aerobic to 0.8 for anaerobic). Default 0.3 for aerobic
            treatment that may be overloaded.
        default_n2o_ef_plant: Default N2O emission factor for wastewater
            treatment plant (kg N2O-N per kg N). IPCC default 0.016.
        default_n2o_ef_effluent: Default N2O emission factor for treated
            effluent discharge (kg N2O-N per kg N). IPCC default 0.005.
        monte_carlo_iterations: Number of Monte Carlo simulation iterations
            for uncertainty quantification. Higher values yield more
            precise confidence intervals at the cost of computation time.
        monte_carlo_seed: Random seed for Monte Carlo reproducibility.
            Set to 0 for non-deterministic runs.
        confidence_levels: Comma-separated confidence level percentages for
            uncertainty analysis output (e.g. "90,95,99").
        enable_biological: When True, biological treatment engines
            (composting, anaerobic digestion, MBT, vermicomposting) are
            available for emission calculations.
        enable_thermal: When True, thermal treatment engines (incineration,
            pyrolysis, gasification, open burning) are available for
            emission calculations.
        enable_wastewater: When True, on-site industrial wastewater
            treatment CH4 and N2O calculations are available.
        enable_methane_recovery: When True, methane recovery tracking
            including capture, flaring, utilization, and venting is
            available for all treatment methods that generate CH4.
        enable_energy_recovery: When True, energy recovery credit
            calculations are available for waste-to-energy and biogas
            utilization, computing displaced grid emissions.
        enable_compliance_checking: When True, calculation results are
            automatically checked against applicable regulatory frameworks
            (IPCC, GHG Protocol, CSRD, EPA, EU IED, DEFRA, ISO 14064).
        enable_uncertainty: When True, Monte Carlo or analytical uncertainty
            quantification is available for waste treatment emission
            calculations.
        enable_provenance: Compute and store SHA-256 provenance hashes for
            all facility registrations, waste stream definitions, treatment
            events, calculation steps, and batch operations.
        enable_metrics: When True, Prometheus metrics are exported under
            the ``gl_wt_`` prefix.
        separate_biogenic_co2: When True, fossil and biogenic CO2 are
            tracked and reported separately. Critical for EU ETS compliance
            where biogenic CO2 from biomass is excluded from cap-and-trade.
        max_facilities: Maximum number of treatment facility registrations
            allowed in the system simultaneously.
        max_waste_streams: Maximum number of waste stream records that
            can be tracked per tenant.
        cache_ttl_seconds: TTL (seconds) for cached emission factor and
            DOC/MCF table lookups in Redis.
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
            as large batch calculations and FOD multi-year projections are
            processed asynchronously via a background task queue.
        health_check_interval: Interval in seconds between periodic health
            checks of database and cache connections.
        genesis_hash: Anchor string used as the root of every provenance
            chain. Uniquely identifies the Waste Treatment Emissions agent.
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
    default_calculation_method: str = "IPCC_TIER_2"
    default_emission_factor_source: str = "IPCC_2019"

    # -- Calculation precision -----------------------------------------------
    decimal_precision: int = 8

    # -- FOD model parameters (landfill and anaerobic decomposition) ---------
    default_doc_f: float = 0.5
    default_mcf: float = 1.0
    default_f_ch4: float = 0.5
    default_oxidation_factor: float = 0.1
    default_collection_efficiency: float = 0.75
    default_flare_efficiency: float = 0.98
    default_utilization_efficiency: float = 0.95
    default_climate_zone: str = "temperate"
    default_decay_rate: float = 0.05

    # -- Biological treatment defaults ---------------------------------------
    composting_ch4_ef: float = 4.0
    composting_n2o_ef: float = 0.24
    ad_ch4_ef: float = 0.8
    default_digestion_efficiency: float = 0.70
    default_biogas_ch4_fraction: float = 0.60

    # -- Thermal treatment defaults ------------------------------------------
    default_incineration_of: float = 1.0
    default_energy_recovery_efficiency: float = 0.25
    default_open_burning_of: float = 0.58

    # -- Wastewater treatment defaults ---------------------------------------
    default_bod_ch4_capacity: float = 0.6
    default_cod_ch4_capacity: float = 0.25
    default_ww_mcf: float = 0.3
    default_n2o_ef_plant: float = 0.016
    default_n2o_ef_effluent: float = 0.005

    # -- Monte Carlo uncertainty analysis ------------------------------------
    monte_carlo_iterations: int = 5_000
    monte_carlo_seed: int = 42
    confidence_levels: str = "90,95,99"

    # -- Feature toggles -----------------------------------------------------
    enable_biological: bool = True
    enable_thermal: bool = True
    enable_wastewater: bool = True
    enable_methane_recovery: bool = True
    enable_energy_recovery: bool = True
    enable_compliance_checking: bool = True
    enable_uncertainty: bool = True
    enable_provenance: bool = True
    enable_metrics: bool = True
    separate_biogenic_co2: bool = True

    # -- Facility and waste stream capacity limits ---------------------------
    max_facilities: int = 10_000
    max_waste_streams: int = 50_000

    # -- Cache ---------------------------------------------------------------
    cache_ttl_seconds: int = 3600

    # -- API settings --------------------------------------------------------
    api_prefix: str = "/api/v1/waste-treatment-emissions"
    api_max_page_size: int = 100
    api_default_page_size: int = 20

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Performance tuning --------------------------------------------------
    worker_threads: int = 4
    enable_background_tasks: bool = True
    health_check_interval: int = 30

    # -- Provenance tracking -------------------------------------------------
    genesis_hash: str = "GL-MRV-X-007-WASTE-TREATMENT-EMISSIONS-GENESIS"

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
        zone, log level), and normalisation of values (e.g. log_level to
        uppercase, climate_zone to lowercase).

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
        normalised_cz = self.default_climate_zone.lower()
        if normalised_cz not in _VALID_CLIMATE_ZONES:
            errors.append(
                f"default_climate_zone must be one of "
                f"{sorted(_VALID_CLIMATE_ZONES)}, "
                f"got '{self.default_climate_zone}'"
            )
        else:
            self.default_climate_zone = normalised_cz

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

        # -- FOD model parameters --------------------------------------------
        for field_name, value in [
            ("default_doc_f", self.default_doc_f),
            ("default_mcf", self.default_mcf),
            ("default_f_ch4", self.default_f_ch4),
            ("default_oxidation_factor", self.default_oxidation_factor),
            ("default_collection_efficiency", self.default_collection_efficiency),
            ("default_flare_efficiency", self.default_flare_efficiency),
            ("default_utilization_efficiency", self.default_utilization_efficiency),
        ]:
            if value < 0.0:
                errors.append(
                    f"{field_name} must be >= 0.0, got {value}"
                )
            if value > 1.0:
                errors.append(
                    f"{field_name} must be <= 1.0, got {value}"
                )

        # -- Decay rate ------------------------------------------------------
        if self.default_decay_rate <= 0.0:
            errors.append(
                f"default_decay_rate must be > 0.0, "
                f"got {self.default_decay_rate}"
            )
        if self.default_decay_rate > 1.0:
            errors.append(
                f"default_decay_rate must be <= 1.0, "
                f"got {self.default_decay_rate}"
            )

        # -- Biological treatment emission factors ---------------------------
        if self.composting_ch4_ef < 0.0:
            errors.append(
                f"composting_ch4_ef must be >= 0.0, "
                f"got {self.composting_ch4_ef}"
            )
        if self.composting_ch4_ef > 100.0:
            errors.append(
                f"composting_ch4_ef must be <= 100.0, "
                f"got {self.composting_ch4_ef}"
            )

        if self.composting_n2o_ef < 0.0:
            errors.append(
                f"composting_n2o_ef must be >= 0.0, "
                f"got {self.composting_n2o_ef}"
            )
        if self.composting_n2o_ef > 10.0:
            errors.append(
                f"composting_n2o_ef must be <= 10.0, "
                f"got {self.composting_n2o_ef}"
            )

        if self.ad_ch4_ef < 0.0:
            errors.append(
                f"ad_ch4_ef must be >= 0.0, "
                f"got {self.ad_ch4_ef}"
            )
        if self.ad_ch4_ef > 50.0:
            errors.append(
                f"ad_ch4_ef must be <= 50.0, "
                f"got {self.ad_ch4_ef}"
            )

        # -- Digestion efficiency and biogas CH4 fraction --------------------
        for field_name, value in [
            ("default_digestion_efficiency", self.default_digestion_efficiency),
            ("default_biogas_ch4_fraction", self.default_biogas_ch4_fraction),
        ]:
            if value <= 0.0:
                errors.append(
                    f"{field_name} must be > 0.0, got {value}"
                )
            if value > 1.0:
                errors.append(
                    f"{field_name} must be <= 1.0, got {value}"
                )

        # -- Thermal treatment parameters ------------------------------------
        for field_name, value in [
            ("default_incineration_of", self.default_incineration_of),
            ("default_energy_recovery_efficiency", self.default_energy_recovery_efficiency),
            ("default_open_burning_of", self.default_open_burning_of),
        ]:
            if value < 0.0:
                errors.append(
                    f"{field_name} must be >= 0.0, got {value}"
                )
            if value > 1.0:
                errors.append(
                    f"{field_name} must be <= 1.0, got {value}"
                )

        # -- Wastewater treatment parameters ---------------------------------
        if self.default_bod_ch4_capacity <= 0.0:
            errors.append(
                f"default_bod_ch4_capacity must be > 0.0, "
                f"got {self.default_bod_ch4_capacity}"
            )
        if self.default_bod_ch4_capacity > 2.0:
            errors.append(
                f"default_bod_ch4_capacity must be <= 2.0, "
                f"got {self.default_bod_ch4_capacity}"
            )

        if self.default_cod_ch4_capacity <= 0.0:
            errors.append(
                f"default_cod_ch4_capacity must be > 0.0, "
                f"got {self.default_cod_ch4_capacity}"
            )
        if self.default_cod_ch4_capacity > 1.0:
            errors.append(
                f"default_cod_ch4_capacity must be <= 1.0, "
                f"got {self.default_cod_ch4_capacity}"
            )

        if self.default_ww_mcf < 0.0:
            errors.append(
                f"default_ww_mcf must be >= 0.0, "
                f"got {self.default_ww_mcf}"
            )
        if self.default_ww_mcf > 1.0:
            errors.append(
                f"default_ww_mcf must be <= 1.0, "
                f"got {self.default_ww_mcf}"
            )

        if self.default_n2o_ef_plant < 0.0:
            errors.append(
                f"default_n2o_ef_plant must be >= 0.0, "
                f"got {self.default_n2o_ef_plant}"
            )
        if self.default_n2o_ef_plant > 1.0:
            errors.append(
                f"default_n2o_ef_plant must be <= 1.0, "
                f"got {self.default_n2o_ef_plant}"
            )

        if self.default_n2o_ef_effluent < 0.0:
            errors.append(
                f"default_n2o_ef_effluent must be >= 0.0, "
                f"got {self.default_n2o_ef_effluent}"
            )
        if self.default_n2o_ef_effluent > 1.0:
            errors.append(
                f"default_n2o_ef_effluent must be <= 1.0, "
                f"got {self.default_n2o_ef_effluent}"
            )

        # -- Capacity limits -------------------------------------------------
        for field_name, value, upper in [
            ("max_batch_size", self.max_batch_size, 100_000),
            ("max_facilities", self.max_facilities, 100_000),
            ("max_waste_streams", self.max_waste_streams, 500_000),
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
                "WasteTreatmentConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "WasteTreatmentConfig validated successfully: "
            "enabled=%s, gwp_source=%s, method=%s, "
            "ef_source=%s, decimal_precision=%d, "
            "doc_f=%.2f, mcf=%.2f, f_ch4=%.2f, "
            "oxidation_factor=%.2f, collection_eff=%.2f, "
            "flare_eff=%.2f, utilization_eff=%.2f, "
            "climate_zone=%s, decay_rate=%.4f, "
            "composting_ch4_ef=%.1f, composting_n2o_ef=%.2f, "
            "ad_ch4_ef=%.1f, digestion_eff=%.2f, "
            "biogas_ch4_frac=%.2f, "
            "incineration_of=%.2f, energy_recovery_eff=%.2f, "
            "open_burning_of=%.2f, "
            "bod_ch4_cap=%.2f, cod_ch4_cap=%.2f, "
            "ww_mcf=%.2f, n2o_ef_plant=%.4f, "
            "n2o_ef_effluent=%.4f, "
            "max_batch_size=%d, monte_carlo_iterations=%d, "
            "confidence_levels=%s, "
            "biological=%s, thermal=%s, wastewater=%s, "
            "methane_recovery=%s, energy_recovery=%s, "
            "separate_biogenic_co2=%s, "
            "compliance=%s, uncertainty=%s, "
            "provenance=%s, metrics=%s",
            self.enabled,
            self.default_gwp_source,
            self.default_calculation_method,
            self.default_emission_factor_source,
            self.decimal_precision,
            self.default_doc_f,
            self.default_mcf,
            self.default_f_ch4,
            self.default_oxidation_factor,
            self.default_collection_efficiency,
            self.default_flare_efficiency,
            self.default_utilization_efficiency,
            self.default_climate_zone,
            self.default_decay_rate,
            self.composting_ch4_ef,
            self.composting_n2o_ef,
            self.ad_ch4_ef,
            self.default_digestion_efficiency,
            self.default_biogas_ch4_fraction,
            self.default_incineration_of,
            self.default_energy_recovery_efficiency,
            self.default_open_burning_of,
            self.default_bod_ch4_capacity,
            self.default_cod_ch4_capacity,
            self.default_ww_mcf,
            self.default_n2o_ef_plant,
            self.default_n2o_ef_effluent,
            self.max_batch_size,
            self.monte_carlo_iterations,
            self.confidence_levels,
            self.enable_biological,
            self.enable_thermal,
            self.enable_wastewater,
            self.enable_methane_recovery,
            self.enable_energy_recovery,
            self.separate_biogenic_co2,
            self.enable_compliance_checking,
            self.enable_uncertainty,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> WasteTreatmentConfig:
        """Build a WasteTreatmentConfig from environment variables.

        Every field can be overridden via
        ``GL_WASTE_TREATMENT_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated WasteTreatmentConfig instance, validated via
            ``__post_init__``.

        Example:
            >>> import os
            >>> os.environ["GL_WASTE_TREATMENT_DEFAULT_CALCULATION_METHOD"] = "FOD"
            >>> cfg = WasteTreatmentConfig.from_env()
            >>> cfg.default_calculation_method
            'FOD'
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
            # FOD model parameters
            default_doc_f=_float(
                "DEFAULT_DOC_F",
                cls.default_doc_f,
            ),
            default_mcf=_float(
                "DEFAULT_MCF",
                cls.default_mcf,
            ),
            default_f_ch4=_float(
                "DEFAULT_F_CH4",
                cls.default_f_ch4,
            ),
            default_oxidation_factor=_float(
                "DEFAULT_OXIDATION_FACTOR",
                cls.default_oxidation_factor,
            ),
            default_collection_efficiency=_float(
                "DEFAULT_COLLECTION_EFFICIENCY",
                cls.default_collection_efficiency,
            ),
            default_flare_efficiency=_float(
                "DEFAULT_FLARE_EFFICIENCY",
                cls.default_flare_efficiency,
            ),
            default_utilization_efficiency=_float(
                "DEFAULT_UTILIZATION_EFFICIENCY",
                cls.default_utilization_efficiency,
            ),
            default_climate_zone=_str(
                "DEFAULT_CLIMATE_ZONE",
                cls.default_climate_zone,
            ),
            default_decay_rate=_float(
                "DEFAULT_DECAY_RATE",
                cls.default_decay_rate,
            ),
            # Biological treatment defaults
            composting_ch4_ef=_float(
                "COMPOSTING_CH4_EF",
                cls.composting_ch4_ef,
            ),
            composting_n2o_ef=_float(
                "COMPOSTING_N2O_EF",
                cls.composting_n2o_ef,
            ),
            ad_ch4_ef=_float(
                "AD_CH4_EF",
                cls.ad_ch4_ef,
            ),
            default_digestion_efficiency=_float(
                "DEFAULT_DIGESTION_EFFICIENCY",
                cls.default_digestion_efficiency,
            ),
            default_biogas_ch4_fraction=_float(
                "DEFAULT_BIOGAS_CH4_FRACTION",
                cls.default_biogas_ch4_fraction,
            ),
            # Thermal treatment defaults
            default_incineration_of=_float(
                "DEFAULT_INCINERATION_OF",
                cls.default_incineration_of,
            ),
            default_energy_recovery_efficiency=_float(
                "DEFAULT_ENERGY_RECOVERY_EFF",
                cls.default_energy_recovery_efficiency,
            ),
            default_open_burning_of=_float(
                "DEFAULT_OPEN_BURNING_OF",
                cls.default_open_burning_of,
            ),
            # Wastewater treatment defaults
            default_bod_ch4_capacity=_float(
                "DEFAULT_BOD_CH4_CAPACITY",
                cls.default_bod_ch4_capacity,
            ),
            default_cod_ch4_capacity=_float(
                "DEFAULT_COD_CH4_CAPACITY",
                cls.default_cod_ch4_capacity,
            ),
            default_ww_mcf=_float(
                "DEFAULT_WW_MCF",
                cls.default_ww_mcf,
            ),
            default_n2o_ef_plant=_float(
                "DEFAULT_N2O_EF_PLANT",
                cls.default_n2o_ef_plant,
            ),
            default_n2o_ef_effluent=_float(
                "DEFAULT_N2O_EF_EFFLUENT",
                cls.default_n2o_ef_effluent,
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
            enable_biological=_bool(
                "ENABLE_BIOLOGICAL",
                cls.enable_biological,
            ),
            enable_thermal=_bool(
                "ENABLE_THERMAL",
                cls.enable_thermal,
            ),
            enable_wastewater=_bool(
                "ENABLE_WASTEWATER",
                cls.enable_wastewater,
            ),
            enable_methane_recovery=_bool(
                "ENABLE_METHANE_RECOVERY",
                cls.enable_methane_recovery,
            ),
            enable_energy_recovery=_bool(
                "ENABLE_ENERGY_RECOVERY",
                cls.enable_energy_recovery,
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
            separate_biogenic_co2=_bool(
                "SEPARATE_BIOGENIC_CO2",
                cls.separate_biogenic_co2,
            ),
            # Facility and waste stream capacity limits
            max_facilities=_int(
                "MAX_FACILITIES",
                cls.max_facilities,
            ),
            max_waste_streams=_int(
                "MAX_WASTE_STREAMS",
                cls.max_waste_streams,
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
            "WasteTreatmentConfig loaded: "
            "enabled=%s, gwp_source=%s, method=%s, "
            "ef_source=%s, decimal_precision=%d, "
            "doc_f=%.2f, mcf=%.2f, f_ch4=%.2f, "
            "oxidation_factor=%.2f, collection_eff=%.2f, "
            "flare_eff=%.2f, utilization_eff=%.2f, "
            "climate_zone=%s, decay_rate=%.4f, "
            "composting_ch4_ef=%.1f, composting_n2o_ef=%.2f, "
            "ad_ch4_ef=%.1f, digestion_eff=%.2f, "
            "biogas_ch4_frac=%.2f, "
            "incineration_of=%.2f, energy_recovery_eff=%.2f, "
            "open_burning_of=%.2f, "
            "bod_ch4_cap=%.2f, cod_ch4_cap=%.2f, "
            "ww_mcf=%.2f, n2o_ef_plant=%.4f, "
            "n2o_ef_effluent=%.4f, "
            "max_batch_size=%d, max_facilities=%d, "
            "max_waste_streams=%d, "
            "monte_carlo_iterations=%d, confidence_levels=%s, "
            "biological=%s, thermal=%s, wastewater=%s, "
            "methane_recovery=%s, energy_recovery=%s, "
            "separate_biogenic_co2=%s, "
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
            config.default_doc_f,
            config.default_mcf,
            config.default_f_ch4,
            config.default_oxidation_factor,
            config.default_collection_efficiency,
            config.default_flare_efficiency,
            config.default_utilization_efficiency,
            config.default_climate_zone,
            config.default_decay_rate,
            config.composting_ch4_ef,
            config.composting_n2o_ef,
            config.ad_ch4_ef,
            config.default_digestion_efficiency,
            config.default_biogas_ch4_fraction,
            config.default_incineration_of,
            config.default_energy_recovery_efficiency,
            config.default_open_burning_of,
            config.default_bod_ch4_capacity,
            config.default_cod_ch4_capacity,
            config.default_ww_mcf,
            config.default_n2o_ef_plant,
            config.default_n2o_ef_effluent,
            config.max_batch_size,
            config.max_facilities,
            config.max_waste_streams,
            config.monte_carlo_iterations,
            config.confidence_levels,
            config.enable_biological,
            config.enable_thermal,
            config.enable_wastewater,
            config.enable_methane_recovery,
            config.enable_energy_recovery,
            config.separate_biogenic_co2,
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
            >>> cfg = WasteTreatmentConfig()
            >>> d = cfg.to_dict()
            >>> d["default_calculation_method"]
            'IPCC_TIER_2'
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
            # -- FOD model parameters ---------------------------------------
            "default_doc_f": self.default_doc_f,
            "default_mcf": self.default_mcf,
            "default_f_ch4": self.default_f_ch4,
            "default_oxidation_factor": self.default_oxidation_factor,
            "default_collection_efficiency": self.default_collection_efficiency,
            "default_flare_efficiency": self.default_flare_efficiency,
            "default_utilization_efficiency": self.default_utilization_efficiency,
            "default_climate_zone": self.default_climate_zone,
            "default_decay_rate": self.default_decay_rate,
            # -- Biological treatment defaults ------------------------------
            "composting_ch4_ef": self.composting_ch4_ef,
            "composting_n2o_ef": self.composting_n2o_ef,
            "ad_ch4_ef": self.ad_ch4_ef,
            "default_digestion_efficiency": self.default_digestion_efficiency,
            "default_biogas_ch4_fraction": self.default_biogas_ch4_fraction,
            # -- Thermal treatment defaults ---------------------------------
            "default_incineration_of": self.default_incineration_of,
            "default_energy_recovery_efficiency": self.default_energy_recovery_efficiency,
            "default_open_burning_of": self.default_open_burning_of,
            # -- Wastewater treatment defaults ------------------------------
            "default_bod_ch4_capacity": self.default_bod_ch4_capacity,
            "default_cod_ch4_capacity": self.default_cod_ch4_capacity,
            "default_ww_mcf": self.default_ww_mcf,
            "default_n2o_ef_plant": self.default_n2o_ef_plant,
            "default_n2o_ef_effluent": self.default_n2o_ef_effluent,
            # -- Monte Carlo uncertainty analysis ---------------------------
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "monte_carlo_seed": self.monte_carlo_seed,
            "confidence_levels": self.confidence_levels,
            # -- Feature toggles --------------------------------------------
            "enable_biological": self.enable_biological,
            "enable_thermal": self.enable_thermal,
            "enable_wastewater": self.enable_wastewater,
            "enable_methane_recovery": self.enable_methane_recovery,
            "enable_energy_recovery": self.enable_energy_recovery,
            "enable_compliance_checking": self.enable_compliance_checking,
            "enable_uncertainty": self.enable_uncertainty,
            "enable_provenance": self.enable_provenance,
            "enable_metrics": self.enable_metrics,
            "separate_biogenic_co2": self.separate_biogenic_co2,
            # -- Facility and waste stream capacity limits ------------------
            "max_facilities": self.max_facilities,
            "max_waste_streams": self.max_waste_streams,
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
        return f"WasteTreatmentConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------


class _WasteTreatmentConfigHolder:
    """Thread-safe singleton holder for WasteTreatmentConfig.

    Uses double-checked locking with a threading.Lock to ensure
    exactly one WasteTreatmentConfig instance is created from environment
    variables. Subsequent calls to ``get()`` return the cached
    instance without lock contention.

    Attributes:
        _instance: Cached WasteTreatmentConfig or None.
        _lock: Threading lock for initialization.
    """

    _instance: Optional[WasteTreatmentConfig] = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get(cls) -> WasteTreatmentConfig:
        """Return the singleton WasteTreatmentConfig, creating from env if needed.

        Uses double-checked locking for thread safety with minimal
        contention on the hot path.

        Returns:
            WasteTreatmentConfig singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = WasteTreatmentConfig.from_env()
        return cls._instance

    @classmethod
    def set(cls, config: WasteTreatmentConfig) -> None:
        """Replace the singleton WasteTreatmentConfig.

        Args:
            config: New WasteTreatmentConfig to install as the singleton.
        """
        with cls._lock:
            cls._instance = config

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton to None for test teardown."""
        with cls._lock:
            cls._instance = None


def get_config() -> WasteTreatmentConfig:
    """Return the singleton WasteTreatmentConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_WASTE_TREATMENT_*`` environment variables via
    :meth:`WasteTreatmentConfig.from_env`.

    Returns:
        WasteTreatmentConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.default_calculation_method
        'IPCC_TIER_2'
    """
    return _WasteTreatmentConfigHolder.get()


def set_config(config: WasteTreatmentConfig) -> None:
    """Replace the singleton WasteTreatmentConfig.

    Primarily intended for testing and dependency injection scenarios
    where a custom configuration must be supplied without relying on
    environment variables.

    Args:
        config: New :class:`WasteTreatmentConfig` to install as the singleton.

    Example:
        >>> cfg = WasteTreatmentConfig(default_calculation_method="FOD")
        >>> set_config(cfg)
        >>> assert get_config().default_calculation_method == "FOD"
    """
    _WasteTreatmentConfigHolder.set(config)
    logger.info(
        "WasteTreatmentConfig replaced programmatically: "
        "enabled=%s, gwp_source=%s, method=%s, "
        "ef_source=%s, max_batch_size=%d, "
        "monte_carlo_iterations=%d, "
        "biological=%s, thermal=%s, wastewater=%s, "
        "separate_biogenic_co2=%s",
        config.enabled,
        config.default_gwp_source,
        config.default_calculation_method,
        config.default_emission_factor_source,
        config.max_batch_size,
        config.monte_carlo_iterations,
        config.enable_biological,
        config.enable_thermal,
        config.enable_wastewater,
        config.separate_biogenic_co2,
    )


def reset_config() -> None:
    """Reset the singleton WasteTreatmentConfig to ``None``.

    The next call to :func:`get_config` will re-read environment variables
    and construct a fresh instance. Intended for test teardown to prevent
    state leakage between test cases.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads GL_WASTE_TREATMENT_* env vars
    """
    _WasteTreatmentConfigHolder.reset()
    logger.debug("WasteTreatmentConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "WasteTreatmentConfig",
    "get_config",
    "set_config",
    "reset_config",
]
