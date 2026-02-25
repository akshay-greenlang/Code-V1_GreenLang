# -*- coding: utf-8 -*-
"""
Steam/Heat Purchase Agent Configuration - AGENT-MRV-011

Centralized configuration for the Steam/Heat Purchase Agent SDK covering:
- General service settings (name, version, logging, tenant)
- Database connection and pool settings (PostgreSQL)
- Calculation settings (GWP source, data quality tier, boiler efficiency,
  distribution loss, condensate return, CHP allocation, biogenic separation)
- Steam settings (fuel type, pressure, quality, multi-fuel blending)
- District heating settings (region, network type, distribution loss,
  supplier emission factors)
- District cooling settings (technology, COP, grid EF, free cooling,
  thermal storage losses)
- CHP/cogeneration settings (electrical/thermal efficiency, allocation
  method, reference efficiencies, primary energy savings)
- Uncertainty quantification (Monte Carlo) parameters
- Regulatory compliance framework toggles (GHG Protocol Scope 2,
  ISO 14064, CSRD ESRS, CDP, SBTi, EU EED, EPA MRR)

This module implements a singleton pattern using ``__new__`` with a
class-level ``_instance`` and ``_initialized`` flag, ensuring exactly one
configuration object exists across the application lifecycle. All settings
can be overridden via environment variables with the ``GL_SHP_`` prefix.

Environment Variable Reference (GL_SHP_ prefix):
    GL_SHP_SERVICE_NAME                     - Service name for tracing
    GL_SHP_SERVICE_VERSION                  - Service version string
    GL_SHP_LOG_LEVEL                        - Logging level
    GL_SHP_DEBUG_MODE                       - Enable debug mode
    GL_SHP_TENANT_ID                        - Default tenant identifier
    GL_SHP_DB_HOST                          - PostgreSQL host
    GL_SHP_DB_PORT                          - PostgreSQL port
    GL_SHP_DB_NAME                          - PostgreSQL database name
    GL_SHP_DB_USER                          - PostgreSQL username
    GL_SHP_DB_PASSWORD                      - PostgreSQL password
    GL_SHP_DB_POOL_MIN                      - Minimum connection pool size
    GL_SHP_DB_POOL_MAX                      - Maximum connection pool size
    GL_SHP_DB_SSL_MODE                      - PostgreSQL SSL mode
    GL_SHP_CALC_DECIMAL_PLACES              - Decimal places for calculations
    GL_SHP_CALC_MAX_BATCH_SIZE              - Maximum records per batch
    GL_SHP_CALC_DEFAULT_GWP_SOURCE          - Default GWP source (AR4/AR5/AR6)
    GL_SHP_CALC_DEFAULT_DATA_QUALITY_TIER   - Default data quality tier
    GL_SHP_CALC_DEFAULT_BOILER_EFFICIENCY   - Default boiler efficiency (0-1)
    GL_SHP_CALC_DEFAULT_DISTRIBUTION_LOSS_PCT - Default distribution loss %
    GL_SHP_CALC_CONDENSATE_RETURN_DEFAULT_PCT - Default condensate return %
    GL_SHP_CALC_DEFAULT_CHP_ALLOC_METHOD    - Default CHP allocation method
    GL_SHP_CALC_DEFAULT_AMBIENT_TEMP_C      - Default ambient temperature (C)
    GL_SHP_CALC_MAX_TRACE_STEPS             - Max provenance trace steps
    GL_SHP_CALC_ENABLE_BIOGENIC_SEPARATION  - Enable biogenic CO2 separation
    GL_SHP_CALC_ENABLE_CONDENSATE_ADJUSTMENT - Enable condensate adjustment
    GL_SHP_STEAM_DEFAULT_FUEL_TYPE          - Default fuel type for steam
    GL_SHP_STEAM_DEFAULT_STEAM_PRESSURE     - Default steam pressure class
    GL_SHP_STEAM_DEFAULT_STEAM_QUALITY      - Default steam quality type
    GL_SHP_STEAM_ENABLE_MULTI_FUEL_BLEND    - Enable multi-fuel blending
    GL_SHP_STEAM_MAX_FUEL_TYPES_PER_BLEND   - Max fuel types in a blend
    GL_SHP_DH_DEFAULT_REGION                - Default district heating region
    GL_SHP_DH_DEFAULT_NETWORK_TYPE          - Default network type
    GL_SHP_DH_ENABLE_DISTRIBUTION_LOSS      - Enable distribution loss calc
    GL_SHP_DH_ENABLE_SUPPLIER_EF            - Enable supplier emission factors
    GL_SHP_DC_DEFAULT_TECHNOLOGY            - Default cooling technology
    GL_SHP_DC_DEFAULT_COP                   - Default coefficient of performance
    GL_SHP_DC_DEFAULT_GRID_EF_KWH           - Default grid EF (kgCO2e/kWh)
    GL_SHP_DC_ENABLE_FREE_COOLING_ADJUSTMENT - Enable free cooling adjustment
    GL_SHP_DC_ENABLE_THERMAL_STORAGE_LOSSES - Enable thermal storage losses
    GL_SHP_CHP_DEFAULT_ELECTRICAL_EFFICIENCY - CHP electrical efficiency
    GL_SHP_CHP_DEFAULT_THERMAL_EFFICIENCY   - CHP thermal efficiency
    GL_SHP_CHP_DEFAULT_ALLOC_METHOD         - CHP allocation method
    GL_SHP_CHP_REFERENCE_ELECTRICAL_EFFICIENCY - Reference elec. efficiency
    GL_SHP_CHP_REFERENCE_THERMAL_EFFICIENCY - Reference thermal efficiency
    GL_SHP_CHP_ENABLE_PRIMARY_ENERGY_SAVINGS - Enable PES calculation
    GL_SHP_UNC_DEFAULT_METHOD               - Uncertainty method
    GL_SHP_UNC_DEFAULT_ITERATIONS           - Monte Carlo iterations
    GL_SHP_UNC_DEFAULT_CONFIDENCE_LEVEL     - Confidence level (0.0-1.0)
    GL_SHP_UNC_SEED                         - Random seed for reproducibility
    GL_SHP_UNC_ACTIVITY_DATA_UNCERTAINTY_PCT - Activity data uncertainty %
    GL_SHP_UNC_EMISSION_FACTOR_UNCERTAINTY_PCT - Emission factor uncertainty %
    GL_SHP_UNC_EFFICIENCY_UNCERTAINTY_PCT   - Efficiency uncertainty %
    GL_SHP_UNC_COP_UNCERTAINTY_PCT          - COP uncertainty %
    GL_SHP_UNC_CHP_ALLOCATION_UNCERTAINTY_PCT - CHP allocation uncertainty %
    GL_SHP_COMP_ENABLED_FRAMEWORKS          - Comma-separated frameworks
    GL_SHP_COMP_STRICT_MODE                 - Enable strict compliance mode
    GL_SHP_COMP_REQUIRE_ALL_FRAMEWORKS      - Require all frameworks to pass
    GL_SHP_ENABLE_METRICS                   - Enable Prometheus metrics export
    GL_SHP_METRICS_PREFIX                   - Prometheus metrics prefix
    GL_SHP_ENABLE_TRACING                   - Enable OpenTelemetry tracing
    GL_SHP_ENABLE_PROVENANCE                - Enable SHA-256 provenance tracking
    GL_SHP_GENESIS_HASH                     - Provenance chain genesis anchor
    GL_SHP_ENABLE_AUTH                      - Enable authentication middleware
    GL_SHP_WORKER_THREADS                   - Worker thread pool size
    GL_SHP_ENABLE_BACKGROUND_TASKS          - Enable background task processing
    GL_SHP_HEALTH_CHECK_INTERVAL            - Health check interval (seconds)
    GL_SHP_API_PREFIX                       - REST API route prefix
    GL_SHP_API_RATE_LIMIT                   - API requests per minute
    GL_SHP_CORS_ORIGINS                     - Comma-separated CORS origins
    GL_SHP_ENABLE_DOCS                      - Enable API documentation
    GL_SHP_ENABLED                          - Master enable/disable switch

Example:
    >>> from greenlang.steam_heat_purchase.config import SteamHeatPurchaseConfig
    >>> cfg = SteamHeatPurchaseConfig()
    >>> print(cfg.service_name, cfg.calc_default_gwp_source)
    steam-heat-purchase-service AR6

    >>> # Check singleton
    >>> cfg2 = SteamHeatPurchaseConfig()
    >>> assert cfg is cfg2

    >>> # Reset for testing
    >>> SteamHeatPurchaseConfig.reset()
    >>> cfg3 = SteamHeatPurchaseConfig()
    >>> assert cfg is not cfg3

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-011 Steam/Heat Purchase Agent (GL-MRV-X-022)
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
import os
import threading
from decimal import ROUND_CEILING
from decimal import ROUND_DOWN
from decimal import ROUND_FLOOR
from decimal import ROUND_HALF_DOWN
from decimal import ROUND_HALF_EVEN
from decimal import ROUND_HALF_UP
from decimal import ROUND_UP
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX: str = "GL_SHP_"

# ---------------------------------------------------------------------------
# Valid enumeration values for configuration validation
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

_VALID_GWP_SOURCES = frozenset({"AR4", "AR5", "AR6"})

_VALID_DATA_QUALITY_TIERS = frozenset({
    "TIER_1",
    "TIER_2",
    "TIER_3",
})

_VALID_CHP_ALLOC_METHODS = frozenset({
    "EFFICIENCY",
    "ENERGY",
    "EXERGY",
    "ECONOMIC",
    "REFERENCE_EFFICIENCY",
})

_VALID_STEAM_PRESSURES = frozenset({
    "LOW",
    "MEDIUM",
    "HIGH",
    "VERY_HIGH",
})

_VALID_STEAM_QUALITIES = frozenset({
    "SATURATED",
    "SUPERHEATED",
    "WET",
})

_VALID_FUEL_TYPES = frozenset({
    "natural_gas",
    "fuel_oil_light",
    "fuel_oil_heavy",
    "coal_bituminous",
    "coal_anthracite",
    "coal_sub_bituminous",
    "coal_lignite",
    "biomass_wood",
    "biomass_waste",
    "biogas",
    "lpg",
    "diesel",
    "propane",
    "kerosene",
    "peat",
    "municipal_solid_waste",
    "petroleum_coke",
    "blast_furnace_gas",
    "coke_oven_gas",
    "refinery_gas",
})

_VALID_NETWORK_TYPES = frozenset({
    "MUNICIPAL",
    "PRIVATE",
    "INDUSTRIAL",
    "CAMPUS",
})

_VALID_COOLING_TECHNOLOGIES = frozenset({
    "centrifugal_chiller",
    "absorption_chiller",
    "screw_chiller",
    "reciprocating_chiller",
    "air_cooled_chiller",
    "district_cooling_network",
    "free_cooling",
    "thermal_storage",
})

_VALID_UNCERTAINTY_METHODS = frozenset({
    "monte_carlo",
    "analytical",
    "bootstrap",
    "latin_hypercube",
})

_VALID_ROUNDING_MODES = frozenset({
    "ROUND_HALF_UP",
    "ROUND_HALF_DOWN",
    "ROUND_HALF_EVEN",
    "ROUND_UP",
    "ROUND_DOWN",
    "ROUND_CEILING",
    "ROUND_FLOOR",
})

_ROUNDING_MODE_MAP: Dict[str, str] = {
    "ROUND_HALF_UP": ROUND_HALF_UP,
    "ROUND_HALF_DOWN": ROUND_HALF_DOWN,
    "ROUND_HALF_EVEN": ROUND_HALF_EVEN,
    "ROUND_UP": ROUND_UP,
    "ROUND_DOWN": ROUND_DOWN,
    "ROUND_CEILING": ROUND_CEILING,
    "ROUND_FLOOR": ROUND_FLOOR,
}

_VALID_SSL_MODES = frozenset({
    "disable",
    "allow",
    "prefer",
    "require",
    "verify-ca",
    "verify-full",
})

_VALID_FRAMEWORKS = frozenset({
    "ghg_protocol_scope2",
    "iso_14064",
    "csrd_esrs",
    "cdp",
    "sbti",
    "eu_eed",
    "epa_mrr",
})

# ---------------------------------------------------------------------------
# Default compliance frameworks
# ---------------------------------------------------------------------------

_DEFAULT_ENABLED_FRAMEWORKS: List[str] = [
    "ghg_protocol_scope2",
    "iso_14064",
    "csrd_esrs",
    "cdp",
    "sbti",
    "eu_eed",
    "epa_mrr",
]


# ---------------------------------------------------------------------------
# SteamHeatPurchaseConfig
# ---------------------------------------------------------------------------


class SteamHeatPurchaseConfig:
    """Singleton configuration for the Steam/Heat Purchase Agent.

    Implements a singleton pattern via ``__new__`` with a class-level
    ``_instance`` and ``_initialized`` flag. On first instantiation, all
    settings are loaded from environment variables with the ``GL_SHP_``
    prefix. Subsequent instantiations return the same object.

    The configuration covers nine domains:
    1. General Settings - service name, version, logging, tenant
    2. Database Settings - PostgreSQL connection and pool sizing
    3. Calculation Settings - GWP source, data quality tier, boiler
       efficiency, distribution loss, condensate return, CHP allocation,
       biogenic separation, ambient temperature
    4. Steam Settings - fuel type, pressure, quality, multi-fuel blending
    5. District Heating Settings - region, network type, distribution
       loss, supplier emission factors
    6. District Cooling Settings - technology, COP, grid EF, free cooling,
       thermal storage losses
    7. CHP Settings - electrical/thermal efficiency, allocation method,
       reference efficiencies, primary energy savings
    8. Uncertainty Settings - Monte Carlo method, iterations, confidence
       level, per-parameter uncertainty percentages
    9. Compliance Settings - enabled regulatory frameworks, strict mode

    The singleton can be reset for testing via :meth:`reset`. Configuration
    can be validated explicitly via :meth:`validate`, which returns a list
    of error strings (empty list means valid). Serialisation is supported
    via :meth:`to_dict` and :meth:`from_dict`.

    Attributes:
        service_name: Service name for tracing and identification.
        service_version: Service version string.
        log_level: Logging verbosity level.
        debug_mode: Enable debug mode for verbose diagnostics.
        tenant_id: Default tenant identifier for multi-tenancy.
        db_host: PostgreSQL server hostname.
        db_port: PostgreSQL server port.
        db_name: PostgreSQL database name.
        db_user: PostgreSQL username.
        db_password: PostgreSQL password (never logged or serialised).
        db_pool_min: Minimum number of connections in the pool.
        db_pool_max: Maximum number of connections in the pool.
        db_ssl_mode: PostgreSQL SSL connection mode.
        calc_decimal_places: Number of decimal places for calculations.
        calc_max_batch_size: Maximum records per batch operation.
        calc_default_gwp_source: Default IPCC Assessment Report for GWP.
        calc_default_data_quality_tier: Default IPCC data quality tier.
        calc_default_boiler_efficiency: Default boiler efficiency (0-1).
        calc_default_distribution_loss_pct: Default distribution loss (0-1).
        calc_condensate_return_default_pct: Default condensate return (0-1).
        calc_default_chp_alloc_method: Default CHP allocation method.
        calc_default_ambient_temp_c: Default ambient temperature in Celsius.
        calc_max_trace_steps: Maximum provenance trace steps.
        calc_enable_biogenic_separation: Enable biogenic CO2 separation.
        calc_enable_condensate_adjustment: Enable condensate return adjustment.
        steam_default_fuel_type: Default fuel type for steam generation.
        steam_default_steam_pressure: Default steam pressure class.
        steam_default_steam_quality: Default steam quality type.
        steam_enable_multi_fuel_blend: Enable multi-fuel blending.
        steam_max_fuel_types_per_blend: Max fuel types in a blend.
        dh_default_region: Default district heating region.
        dh_default_network_type: Default network type.
        dh_enable_distribution_loss: Enable distribution loss calculation.
        dh_enable_supplier_ef: Enable supplier-specific emission factors.
        dc_default_technology: Default cooling technology.
        dc_default_cop: Default coefficient of performance.
        dc_default_grid_ef_kwh: Default grid EF (kgCO2e/kWh).
        dc_enable_free_cooling_adjustment: Enable free cooling adjustment.
        dc_enable_thermal_storage_losses: Enable thermal storage losses.
        chp_default_electrical_efficiency: CHP electrical efficiency (0-1).
        chp_default_thermal_efficiency: CHP thermal efficiency (0-1).
        chp_default_alloc_method: CHP allocation method.
        chp_reference_electrical_efficiency: Reference electrical efficiency.
        chp_reference_thermal_efficiency: Reference thermal efficiency.
        chp_enable_primary_energy_savings: Enable PES calculation.
        unc_default_method: Uncertainty quantification method.
        unc_default_iterations: Number of Monte Carlo iterations.
        unc_default_confidence_level: Confidence level for intervals.
        unc_seed: Random seed for reproducibility.
        unc_activity_data_uncertainty_pct: Activity data uncertainty %.
        unc_emission_factor_uncertainty_pct: Emission factor uncertainty %.
        unc_efficiency_uncertainty_pct: Efficiency uncertainty %.
        unc_cop_uncertainty_pct: COP uncertainty %.
        unc_chp_allocation_uncertainty_pct: CHP allocation uncertainty %.
        comp_enabled_frameworks: List of enabled compliance frameworks.
        comp_strict_mode: Enable strict compliance checking.
        comp_require_all_frameworks: Require all frameworks to pass.
        enable_metrics: Enable Prometheus metrics export.
        metrics_prefix: Prometheus metric name prefix.
        enable_tracing: Enable OpenTelemetry distributed tracing.
        enable_provenance: Enable SHA-256 provenance hash tracking.
        genesis_hash: Genesis anchor for the provenance chain.
        enable_auth: Enable authentication middleware.
        worker_threads: Thread pool size for parallel operations.
        enable_background_tasks: Enable background task processing.
        health_check_interval: Health check interval in seconds.
        api_prefix: REST API URL prefix.
        api_rate_limit: Maximum API requests per minute.
        cors_origins: Allowed CORS origins.
        enable_docs: Enable interactive API documentation.
        enabled: Master enable/disable switch for the agent.

    Example:
        >>> cfg = SteamHeatPurchaseConfig()
        >>> cfg.calc_default_gwp_source
        'AR6'
        >>> cfg.get_db_url()
        'postgresql://greenlang@localhost:5432/greenlang?sslmode=prefer'
        >>> cfg.is_framework_enabled("cdp")
        True
        >>> cfg.get_steam_config()
        {'default_fuel_type': 'natural_gas', ...}
    """

    _instance: Optional[SteamHeatPurchaseConfig] = None
    _initialized: bool = False
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton constructor
    # ------------------------------------------------------------------

    def __new__(cls) -> SteamHeatPurchaseConfig:
        """Return the singleton instance, creating it on first call.

        Uses a threading RLock to ensure thread-safe initialisation. Only
        one instance is ever created; subsequent calls return the same
        object without acquiring the lock (double-checked locking).

        Returns:
            The singleton SteamHeatPurchaseConfig instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialise configuration from environment variables.

        Guarded by the ``_initialized`` flag so that repeated calls to
        ``__init__`` (from repeated ``SteamHeatPurchaseConfig()`` calls)
        do not re-read environment variables or overwrite customised
        values.
        """
        if self.__class__._initialized:
            return
        self._load_from_env()
        self.__class__._initialized = True
        logger.info(
            "SteamHeatPurchaseConfig initialised from environment: "
            "service=%s, version=%s, "
            "gwp=%s, tier=%s, "
            "boiler_eff=%s, dist_loss=%s, "
            "chp_alloc=%s, frameworks=%s, "
            "metrics=%s, tracing=%s",
            self.service_name,
            self.service_version,
            self.calc_default_gwp_source,
            self.calc_default_data_quality_tier,
            self.calc_default_boiler_efficiency,
            self.calc_default_distribution_loss_pct,
            self.calc_default_chp_alloc_method,
            self.comp_enabled_frameworks,
            self.enable_metrics,
            self.enable_tracing,
        )

    # ------------------------------------------------------------------
    # Environment loading
    # ------------------------------------------------------------------

    def _load_from_env(self) -> None:
        """Load all configuration from environment variables.

        Reads environment variables with the ``GL_SHP_`` prefix and
        populates all instance attributes. Each setting has a sensible
        default so the agent can start with zero environment configuration.

        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer and float values are parsed with fallback to defaults on
        malformed input, emitting a WARNING log.
        List values are parsed from comma-separated strings.
        """
        # -- 1. General Settings -------------------------------------------------
        self.service_name: str = self._env_str(
            "SERVICE_NAME", "steam-heat-purchase-service"
        )
        self.service_version: str = self._env_str(
            "SERVICE_VERSION", "1.0.0"
        )
        self.log_level: str = self._env_str("LOG_LEVEL", "INFO")
        self.debug_mode: bool = self._env_bool("DEBUG_MODE", False)
        self.tenant_id: str = self._env_str("TENANT_ID", "default")

        # -- 2. Database Settings ------------------------------------------------
        self.db_host: str = self._env_str("DB_HOST", "localhost")
        self.db_port: int = self._env_int("DB_PORT", 5432)
        self.db_name: str = self._env_str("DB_NAME", "greenlang")
        self.db_user: str = self._env_str("DB_USER", "greenlang")
        self.db_password: str = self._env_str("DB_PASSWORD", "")
        self.db_pool_min: int = self._env_int("DB_POOL_MIN", 2)
        self.db_pool_max: int = self._env_int("DB_POOL_MAX", 10)
        self.db_ssl_mode: str = self._env_str("DB_SSL_MODE", "prefer")

        # -- 3. Calculation Settings ---------------------------------------------
        self.calc_decimal_places: int = self._env_int(
            "CALC_DECIMAL_PLACES", 8
        )
        self.calc_max_batch_size: int = self._env_int(
            "CALC_MAX_BATCH_SIZE", 10000
        )
        self.calc_default_gwp_source: str = self._env_str(
            "CALC_DEFAULT_GWP_SOURCE", "AR6"
        )
        self.calc_default_data_quality_tier: str = self._env_str(
            "CALC_DEFAULT_DATA_QUALITY_TIER", "TIER_1"
        )
        self.calc_default_boiler_efficiency: float = self._env_float(
            "CALC_DEFAULT_BOILER_EFFICIENCY", 0.85
        )
        self.calc_default_distribution_loss_pct: float = self._env_float(
            "CALC_DEFAULT_DISTRIBUTION_LOSS_PCT", 0.12
        )
        self.calc_condensate_return_default_pct: float = self._env_float(
            "CALC_CONDENSATE_RETURN_DEFAULT_PCT", 0.0
        )
        self.calc_default_chp_alloc_method: str = self._env_str(
            "CALC_DEFAULT_CHP_ALLOC_METHOD", "EFFICIENCY"
        )
        self.calc_default_ambient_temp_c: float = self._env_float(
            "CALC_DEFAULT_AMBIENT_TEMP_C", 25.0
        )
        self.calc_max_trace_steps: int = self._env_int(
            "CALC_MAX_TRACE_STEPS", 200
        )
        self.calc_enable_biogenic_separation: bool = self._env_bool(
            "CALC_ENABLE_BIOGENIC_SEPARATION", True
        )
        self.calc_enable_condensate_adjustment: bool = self._env_bool(
            "CALC_ENABLE_CONDENSATE_ADJUSTMENT", True
        )

        # -- 4. Steam Settings ---------------------------------------------------
        self.steam_default_fuel_type: str = self._env_str(
            "STEAM_DEFAULT_FUEL_TYPE", "natural_gas"
        )
        self.steam_default_steam_pressure: str = self._env_str(
            "STEAM_DEFAULT_STEAM_PRESSURE", "MEDIUM"
        )
        self.steam_default_steam_quality: str = self._env_str(
            "STEAM_DEFAULT_STEAM_QUALITY", "SATURATED"
        )
        self.steam_enable_multi_fuel_blend: bool = self._env_bool(
            "STEAM_ENABLE_MULTI_FUEL_BLEND", True
        )
        self.steam_max_fuel_types_per_blend: int = self._env_int(
            "STEAM_MAX_FUEL_TYPES_PER_BLEND", 5
        )

        # -- 5. District Heating Settings ----------------------------------------
        self.dh_default_region: str = self._env_str(
            "DH_DEFAULT_REGION", "global_default"
        )
        self.dh_default_network_type: str = self._env_str(
            "DH_DEFAULT_NETWORK_TYPE", "MUNICIPAL"
        )
        self.dh_enable_distribution_loss: bool = self._env_bool(
            "DH_ENABLE_DISTRIBUTION_LOSS", True
        )
        self.dh_enable_supplier_ef: bool = self._env_bool(
            "DH_ENABLE_SUPPLIER_EF", True
        )

        # -- 6. District Cooling Settings ----------------------------------------
        self.dc_default_technology: str = self._env_str(
            "DC_DEFAULT_TECHNOLOGY", "centrifugal_chiller"
        )
        self.dc_default_cop: float = self._env_float(
            "DC_DEFAULT_COP", 6.0
        )
        self.dc_default_grid_ef_kwh: float = self._env_float(
            "DC_DEFAULT_GRID_EF_KWH", 0.436
        )
        self.dc_enable_free_cooling_adjustment: bool = self._env_bool(
            "DC_ENABLE_FREE_COOLING_ADJUSTMENT", True
        )
        self.dc_enable_thermal_storage_losses: bool = self._env_bool(
            "DC_ENABLE_THERMAL_STORAGE_LOSSES", True
        )

        # -- 7. CHP Settings ----------------------------------------------------
        self.chp_default_electrical_efficiency: float = self._env_float(
            "CHP_DEFAULT_ELECTRICAL_EFFICIENCY", 0.35
        )
        self.chp_default_thermal_efficiency: float = self._env_float(
            "CHP_DEFAULT_THERMAL_EFFICIENCY", 0.45
        )
        self.chp_default_alloc_method: str = self._env_str(
            "CHP_DEFAULT_ALLOC_METHOD", "EFFICIENCY"
        )
        self.chp_reference_electrical_efficiency: float = self._env_float(
            "CHP_REFERENCE_ELECTRICAL_EFFICIENCY", 0.525
        )
        self.chp_reference_thermal_efficiency: float = self._env_float(
            "CHP_REFERENCE_THERMAL_EFFICIENCY", 0.90
        )
        self.chp_enable_primary_energy_savings: bool = self._env_bool(
            "CHP_ENABLE_PRIMARY_ENERGY_SAVINGS", True
        )

        # -- 8. Uncertainty Settings ---------------------------------------------
        self.unc_default_method: str = self._env_str(
            "UNC_DEFAULT_METHOD", "monte_carlo"
        )
        self.unc_default_iterations: int = self._env_int(
            "UNC_DEFAULT_ITERATIONS", 10000
        )
        self.unc_default_confidence_level: float = self._env_float(
            "UNC_DEFAULT_CONFIDENCE_LEVEL", 0.95
        )
        self.unc_seed: int = self._env_int("UNC_SEED", 42)
        self.unc_activity_data_uncertainty_pct: float = self._env_float(
            "UNC_ACTIVITY_DATA_UNCERTAINTY_PCT", 5.0
        )
        self.unc_emission_factor_uncertainty_pct: float = self._env_float(
            "UNC_EMISSION_FACTOR_UNCERTAINTY_PCT", 10.0
        )
        self.unc_efficiency_uncertainty_pct: float = self._env_float(
            "UNC_EFFICIENCY_UNCERTAINTY_PCT", 5.0
        )
        self.unc_cop_uncertainty_pct: float = self._env_float(
            "UNC_COP_UNCERTAINTY_PCT", 8.0
        )
        self.unc_chp_allocation_uncertainty_pct: float = self._env_float(
            "UNC_CHP_ALLOCATION_UNCERTAINTY_PCT", 10.0
        )

        # -- 9. Compliance Settings ----------------------------------------------
        self.comp_enabled_frameworks: List[str] = self._env_list(
            "COMP_ENABLED_FRAMEWORKS",
            _DEFAULT_ENABLED_FRAMEWORKS,
        )
        self.comp_strict_mode: bool = self._env_bool(
            "COMP_STRICT_MODE", False
        )
        self.comp_require_all_frameworks: bool = self._env_bool(
            "COMP_REQUIRE_ALL_FRAMEWORKS", False
        )

        # -- Logging & Observability ---------------------------------------------
        self.enable_metrics: bool = self._env_bool(
            "ENABLE_METRICS", True
        )
        self.metrics_prefix: str = self._env_str(
            "METRICS_PREFIX", "gl_shp_"
        )
        self.enable_tracing: bool = self._env_bool(
            "ENABLE_TRACING", True
        )

        # -- Provenance Tracking -------------------------------------------------
        self.enable_provenance: bool = self._env_bool(
            "ENABLE_PROVENANCE", True
        )
        self.genesis_hash: str = self._env_str(
            "GENESIS_HASH",
            "GL-MRV-X-022-STEAM-HEAT-PURCHASE-GENESIS",
        )

        # -- Auth & Background Tasks ---------------------------------------------
        self.enable_auth: bool = self._env_bool("ENABLE_AUTH", True)
        self.worker_threads: int = self._env_int("WORKER_THREADS", 4)
        self.enable_background_tasks: bool = self._env_bool(
            "ENABLE_BACKGROUND_TASKS", True
        )
        self.health_check_interval: int = self._env_int(
            "HEALTH_CHECK_INTERVAL", 30
        )

        # -- API Settings --------------------------------------------------------
        self.api_prefix: str = self._env_str(
            "API_PREFIX", "/api/v1/steam-heat-purchase"
        )
        self.api_rate_limit: int = self._env_int("API_RATE_LIMIT", 100)
        self.cors_origins: List[str] = self._env_list(
            "CORS_ORIGINS", ["*"]
        )
        self.enable_docs: bool = self._env_bool("ENABLE_DOCS", True)

        # -- Master switch -------------------------------------------------------
        self.enabled: bool = self._env_bool("ENABLED", True)

    # ------------------------------------------------------------------
    # Environment variable parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _env_str(name: str, default: str) -> str:
        """Read a string environment variable with the GL_SHP_ prefix.

        Args:
            name: Variable name suffix (after GL_SHP_).
            default: Default value if not set.

        Returns:
            The environment variable value or the default.
        """
        val = os.environ.get(f"{_ENV_PREFIX}{name}")
        if val is None:
            return default
        return val.strip()

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        """Read an integer environment variable with the GL_SHP_ prefix.

        Args:
            name: Variable name suffix (after GL_SHP_).
            default: Default value if not set or parse fails.

        Returns:
            Parsed integer value or the default.
        """
        val = os.environ.get(f"{_ENV_PREFIX}{name}")
        if val is None:
            return default
        try:
            return int(val.strip())
        except ValueError:
            logger.warning(
                "Invalid integer for %s%s=%r, using default %d",
                _ENV_PREFIX,
                name,
                val,
                default,
            )
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        """Read a float environment variable with the GL_SHP_ prefix.

        Args:
            name: Variable name suffix (after GL_SHP_).
            default: Default value if not set or parse fails.

        Returns:
            Parsed float value or the default.
        """
        val = os.environ.get(f"{_ENV_PREFIX}{name}")
        if val is None:
            return default
        try:
            return float(val.strip())
        except ValueError:
            logger.warning(
                "Invalid float for %s%s=%r, using default %f",
                _ENV_PREFIX,
                name,
                val,
                default,
            )
            return default

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        """Read a boolean environment variable with the GL_SHP_ prefix.

        Accepts ``true``, ``1``, ``yes`` (case-insensitive) as True.
        All other non-None values are treated as False.

        Args:
            name: Variable name suffix (after GL_SHP_).
            default: Default value if not set.

        Returns:
            Parsed boolean value or the default.
        """
        val = os.environ.get(f"{_ENV_PREFIX}{name}")
        if val is None:
            return default
        return val.strip().lower() in ("true", "1", "yes")

    @staticmethod
    def _env_list(name: str, default: List[str]) -> List[str]:
        """Read a comma-separated list environment variable.

        Args:
            name: Variable name suffix (after GL_SHP_).
            default: Default list if not set.

        Returns:
            Parsed list of stripped strings, or the default.
        """
        val = os.environ.get(f"{_ENV_PREFIX}{name}")
        if val is None:
            return list(default)
        items = [item.strip() for item in val.split(",") if item.strip()]
        if not items:
            return list(default)
        return items

    # ------------------------------------------------------------------
    # Singleton lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance for test teardown.

        After calling ``reset()``, the next instantiation of
        ``SteamHeatPurchaseConfig()`` will re-read all environment
        variables and construct a fresh configuration object. Thread-safe.

        Example:
            >>> SteamHeatPurchaseConfig.reset()
            >>> cfg = SteamHeatPurchaseConfig()  # fresh instance
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
        logger.debug("SteamHeatPurchaseConfig singleton reset")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate all configuration settings.

        Performs comprehensive checks across all configuration domains:
        general settings, database connectivity parameters, calculation
        settings, steam settings, district heating settings, district
        cooling settings, CHP settings, uncertainty parameters,
        compliance frameworks, logging levels, provenance, and
        performance tuning.

        Returns:
            A list of human-readable error strings. An empty list means
            all validation checks passed.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> errors = cfg.validate()
            >>> assert len(errors) == 0
        """
        errors: List[str] = []

        # -- General Settings ------------------------------------------------
        errors.extend(self._validate_general_settings())

        # -- Database Settings -----------------------------------------------
        errors.extend(self._validate_database_settings())

        # -- Calculation Settings --------------------------------------------
        errors.extend(self._validate_calculation_settings())

        # -- Steam Settings --------------------------------------------------
        errors.extend(self._validate_steam_settings())

        # -- District Heating Settings ---------------------------------------
        errors.extend(self._validate_district_heating_settings())

        # -- District Cooling Settings ---------------------------------------
        errors.extend(self._validate_district_cooling_settings())

        # -- CHP Settings ----------------------------------------------------
        errors.extend(self._validate_chp_settings())

        # -- Uncertainty Settings --------------------------------------------
        errors.extend(self._validate_uncertainty_settings())

        # -- Compliance Settings ---------------------------------------------
        errors.extend(self._validate_compliance_settings())

        # -- Logging & Observability -----------------------------------------
        errors.extend(self._validate_logging_settings())

        # -- Provenance Tracking ---------------------------------------------
        errors.extend(self._validate_provenance_settings())

        # -- Performance Tuning ----------------------------------------------
        errors.extend(self._validate_performance_settings())

        # -- API Settings ----------------------------------------------------
        errors.extend(self._validate_api_settings())

        if errors:
            logger.warning(
                "SteamHeatPurchaseConfig validation found %d error(s):\n%s",
                len(errors),
                "\n".join(f"  - {e}" for e in errors),
            )
        else:
            logger.debug(
                "SteamHeatPurchaseConfig validation passed: "
                "all %d checks OK",
                self._count_validation_checks(),
            )

        return errors

    def _validate_general_settings(self) -> List[str]:
        """Validate general service settings.

        Checks service name and version are non-empty, log level is
        valid, and tenant ID is non-empty.

        Returns:
            List of error strings for invalid general settings.
        """
        errors: List[str] = []

        if not self.service_name:
            errors.append("service_name must not be empty")

        if self.service_name and len(self.service_name) > 128:
            errors.append(
                f"service_name must be <= 128 characters, "
                f"got {len(self.service_name)}"
            )

        if not self.service_version:
            errors.append("service_version must not be empty")

        normalised_log = self.log_level.upper()
        if normalised_log not in _VALID_LOG_LEVELS:
            errors.append(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got '{self.log_level}'"
            )

        if not self.tenant_id:
            errors.append("tenant_id must not be empty")

        return errors

    def _validate_database_settings(self) -> List[str]:
        """Validate database connection settings.

        Checks host non-empty, port in valid TCP range (1-65535), name
        and user non-empty, pool sizes within reasonable bounds, and
        that pool_min does not exceed pool_max. SSL mode is validated
        against the PostgreSQL-accepted set.

        Returns:
            List of error strings for invalid database settings.
        """
        errors: List[str] = []

        if not self.db_host:
            errors.append("db_host must not be empty")

        if self.db_port <= 0:
            errors.append(
                f"db_port must be > 0, got {self.db_port}"
            )
        if self.db_port > 65535:
            errors.append(
                f"db_port must be <= 65535, got {self.db_port}"
            )

        if not self.db_name:
            errors.append("db_name must not be empty")

        if not self.db_user:
            errors.append("db_user must not be empty")

        if self.db_pool_min < 0:
            errors.append(
                f"db_pool_min must be >= 0, got {self.db_pool_min}"
            )
        if self.db_pool_min > 100:
            errors.append(
                f"db_pool_min must be <= 100, got {self.db_pool_min}"
            )

        if self.db_pool_max <= 0:
            errors.append(
                f"db_pool_max must be > 0, got {self.db_pool_max}"
            )
        if self.db_pool_max > 500:
            errors.append(
                f"db_pool_max must be <= 500, got {self.db_pool_max}"
            )

        if self.db_pool_min > self.db_pool_max:
            errors.append(
                f"db_pool_min ({self.db_pool_min}) must be <= "
                f"db_pool_max ({self.db_pool_max})"
            )

        normalised_ssl = self.db_ssl_mode.lower()
        if normalised_ssl not in _VALID_SSL_MODES:
            errors.append(
                f"db_ssl_mode must be one of {sorted(_VALID_SSL_MODES)}, "
                f"got '{self.db_ssl_mode}'"
            )

        return errors

    def _validate_calculation_settings(self) -> List[str]:
        """Validate calculation settings.

        Checks GWP source, data quality tier, boiler efficiency bounds,
        distribution loss percentage, condensate return percentage,
        CHP allocation method, ambient temperature, max trace steps,
        decimal places, and batch size.

        Returns:
            List of error strings for invalid calculation settings.
        """
        errors: List[str] = []

        # -- GWP source ------------------------------------------------------
        normalised_gwp = self.calc_default_gwp_source.upper()
        if normalised_gwp not in _VALID_GWP_SOURCES:
            errors.append(
                f"calc_default_gwp_source must be one of "
                f"{sorted(_VALID_GWP_SOURCES)}, "
                f"got '{self.calc_default_gwp_source}'"
            )

        # -- Data quality tier -----------------------------------------------
        normalised_tier = self.calc_default_data_quality_tier.upper()
        if normalised_tier not in _VALID_DATA_QUALITY_TIERS:
            errors.append(
                f"calc_default_data_quality_tier must be one of "
                f"{sorted(_VALID_DATA_QUALITY_TIERS)}, "
                f"got '{self.calc_default_data_quality_tier}'"
            )

        # -- Decimal places --------------------------------------------------
        if self.calc_decimal_places < 0:
            errors.append(
                f"calc_decimal_places must be >= 0, "
                f"got {self.calc_decimal_places}"
            )
        if self.calc_decimal_places > 28:
            errors.append(
                f"calc_decimal_places must be <= 28, "
                f"got {self.calc_decimal_places}"
            )

        # -- Max batch size --------------------------------------------------
        if self.calc_max_batch_size <= 0:
            errors.append(
                f"calc_max_batch_size must be > 0, "
                f"got {self.calc_max_batch_size}"
            )
        if self.calc_max_batch_size > 100_000:
            errors.append(
                f"calc_max_batch_size must be <= 100000, "
                f"got {self.calc_max_batch_size}"
            )

        # -- Boiler efficiency -----------------------------------------------
        if self.calc_default_boiler_efficiency <= 0.0:
            errors.append(
                f"calc_default_boiler_efficiency must be > 0.0, "
                f"got {self.calc_default_boiler_efficiency}"
            )
        if self.calc_default_boiler_efficiency > 1.0:
            errors.append(
                f"calc_default_boiler_efficiency must be <= 1.0, "
                f"got {self.calc_default_boiler_efficiency}"
            )

        # -- Distribution loss percentage ------------------------------------
        if self.calc_default_distribution_loss_pct < 0.0:
            errors.append(
                f"calc_default_distribution_loss_pct must be >= 0.0, "
                f"got {self.calc_default_distribution_loss_pct}"
            )
        if self.calc_default_distribution_loss_pct > 1.0:
            errors.append(
                f"calc_default_distribution_loss_pct must be <= 1.0, "
                f"got {self.calc_default_distribution_loss_pct}"
            )

        # -- Condensate return percentage ------------------------------------
        if self.calc_condensate_return_default_pct < 0.0:
            errors.append(
                f"calc_condensate_return_default_pct must be >= 0.0, "
                f"got {self.calc_condensate_return_default_pct}"
            )
        if self.calc_condensate_return_default_pct > 1.0:
            errors.append(
                f"calc_condensate_return_default_pct must be <= 1.0, "
                f"got {self.calc_condensate_return_default_pct}"
            )

        # -- CHP allocation method -------------------------------------------
        normalised_alloc = self.calc_default_chp_alloc_method.upper()
        if normalised_alloc not in _VALID_CHP_ALLOC_METHODS:
            errors.append(
                f"calc_default_chp_alloc_method must be one of "
                f"{sorted(_VALID_CHP_ALLOC_METHODS)}, "
                f"got '{self.calc_default_chp_alloc_method}'"
            )

        # -- Ambient temperature ---------------------------------------------
        if self.calc_default_ambient_temp_c < -80.0:
            errors.append(
                f"calc_default_ambient_temp_c must be >= -80.0, "
                f"got {self.calc_default_ambient_temp_c}"
            )
        if self.calc_default_ambient_temp_c > 60.0:
            errors.append(
                f"calc_default_ambient_temp_c must be <= 60.0, "
                f"got {self.calc_default_ambient_temp_c}"
            )

        # -- Max trace steps -------------------------------------------------
        if self.calc_max_trace_steps <= 0:
            errors.append(
                f"calc_max_trace_steps must be > 0, "
                f"got {self.calc_max_trace_steps}"
            )
        if self.calc_max_trace_steps > 10000:
            errors.append(
                f"calc_max_trace_steps must be <= 10000, "
                f"got {self.calc_max_trace_steps}"
            )

        return errors

    def _validate_steam_settings(self) -> List[str]:
        """Validate steam-specific settings.

        Checks fuel type, steam pressure class, steam quality type,
        and multi-fuel blend limits.

        Returns:
            List of error strings for invalid steam settings.
        """
        errors: List[str] = []

        # -- Default fuel type -----------------------------------------------
        normalised_fuel = self.steam_default_fuel_type.lower()
        if normalised_fuel not in _VALID_FUEL_TYPES:
            errors.append(
                f"steam_default_fuel_type must be one of "
                f"{sorted(_VALID_FUEL_TYPES)}, "
                f"got '{self.steam_default_fuel_type}'"
            )

        # -- Steam pressure --------------------------------------------------
        normalised_pressure = self.steam_default_steam_pressure.upper()
        if normalised_pressure not in _VALID_STEAM_PRESSURES:
            errors.append(
                f"steam_default_steam_pressure must be one of "
                f"{sorted(_VALID_STEAM_PRESSURES)}, "
                f"got '{self.steam_default_steam_pressure}'"
            )

        # -- Steam quality ---------------------------------------------------
        normalised_quality = self.steam_default_steam_quality.upper()
        if normalised_quality not in _VALID_STEAM_QUALITIES:
            errors.append(
                f"steam_default_steam_quality must be one of "
                f"{sorted(_VALID_STEAM_QUALITIES)}, "
                f"got '{self.steam_default_steam_quality}'"
            )

        # -- Max fuel types per blend ----------------------------------------
        if self.steam_max_fuel_types_per_blend <= 0:
            errors.append(
                f"steam_max_fuel_types_per_blend must be > 0, "
                f"got {self.steam_max_fuel_types_per_blend}"
            )
        if self.steam_max_fuel_types_per_blend > 20:
            errors.append(
                f"steam_max_fuel_types_per_blend must be <= 20, "
                f"got {self.steam_max_fuel_types_per_blend}"
            )

        # -- Multi-fuel blend requires at least 2 types ----------------------
        if (
            self.steam_enable_multi_fuel_blend
            and self.steam_max_fuel_types_per_blend < 2
        ):
            errors.append(
                "steam_enable_multi_fuel_blend requires "
                "steam_max_fuel_types_per_blend >= 2"
            )

        return errors

    def _validate_district_heating_settings(self) -> List[str]:
        """Validate district heating settings.

        Checks region is non-empty and network type is a valid
        recognised type.

        Returns:
            List of error strings for invalid district heating settings.
        """
        errors: List[str] = []

        # -- Default region --------------------------------------------------
        if not self.dh_default_region:
            errors.append("dh_default_region must not be empty")

        # -- Network type ----------------------------------------------------
        normalised_network = self.dh_default_network_type.upper()
        if normalised_network not in _VALID_NETWORK_TYPES:
            errors.append(
                f"dh_default_network_type must be one of "
                f"{sorted(_VALID_NETWORK_TYPES)}, "
                f"got '{self.dh_default_network_type}'"
            )

        # -- Supplier EF requires distribution loss --------------------------
        if self.dh_enable_supplier_ef and not self.dh_enable_distribution_loss:
            logger.debug(
                "dh_enable_supplier_ef is True but "
                "dh_enable_distribution_loss is False; supplier "
                "emission factors will not account for distribution losses"
            )

        return errors

    def _validate_district_cooling_settings(self) -> List[str]:
        """Validate district cooling settings.

        Checks cooling technology, COP bounds, grid emission factor
        bounds, and feature flag consistency.

        Returns:
            List of error strings for invalid district cooling settings.
        """
        errors: List[str] = []

        # -- Default technology ----------------------------------------------
        normalised_tech = self.dc_default_technology.lower()
        if normalised_tech not in _VALID_COOLING_TECHNOLOGIES:
            errors.append(
                f"dc_default_technology must be one of "
                f"{sorted(_VALID_COOLING_TECHNOLOGIES)}, "
                f"got '{self.dc_default_technology}'"
            )

        # -- COP bounds ------------------------------------------------------
        if self.dc_default_cop <= 0.0:
            errors.append(
                f"dc_default_cop must be > 0.0, "
                f"got {self.dc_default_cop}"
            )
        if self.dc_default_cop > 30.0:
            errors.append(
                f"dc_default_cop must be <= 30.0, "
                f"got {self.dc_default_cop}"
            )

        # -- Grid EF bounds --------------------------------------------------
        if self.dc_default_grid_ef_kwh < 0.0:
            errors.append(
                f"dc_default_grid_ef_kwh must be >= 0.0, "
                f"got {self.dc_default_grid_ef_kwh}"
            )
        if self.dc_default_grid_ef_kwh > 5.0:
            errors.append(
                f"dc_default_grid_ef_kwh must be <= 5.0, "
                f"got {self.dc_default_grid_ef_kwh}"
            )

        # -- Free cooling with thermal storage -------------------------------
        if (
            self.dc_enable_free_cooling_adjustment
            and self.dc_enable_thermal_storage_losses
        ):
            # Both can be enabled; this is an informational check only
            logger.debug(
                "Both free_cooling_adjustment and thermal_storage_losses "
                "are enabled; cooling calculations will include both "
                "adjustments"
            )

        return errors

    def _validate_chp_settings(self) -> List[str]:
        """Validate CHP/cogeneration settings.

        Checks electrical and thermal efficiency bounds, allocation
        method validity, reference efficiencies, and combined
        efficiency constraints.

        Returns:
            List of error strings for invalid CHP settings.
        """
        errors: List[str] = []

        # -- Electrical efficiency -------------------------------------------
        if self.chp_default_electrical_efficiency <= 0.0:
            errors.append(
                f"chp_default_electrical_efficiency must be > 0.0, "
                f"got {self.chp_default_electrical_efficiency}"
            )
        if self.chp_default_electrical_efficiency > 0.70:
            errors.append(
                f"chp_default_electrical_efficiency must be <= 0.70, "
                f"got {self.chp_default_electrical_efficiency}"
            )

        # -- Thermal efficiency ----------------------------------------------
        if self.chp_default_thermal_efficiency <= 0.0:
            errors.append(
                f"chp_default_thermal_efficiency must be > 0.0, "
                f"got {self.chp_default_thermal_efficiency}"
            )
        if self.chp_default_thermal_efficiency > 0.80:
            errors.append(
                f"chp_default_thermal_efficiency must be <= 0.80, "
                f"got {self.chp_default_thermal_efficiency}"
            )

        # -- Combined efficiency must not exceed 1.0 -------------------------
        combined_efficiency = (
            self.chp_default_electrical_efficiency
            + self.chp_default_thermal_efficiency
        )
        if combined_efficiency > 1.0:
            errors.append(
                f"Combined CHP efficiency "
                f"(electrical {self.chp_default_electrical_efficiency} + "
                f"thermal {self.chp_default_thermal_efficiency} = "
                f"{combined_efficiency:.4f}) must be <= 1.0"
            )

        # -- CHP allocation method -------------------------------------------
        normalised_alloc = self.chp_default_alloc_method.upper()
        if normalised_alloc not in _VALID_CHP_ALLOC_METHODS:
            errors.append(
                f"chp_default_alloc_method must be one of "
                f"{sorted(_VALID_CHP_ALLOC_METHODS)}, "
                f"got '{self.chp_default_alloc_method}'"
            )

        # -- Reference electrical efficiency ---------------------------------
        if self.chp_reference_electrical_efficiency <= 0.0:
            errors.append(
                f"chp_reference_electrical_efficiency must be > 0.0, "
                f"got {self.chp_reference_electrical_efficiency}"
            )
        if self.chp_reference_electrical_efficiency > 1.0:
            errors.append(
                f"chp_reference_electrical_efficiency must be <= 1.0, "
                f"got {self.chp_reference_electrical_efficiency}"
            )

        # -- Reference thermal efficiency ------------------------------------
        if self.chp_reference_thermal_efficiency <= 0.0:
            errors.append(
                f"chp_reference_thermal_efficiency must be > 0.0, "
                f"got {self.chp_reference_thermal_efficiency}"
            )
        if self.chp_reference_thermal_efficiency > 1.0:
            errors.append(
                f"chp_reference_thermal_efficiency must be <= 1.0, "
                f"got {self.chp_reference_thermal_efficiency}"
            )

        # -- PES requires reference efficiencies > actual --------------------
        if self.chp_enable_primary_energy_savings:
            if (
                self.chp_reference_electrical_efficiency
                <= self.chp_default_electrical_efficiency
            ):
                errors.append(
                    f"chp_reference_electrical_efficiency "
                    f"({self.chp_reference_electrical_efficiency}) must be "
                    f"> chp_default_electrical_efficiency "
                    f"({self.chp_default_electrical_efficiency}) for "
                    f"primary energy savings calculation"
                )

        return errors

    def _validate_uncertainty_settings(self) -> List[str]:
        """Validate uncertainty quantification settings.

        Checks uncertainty method, Monte Carlo iterations, confidence
        level, random seed, and per-parameter uncertainty percentages.

        Returns:
            List of error strings for invalid uncertainty settings.
        """
        errors: List[str] = []

        # -- Uncertainty method ----------------------------------------------
        normalised_method = self.unc_default_method.lower()
        if normalised_method not in _VALID_UNCERTAINTY_METHODS:
            errors.append(
                f"unc_default_method must be one of "
                f"{sorted(_VALID_UNCERTAINTY_METHODS)}, "
                f"got '{self.unc_default_method}'"
            )

        # -- Monte Carlo iterations ------------------------------------------
        if self.unc_default_iterations <= 0:
            errors.append(
                f"unc_default_iterations must be > 0, "
                f"got {self.unc_default_iterations}"
            )
        if self.unc_default_iterations > 1_000_000:
            errors.append(
                f"unc_default_iterations must be <= 1000000, "
                f"got {self.unc_default_iterations}"
            )

        # -- Confidence level ------------------------------------------------
        if self.unc_default_confidence_level <= 0.0:
            errors.append(
                f"unc_default_confidence_level must be > 0.0, "
                f"got {self.unc_default_confidence_level}"
            )
        if self.unc_default_confidence_level >= 1.0:
            errors.append(
                f"unc_default_confidence_level must be < 1.0, "
                f"got {self.unc_default_confidence_level}"
            )

        # -- Seed bounds -----------------------------------------------------
        if self.unc_seed < 0:
            errors.append(
                f"unc_seed must be >= 0, got {self.unc_seed}"
            )

        # -- Activity data uncertainty percentage ----------------------------
        if self.unc_activity_data_uncertainty_pct < 0.0:
            errors.append(
                f"unc_activity_data_uncertainty_pct must be >= 0.0, "
                f"got {self.unc_activity_data_uncertainty_pct}"
            )
        if self.unc_activity_data_uncertainty_pct > 100.0:
            errors.append(
                f"unc_activity_data_uncertainty_pct must be <= 100.0, "
                f"got {self.unc_activity_data_uncertainty_pct}"
            )

        # -- Emission factor uncertainty percentage --------------------------
        if self.unc_emission_factor_uncertainty_pct < 0.0:
            errors.append(
                f"unc_emission_factor_uncertainty_pct must be >= 0.0, "
                f"got {self.unc_emission_factor_uncertainty_pct}"
            )
        if self.unc_emission_factor_uncertainty_pct > 100.0:
            errors.append(
                f"unc_emission_factor_uncertainty_pct must be <= 100.0, "
                f"got {self.unc_emission_factor_uncertainty_pct}"
            )

        # -- Efficiency uncertainty percentage -------------------------------
        if self.unc_efficiency_uncertainty_pct < 0.0:
            errors.append(
                f"unc_efficiency_uncertainty_pct must be >= 0.0, "
                f"got {self.unc_efficiency_uncertainty_pct}"
            )
        if self.unc_efficiency_uncertainty_pct > 100.0:
            errors.append(
                f"unc_efficiency_uncertainty_pct must be <= 100.0, "
                f"got {self.unc_efficiency_uncertainty_pct}"
            )

        # -- COP uncertainty percentage --------------------------------------
        if self.unc_cop_uncertainty_pct < 0.0:
            errors.append(
                f"unc_cop_uncertainty_pct must be >= 0.0, "
                f"got {self.unc_cop_uncertainty_pct}"
            )
        if self.unc_cop_uncertainty_pct > 100.0:
            errors.append(
                f"unc_cop_uncertainty_pct must be <= 100.0, "
                f"got {self.unc_cop_uncertainty_pct}"
            )

        # -- CHP allocation uncertainty percentage ---------------------------
        if self.unc_chp_allocation_uncertainty_pct < 0.0:
            errors.append(
                f"unc_chp_allocation_uncertainty_pct must be >= 0.0, "
                f"got {self.unc_chp_allocation_uncertainty_pct}"
            )
        if self.unc_chp_allocation_uncertainty_pct > 100.0:
            errors.append(
                f"unc_chp_allocation_uncertainty_pct must be <= 100.0, "
                f"got {self.unc_chp_allocation_uncertainty_pct}"
            )

        return errors

    def _validate_compliance_settings(self) -> List[str]:
        """Validate compliance framework settings.

        Checks that enabled frameworks are all recognised identifiers,
        checks for duplicates, and validates that strict mode has
        at least one framework enabled.

        Returns:
            List of error strings for invalid compliance settings.
        """
        errors: List[str] = []

        if not self.comp_enabled_frameworks:
            errors.append("comp_enabled_frameworks must not be empty")
        else:
            for fw in self.comp_enabled_frameworks:
                normalised = fw.lower()
                if normalised not in _VALID_FRAMEWORKS:
                    errors.append(
                        f"Framework '{fw}' is not valid; "
                        f"must be one of {sorted(_VALID_FRAMEWORKS)}"
                    )

            # Check for duplicates
            seen: set = set()
            for fw in self.comp_enabled_frameworks:
                normalised = fw.lower()
                if normalised in seen:
                    errors.append(
                        f"Duplicate framework '{fw}' in "
                        f"comp_enabled_frameworks"
                    )
                seen.add(normalised)

        # Strict mode requires at least one framework
        if self.comp_strict_mode and not self.comp_enabled_frameworks:
            errors.append(
                "comp_strict_mode requires at least one framework "
                "in comp_enabled_frameworks"
            )

        # Require all frameworks needs strict mode
        if self.comp_require_all_frameworks and not self.comp_strict_mode:
            errors.append(
                "comp_require_all_frameworks requires "
                "comp_strict_mode to be True"
            )

        # SBTi requires ghg_protocol_scope2
        normalised_fws = [
            fw.lower() for fw in self.comp_enabled_frameworks
        ]
        if "sbti" in normalised_fws:
            if "ghg_protocol_scope2" not in normalised_fws:
                errors.append(
                    "SBTi framework requires 'ghg_protocol_scope2' "
                    "in comp_enabled_frameworks"
                )

        return errors

    def _validate_logging_settings(self) -> List[str]:
        """Validate logging and observability settings.

        Checks metrics prefix naming conventions and service name
        length constraints.

        Returns:
            List of error strings for invalid logging settings.
        """
        errors: List[str] = []

        if not self.metrics_prefix:
            errors.append("metrics_prefix must not be empty")

        if self.metrics_prefix and not self.metrics_prefix.replace(
            "_", ""
        ).isalnum():
            errors.append(
                f"metrics_prefix must contain only alphanumeric "
                f"characters and underscores, "
                f"got '{self.metrics_prefix}'"
            )

        return errors

    def _validate_provenance_settings(self) -> List[str]:
        """Validate provenance tracking settings.

        Checks that genesis hash is provided when provenance is enabled
        and that the hash length is within bounds.

        Returns:
            List of error strings for invalid provenance settings.
        """
        errors: List[str] = []

        if self.enable_provenance and not self.genesis_hash:
            errors.append(
                "genesis_hash must not be empty when "
                "enable_provenance is True"
            )

        if self.genesis_hash and len(self.genesis_hash) > 256:
            errors.append(
                f"genesis_hash must be <= 256 characters, "
                f"got {len(self.genesis_hash)}"
            )

        return errors

    def _validate_performance_settings(self) -> List[str]:
        """Validate performance tuning settings.

        Checks worker threads and health check interval are within
        reasonable bounds for production deployments.

        Returns:
            List of error strings for invalid performance settings.
        """
        errors: List[str] = []

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
        if self.health_check_interval > 3600:
            errors.append(
                f"health_check_interval must be <= 3600, "
                f"got {self.health_check_interval}"
            )

        return errors

    def _validate_api_settings(self) -> List[str]:
        """Validate API settings.

        Checks API prefix is non-empty and starts with slash, rate limit
        is within reasonable bounds, and CORS origins list is non-empty.

        Returns:
            List of error strings for invalid API settings.
        """
        errors: List[str] = []

        if not self.api_prefix:
            errors.append("api_prefix must not be empty")

        if self.api_prefix and not self.api_prefix.startswith("/"):
            errors.append(
                f"api_prefix must start with '/', "
                f"got '{self.api_prefix}'"
            )

        if self.api_rate_limit <= 0:
            errors.append(
                f"api_rate_limit must be > 0, "
                f"got {self.api_rate_limit}"
            )
        if self.api_rate_limit > 10_000:
            errors.append(
                f"api_rate_limit must be <= 10000, "
                f"got {self.api_rate_limit}"
            )

        if not self.cors_origins:
            errors.append("cors_origins must not be empty")

        return errors

    @staticmethod
    def _count_validation_checks() -> int:
        """Return the approximate number of validation checks performed.

        Returns:
            Count of individual validation assertions.
        """
        # General: 5, Database: 10, Calculation: 20, Steam: 6,
        # District Heating: 2, District Cooling: 5, CHP: 12,
        # Uncertainty: 15, Compliance: 5, Logging: 2,
        # Provenance: 2, Performance: 4, API: 5
        return 93

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a plain Python dictionary.

        The returned dictionary is safe to pass to ``json.dumps``,
        ``yaml.dump``, or any structured logging framework. Sensitive
        fields (``db_password``) are redacted to prevent accidental
        credential leakage.

        Returns:
            Dictionary representation of the configuration with sensitive
            fields redacted.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> d = cfg.to_dict()
            >>> d["calc_default_gwp_source"]
            'AR6'
            >>> d["db_password"]
            '***'
        """
        return {
            # -- Master switch -----------------------------------------------
            "enabled": self.enabled,
            # -- 1. General Settings -----------------------------------------
            "service_name": self.service_name,
            "service_version": self.service_version,
            "log_level": self.log_level,
            "debug_mode": self.debug_mode,
            "tenant_id": self.tenant_id,
            # -- 2. Database Settings ----------------------------------------
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_name": self.db_name,
            "db_user": self.db_user,
            "db_password": "***" if self.db_password else "",
            "db_pool_min": self.db_pool_min,
            "db_pool_max": self.db_pool_max,
            "db_ssl_mode": self.db_ssl_mode,
            # -- 3. Calculation Settings -------------------------------------
            "calc_decimal_places": self.calc_decimal_places,
            "calc_max_batch_size": self.calc_max_batch_size,
            "calc_default_gwp_source": self.calc_default_gwp_source,
            "calc_default_data_quality_tier": (
                self.calc_default_data_quality_tier
            ),
            "calc_default_boiler_efficiency": (
                self.calc_default_boiler_efficiency
            ),
            "calc_default_distribution_loss_pct": (
                self.calc_default_distribution_loss_pct
            ),
            "calc_condensate_return_default_pct": (
                self.calc_condensate_return_default_pct
            ),
            "calc_default_chp_alloc_method": (
                self.calc_default_chp_alloc_method
            ),
            "calc_default_ambient_temp_c": (
                self.calc_default_ambient_temp_c
            ),
            "calc_max_trace_steps": self.calc_max_trace_steps,
            "calc_enable_biogenic_separation": (
                self.calc_enable_biogenic_separation
            ),
            "calc_enable_condensate_adjustment": (
                self.calc_enable_condensate_adjustment
            ),
            # -- 4. Steam Settings -------------------------------------------
            "steam_default_fuel_type": self.steam_default_fuel_type,
            "steam_default_steam_pressure": (
                self.steam_default_steam_pressure
            ),
            "steam_default_steam_quality": (
                self.steam_default_steam_quality
            ),
            "steam_enable_multi_fuel_blend": (
                self.steam_enable_multi_fuel_blend
            ),
            "steam_max_fuel_types_per_blend": (
                self.steam_max_fuel_types_per_blend
            ),
            # -- 5. District Heating Settings --------------------------------
            "dh_default_region": self.dh_default_region,
            "dh_default_network_type": self.dh_default_network_type,
            "dh_enable_distribution_loss": (
                self.dh_enable_distribution_loss
            ),
            "dh_enable_supplier_ef": self.dh_enable_supplier_ef,
            # -- 6. District Cooling Settings --------------------------------
            "dc_default_technology": self.dc_default_technology,
            "dc_default_cop": self.dc_default_cop,
            "dc_default_grid_ef_kwh": self.dc_default_grid_ef_kwh,
            "dc_enable_free_cooling_adjustment": (
                self.dc_enable_free_cooling_adjustment
            ),
            "dc_enable_thermal_storage_losses": (
                self.dc_enable_thermal_storage_losses
            ),
            # -- 7. CHP Settings ---------------------------------------------
            "chp_default_electrical_efficiency": (
                self.chp_default_electrical_efficiency
            ),
            "chp_default_thermal_efficiency": (
                self.chp_default_thermal_efficiency
            ),
            "chp_default_alloc_method": self.chp_default_alloc_method,
            "chp_reference_electrical_efficiency": (
                self.chp_reference_electrical_efficiency
            ),
            "chp_reference_thermal_efficiency": (
                self.chp_reference_thermal_efficiency
            ),
            "chp_enable_primary_energy_savings": (
                self.chp_enable_primary_energy_savings
            ),
            # -- 8. Uncertainty Settings -------------------------------------
            "unc_default_method": self.unc_default_method,
            "unc_default_iterations": self.unc_default_iterations,
            "unc_default_confidence_level": (
                self.unc_default_confidence_level
            ),
            "unc_seed": self.unc_seed,
            "unc_activity_data_uncertainty_pct": (
                self.unc_activity_data_uncertainty_pct
            ),
            "unc_emission_factor_uncertainty_pct": (
                self.unc_emission_factor_uncertainty_pct
            ),
            "unc_efficiency_uncertainty_pct": (
                self.unc_efficiency_uncertainty_pct
            ),
            "unc_cop_uncertainty_pct": self.unc_cop_uncertainty_pct,
            "unc_chp_allocation_uncertainty_pct": (
                self.unc_chp_allocation_uncertainty_pct
            ),
            # -- 9. Compliance Settings --------------------------------------
            "comp_enabled_frameworks": list(
                self.comp_enabled_frameworks
            ),
            "comp_strict_mode": self.comp_strict_mode,
            "comp_require_all_frameworks": (
                self.comp_require_all_frameworks
            ),
            # -- Logging & Observability -------------------------------------
            "enable_metrics": self.enable_metrics,
            "metrics_prefix": self.metrics_prefix,
            "enable_tracing": self.enable_tracing,
            # -- Provenance Tracking -----------------------------------------
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # -- Auth & Background Tasks -------------------------------------
            "enable_auth": self.enable_auth,
            "worker_threads": self.worker_threads,
            "enable_background_tasks": self.enable_background_tasks,
            "health_check_interval": self.health_check_interval,
            # -- API Settings ------------------------------------------------
            "api_prefix": self.api_prefix,
            "api_rate_limit": self.api_rate_limit,
            "cors_origins": list(self.cors_origins),
            "enable_docs": self.enable_docs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SteamHeatPurchaseConfig:
        """Deserialise a configuration from a dictionary.

        Creates a new SteamHeatPurchaseConfig instance and populates it
        from the provided dictionary. The singleton is reset first to
        allow the new configuration to be installed. Keys not present
        in the dictionary retain their environment-loaded defaults.

        Args:
            data: Dictionary of configuration key-value pairs. Keys
                correspond to attribute names on the config object.

        Returns:
            A new SteamHeatPurchaseConfig instance with values from the
            dictionary.

        Example:
            >>> d = {"calc_default_gwp_source": "AR5", "calc_decimal_places": 12}
            >>> cfg = SteamHeatPurchaseConfig.from_dict(d)
            >>> cfg.calc_default_gwp_source
            'AR5'
            >>> cfg.calc_decimal_places
            12
        """
        # Reset singleton to allow fresh construction
        cls.reset()

        # Create fresh instance (triggers _load_from_env via __init__)
        instance = cls()

        # Override with provided dictionary values
        instance._apply_dict(data)

        logger.info(
            "SteamHeatPurchaseConfig loaded from dict: %d keys applied",
            len(data),
        )
        return instance

    def _apply_dict(self, data: Dict[str, Any]) -> None:
        """Apply dictionary values to the configuration instance.

        Only applies values for known attribute names. Unknown keys
        are logged as warnings and skipped. Redacted password fields
        (``'***'``) are skipped to avoid overwriting real credentials.

        Args:
            data: Dictionary of configuration key-value pairs.
        """
        known_attrs = set(self.to_dict().keys())

        for key, value in data.items():
            if key in known_attrs:
                # Handle password fields - skip '***' redacted values
                if key in ("db_password",):
                    if value == "***":
                        continue
                setattr(self, key, value)
            else:
                logger.warning(
                    "Unknown configuration key '%s' in from_dict, skipping",
                    key,
                )

    # ------------------------------------------------------------------
    # Connection URL builders
    # ------------------------------------------------------------------

    def get_db_url(self) -> str:
        """Build a PostgreSQL connection URL from individual settings.

        Constructs a standard PostgreSQL connection URL from the
        configured host, port, database, user, and password fields.
        The password is URL-encoded to handle special characters.
        The SSL mode is appended as a query parameter.

        Returns:
            PostgreSQL connection URL string.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> url = cfg.get_db_url()
            >>> url.startswith("postgresql://")
            True
        """
        if self.db_password:
            encoded_password = quote_plus(self.db_password)
            auth = f"{self.db_user}:{encoded_password}"
        else:
            auth = self.db_user

        url = (
            f"postgresql://{auth}@{self.db_host}:{self.db_port}"
            f"/{self.db_name}"
        )

        # Append SSL mode as query parameter
        if self.db_ssl_mode:
            url += f"?sslmode={self.db_ssl_mode}"

        return url

    def get_async_db_url(self) -> str:
        """Build an async PostgreSQL connection URL for asyncpg/psycopg.

        Identical to :meth:`get_db_url` but uses the
        ``postgresql+asyncpg://`` scheme for async driver compatibility.

        Returns:
            Async PostgreSQL connection URL string.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> url = cfg.get_async_db_url()
            >>> url.startswith("postgresql+asyncpg://")
            True
        """
        if self.db_password:
            encoded_password = quote_plus(self.db_password)
            auth = f"{self.db_user}:{encoded_password}"
        else:
            auth = self.db_user

        url = (
            f"postgresql+asyncpg://{auth}@{self.db_host}:{self.db_port}"
            f"/{self.db_name}"
        )

        if self.db_ssl_mode and self.db_ssl_mode != "disable":
            url += f"?ssl={self.db_ssl_mode}"

        return url

    # ------------------------------------------------------------------
    # Framework accessors
    # ------------------------------------------------------------------

    def get_enabled_frameworks(self) -> List[str]:
        """Return a copy of the enabled compliance frameworks list.

        Returns:
            List of enabled framework identifiers.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> "cdp" in cfg.get_enabled_frameworks()
            True
        """
        return list(self.comp_enabled_frameworks)

    def is_framework_enabled(self, framework: str) -> bool:
        """Check if a specific compliance framework is enabled.

        Performs a case-insensitive comparison against the configured
        list of enabled frameworks.

        Args:
            framework: Framework identifier to check (e.g. "cdp",
                "ghg_protocol_scope2", "csrd_esrs").

        Returns:
            True if the framework is in the enabled list, False otherwise.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> cfg.is_framework_enabled("cdp")
            True
            >>> cfg.is_framework_enabled("unknown_framework")
            False
        """
        normalised = framework.lower()
        return normalised in [
            fw.lower() for fw in self.comp_enabled_frameworks
        ]

    # ------------------------------------------------------------------
    # Rounding mode accessor
    # ------------------------------------------------------------------

    def get_rounding_mode(self) -> str:
        """Return the Python Decimal rounding mode constant.

        Maps ``ROUND_HALF_UP`` to the corresponding ``decimal`` module
        constant for use in ``Decimal.quantize()``.

        Returns:
            Decimal rounding mode constant string.

        Raises:
            ValueError: If no valid rounding mode is configured.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> cfg.get_rounding_mode()
            'ROUND_HALF_UP'
        """
        # Default rounding mode for steam/heat purchase calculations
        return _ROUNDING_MODE_MAP.get("ROUND_HALF_UP", ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Steam configuration accessor
    # ------------------------------------------------------------------

    def get_steam_config(self) -> Dict[str, Any]:
        """Return steam-specific configuration as a dictionary.

        Returns:
            Dictionary containing all steam-related settings suitable
            for initialising the SteamCalculationEngine.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> steam = cfg.get_steam_config()
            >>> steam["default_fuel_type"]
            'natural_gas'
        """
        return {
            "default_fuel_type": self.steam_default_fuel_type,
            "default_steam_pressure": self.steam_default_steam_pressure,
            "default_steam_quality": self.steam_default_steam_quality,
            "enable_multi_fuel_blend": self.steam_enable_multi_fuel_blend,
            "max_fuel_types_per_blend": (
                self.steam_max_fuel_types_per_blend
            ),
            "default_boiler_efficiency": (
                self.calc_default_boiler_efficiency
            ),
        }

    # ------------------------------------------------------------------
    # District heating configuration accessor
    # ------------------------------------------------------------------

    def get_district_heating_config(self) -> Dict[str, Any]:
        """Return district heating configuration as a dictionary.

        Returns:
            Dictionary containing all district-heating-related settings
            suitable for initialising the DistrictHeatingEngine.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> dh = cfg.get_district_heating_config()
            >>> dh["default_region"]
            'global_default'
        """
        return {
            "default_region": self.dh_default_region,
            "default_network_type": self.dh_default_network_type,
            "enable_distribution_loss": self.dh_enable_distribution_loss,
            "enable_supplier_ef": self.dh_enable_supplier_ef,
            "default_distribution_loss_pct": (
                self.calc_default_distribution_loss_pct
            ),
        }

    # ------------------------------------------------------------------
    # District cooling configuration accessor
    # ------------------------------------------------------------------

    def get_district_cooling_config(self) -> Dict[str, Any]:
        """Return district cooling configuration as a dictionary.

        Returns:
            Dictionary containing all district-cooling-related settings
            suitable for initialising the DistrictCoolingEngine.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> dc = cfg.get_district_cooling_config()
            >>> dc["default_technology"]
            'centrifugal_chiller'
        """
        return {
            "default_technology": self.dc_default_technology,
            "default_cop": self.dc_default_cop,
            "default_grid_ef_kwh": self.dc_default_grid_ef_kwh,
            "enable_free_cooling_adjustment": (
                self.dc_enable_free_cooling_adjustment
            ),
            "enable_thermal_storage_losses": (
                self.dc_enable_thermal_storage_losses
            ),
        }

    # ------------------------------------------------------------------
    # CHP configuration accessor
    # ------------------------------------------------------------------

    def get_chp_config(self) -> Dict[str, Any]:
        """Return CHP/cogeneration configuration as a dictionary.

        Returns:
            Dictionary containing all CHP-related settings suitable
            for initialising the CHPAllocationEngine.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> chp = cfg.get_chp_config()
            >>> chp["default_alloc_method"]
            'EFFICIENCY'
        """
        return {
            "default_electrical_efficiency": (
                self.chp_default_electrical_efficiency
            ),
            "default_thermal_efficiency": (
                self.chp_default_thermal_efficiency
            ),
            "default_alloc_method": self.chp_default_alloc_method,
            "reference_electrical_efficiency": (
                self.chp_reference_electrical_efficiency
            ),
            "reference_thermal_efficiency": (
                self.chp_reference_thermal_efficiency
            ),
            "enable_primary_energy_savings": (
                self.chp_enable_primary_energy_savings
            ),
        }

    # ------------------------------------------------------------------
    # Uncertainty parameter accessor
    # ------------------------------------------------------------------

    def get_uncertainty_params(self) -> Dict[str, Any]:
        """Return uncertainty quantification parameters as a dictionary.

        Returns:
            Dictionary containing all uncertainty-related settings.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> params = cfg.get_uncertainty_params()
            >>> params["default_method"]
            'monte_carlo'
        """
        return {
            "default_method": self.unc_default_method,
            "iterations": self.unc_default_iterations,
            "confidence_level": self.unc_default_confidence_level,
            "seed": self.unc_seed,
            "activity_data_uncertainty_pct": (
                self.unc_activity_data_uncertainty_pct
            ),
            "emission_factor_uncertainty_pct": (
                self.unc_emission_factor_uncertainty_pct
            ),
            "efficiency_uncertainty_pct": (
                self.unc_efficiency_uncertainty_pct
            ),
            "cop_uncertainty_pct": self.unc_cop_uncertainty_pct,
            "chp_allocation_uncertainty_pct": (
                self.unc_chp_allocation_uncertainty_pct
            ),
        }

    # ------------------------------------------------------------------
    # Calculation configuration accessor
    # ------------------------------------------------------------------

    def get_calculation_config(self) -> Dict[str, Any]:
        """Return calculation engine configuration as a dictionary.

        Returns:
            Dictionary containing all calculation-related settings
            suitable for initialising the calculation engines.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> calc = cfg.get_calculation_config()
            >>> calc["gwp_source"]
            'AR6'
        """
        return {
            "gwp_source": self.calc_default_gwp_source,
            "data_quality_tier": self.calc_default_data_quality_tier,
            "decimal_places": self.calc_decimal_places,
            "max_batch_size": self.calc_max_batch_size,
            "boiler_efficiency": self.calc_default_boiler_efficiency,
            "distribution_loss_pct": (
                self.calc_default_distribution_loss_pct
            ),
            "condensate_return_pct": (
                self.calc_condensate_return_default_pct
            ),
            "chp_alloc_method": self.calc_default_chp_alloc_method,
            "ambient_temp_c": self.calc_default_ambient_temp_c,
            "max_trace_steps": self.calc_max_trace_steps,
            "enable_biogenic_separation": (
                self.calc_enable_biogenic_separation
            ),
            "enable_condensate_adjustment": (
                self.calc_enable_condensate_adjustment
            ),
        }

    # ------------------------------------------------------------------
    # Database pool parameters accessor
    # ------------------------------------------------------------------

    def get_db_pool_params(self) -> Dict[str, Any]:
        """Return database connection pool parameters.

        Returns:
            Dictionary containing pool configuration suitable for
            passing to psycopg_pool or similar connection pool libraries.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> params = cfg.get_db_pool_params()
            >>> params["min_size"]
            2
        """
        return {
            "min_size": self.db_pool_min,
            "max_size": self.db_pool_max,
            "conninfo": self.get_db_url(),
        }

    # ------------------------------------------------------------------
    # API configuration accessor
    # ------------------------------------------------------------------

    def get_api_config(self) -> Dict[str, Any]:
        """Return API configuration as a dictionary.

        Returns:
            Dictionary of API-related settings suitable for FastAPI
            application construction.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> api_cfg = cfg.get_api_config()
            >>> api_cfg["prefix"]
            '/api/v1/steam-heat-purchase'
        """
        return {
            "prefix": self.api_prefix,
            "rate_limit": self.api_rate_limit,
            "cors_origins": list(self.cors_origins),
            "enable_docs": self.enable_docs,
        }

    # ------------------------------------------------------------------
    # Observability configuration accessor
    # ------------------------------------------------------------------

    def get_observability_config(self) -> Dict[str, Any]:
        """Return observability configuration as a dictionary.

        Returns:
            Dictionary of observability-related settings for metrics
            and tracing initialisation.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> obs_cfg = cfg.get_observability_config()
            >>> obs_cfg["metrics_prefix"]
            'gl_shp_'
        """
        return {
            "log_level": self.log_level,
            "enable_metrics": self.enable_metrics,
            "metrics_prefix": self.metrics_prefix,
            "enable_tracing": self.enable_tracing,
            "service_name": self.service_name,
        }

    # ------------------------------------------------------------------
    # Feature flag summary
    # ------------------------------------------------------------------

    def get_feature_flags(self) -> Dict[str, bool]:
        """Return all feature flags as a dictionary.

        Returns:
            Dictionary mapping feature flag name to boolean state.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> flags = cfg.get_feature_flags()
            >>> flags["calc_enable_biogenic_separation"]
            True
        """
        return {
            "debug_mode": self.debug_mode,
            "calc_enable_biogenic_separation": (
                self.calc_enable_biogenic_separation
            ),
            "calc_enable_condensate_adjustment": (
                self.calc_enable_condensate_adjustment
            ),
            "steam_enable_multi_fuel_blend": (
                self.steam_enable_multi_fuel_blend
            ),
            "dh_enable_distribution_loss": (
                self.dh_enable_distribution_loss
            ),
            "dh_enable_supplier_ef": self.dh_enable_supplier_ef,
            "dc_enable_free_cooling_adjustment": (
                self.dc_enable_free_cooling_adjustment
            ),
            "dc_enable_thermal_storage_losses": (
                self.dc_enable_thermal_storage_losses
            ),
            "chp_enable_primary_energy_savings": (
                self.chp_enable_primary_energy_savings
            ),
            "comp_strict_mode": self.comp_strict_mode,
            "comp_require_all_frameworks": (
                self.comp_require_all_frameworks
            ),
            "enable_metrics": self.enable_metrics,
            "enable_tracing": self.enable_tracing,
            "enable_provenance": self.enable_provenance,
            "enable_auth": self.enable_auth,
            "enable_docs": self.enable_docs,
            "enable_background_tasks": self.enable_background_tasks,
        }

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Sensitive fields (db_password) are replaced with ``'***'`` so
        that repr output is safe to include in log messages and
        exception tracebacks.

        Returns:
            String representation of the configuration.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"SteamHeatPurchaseConfig({pairs})"

    def __str__(self) -> str:
        """Return a human-readable summary of the configuration.

        Returns:
            Multi-line string summary of key settings.
        """
        return (
            f"SteamHeatPurchaseConfig("
            f"enabled={self.enabled}, "
            f"service={self.service_name}, "
            f"gwp={self.calc_default_gwp_source}, "
            f"tier={self.calc_default_data_quality_tier}, "
            f"boiler_eff={self.calc_default_boiler_efficiency}, "
            f"chp_alloc={self.calc_default_chp_alloc_method}, "
            f"batch={self.calc_max_batch_size}, "
            f"frameworks={len(self.comp_enabled_frameworks)}, "
            f"uncertainty={self.unc_default_method}"
            f")"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality by comparing all serialised values.

        Args:
            other: Object to compare against.

        Returns:
            True if other is a SteamHeatPurchaseConfig with identical
            settings.
        """
        if not isinstance(other, SteamHeatPurchaseConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    # ------------------------------------------------------------------
    # JSON serialisation helpers
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Serialise the configuration to a JSON string.

        Sensitive fields are redacted in the output. Uses the same
        redaction rules as :meth:`to_dict`.

        Args:
            indent: JSON indentation level. Defaults to 2.

        Returns:
            JSON string representation.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> json_str = cfg.to_json()
            >>> '"calc_default_gwp_source": "AR6"' in json_str
            True
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, json_str: str) -> SteamHeatPurchaseConfig:
        """Deserialise a configuration from a JSON string.

        Args:
            json_str: JSON string containing configuration key-value
                pairs.

        Returns:
            A new SteamHeatPurchaseConfig instance.

        Example:
            >>> json_str = '{"calc_default_gwp_source": "AR5"}'
            >>> cfg = SteamHeatPurchaseConfig.from_json(json_str)
            >>> cfg.calc_default_gwp_source
            'AR5'
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Normalisation helper
    # ------------------------------------------------------------------

    def normalise(self) -> None:
        """Normalise configuration values to canonical forms.

        Converts string enumerations to their expected case (e.g. GWP
        source to uppercase, fuel types to lowercase, log level to
        uppercase, SSL mode to lowercase). This method is idempotent.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> cfg.calc_default_gwp_source = "ar6"
            >>> cfg.normalise()
            >>> cfg.calc_default_gwp_source
            'AR6'
        """
        # GWP source -> uppercase
        self.calc_default_gwp_source = (
            self.calc_default_gwp_source.upper()
        )

        # Data quality tier -> uppercase
        self.calc_default_data_quality_tier = (
            self.calc_default_data_quality_tier.upper()
        )

        # CHP allocation method -> uppercase
        self.calc_default_chp_alloc_method = (
            self.calc_default_chp_alloc_method.upper()
        )

        # Steam pressure -> uppercase
        self.steam_default_steam_pressure = (
            self.steam_default_steam_pressure.upper()
        )

        # Steam quality -> uppercase
        self.steam_default_steam_quality = (
            self.steam_default_steam_quality.upper()
        )

        # Fuel type -> lowercase
        self.steam_default_fuel_type = (
            self.steam_default_fuel_type.lower()
        )

        # Network type -> uppercase
        self.dh_default_network_type = (
            self.dh_default_network_type.upper()
        )

        # Cooling technology -> lowercase
        self.dc_default_technology = self.dc_default_technology.lower()

        # CHP allocation method -> uppercase
        self.chp_default_alloc_method = (
            self.chp_default_alloc_method.upper()
        )

        # Uncertainty method -> lowercase
        self.unc_default_method = self.unc_default_method.lower()

        # Log level -> uppercase
        self.log_level = self.log_level.upper()

        # SSL mode -> lowercase
        self.db_ssl_mode = self.db_ssl_mode.lower()

        # Frameworks -> lowercase
        self.comp_enabled_frameworks = [
            fw.lower() for fw in self.comp_enabled_frameworks
        ]

        # Metrics prefix -> lowercase (convention)
        self.metrics_prefix = self.metrics_prefix.lower()

        logger.debug(
            "SteamHeatPurchaseConfig normalised: gwp=%s, tier=%s, "
            "log_level=%s, ssl_mode=%s, chp_alloc=%s",
            self.calc_default_gwp_source,
            self.calc_default_data_quality_tier,
            self.log_level,
            self.db_ssl_mode,
            self.calc_default_chp_alloc_method,
        )

    # ------------------------------------------------------------------
    # Merge with overrides
    # ------------------------------------------------------------------

    def merge(self, overrides: Dict[str, Any]) -> None:
        """Merge override values into the current configuration.

        Only known attributes are applied. Unknown keys are logged
        as warnings. Sensitive fields cannot be set to the redacted
        placeholder ``'***'``.

        Args:
            overrides: Dictionary of override key-value pairs.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> cfg.merge({"calc_decimal_places": 12})
            >>> cfg.calc_decimal_places
            12
        """
        self._apply_dict(overrides)
        logger.debug(
            "SteamHeatPurchaseConfig merged %d overrides",
            len(overrides),
        )

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> Dict[str, Any]:
        """Return a deep copy of the configuration as a dictionary.

        Unlike :meth:`to_dict`, this method does NOT redact sensitive
        fields. It is intended for internal use where the full
        configuration needs to be preserved (e.g. serialisation to
        encrypted storage).

        Returns:
            Dictionary with all values including sensitive fields.
        """
        d = self.to_dict()
        # Restore actual sensitive values
        d["db_password"] = self.db_password
        return d

    # ------------------------------------------------------------------
    # Summary for health checks
    # ------------------------------------------------------------------

    def health_summary(self) -> Dict[str, Any]:
        """Return a health check summary of the configuration.

        Includes validation status, enabled feature counts, and
        key operational parameters. Suitable for inclusion in
        ``/health`` endpoint responses.

        Returns:
            Dictionary with health-relevant configuration summary.

        Example:
            >>> cfg = SteamHeatPurchaseConfig()
            >>> summary = cfg.health_summary()
            >>> summary["validation_status"]
            'PASS'
        """
        errors = self.validate()
        flags = self.get_feature_flags()
        enabled_flag_count = sum(1 for v in flags.values() if v)

        return {
            "agent": "steam-heat-purchase",
            "agent_id": "AGENT-MRV-011",
            "enabled": self.enabled,
            "validation_status": "PASS" if not errors else "FAIL",
            "validation_errors": len(errors),
            "service_name": self.service_name,
            "service_version": self.service_version,
            "gwp_source": self.calc_default_gwp_source,
            "data_quality_tier": self.calc_default_data_quality_tier,
            "boiler_efficiency": self.calc_default_boiler_efficiency,
            "chp_alloc_method": self.calc_default_chp_alloc_method,
            "decimal_places": self.calc_decimal_places,
            "max_batch_size": self.calc_max_batch_size,
            "enabled_frameworks": len(self.comp_enabled_frameworks),
            "enabled_features": enabled_flag_count,
            "total_features": len(flags),
            "db_host": self.db_host,
            "db_port": self.db_port,
            "worker_threads": self.worker_threads,
            "provenance_enabled": self.enable_provenance,
            "metrics_prefix": self.metrics_prefix,
            "uncertainty_method": self.unc_default_method,
            "uncertainty_iterations": self.unc_default_iterations,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def get_config() -> SteamHeatPurchaseConfig:
    """Return the singleton SteamHeatPurchaseConfig.

    Convenience function that delegates to the singleton constructor.
    Thread-safe via the class-level lock in ``__new__``.

    Returns:
        SteamHeatPurchaseConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.calc_default_gwp_source
        'AR6'
    """
    return SteamHeatPurchaseConfig()


def set_config(
    overrides: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> SteamHeatPurchaseConfig:
    """Reset and re-create the singleton with optional overrides.

    Resets the singleton, creates a fresh instance from environment
    variables, then applies any provided overrides. This is the
    primary entry point for test setup.

    Args:
        overrides: Dictionary of configuration overrides.
        **kwargs: Additional keyword overrides (merged with overrides).

    Returns:
        The new SteamHeatPurchaseConfig singleton.

    Example:
        >>> cfg = set_config(calc_default_gwp_source="AR5")
        >>> cfg.calc_default_gwp_source
        'AR5'
    """
    SteamHeatPurchaseConfig.reset()
    cfg = SteamHeatPurchaseConfig()

    merged: Dict[str, Any] = {}
    if overrides:
        merged.update(overrides)
    merged.update(kwargs)

    if merged:
        cfg._apply_dict(merged)

    logger.info(
        "SteamHeatPurchaseConfig set with %d overrides: "
        "enabled=%s, gwp=%s, tier=%s, "
        "boiler_eff=%s, batch_size=%d",
        len(merged),
        cfg.enabled,
        cfg.calc_default_gwp_source,
        cfg.calc_default_data_quality_tier,
        cfg.calc_default_boiler_efficiency,
        cfg.calc_max_batch_size,
    )
    return cfg


def reset_config() -> None:
    """Reset the singleton SteamHeatPurchaseConfig to None.

    The next call to :func:`get_config` or
    ``SteamHeatPurchaseConfig()`` will re-read environment variables
    and construct a fresh instance. Intended for test teardown to
    prevent state leakage.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # fresh instance from env vars
    """
    SteamHeatPurchaseConfig.reset()


def validate_config() -> List[str]:
    """Validate the current singleton configuration.

    Convenience function that calls
    :meth:`SteamHeatPurchaseConfig.validate` on the current singleton
    instance.

    Returns:
        List of validation error strings (empty if valid).

    Example:
        >>> errors = validate_config()
        >>> assert len(errors) == 0
    """
    return get_config().validate()


# ---------------------------------------------------------------------------
# Constants for external consumers
# ---------------------------------------------------------------------------

#: Default enabled compliance frameworks
DEFAULT_ENABLED_FRAMEWORKS: List[str] = list(_DEFAULT_ENABLED_FRAMEWORKS)

#: Valid GWP sources
VALID_GWP_SOURCES: frozenset = _VALID_GWP_SOURCES

#: Valid data quality tiers
VALID_DATA_QUALITY_TIERS: frozenset = _VALID_DATA_QUALITY_TIERS

#: Valid CHP allocation methods
VALID_CHP_ALLOC_METHODS: frozenset = _VALID_CHP_ALLOC_METHODS

#: Valid steam pressure classes
VALID_STEAM_PRESSURES: frozenset = _VALID_STEAM_PRESSURES

#: Valid steam quality types
VALID_STEAM_QUALITIES: frozenset = _VALID_STEAM_QUALITIES

#: Valid fuel types
VALID_FUEL_TYPES: frozenset = _VALID_FUEL_TYPES

#: Valid network types
VALID_NETWORK_TYPES: frozenset = _VALID_NETWORK_TYPES

#: Valid cooling technologies
VALID_COOLING_TECHNOLOGIES: frozenset = _VALID_COOLING_TECHNOLOGIES

#: Valid uncertainty methods
VALID_UNCERTAINTY_METHODS: frozenset = _VALID_UNCERTAINTY_METHODS

#: Valid rounding modes
VALID_ROUNDING_MODES: frozenset = _VALID_ROUNDING_MODES

#: Valid SSL modes
VALID_SSL_MODES: frozenset = _VALID_SSL_MODES

#: Valid compliance frameworks
VALID_FRAMEWORKS: frozenset = _VALID_FRAMEWORKS

#: Environment variable prefix
ENV_PREFIX: str = _ENV_PREFIX


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    # Core class
    "SteamHeatPurchaseConfig",
    # Module-level functions
    "get_config",
    "set_config",
    "reset_config",
    "validate_config",
    # Constants
    "DEFAULT_ENABLED_FRAMEWORKS",
    "VALID_GWP_SOURCES",
    "VALID_DATA_QUALITY_TIERS",
    "VALID_CHP_ALLOC_METHODS",
    "VALID_STEAM_PRESSURES",
    "VALID_STEAM_QUALITIES",
    "VALID_FUEL_TYPES",
    "VALID_NETWORK_TYPES",
    "VALID_COOLING_TECHNOLOGIES",
    "VALID_UNCERTAINTY_METHODS",
    "VALID_ROUNDING_MODES",
    "VALID_SSL_MODES",
    "VALID_FRAMEWORKS",
    "ENV_PREFIX",
]
