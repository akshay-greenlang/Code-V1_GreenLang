"""
Metrics Collection for Fuel & Energy Activities Agent (AGENT-MRV-016, GL-MRV-S3-003).

This module provides Prometheus metrics tracking for fuel- and energy-related
activities emissions calculations (Scope 3, Category 3) including upstream
emissions of purchased fuels (well-to-tank / WTT), upstream emissions of
purchased electricity (generation not covered in Scope 2), and transmission
& distribution (T&D) losses associated with purchased electricity, steam,
heating, and cooling.

Thread-safe singleton pattern with graceful fallback if Prometheus unavailable.

Metrics prefix: gl_fea_

12 Prometheus Metrics:
    1.  gl_fea_calculations_total              - Counter: total calculations performed
    2.  gl_fea_calculation_duration_seconds     - Histogram: calculation durations
    3.  gl_fea_calculation_errors_total         - Counter: calculation errors
    4.  gl_fea_emissions_kgco2e_total           - Counter: total emissions in kgCO2e
    5.  gl_fea_fuel_consumed_kwh_total          - Counter: total fuel consumed (kWh)
    6.  gl_fea_electricity_consumed_kwh_total   - Counter: electricity consumed (kWh)
    7.  gl_fea_td_losses_kwh_total              - Counter: T&D losses (kWh)
    8.  gl_fea_wtt_factor_lookups_total         - Counter: WTT factor lookups
    9.  gl_fea_upstream_factor_lookups_total    - Counter: upstream electricity factor lookups
    10. gl_fea_compliance_checks_total          - Counter: compliance checks
    11. gl_fea_batch_jobs_total                 - Counter: batch processing jobs
    12. gl_fea_active_calculations              - Gauge: active calculations in progress

GHG Protocol Scope 3 Category 3 covers three sub-categories:
    A. Upstream emissions of purchased fuels (WTT): Extraction, production,
       and transportation of fuels consumed by the reporting company.
    B. Upstream emissions of purchased electricity: Generation-related
       activities not included in Scope 2 (e.g., extraction, processing,
       and transport of fuels consumed in electricity generation).
    C. T&D losses: Electricity lost in transmission and distribution
       that is not captured in Scope 2.

Example:
    >>> metrics = FuelEnergyActivitiesMetrics()
    >>> metrics.record_calculation(
    ...     activity_type="wtt_fuel",
    ...     calculation_method="emission_factor",
    ...     status="success",
    ...     duration_s=0.45,
    ...     emissions_kgco2e=1250.8
    ... )
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional, Generator
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful Prometheus import -- fall back to no-op stubs when the client
# library is not installed, ensuring the agent still operates correctly.
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not available, metrics will be no-ops")
    PROMETHEUS_AVAILABLE = False

    class Counter:  # type: ignore[no-redef]
        """No-op Counter stub for environments without prometheus_client."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, **kwargs: Any) -> "Counter":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

    class Histogram:  # type: ignore[no-redef]
        """No-op Histogram stub for environments without prometheus_client."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, **kwargs: Any) -> "Histogram":
            return self

        def observe(self, amount: float) -> None:
            pass

    class Gauge:  # type: ignore[no-redef]
        """No-op Gauge stub for environments without prometheus_client."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, **kwargs: Any) -> "Gauge":
            return self

        def set(self, value: float) -> None:
            pass

        def inc(self, amount: float = 1) -> None:
            pass

        def dec(self, amount: float = 1) -> None:
            pass

    class Info:  # type: ignore[no-redef]
        """No-op Info stub for environments without prometheus_client."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def info(self, data: Dict[str, str]) -> None:
            pass


# ===========================================================================
# Enumerations -- Fuel & Energy Activities domain-specific label value sets
# ===========================================================================

class ActivityType(str, Enum):
    """
    Activity types for Scope 3 Category 3 fuel- and energy-related activities.

    These correspond to the three sub-categories defined in the GHG Protocol
    Corporate Value Chain (Scope 3) Accounting and Reporting Standard:
        A. Upstream emissions of purchased fuels (WTT)
        B. Upstream emissions of purchased electricity
        C. Transmission & distribution losses
    """
    WTT_FUEL = "wtt_fuel"
    WTT_ELECTRICITY = "wtt_electricity"
    WTT_HEAT = "wtt_heat"
    WTT_STEAM = "wtt_steam"
    WTT_COOLING = "wtt_cooling"
    TD_LOSS_ELECTRICITY = "td_loss_electricity"
    TD_LOSS_HEAT = "td_loss_heat"
    TD_LOSS_STEAM = "td_loss_steam"
    TD_LOSS_COOLING = "td_loss_cooling"
    UPSTREAM_ELECTRICITY = "upstream_electricity"
    UPSTREAM_HEAT = "upstream_heat"
    UPSTREAM_STEAM = "upstream_steam"
    UPSTREAM_COOLING = "upstream_cooling"
    COMBINED = "combined"
    OTHER = "other"


class CalculationMethod(str, Enum):
    """
    Calculation methods for fuel- and energy-related activities.

    GHG Protocol Scope 3 Category 3 supports several approaches:
        - Emission factor method: activity data x WTT or T&D loss factors
        - Supplier-specific method: using primary data from energy suppliers
        - Average-data method: using country/regional average factors
        - Hybrid method: combining multiple approaches
    """
    EMISSION_FACTOR = "emission_factor"
    SUPPLIER_SPECIFIC = "supplier_specific"
    AVERAGE_DATA = "average_data"
    HYBRID = "hybrid"
    DIRECT_MEASUREMENT = "direct_measurement"
    MARKET_BASED = "market_based"
    LOCATION_BASED = "location_based"


class CalculationStatus(str, Enum):
    """Calculation operation status."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    INSUFFICIENT_DATA = "insufficient_data"
    SKIPPED = "skipped"


class FuelType(str, Enum):
    """
    Fuel types tracked for WTT upstream emissions.

    Covers primary energy carriers whose extraction, refining, and
    transportation generate upstream (well-to-tank) emissions.
    """
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    PETROL = "petrol"
    FUEL_OIL = "fuel_oil"
    LPG = "lpg"
    COAL = "coal"
    ANTHRACITE = "anthracite"
    LIGNITE = "lignite"
    KEROSENE = "kerosene"
    JET_FUEL = "jet_fuel"
    AVIATION_GASOLINE = "aviation_gasoline"
    BIODIESEL = "biodiesel"
    BIOETHANOL = "bioethanol"
    BIOMASS_WOOD = "biomass_wood"
    BIOMASS_PELLETS = "biomass_pellets"
    BIOGAS = "biogas"
    HYDROGEN = "hydrogen"
    CNG = "cng"
    LNG = "lng"
    PROPANE = "propane"
    BUTANE = "butane"
    PEAT = "peat"
    WASTE = "waste"
    OTHER = "other"


class FuelCategory(str, Enum):
    """
    Broad fuel categories for aggregation and reporting.

    These align with IPCC and GHG Protocol fuel classification hierarchies.
    """
    GASEOUS_FOSSIL = "gaseous_fossil"
    LIQUID_FOSSIL = "liquid_fossil"
    SOLID_FOSSIL = "solid_fossil"
    BIOFUEL = "biofuel"
    WASTE_DERIVED = "waste_derived"
    HYDROGEN = "hydrogen"
    OTHER = "other"


class EnergyType(str, Enum):
    """
    Types of purchased energy tracked for upstream emissions and T&D losses.

    These represent the forms of final energy delivery to the reporting
    company boundary.
    """
    GRID_ELECTRICITY = "grid_electricity"
    RENEWABLE_ELECTRICITY = "renewable_electricity"
    DISTRICT_HEAT = "district_heat"
    DISTRICT_COOLING = "district_cooling"
    STEAM = "steam"
    CHILLED_WATER = "chilled_water"
    CHP_ELECTRICITY = "chp_electricity"
    CHP_HEAT = "chp_heat"
    ON_SITE_SOLAR = "on_site_solar"
    ON_SITE_WIND = "on_site_wind"
    OTHER = "other"


class GridRegion(str, Enum):
    """
    Grid regions for location-based electricity emission factors.

    Major regional/national grids used for T&D loss factors and upstream
    electricity emission factor lookups.
    """
    US_NERC_RFC = "us_nerc_rfc"
    US_NERC_SERC = "us_nerc_serc"
    US_NERC_WECC = "us_nerc_wecc"
    US_NERC_TRE = "us_nerc_tre"
    US_NERC_MRO = "us_nerc_mro"
    US_NERC_NPCC = "us_nerc_npcc"
    US_NERC_SPP = "us_nerc_spp"
    US_AVERAGE = "us_average"
    EU_AVERAGE = "eu_average"
    UK_GRID = "uk_grid"
    DE_GRID = "de_grid"
    FR_GRID = "fr_grid"
    CN_GRID = "cn_grid"
    CN_NORTH = "cn_north"
    CN_CENTRAL = "cn_central"
    CN_SOUTH = "cn_south"
    IN_GRID = "in_grid"
    JP_GRID = "jp_grid"
    AU_GRID = "au_grid"
    BR_GRID = "br_grid"
    CA_GRID = "ca_grid"
    GLOBAL_AVERAGE = "global_average"
    OTHER = "other"


class EmissionGas(str, Enum):
    """Greenhouse gas types tracked in fuel & energy activities calculations."""
    CO2 = "co2"
    CH4 = "ch4"
    N2O = "n2o"
    HFCS = "hfcs"
    PFCS = "pfcs"
    SF6 = "sf6"
    NF3 = "nf3"
    CO2E = "co2e"


class WTTFactorSource(str, Enum):
    """
    Sources for well-to-tank (WTT) emission factors.

    WTT factors represent the upstream emissions associated with extraction,
    refining, processing, and transportation of fuels before combustion.
    """
    DEFRA = "defra"
    EPA = "epa"
    IPCC = "ipcc"
    ECOINVENT = "ecoinvent"
    GHG_PROTOCOL = "ghg_protocol"
    IEA = "iea"
    EXIOBASE = "exiobase"
    GEMIS = "gemis"
    GREET = "greet"
    INDUSTRY_AVERAGE = "industry_average"
    SUPPLIER_SPECIFIC = "supplier_specific"
    CUSTOM = "custom"


class UpstreamFactorSource(str, Enum):
    """
    Sources for upstream electricity and T&D loss emission factors.

    Upstream factors represent the life-cycle emissions from electricity
    generation excluding direct combustion (already in Scope 2), plus
    transmission and distribution loss factors.
    """
    IEA = "iea"
    EPA_EGRID = "epa_egrid"
    DEFRA = "defra"
    AIB_RE_DISS = "aib_re_diss"
    EMBER = "ember"
    ENTSOE = "entsoe"
    ECOINVENT = "ecoinvent"
    GHG_PROTOCOL = "ghg_protocol"
    NATIONAL_GRID = "national_grid"
    UTILITY_SPECIFIC = "utility_specific"
    CUSTOM = "custom"


class Country(str, Enum):
    """
    Countries for T&D loss factor and upstream electricity lookups.

    Top 25 reporting countries plus aggregates for regional and global
    averages used in location-based calculations.
    """
    US = "us"
    GB = "gb"
    DE = "de"
    FR = "fr"
    IT = "it"
    ES = "es"
    NL = "nl"
    BE = "be"
    AT = "at"
    CH = "ch"
    SE = "se"
    NO = "no"
    DK = "dk"
    FI = "fi"
    PL = "pl"
    CN = "cn"
    JP = "jp"
    KR = "kr"
    IN = "in"
    AU = "au"
    NZ = "nz"
    CA = "ca"
    BR = "br"
    MX = "mx"
    ZA = "za"
    AE = "ae"
    SG = "sg"
    EU_AVERAGE = "eu_average"
    OECD_AVERAGE = "oecd_average"
    GLOBAL_AVERAGE = "global_average"
    OTHER = "other"


class Framework(str, Enum):
    """Compliance frameworks for fuel & energy activities reporting."""
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    ISO_14067 = "iso_14067"
    CSRD = "csrd"
    TCFD = "tcfd"
    CDP = "cdp"
    SBTI = "sbti"
    ESRS = "esrs"
    SEC_CLIMATE = "sec_climate"


class ComplianceStatus(str, Enum):
    """Compliance check result status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    NEEDS_REVIEW = "needs_review"


class BatchStatus(str, Enum):
    """Batch processing job status."""
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


class ErrorType(str, Enum):
    """Error types for detailed tracking of fuel & energy calculation failures."""
    VALIDATION_ERROR = "validation_error"
    CALCULATION_ERROR = "calculation_error"
    DATABASE_ERROR = "database_error"
    API_ERROR = "api_error"
    TIMEOUT_ERROR = "timeout_error"
    CONFIGURATION_ERROR = "configuration_error"
    DATA_NOT_FOUND = "data_not_found"
    WTT_FACTOR_UNAVAILABLE = "wtt_factor_unavailable"
    TD_LOSS_FACTOR_UNAVAILABLE = "td_loss_factor_unavailable"
    UPSTREAM_FACTOR_UNAVAILABLE = "upstream_factor_unavailable"
    FUEL_CONVERSION_ERROR = "fuel_conversion_error"
    GRID_REGION_UNKNOWN = "grid_region_unknown"
    ENERGY_UNIT_CONVERSION_ERROR = "energy_unit_conversion_error"
    SUPPLIER_DATA_UNAVAILABLE = "supplier_data_unavailable"


# ===========================================================================
# FuelEnergyActivitiesMetrics -- Thread-safe Singleton
# ===========================================================================

class FuelEnergyActivitiesMetrics:
    """
    Thread-safe singleton metrics collector for Fuel & Energy Activities Agent (MRV-016).

    Provides 12 Prometheus metrics for tracking Scope 3 Category 3
    fuel- and energy-related activities emissions calculations, including
    upstream fuel emissions (WTT), upstream electricity emissions,
    transmission & distribution losses, factor lookups, compliance
    checks, batch processing, and active calculation gauges.

    All metrics use the ``gl_fea_`` prefix to ensure namespace isolation
    within the GreenLang Prometheus ecosystem.

    Scope 3 Category 3 Sub-Categories Tracked:
        A. Upstream emissions of purchased fuels (WTT)
        B. Upstream emissions of purchased electricity
        C. Transmission & distribution (T&D) losses

    Attributes:
        calculations_total: Counter for total calculation operations
        calculation_duration_seconds: Histogram for operation durations
        calculation_errors_total: Counter for calculation errors
        emissions_kgco2e_total: Counter for total emissions calculated (kgCO2e)
        fuel_consumed_kwh_total: Counter for total fuel consumed (kWh)
        electricity_consumed_kwh_total: Counter for electricity consumed (kWh)
        td_losses_kwh_total: Counter for T&D losses calculated (kWh)
        wtt_factor_lookups_total: Counter for WTT factor lookups
        upstream_factor_lookups_total: Counter for upstream electricity factor lookups
        compliance_checks_total: Counter for compliance checks
        batch_jobs_total: Counter for batch processing jobs
        active_calculations: Gauge for currently active calculations

    Example:
        >>> metrics = FuelEnergyActivitiesMetrics()
        >>> metrics.record_calculation(
        ...     activity_type="wtt_fuel",
        ...     calculation_method="emission_factor",
        ...     status="success",
        ...     duration_s=0.45,
        ...     emissions_kgco2e=1250.8
        ... )
        >>> summary = metrics.get_metrics_summary()
        >>> assert summary['calculations'] == 1
    """

    _instance: Optional["FuelEnergyActivitiesMetrics"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "FuelEnergyActivitiesMetrics":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize metrics (only once due to singleton)."""
        if hasattr(self, '_initialized'):
            return

        self._initialized: bool = True
        self._start_time: datetime = datetime.utcnow()
        self._stats_lock: threading.Lock = threading.Lock()
        self._in_memory_stats: Dict[str, Any] = {
            'calculations': 0,
            'emissions_kgco2e': 0.0,
            'fuel_consumed_kwh': 0.0,
            'electricity_consumed_kwh': 0.0,
            'td_losses_kwh': 0.0,
            'wtt_factor_lookups': 0,
            'upstream_factor_lookups': 0,
            'compliance_checks': 0,
            'batch_jobs': 0,
            'errors': 0,
        }

        # Initialize Prometheus metrics
        self._init_metrics()

        logger.info(
            "FuelEnergyActivitiesMetrics initialized (Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable"
        )

    def _init_metrics(self) -> None:
        """
        Initialize all 12 Prometheus metrics with gl_fea_ prefix.

        Each metric is documented with its purpose, label set, and the
        Scope 3 Category 3 sub-category it supports.

        When Prometheus is available, metric registration may fail if the
        metrics were previously registered (e.g., after a reset() call in
        tests). In that case we unregister from the default registry and
        re-register to obtain fresh collector objects.
        """
        # Helper to safely create a Prometheus metric, handling the case
        # where a metric with the same name is already registered.
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import REGISTRY

            def _safe_create(metric_cls: type, name: str, *args: Any, **kwargs: Any) -> Any:
                """Create a metric, unregistering any prior collector on conflict."""
                try:
                    return metric_cls(name, *args, **kwargs)
                except ValueError:
                    # Already registered -- unregister the old collector and retry
                    try:
                        REGISTRY.unregister(REGISTRY._names_to_collectors.get(name))
                    except Exception:
                        # Fallback: walk collectors to find the one owning this name
                        for collector in list(REGISTRY._names_to_collectors.values()):
                            try:
                                REGISTRY.unregister(collector)
                            except Exception:
                                pass
                    return metric_cls(name, *args, **kwargs)
        else:
            def _safe_create(metric_cls: type, name: str, *args: Any, **kwargs: Any) -> Any:
                """No-op stub creation (Prometheus not available)."""
                return metric_cls(name, *args, **kwargs)

        # ------------------------------------------------------------------
        # 1. gl_fea_calculations_total (Counter)
        #    Total fuel & energy activities emission calculations performed.
        #    Labels:
        #      - activity_type: WTT fuel, WTT electricity, T&D loss, etc.
        #      - calculation_method: emission_factor, supplier_specific, etc.
        #      - status: success, failed, partial, insufficient_data, skipped
        # ------------------------------------------------------------------
        self.calculations_total = _safe_create(Counter,
            'gl_fea_calculations_total',
            'Total fuel and energy activities emission calculations performed',
            ['activity_type', 'calculation_method', 'status']
        )

        # ------------------------------------------------------------------
        # 2. gl_fea_calculation_duration_seconds (Histogram)
        #    Duration of calculation operations in seconds.
        #    Labels:
        #      - activity_type: The type of activity being calculated
        #      - calculation_method: Method used for the calculation
        #    Buckets tuned for typical F&EA calculation latencies:
        #      - Sub-10ms for cached factor lookups
        #      - 10-100ms for single calculations
        #      - 100ms-1s for complex multi-fuel calculations
        #      - 1-60s for batch operations and cross-reference lookups
        # ------------------------------------------------------------------
        self.calculation_duration_seconds = _safe_create(Histogram,
            'gl_fea_calculation_duration_seconds',
            'Duration of fuel and energy activities calculation operations',
            ['activity_type', 'calculation_method'],
            buckets=[
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
                1.0, 2.5, 5.0, 10.0, 30.0, 60.0
            ]
        )

        # ------------------------------------------------------------------
        # 3. gl_fea_calculation_errors_total (Counter)
        #    Total calculation errors by type.
        #    Labels:
        #      - error_type: validation_error, wtt_factor_unavailable, etc.
        #    Provides breakdown of failure modes for diagnostics and
        #    alerting on systematic issues (e.g., missing WTT factors).
        # ------------------------------------------------------------------
        self.calculation_errors_total = _safe_create(Counter,
            'gl_fea_calculation_errors_total',
            'Total fuel and energy activities calculation errors',
            ['error_type']
        )

        # ------------------------------------------------------------------
        # 4. gl_fea_emissions_kgco2e_total (Counter)
        #    Total emissions calculated in kilograms CO2-equivalent.
        #    Labels:
        #      - activity_type: WTT fuel, T&D loss, upstream electricity, etc.
        #      - gas: CO2, CH4, N2O, or aggregate CO2e
        #      - fuel_type: natural_gas, diesel, coal, etc. (or "electricity"
        #        for grid-related activity types)
        #    Tracks cumulative emissions output enabling rate calculation
        #    and comparison across activity types and fuel categories.
        # ------------------------------------------------------------------
        self.emissions_kgco2e_total = _safe_create(Counter,
            'gl_fea_emissions_kgco2e_total',
            'Total fuel and energy activities emissions calculated in kgCO2e',
            ['activity_type', 'gas', 'fuel_type']
        )

        # ------------------------------------------------------------------
        # 5. gl_fea_fuel_consumed_kwh_total (Counter)
        #    Total fuel energy consumed in kilowatt-hours.
        #    Labels:
        #      - fuel_type: Specific fuel (natural_gas, diesel, coal, etc.)
        #      - fuel_category: Broad category (gaseous_fossil, liquid_fossil, etc.)
        #    Used as the activity data denominator for WTT fuel calculations.
        #    All fuel quantities are normalized to kWh for comparability.
        # ------------------------------------------------------------------
        self.fuel_consumed_kwh_total = _safe_create(Counter,
            'gl_fea_fuel_consumed_kwh_total',
            'Total fuel energy consumed in kWh for WTT calculations',
            ['fuel_type', 'fuel_category']
        )

        # ------------------------------------------------------------------
        # 6. gl_fea_electricity_consumed_kwh_total (Counter)
        #    Total electricity consumed in kilowatt-hours.
        #    Labels:
        #      - energy_type: grid_electricity, renewable, CHP, district, etc.
        #      - grid_region: Regional/national grid identifier
        #    Tracks electricity input for upstream emission and T&D loss
        #    calculations, broken down by energy delivery type and region.
        # ------------------------------------------------------------------
        self.electricity_consumed_kwh_total = _safe_create(Counter,
            'gl_fea_electricity_consumed_kwh_total',
            'Total electricity consumed in kWh for upstream/T&D calculations',
            ['energy_type', 'grid_region']
        )

        # ------------------------------------------------------------------
        # 7. gl_fea_td_losses_kwh_total (Counter)
        #    Total transmission & distribution losses in kilowatt-hours.
        #    Labels:
        #      - country: Country/region where T&D losses occur
        #    T&D losses represent the electricity lost between the power
        #    plant and the end consumer. Typical loss rates: 2-15% depending
        #    on grid infrastructure maturity.
        # ------------------------------------------------------------------
        self.td_losses_kwh_total = _safe_create(Counter,
            'gl_fea_td_losses_kwh_total',
            'Total transmission and distribution losses calculated in kWh',
            ['country']
        )

        # ------------------------------------------------------------------
        # 8. gl_fea_wtt_factor_lookups_total (Counter)
        #    Total well-to-tank (WTT) emission factor lookups.
        #    Labels:
        #      - source: Factor database/source (DEFRA, EPA, IPCC, etc.)
        #      - fuel_type: Fuel for which the WTT factor was looked up
        #    Tracks the frequency and source distribution of WTT factor
        #    retrievals, enabling cache hit ratio optimization and source
        #    coverage analysis.
        # ------------------------------------------------------------------
        self.wtt_factor_lookups_total = _safe_create(Counter,
            'gl_fea_wtt_factor_lookups_total',
            'Total well-to-tank emission factor lookups performed',
            ['source', 'fuel_type']
        )

        # ------------------------------------------------------------------
        # 9. gl_fea_upstream_factor_lookups_total (Counter)
        #    Total upstream electricity emission factor lookups.
        #    Labels:
        #      - source: Factor database/source (IEA, eGRID, Ember, etc.)
        #      - country: Country/region for the lookup
        #    Tracks retrieval of upstream generation and T&D loss factors
        #    by source and geography for quality and coverage monitoring.
        # ------------------------------------------------------------------
        self.upstream_factor_lookups_total = _safe_create(Counter,
            'gl_fea_upstream_factor_lookups_total',
            'Total upstream electricity emission factor lookups performed',
            ['source', 'country']
        )

        # ------------------------------------------------------------------
        # 10. gl_fea_compliance_checks_total (Counter)
        #     Total compliance checks performed.
        #     Labels:
        #       - framework: GHG Protocol, ISO 14064, CSRD, CDP, etc.
        #       - status: compliant, non_compliant, warning, etc.
        #     Tracks regulatory compliance validation for Scope 3 Cat 3
        #     reporting across multiple frameworks.
        # ------------------------------------------------------------------
        self.compliance_checks_total = _safe_create(Counter,
            'gl_fea_compliance_checks_total',
            'Total fuel and energy activities compliance checks performed',
            ['framework', 'status']
        )

        # ------------------------------------------------------------------
        # 11. gl_fea_batch_jobs_total (Counter)
        #     Total batch processing jobs.
        #     Labels:
        #       - status: completed, failed, partial, timeout
        #     Tracks batch processing operations for multi-facility,
        #     multi-fuel, or multi-period calculations.
        # ------------------------------------------------------------------
        self.batch_jobs_total = _safe_create(Counter,
            'gl_fea_batch_jobs_total',
            'Total fuel and energy activities batch processing jobs',
            ['status']
        )

        # ------------------------------------------------------------------
        # 12. gl_fea_active_calculations (Gauge)
        #     Number of currently active calculations in progress.
        #     No labels (simple gauge).
        #     Provides real-time concurrency visibility for capacity
        #     planning and load monitoring.
        # ------------------------------------------------------------------
        self.active_calculations = _safe_create(Gauge,
            'gl_fea_active_calculations',
            'Number of currently active fuel and energy activities calculations'
        )

    # ======================================================================
    # Primary recording methods
    # ======================================================================

    def record_calculation(
        self,
        activity_type: str,
        calculation_method: str,
        status: str,
        duration_s: float,
        emissions_kgco2e: Optional[float] = None,
        gas: Optional[str] = None,
        fuel_type: Optional[str] = None,
        operation: Optional[str] = None
    ) -> None:
        """
        Record a fuel & energy activities emission calculation operation.

        This is the primary recording method that increments the calculations
        counter, observes the duration histogram, and optionally tracks
        emissions output. It covers all three Scope 3 Category 3
        sub-categories (WTT fuel, upstream electricity, T&D losses).

        Args:
            activity_type: Activity type (wtt_fuel/wtt_electricity/td_loss_electricity/etc)
            calculation_method: Method used (emission_factor/supplier_specific/average_data/etc)
            status: Calculation status (success/failed/partial/insufficient_data/skipped)
            duration_s: Operation duration in seconds
            emissions_kgco2e: Emissions calculated in kgCO2e (optional)
            gas: Greenhouse gas type for emissions (co2/ch4/n2o/co2e) (optional)
            fuel_type: Fuel type associated with calculation (optional)
            operation: Operation name for duration tracking (optional, for compatibility)

        Example:
            >>> metrics.record_calculation(
            ...     activity_type="wtt_fuel",
            ...     calculation_method="emission_factor",
            ...     status="success",
            ...     duration_s=0.45,
            ...     emissions_kgco2e=1250.8,
            ...     gas="co2e",
            ...     fuel_type="natural_gas"
            ... )
        """
        try:
            # Validate and normalize activity_type
            activity_type = self._validate_enum_value(
                activity_type, ActivityType, ActivityType.OTHER.value
            )

            # Validate and normalize calculation_method
            calculation_method = self._validate_enum_value(
                calculation_method, CalculationMethod, CalculationMethod.EMISSION_FACTOR.value
            )

            # Validate and normalize status
            status = self._validate_enum_value(
                status, CalculationStatus, CalculationStatus.FAILED.value
            )

            # 1. Increment calculation counter
            self.calculations_total.labels(
                activity_type=activity_type,
                calculation_method=calculation_method,
                status=status
            ).inc()

            # 2. Observe duration
            self.calculation_duration_seconds.labels(
                activity_type=activity_type,
                calculation_method=calculation_method
            ).observe(duration_s)

            # 3. Record emissions if provided
            if emissions_kgco2e is not None and emissions_kgco2e > 0:
                gas = self._validate_enum_value(
                    gas or EmissionGas.CO2E.value,
                    EmissionGas,
                    EmissionGas.CO2E.value
                )
                fuel_type_val = self._validate_enum_value(
                    fuel_type or FuelType.OTHER.value,
                    FuelType,
                    FuelType.OTHER.value
                )

                self.emissions_kgco2e_total.labels(
                    activity_type=activity_type,
                    gas=gas,
                    fuel_type=fuel_type_val
                ).inc(emissions_kgco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_kgco2e'] += emissions_kgco2e

            # 4. Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['calculations'] += 1

            logger.debug(
                "Recorded calculation: activity=%s, method=%s, status=%s, "
                "duration=%.3fs, emissions=%.4f kgCO2e",
                activity_type, calculation_method, status, duration_s,
                emissions_kgco2e if emissions_kgco2e else 0.0
            )

        except Exception as e:
            logger.error("Failed to record calculation metrics: %s", e, exc_info=True)

    def record_fuel_consumption(
        self,
        fuel_type: str,
        fuel_category: str,
        kwh: float,
        duration_s: Optional[float] = None,
        emissions_kgco2e: Optional[float] = None
    ) -> None:
        """
        Record fuel consumption for WTT upstream emissions tracking.

        Tracks the quantity of fuel consumed (normalized to kWh) as the
        activity data input for well-to-tank emission calculations. This
        corresponds to Scope 3 Category 3 Sub-Category A.

        Args:
            fuel_type: Specific fuel (natural_gas/diesel/coal/lpg/etc)
            fuel_category: Broad fuel category (gaseous_fossil/liquid_fossil/etc)
            kwh: Energy content of fuel consumed in kWh
            duration_s: Operation duration in seconds (optional)
            emissions_kgco2e: WTT emissions in kgCO2e if already calculated (optional)

        Example:
            >>> metrics.record_fuel_consumption(
            ...     fuel_type="natural_gas",
            ...     fuel_category="gaseous_fossil",
            ...     kwh=50000.0,
            ...     emissions_kgco2e=2850.5
            ... )
        """
        try:
            # Validate and normalize inputs
            fuel_type = self._validate_enum_value(
                fuel_type, FuelType, FuelType.OTHER.value
            )
            fuel_category = self._validate_enum_value(
                fuel_category, FuelCategory, FuelCategory.OTHER.value
            )

            # Clamp kWh to non-negative
            kwh = max(0.0, kwh)

            # Increment fuel consumption counter
            self.fuel_consumed_kwh_total.labels(
                fuel_type=fuel_type,
                fuel_category=fuel_category
            ).inc(kwh)

            # Record associated emissions if provided
            if emissions_kgco2e is not None and emissions_kgco2e > 0:
                self.emissions_kgco2e_total.labels(
                    activity_type=ActivityType.WTT_FUEL.value,
                    gas=EmissionGas.CO2E.value,
                    fuel_type=fuel_type
                ).inc(emissions_kgco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_kgco2e'] += emissions_kgco2e

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    activity_type=ActivityType.WTT_FUEL.value,
                    calculation_method=CalculationMethod.EMISSION_FACTOR.value
                ).observe(duration_s)

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['fuel_consumed_kwh'] += kwh

            logger.debug(
                "Recorded fuel consumption: type=%s, category=%s, "
                "kwh=%.2f, emissions=%.4f kgCO2e",
                fuel_type, fuel_category, kwh,
                emissions_kgco2e if emissions_kgco2e else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record fuel consumption metrics: %s",
                e, exc_info=True
            )

    def record_electricity_consumption(
        self,
        energy_type: str,
        grid_region: str,
        kwh: float,
        duration_s: Optional[float] = None,
        emissions_kgco2e: Optional[float] = None
    ) -> None:
        """
        Record electricity consumption for upstream emission and T&D tracking.

        Tracks the quantity of electricity consumed (in kWh) as the activity
        data input for upstream electricity emissions and T&D loss
        calculations. Corresponds to Scope 3 Category 3 Sub-Categories B and C.

        Args:
            energy_type: Type of energy (grid_electricity/renewable/district_heat/etc)
            grid_region: Grid region identifier for location-based factors
            kwh: Electricity consumed in kWh
            duration_s: Operation duration in seconds (optional)
            emissions_kgco2e: Upstream emissions in kgCO2e if calculated (optional)

        Example:
            >>> metrics.record_electricity_consumption(
            ...     energy_type="grid_electricity",
            ...     grid_region="uk_grid",
            ...     kwh=100000.0,
            ...     emissions_kgco2e=4500.0
            ... )
        """
        try:
            # Validate and normalize inputs
            energy_type = self._validate_enum_value(
                energy_type, EnergyType, EnergyType.OTHER.value
            )
            grid_region = self._validate_enum_value(
                grid_region, GridRegion, GridRegion.OTHER.value
            )

            # Clamp kWh to non-negative
            kwh = max(0.0, kwh)

            # Increment electricity consumption counter
            self.electricity_consumed_kwh_total.labels(
                energy_type=energy_type,
                grid_region=grid_region
            ).inc(kwh)

            # Record associated emissions if provided
            if emissions_kgco2e is not None and emissions_kgco2e > 0:
                self.emissions_kgco2e_total.labels(
                    activity_type=ActivityType.UPSTREAM_ELECTRICITY.value,
                    gas=EmissionGas.CO2E.value,
                    fuel_type=FuelType.OTHER.value
                ).inc(emissions_kgco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_kgco2e'] += emissions_kgco2e

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    activity_type=ActivityType.UPSTREAM_ELECTRICITY.value,
                    calculation_method=CalculationMethod.LOCATION_BASED.value
                ).observe(duration_s)

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['electricity_consumed_kwh'] += kwh

            logger.debug(
                "Recorded electricity consumption: type=%s, region=%s, "
                "kwh=%.2f, emissions=%.4f kgCO2e",
                energy_type, grid_region, kwh,
                emissions_kgco2e if emissions_kgco2e else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record electricity consumption metrics: %s",
                e, exc_info=True
            )

    def record_td_loss(
        self,
        country: str,
        kwh: float,
        loss_rate_pct: Optional[float] = None,
        duration_s: Optional[float] = None,
        emissions_kgco2e: Optional[float] = None
    ) -> None:
        """
        Record a transmission & distribution loss calculation.

        T&D losses represent the electricity lost between the power plant
        and the end consumer. Typical global loss rates range from 2% to 15%
        depending on grid infrastructure maturity. This corresponds to
        Scope 3 Category 3 Sub-Category C.

        Args:
            country: Country/region where T&D losses occur
            kwh: T&D losses in kilowatt-hours
            loss_rate_pct: T&D loss rate as percentage (optional, for logging)
            duration_s: Calculation duration in seconds (optional)
            emissions_kgco2e: Emissions from T&D losses in kgCO2e (optional)

        Example:
            >>> metrics.record_td_loss(
            ...     country="gb",
            ...     kwh=7500.0,
            ...     loss_rate_pct=7.5,
            ...     emissions_kgco2e=3375.0
            ... )
        """
        try:
            # Validate and normalize inputs
            country = self._validate_enum_value(
                country, Country, Country.OTHER.value
            )

            # Clamp kWh to non-negative
            kwh = max(0.0, kwh)

            # Increment T&D loss counter
            self.td_losses_kwh_total.labels(
                country=country
            ).inc(kwh)

            # Record associated emissions if provided
            if emissions_kgco2e is not None and emissions_kgco2e > 0:
                self.emissions_kgco2e_total.labels(
                    activity_type=ActivityType.TD_LOSS_ELECTRICITY.value,
                    gas=EmissionGas.CO2E.value,
                    fuel_type=FuelType.OTHER.value
                ).inc(emissions_kgco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_kgco2e'] += emissions_kgco2e

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    activity_type=ActivityType.TD_LOSS_ELECTRICITY.value,
                    calculation_method=CalculationMethod.EMISSION_FACTOR.value
                ).observe(duration_s)

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['td_losses_kwh'] += kwh

            logger.debug(
                "Recorded T&D loss: country=%s, kwh=%.2f, loss_rate=%.2f%%, "
                "emissions=%.4f kgCO2e",
                country, kwh,
                loss_rate_pct if loss_rate_pct is not None else 0.0,
                emissions_kgco2e if emissions_kgco2e else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record T&D loss metrics: %s",
                e, exc_info=True
            )

    def record_wtt_lookup(
        self,
        source: str,
        fuel_type: str,
        count: int = 1,
        duration_s: Optional[float] = None
    ) -> None:
        """
        Record a well-to-tank (WTT) emission factor lookup.

        WTT factors represent the upstream emissions associated with
        extraction, refining, processing, and transportation of fuels
        before combustion at the reporting company's facilities.

        Args:
            source: Factor source database (defra/epa/ipcc/ecoinvent/etc)
            fuel_type: Fuel type for which the WTT factor was looked up
            count: Number of lookups (default: 1)
            duration_s: Lookup duration in seconds (optional)

        Example:
            >>> metrics.record_wtt_lookup(
            ...     source="defra",
            ...     fuel_type="natural_gas",
            ...     count=1,
            ...     duration_s=0.003
            ... )
        """
        try:
            # Validate and normalize inputs
            source = self._validate_enum_value(
                source, WTTFactorSource, WTTFactorSource.CUSTOM.value
            )
            fuel_type = self._validate_enum_value(
                fuel_type, FuelType, FuelType.OTHER.value
            )

            # Increment WTT factor lookup counter
            self.wtt_factor_lookups_total.labels(
                source=source,
                fuel_type=fuel_type
            ).inc(count)

            # Record lookup duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    activity_type=ActivityType.WTT_FUEL.value,
                    calculation_method=CalculationMethod.EMISSION_FACTOR.value
                ).observe(duration_s)

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['wtt_factor_lookups'] += count

            logger.debug(
                "Recorded WTT factor lookup: source=%s, fuel=%s, count=%d",
                source, fuel_type, count
            )

        except Exception as e:
            logger.error(
                "Failed to record WTT factor lookup metrics: %s",
                e, exc_info=True
            )

    def record_upstream_lookup(
        self,
        source: str,
        country: str,
        count: int = 1,
        duration_s: Optional[float] = None
    ) -> None:
        """
        Record an upstream electricity emission factor lookup.

        Upstream factors are used for calculating the generation-related
        upstream emissions of purchased electricity, as well as T&D loss
        emission factors for specific countries and grid regions.

        Args:
            source: Factor source database (iea/epa_egrid/defra/ember/etc)
            country: Country for which the factor was looked up
            count: Number of lookups (default: 1)
            duration_s: Lookup duration in seconds (optional)

        Example:
            >>> metrics.record_upstream_lookup(
            ...     source="iea",
            ...     country="gb",
            ...     count=1,
            ...     duration_s=0.005
            ... )
        """
        try:
            # Validate and normalize inputs
            source = self._validate_enum_value(
                source, UpstreamFactorSource, UpstreamFactorSource.CUSTOM.value
            )
            country = self._validate_enum_value(
                country, Country, Country.OTHER.value
            )

            # Increment upstream factor lookup counter
            self.upstream_factor_lookups_total.labels(
                source=source,
                country=country
            ).inc(count)

            # Record lookup duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    activity_type=ActivityType.UPSTREAM_ELECTRICITY.value,
                    calculation_method=CalculationMethod.LOCATION_BASED.value
                ).observe(duration_s)

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['upstream_factor_lookups'] += count

            logger.debug(
                "Recorded upstream factor lookup: source=%s, country=%s, count=%d",
                source, country, count
            )

        except Exception as e:
            logger.error(
                "Failed to record upstream factor lookup metrics: %s",
                e, exc_info=True
            )

    def record_compliance_check(
        self,
        framework: str,
        status: str,
        duration_s: Optional[float] = None
    ) -> None:
        """
        Record a compliance check operation.

        Tracks validation of Scope 3 Category 3 calculations against
        regulatory and voluntary reporting frameworks including
        GHG Protocol, ISO 14064, CSRD, CDP, and SBTi.

        Args:
            framework: Compliance framework (ghg_protocol/iso_14064/csrd/etc)
            status: Compliance check result (compliant/non_compliant/warning/etc)
            duration_s: Check duration in seconds (optional)

        Example:
            >>> metrics.record_compliance_check(
            ...     framework="ghg_protocol",
            ...     status="compliant",
            ...     duration_s=0.12
            ... )
        """
        try:
            # Validate and normalize inputs
            framework = self._validate_enum_value(
                framework, Framework, Framework.GHG_PROTOCOL.value
            )
            status = self._validate_enum_value(
                status, ComplianceStatus, ComplianceStatus.NOT_APPLICABLE.value
            )

            # Increment compliance check counter
            self.compliance_checks_total.labels(
                framework=framework,
                status=status
            ).inc()

            # Record check duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    activity_type=ActivityType.OTHER.value,
                    calculation_method=CalculationMethod.EMISSION_FACTOR.value
                ).observe(duration_s)

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['compliance_checks'] += 1

            logger.debug(
                "Recorded compliance check: framework=%s, status=%s",
                framework, status
            )

        except Exception as e:
            logger.error(
                "Failed to record compliance check metrics: %s",
                e, exc_info=True
            )

    def record_batch_job(
        self,
        status: str,
        size: Optional[int] = None,
        successful: Optional[int] = None,
        failed: Optional[int] = None,
        duration_s: Optional[float] = None,
        total_emissions_kgco2e: Optional[float] = None
    ) -> None:
        """
        Record a batch processing job.

        Batch jobs process multiple fuel/energy records in a single
        operation, typically for multi-facility, multi-fuel, or
        multi-period Scope 3 Category 3 calculations.

        Args:
            status: Job status (completed/failed/partial/timeout)
            size: Number of items in the batch (optional)
            successful: Number of successful calculations (optional)
            failed: Number of failed calculations (optional)
            duration_s: Total batch duration in seconds (optional)
            total_emissions_kgco2e: Total emissions for the batch in kgCO2e (optional)

        Example:
            >>> metrics.record_batch_job(
            ...     status="completed",
            ...     size=250,
            ...     successful=248,
            ...     failed=2,
            ...     duration_s=15.3,
            ...     total_emissions_kgco2e=87500.0
            ... )
        """
        try:
            # Validate and normalize status
            status = self._validate_enum_value(
                status, BatchStatus, BatchStatus.FAILED.value
            )

            # Increment batch job counter
            self.batch_jobs_total.labels(
                status=status
            ).inc()

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    activity_type=ActivityType.COMBINED.value,
                    calculation_method=CalculationMethod.EMISSION_FACTOR.value
                ).observe(duration_s)

            # Record successful calculations in primary counter
            if successful is not None and successful > 0:
                self.calculations_total.labels(
                    activity_type=ActivityType.COMBINED.value,
                    calculation_method=CalculationMethod.EMISSION_FACTOR.value,
                    status=CalculationStatus.SUCCESS.value
                ).inc(successful)

            # Record failed calculations in primary counter
            if failed is not None and failed > 0:
                self.calculations_total.labels(
                    activity_type=ActivityType.COMBINED.value,
                    calculation_method=CalculationMethod.EMISSION_FACTOR.value,
                    status=CalculationStatus.FAILED.value
                ).inc(failed)

            # Record aggregate emissions if provided
            if total_emissions_kgco2e is not None and total_emissions_kgco2e > 0:
                self.emissions_kgco2e_total.labels(
                    activity_type=ActivityType.COMBINED.value,
                    gas=EmissionGas.CO2E.value,
                    fuel_type=FuelType.OTHER.value
                ).inc(total_emissions_kgco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_kgco2e'] += total_emissions_kgco2e

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['batch_jobs'] += 1
                if size is not None:
                    self._in_memory_stats['calculations'] += size

            logger.info(
                "Recorded batch job: status=%s, size=%s, successful=%s, "
                "failed=%s, duration=%.2fs, emissions=%.4f kgCO2e",
                status,
                size if size is not None else "N/A",
                successful if successful is not None else "N/A",
                failed if failed is not None else "N/A",
                duration_s if duration_s else 0.0,
                total_emissions_kgco2e if total_emissions_kgco2e else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record batch job metrics: %s",
                e, exc_info=True
            )

    def record_error(
        self,
        error_type: str,
        operation: Optional[str] = None,
        activity_type: Optional[str] = None,
        fuel_type: Optional[str] = None
    ) -> None:
        """
        Record a calculation error occurrence.

        Tracks error frequency by type for diagnostics, alerting, and
        root cause analysis. Error types specific to Scope 3 Category 3
        include WTT factor unavailable, T&D loss factor unavailable,
        upstream factor unavailable, and energy unit conversion errors.

        Args:
            error_type: Type of error (validation_error/wtt_factor_unavailable/etc)
            operation: Operation where the error occurred (optional, for logging)
            activity_type: Activity type where error occurred (optional, for logging)
            fuel_type: Fuel type related to error (optional, for logging)

        Example:
            >>> metrics.record_error(
            ...     error_type="wtt_factor_unavailable",
            ...     operation="calculate_wtt_emissions",
            ...     fuel_type="hydrogen"
            ... )
        """
        try:
            # Validate error type
            error_type = self._validate_enum_value(
                error_type, ErrorType, ErrorType.VALIDATION_ERROR.value
            )

            # Increment error counter
            self.calculation_errors_total.labels(
                error_type=error_type
            ).inc()

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['errors'] += 1

            logger.debug(
                "Recorded error: type=%s, operation=%s, activity=%s, fuel=%s",
                error_type,
                operation if operation else "N/A",
                activity_type if activity_type else "N/A",
                fuel_type if fuel_type else "N/A"
            )

        except Exception as e:
            logger.error("Failed to record error metrics: %s", e, exc_info=True)

    # ======================================================================
    # Gauge methods (active calculations)
    # ======================================================================

    def inc_active(self, amount: float = 1) -> None:
        """
        Increment the active calculations gauge.

        Call this when a new calculation begins. Pair with dec_active()
        when the calculation completes (use track_calculation context
        manager for automatic inc/dec).

        Args:
            amount: Amount to increment by (default: 1)

        Example:
            >>> metrics.inc_active()
            >>> try:
            ...     result = perform_wtt_calculation()
            ... finally:
            ...     metrics.dec_active()
        """
        try:
            self.active_calculations.inc(amount)

            logger.debug("Active calculations incremented by %.0f", amount)

        except Exception as e:
            logger.error(
                "Failed to increment active calculations: %s",
                e, exc_info=True
            )

    def dec_active(self, amount: float = 1) -> None:
        """
        Decrement the active calculations gauge.

        Call this when a calculation completes (successfully or not).

        Args:
            amount: Amount to decrement by (default: 1)

        Example:
            >>> metrics.dec_active()
        """
        try:
            self.active_calculations.dec(amount)

            logger.debug("Active calculations decremented by %.0f", amount)

        except Exception as e:
            logger.error(
                "Failed to decrement active calculations: %s",
                e, exc_info=True
            )

    # ======================================================================
    # Summary and reset
    # ======================================================================

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of metrics collected since initialization.

        Returns a dictionary with all in-memory counters, uptime information,
        calculated rates (per-hour throughput), and activity type breakdown.

        Returns:
            Dictionary with metrics summary including counts, uptime, rates,
            and activity breakdown.

        Example:
            >>> summary = metrics.get_metrics_summary()
            >>> print(summary['calculations'])
            5432
            >>> print(summary['rates']['calculations_per_hour'])
            271.6
            >>> print(summary['emissions_kgco2e'])
            1250000.0
        """
        try:
            uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()
            uptime_hours = uptime_seconds / 3600 if uptime_seconds > 0 else 1.0

            with self._stats_lock:
                stats_snapshot = dict(self._in_memory_stats)

            summary: Dict[str, Any] = {
                'prometheus_available': PROMETHEUS_AVAILABLE,
                'agent': 'fuel_energy_activities',
                'agent_id': 'GL-MRV-S3-003',
                'prefix': 'gl_fea_',
                'scope': 'Scope 3 Category 3',
                'description': 'Fuel- and Energy-Related Activities',
                'metrics_count': 12,
                'uptime_seconds': uptime_seconds,
                'uptime_hours': uptime_seconds / 3600,
                'start_time': self._start_time.isoformat(),
                'current_time': datetime.utcnow().isoformat(),
                **stats_snapshot,
                'rates': {
                    'calculations_per_hour': (
                        stats_snapshot['calculations'] / uptime_hours
                    ),
                    'emissions_kgco2e_per_hour': (
                        stats_snapshot['emissions_kgco2e'] / uptime_hours
                    ),
                    'fuel_consumed_kwh_per_hour': (
                        stats_snapshot['fuel_consumed_kwh'] / uptime_hours
                    ),
                    'electricity_consumed_kwh_per_hour': (
                        stats_snapshot['electricity_consumed_kwh'] / uptime_hours
                    ),
                    'td_losses_kwh_per_hour': (
                        stats_snapshot['td_losses_kwh'] / uptime_hours
                    ),
                    'wtt_lookups_per_hour': (
                        stats_snapshot['wtt_factor_lookups'] / uptime_hours
                    ),
                    'upstream_lookups_per_hour': (
                        stats_snapshot['upstream_factor_lookups'] / uptime_hours
                    ),
                    'compliance_checks_per_hour': (
                        stats_snapshot['compliance_checks'] / uptime_hours
                    ),
                    'batch_jobs_per_hour': (
                        stats_snapshot['batch_jobs'] / uptime_hours
                    ),
                    'errors_per_hour': (
                        stats_snapshot['errors'] / uptime_hours
                    ),
                },
                'sub_category_breakdown': {
                    'a_upstream_fuels_wtt': {
                        'fuel_consumed_kwh': stats_snapshot['fuel_consumed_kwh'],
                        'wtt_factor_lookups': stats_snapshot['wtt_factor_lookups'],
                    },
                    'b_upstream_electricity': {
                        'electricity_consumed_kwh': stats_snapshot['electricity_consumed_kwh'],
                        'upstream_factor_lookups': stats_snapshot['upstream_factor_lookups'],
                    },
                    'c_td_losses': {
                        'td_losses_kwh': stats_snapshot['td_losses_kwh'],
                    },
                },
            }

            logger.debug(
                "Generated metrics summary: %d calculations tracked",
                stats_snapshot['calculations']
            )
            return summary

        except Exception as e:
            logger.error("Failed to generate metrics summary: %s", e, exc_info=True)
            return {
                'error': str(e),
                'prometheus_available': PROMETHEUS_AVAILABLE,
                'agent': 'fuel_energy_activities',
            }

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance for testing purposes.

        This destroys the existing singleton so that a fresh instance
        will be created on next access. Primarily used in unit tests.

        WARNING: This is NOT safe for concurrent use. It should only
        be called in test teardown when no other threads are accessing
        the metrics instance.

        Example:
            >>> FuelEnergyActivitiesMetrics.reset()
            >>> metrics = FuelEnergyActivitiesMetrics()  # Fresh instance
        """
        with cls._lock:
            if cls._instance is not None:
                # Clear the initialized flag so __init__ runs again
                if hasattr(cls._instance, '_initialized'):
                    del cls._instance._initialized
                cls._instance = None

                # Also reset the module-level singleton
                global _metrics_instance
                _metrics_instance = None

                logger.info("FuelEnergyActivitiesMetrics singleton reset")

    def reset_stats(self) -> None:
        """
        Reset in-memory statistics (not Prometheus metrics).

        Note: This only resets the in-memory counters used for get_metrics_summary().
        Prometheus metrics are cumulative and cannot be reset without restarting
        the process. This method also resets the start_time for rate calculations.

        Example:
            >>> metrics.reset_stats()
            >>> summary = metrics.get_metrics_summary()
            >>> assert summary['calculations'] == 0
        """
        try:
            with self._stats_lock:
                self._in_memory_stats = {
                    'calculations': 0,
                    'emissions_kgco2e': 0.0,
                    'fuel_consumed_kwh': 0.0,
                    'electricity_consumed_kwh': 0.0,
                    'td_losses_kwh': 0.0,
                    'wtt_factor_lookups': 0,
                    'upstream_factor_lookups': 0,
                    'compliance_checks': 0,
                    'batch_jobs': 0,
                    'errors': 0,
                }
            self._start_time = datetime.utcnow()

            logger.info("Reset in-memory statistics for FuelEnergyActivitiesMetrics")

        except Exception as e:
            logger.error("Failed to reset statistics: %s", e, exc_info=True)

    # ======================================================================
    # Internal helpers
    # ======================================================================

    @staticmethod
    def _validate_enum_value(
        value: Optional[str],
        enum_class: type,
        default: str
    ) -> str:
        """
        Validate a string value against an Enum class.

        If the value is None or not a valid member of the enum, logs a
        warning and returns the default. This ensures that label values
        in Prometheus metrics always have bounded cardinality.

        Args:
            value: The string value to validate
            enum_class: The Enum class to validate against
            default: The default value if validation fails

        Returns:
            Validated value or default

        Example:
            >>> FuelEnergyActivitiesMetrics._validate_enum_value(
            ...     "natural_gas", FuelType, "other"
            ... )
            'natural_gas'
            >>> FuelEnergyActivitiesMetrics._validate_enum_value(
            ...     "invalid_fuel", FuelType, "other"
            ... )
            'other'
        """
        if value is None:
            return default

        valid_values = [m.value for m in enum_class]
        if value not in valid_values:
            logger.warning(
                "Invalid %s value '%s', using default '%s'",
                enum_class.__name__, value, default
            )
            return default

        return value


# ===========================================================================
# Module-level singleton accessor
# ===========================================================================

_metrics_instance: Optional[FuelEnergyActivitiesMetrics] = None
_metrics_lock: threading.Lock = threading.Lock()


def get_metrics() -> FuelEnergyActivitiesMetrics:
    """
    Get the singleton FuelEnergyActivitiesMetrics instance.

    Thread-safe accessor for the global metrics instance. Prefer this
    function over direct instantiation for consistency across the
    fuel & energy activities agent codebase.

    Returns:
        FuelEnergyActivitiesMetrics singleton instance

    Example:
        >>> from greenlang.fuel_energy_activities.metrics import get_metrics
        >>> metrics = get_metrics()
        >>> metrics.record_calculation(
        ...     activity_type="wtt_fuel",
        ...     calculation_method="emission_factor",
        ...     status="success",
        ...     duration_s=0.45,
        ...     emissions_kgco2e=1250.8
        ... )
    """
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = FuelEnergyActivitiesMetrics()

    return _metrics_instance


# ===========================================================================
# Context manager helpers
# ===========================================================================

@contextmanager
def track_calculation(
    activity_type: str = "wtt_fuel",
    calculation_method: str = "emission_factor"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a calculation's lifecycle.

    Automatically increments/decrements the active calculations gauge
    and records duration when the context exits. The caller can set
    ``context['emissions_kgco2e']``, ``context['status']``, ``context['gas']``,
    and ``context['fuel_type']`` before exiting to record those values.

    Args:
        activity_type: Activity type being calculated
        calculation_method: Calculation method being used

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_calculation("wtt_fuel", "emission_factor") as ctx:
        ...     result = perform_wtt_calculation()
        ...     ctx['emissions_kgco2e'] = result.emissions
        ...     ctx['fuel_type'] = "natural_gas"
        ...     ctx['status'] = "success"
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'activity_type': activity_type,
        'calculation_method': calculation_method,
        'status': 'success',
        'emissions_kgco2e': None,
        'gas': None,
        'fuel_type': None,
        'start_time': time.monotonic(),
    }

    # Increment active gauge
    metrics.inc_active()

    try:
        yield context

    except Exception as exc:
        context['status'] = 'failed'
        context['error'] = str(exc)
        logger.error(
            "Calculation failed in track_calculation context: %s",
            exc, exc_info=True
        )
        raise

    finally:
        # Calculate duration
        duration_s = time.monotonic() - context['start_time']

        # Decrement active gauge
        metrics.dec_active()

        # Record the calculation
        metrics.record_calculation(
            activity_type=context['activity_type'],
            calculation_method=context['calculation_method'],
            status=context['status'],
            duration_s=duration_s,
            emissions_kgco2e=context.get('emissions_kgco2e'),
            gas=context.get('gas'),
            fuel_type=context.get('fuel_type')
        )


@contextmanager
def track_duration(
    activity_type: str = "wtt_fuel",
    calculation_method: str = "emission_factor"
) -> Generator[None, None, None]:
    """
    Context manager that tracks the duration of an arbitrary operation.

    Records the elapsed time in the calculation_duration_seconds histogram
    when the context exits.

    Args:
        activity_type: Activity type label for the duration histogram
        calculation_method: Calculation method label for the duration histogram

    Yields:
        None

    Example:
        >>> with track_duration("wtt_fuel", "emission_factor"):
        ...     factors = load_wtt_factors("natural_gas")
    """
    metrics = get_metrics()
    start = time.monotonic()

    try:
        yield

    finally:
        duration_s = time.monotonic() - start

        try:
            metrics.calculation_duration_seconds.labels(
                activity_type=activity_type,
                calculation_method=calculation_method
            ).observe(duration_s)

            logger.debug(
                "Tracked duration: activity=%s, method=%s, duration=%.4fs",
                activity_type, calculation_method, duration_s
            )

        except Exception as e:
            logger.error(
                "Failed to record duration for activity=%s: %s",
                activity_type, e, exc_info=True
            )


@contextmanager
def track_wtt_fuel(
    fuel_type: str = "natural_gas",
    fuel_category: str = "gaseous_fossil",
    calculation_method: str = "emission_factor"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a WTT fuel calculation lifecycle.

    Automatically records the WTT fuel calculation metrics when the
    context exits. The caller should populate ``context['emissions_kgco2e']``
    and ``context['kwh']`` before exiting.

    Corresponds to Scope 3 Category 3 Sub-Category A: Upstream emissions
    of purchased fuels.

    Args:
        fuel_type: Specific fuel being calculated
        fuel_category: Broad fuel category
        calculation_method: Method used for the calculation

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_wtt_fuel("diesel", "liquid_fossil") as ctx:
        ...     result = wtt_engine.calculate(fuel_data)
        ...     ctx['emissions_kgco2e'] = result.total_kgco2e
        ...     ctx['kwh'] = result.energy_kwh
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'fuel_type': fuel_type,
        'fuel_category': fuel_category,
        'calculation_method': calculation_method,
        'status': 'success',
        'emissions_kgco2e': None,
        'kwh': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active()

    try:
        yield context

    except Exception as exc:
        context['status'] = 'failed'
        context['error'] = str(exc)
        logger.error(
            "WTT fuel calculation failed: %s",
            exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active()

        # Record the primary calculation
        metrics.record_calculation(
            activity_type=ActivityType.WTT_FUEL.value,
            calculation_method=context['calculation_method'],
            status=context['status'],
            duration_s=duration_s,
            emissions_kgco2e=context.get('emissions_kgco2e'),
            gas=EmissionGas.CO2E.value,
            fuel_type=context['fuel_type']
        )

        # Record fuel consumption if kWh provided
        kwh = context.get('kwh')
        if kwh is not None and kwh > 0:
            metrics.record_fuel_consumption(
                fuel_type=context['fuel_type'],
                fuel_category=context['fuel_category'],
                kwh=kwh,
                emissions_kgco2e=context.get('emissions_kgco2e')
            )


@contextmanager
def track_upstream_electricity(
    energy_type: str = "grid_electricity",
    grid_region: str = "other",
    calculation_method: str = "location_based"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks an upstream electricity calculation lifecycle.

    Corresponds to Scope 3 Category 3 Sub-Category B: Upstream emissions
    of purchased electricity (generation-related activities not in Scope 2).

    Args:
        energy_type: Type of energy being calculated
        grid_region: Grid region for location-based factors
        calculation_method: Method used (location_based/market_based/etc)

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_upstream_electricity("grid_electricity", "uk_grid") as ctx:
        ...     result = upstream_engine.calculate(electricity_data)
        ...     ctx['emissions_kgco2e'] = result.total_kgco2e
        ...     ctx['kwh'] = result.consumption_kwh
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'energy_type': energy_type,
        'grid_region': grid_region,
        'calculation_method': calculation_method,
        'status': 'success',
        'emissions_kgco2e': None,
        'kwh': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active()

    try:
        yield context

    except Exception as exc:
        context['status'] = 'failed'
        context['error'] = str(exc)
        logger.error(
            "Upstream electricity calculation failed: %s",
            exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active()

        # Record the primary calculation
        metrics.record_calculation(
            activity_type=ActivityType.UPSTREAM_ELECTRICITY.value,
            calculation_method=context['calculation_method'],
            status=context['status'],
            duration_s=duration_s,
            emissions_kgco2e=context.get('emissions_kgco2e')
        )

        # Record electricity consumption if kWh provided
        kwh = context.get('kwh')
        if kwh is not None and kwh > 0:
            metrics.record_electricity_consumption(
                energy_type=context['energy_type'],
                grid_region=context['grid_region'],
                kwh=kwh,
                emissions_kgco2e=context.get('emissions_kgco2e')
            )


@contextmanager
def track_td_loss(
    country: str = "other",
    calculation_method: str = "emission_factor"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a T&D loss calculation lifecycle.

    Corresponds to Scope 3 Category 3 Sub-Category C: Transmission
    and distribution losses for purchased electricity.

    Args:
        country: Country where T&D losses occur
        calculation_method: Method used for the calculation

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_td_loss("gb") as ctx:
        ...     result = td_loss_engine.calculate(grid_data)
        ...     ctx['emissions_kgco2e'] = result.total_kgco2e
        ...     ctx['kwh'] = result.losses_kwh
        ...     ctx['loss_rate_pct'] = result.loss_rate
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'country': country,
        'calculation_method': calculation_method,
        'status': 'success',
        'emissions_kgco2e': None,
        'kwh': None,
        'loss_rate_pct': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active()

    try:
        yield context

    except Exception as exc:
        context['status'] = 'failed'
        context['error'] = str(exc)
        logger.error(
            "T&D loss calculation failed: %s",
            exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active()

        # Record the primary calculation
        metrics.record_calculation(
            activity_type=ActivityType.TD_LOSS_ELECTRICITY.value,
            calculation_method=context['calculation_method'],
            status=context['status'],
            duration_s=duration_s,
            emissions_kgco2e=context.get('emissions_kgco2e')
        )

        # Record T&D loss if kWh provided
        kwh = context.get('kwh')
        if kwh is not None and kwh > 0:
            metrics.record_td_loss(
                country=context['country'],
                kwh=kwh,
                loss_rate_pct=context.get('loss_rate_pct'),
                emissions_kgco2e=context.get('emissions_kgco2e')
            )


@contextmanager
def track_compliance_check(
    framework: str = "ghg_protocol"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a compliance check lifecycle.

    Args:
        framework: Compliance framework being checked

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_compliance_check("csrd") as ctx:
        ...     result = compliance_engine.check(report_data)
        ...     ctx['status'] = result.status
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'framework': framework,
        'status': 'compliant',
        'start_time': time.monotonic(),
    }

    try:
        yield context

    except Exception as exc:
        context['status'] = 'non_compliant'
        context['error'] = str(exc)
        logger.error(
            "Compliance check failed: %s",
            exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']

        metrics.record_compliance_check(
            framework=context['framework'],
            status=context['status'],
            duration_s=duration_s
        )


@contextmanager
def track_batch(
    activity_type: str = "combined"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a batch calculation lifecycle.

    Args:
        activity_type: Primary activity type for the batch

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_batch("wtt_fuel") as ctx:
        ...     results = batch_engine.process(fuel_records)
        ...     ctx['size'] = len(fuel_records)
        ...     ctx['successful'] = sum(1 for r in results if r.ok)
        ...     ctx['failed'] = sum(1 for r in results if not r.ok)
        ...     ctx['total_emissions_kgco2e'] = sum(r.kgco2e for r in results if r.ok)
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'activity_type': activity_type,
        'status': 'completed',
        'size': 0,
        'successful': None,
        'failed': None,
        'total_emissions_kgco2e': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active()

    try:
        yield context

    except Exception as exc:
        context['status'] = 'failed'
        context['error'] = str(exc)
        logger.error(
            "Batch calculation failed: %s",
            exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active()

        size = context.get('size', 0)
        if size > 0:
            metrics.record_batch_job(
                status=context['status'],
                size=size,
                successful=context.get('successful'),
                failed=context.get('failed'),
                duration_s=duration_s,
                total_emissions_kgco2e=context.get('total_emissions_kgco2e')
            )


@contextmanager
def track_wtt_lookup(
    source: str = "defra",
    fuel_type: str = "natural_gas"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a WTT factor lookup lifecycle.

    Args:
        source: Factor source database
        fuel_type: Fuel type being looked up

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_wtt_lookup("defra", "diesel") as ctx:
        ...     factor = wtt_db.lookup(fuel_type="diesel", year=2024)
        ...     ctx['count'] = 1
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'source': source,
        'fuel_type': fuel_type,
        'count': 1,
        'start_time': time.monotonic(),
    }

    try:
        yield context

    except Exception as exc:
        context['error'] = str(exc)
        metrics.record_error(
            error_type=ErrorType.WTT_FACTOR_UNAVAILABLE.value,
            operation="wtt_factor_lookup",
            fuel_type=fuel_type
        )
        logger.error(
            "WTT factor lookup failed: %s",
            exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']

        if 'error' not in context:
            metrics.record_wtt_lookup(
                source=context['source'],
                fuel_type=context['fuel_type'],
                count=context.get('count', 1),
                duration_s=duration_s
            )


@contextmanager
def track_upstream_lookup(
    source: str = "iea",
    country: str = "other"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks an upstream electricity factor lookup lifecycle.

    Args:
        source: Factor source database
        country: Country being looked up

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_upstream_lookup("iea", "gb") as ctx:
        ...     factor = upstream_db.lookup(country="gb", year=2024)
        ...     ctx['count'] = 1
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'source': source,
        'country': country,
        'count': 1,
        'start_time': time.monotonic(),
    }

    try:
        yield context

    except Exception as exc:
        context['error'] = str(exc)
        metrics.record_error(
            error_type=ErrorType.UPSTREAM_FACTOR_UNAVAILABLE.value,
            operation="upstream_factor_lookup",
            activity_type="upstream_electricity"
        )
        logger.error(
            "Upstream factor lookup failed: %s",
            exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']

        if 'error' not in context:
            metrics.record_upstream_lookup(
                source=context['source'],
                country=context['country'],
                count=context.get('count', 1),
                duration_s=duration_s
            )


# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    # Main class and accessor
    'FuelEnergyActivitiesMetrics',
    'get_metrics',

    # Context managers
    'track_calculation',
    'track_duration',
    'track_wtt_fuel',
    'track_upstream_electricity',
    'track_td_loss',
    'track_compliance_check',
    'track_batch',
    'track_wtt_lookup',
    'track_upstream_lookup',

    # Enumerations
    'ActivityType',
    'CalculationMethod',
    'CalculationStatus',
    'FuelType',
    'FuelCategory',
    'EnergyType',
    'GridRegion',
    'EmissionGas',
    'WTTFactorSource',
    'UpstreamFactorSource',
    'Country',
    'Framework',
    'ComplianceStatus',
    'BatchStatus',
    'ErrorType',

    # Availability flag
    'PROMETHEUS_AVAILABLE',
]
