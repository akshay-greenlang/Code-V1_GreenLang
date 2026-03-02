# -*- coding: utf-8 -*-
"""
Prometheus Metrics - Employee Commuting Agent (AGENT-MRV-020)

Thread-safe singleton Prometheus metrics collection with graceful fallback
when prometheus_client is not installed. All metrics use the gl_ec_ prefix.

This module provides Prometheus metrics tracking for employee commuting
emissions calculations (Scope 3, Category 7) including distance-based,
fuel-based, average-data, survey-based, and spend-based calculation methods
across all commute modes (personal car, motorcycle, bus, rail, subway, tram,
light rail, ferry, cycling, walking, e-bike, e-scooter, carpool, vanpool,
shuttle, telework, multi-modal) and telework/remote work offsets.

Thread-safe singleton pattern with graceful fallback if Prometheus
unavailable.

Metrics prefix: gl_ec_

14 Prometheus Metrics:
    1.  gl_ec_calculations_total                - Counter: total calculations performed
    2.  gl_ec_calculation_duration_seconds       - Histogram: calculation latency
    3.  gl_ec_emissions_kg_co2e                 - Counter: total emissions in kgCO2e
    4.  gl_ec_batch_size                        - Histogram: batch sizes
    5.  gl_ec_batch_duration_seconds            - Histogram: batch processing time
    6.  gl_ec_ef_lookups_total                  - Counter: EF lookup count
    7.  gl_ec_ef_cache_hits_total               - Counter: cache hits
    8.  gl_ec_ef_cache_misses_total             - Counter: cache misses
    9.  gl_ec_compliance_checks_total           - Counter: compliance check count
    10. gl_ec_pipeline_stage_duration_seconds   - Histogram: per-stage timing
    11. gl_ec_data_quality_score                - Histogram: DQI scores
    12. gl_ec_survey_response_rate              - Histogram: survey response rates
    13. gl_ec_telework_fraction                 - Histogram: telework fraction dist
    14. gl_ec_active_calculations               - Gauge: currently active calcs

GHG Protocol Scope 3 Category 7 covers employee commuting:
    A. Transportation of employees between their homes and their
       worksites in vehicles not owned or operated by the
       reporting company.
    B. Includes all modes of commute transport: personal vehicles
       (car, motorcycle), public transit (bus, rail, subway, tram,
       light rail, ferry), shared transport (carpool, vanpool, shuttle),
       micro-mobility (e-bike, e-scooter), and active transport
       (cycling, walking).
    C. Telework/remote work emissions (home office energy use).
    D. Multi-modal commuting (combined modes in a single trip).
    E. Excludes business travel (Category 6) and upstream
       transportation (Category 4).

Calculation methods defined by GHG Protocol:
    - Distance-based: commute distance x mode-specific EF x working days
    - Fuel-based: fuel consumed x fuel EF (for personal vehicles)
    - Average-data: national/regional average commute emissions per employee
    - Survey-based: company-specific survey data aggregation
    - Spend-based: transit spend x spend-based EF per mode

Example:
    >>> metrics = EmployeeCommutingMetrics()
    >>> metrics.record_calculation(
    ...     mode="personal_car",
    ...     method="distance_based",
    ...     status="success",
    ...     duration=0.025,
    ...     co2e=1845.2,
    ...     tenant_id="tenant_abc"
    ... )

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-020 Employee Commuting (GL-MRV-S3-007)
Status: Production Ready
Version: 1.0.0
Agent: GL-MRV-S3-007
"""

import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generator, Optional

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

    class _NoOpMetric:
        """
        No-op metric stub for environments without prometheus_client.

        Implements all metric interface methods (labels, observe, inc, dec,
        set, time, info) as silent no-ops so that calling code does not
        need to check for Prometheus availability before recording metrics.

        This allows the Employee Commuting Agent to operate correctly in
        environments where prometheus_client is not installed (e.g.,
        lightweight test environments, CLI tools, or edge deployments).

        Example:
            >>> metric = _NoOpMetric("test_metric", "A test metric")
            >>> metric.labels(mode="car").inc()  # Silent no-op
            >>> metric.observe(0.5)              # Silent no-op
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Accept and discard all constructor arguments."""
            pass

        def labels(self, *args: Any, **kwargs: Any) -> "_NoOpMetric":
            """Return self for label chaining (no-op)."""
            return self

        def observe(self, amount: float = 0) -> None:
            """Accept and discard an observation value (no-op)."""
            pass

        def inc(self, amount: float = 1) -> None:
            """Accept and discard an increment value (no-op)."""
            pass

        def dec(self, amount: float = 1) -> None:
            """Accept and discard a decrement value (no-op)."""
            pass

        def set(self, value: float = 0) -> None:
            """Accept and discard a gauge set value (no-op)."""
            pass

        def time(self) -> "contextmanager":
            """Return a no-op context manager for timing (no-op)."""
            import contextlib
            return contextlib.nullcontext()

        def info(self, data: Optional[Dict[str, str]] = None) -> None:
            """Accept and discard info metric data (no-op)."""
            pass

    Counter = _NoOpMetric  # type: ignore[misc,assignment]
    Histogram = _NoOpMetric  # type: ignore[misc,assignment]
    Gauge = _NoOpMetric  # type: ignore[misc,assignment]
    Info = _NoOpMetric  # type: ignore[misc,assignment]


# ===========================================================================
# Enumerations -- Employee Commuting domain-specific label value sets
# ===========================================================================
# 11 bounded-cardinality enums for Prometheus label values.
# Each enum constrains the label values that can be written to a metric,
# preventing cardinality explosion in Prometheus TSDB.
# ===========================================================================


class CommuteModeLabelEnum(str, Enum):
    """
    Commute transport modes tracked for employee commuting emissions.

    Covers the primary commute modes defined in the GHG Protocol
    Technical Guidance for Calculating Scope 3 Emissions, Category 7.
    Includes personal vehicles, public transit, shared transport,
    micro-mobility, active transport, and telework as a pseudo-mode
    for remote work offset tracking.

    Values:
        PERSONAL_CAR: Single-occupancy personal vehicle commute
        MOTORCYCLE: Powered two-wheeler commute (scooter/motorcycle)
        BUS: Public bus transit (local, express, or coach)
        RAIL: Heavy rail / commuter rail transit
        SUBWAY: Underground metro / subway transit
        TRAM: Tram / streetcar transit
        LIGHT_RAIL: Light rail transit (surface or elevated)
        FERRY: Water-based commuter ferry or water taxi
        CYCLING: Human-powered bicycle commute (zero emissions)
        WALKING: Pedestrian commute (zero emissions)
        E_BIKE: Electric-assist bicycle commute
        E_SCOOTER: Electric kick-scooter commute
        CARPOOL: Shared personal vehicle (2+ occupants)
        VANPOOL: Employer/third-party operated shared van
        SHUTTLE: Company-provided shuttle bus service
        TELEWORK: Remote / work-from-home (home office energy)
        MULTI_MODAL: Combined modes in a single commute trip
        UNKNOWN: Unclassified or unreported commute mode
    """
    PERSONAL_CAR = "personal_car"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    RAIL = "rail"
    SUBWAY = "subway"
    TRAM = "tram"
    LIGHT_RAIL = "light_rail"
    FERRY = "ferry"
    CYCLING = "cycling"
    WALKING = "walking"
    E_BIKE = "e_bike"
    E_SCOOTER = "e_scooter"
    CARPOOL = "carpool"
    VANPOOL = "vanpool"
    SHUTTLE = "shuttle"
    TELEWORK = "telework"
    MULTI_MODAL = "multi_modal"
    UNKNOWN = "unknown"


class VehicleTypeLabelEnum(str, Enum):
    """
    Vehicle types for personal commute transport emissions tracking.

    Differentiates emission factors by vehicle size for personal
    vehicles. Vehicle type affects fuel consumption rates and
    therefore per-kilometre emission factors.

    Values:
        SMALL_CAR: Compact / sub-compact (e.g., Honda Fit, VW Polo)
        MEDIUM_CAR: Mid-size sedan (e.g., Toyota Camry, VW Golf)
        LARGE_CAR: Full-size sedan / estate (e.g., BMW 5-series)
        SUV: Sport utility vehicle / crossover
        PICKUP_TRUCK: Light-duty pickup truck
        MINIVAN: Passenger minivan / MPV
        MOTORCYCLE_SMALL: Motorcycle / scooter <= 125cc
        MOTORCYCLE_LARGE: Motorcycle > 125cc
        UNKNOWN: Unspecified or unreported vehicle type
    """
    SMALL_CAR = "small_car"
    MEDIUM_CAR = "medium_car"
    LARGE_CAR = "large_car"
    SUV = "suv"
    PICKUP_TRUCK = "pickup_truck"
    MINIVAN = "minivan"
    MOTORCYCLE_SMALL = "motorcycle_small"
    MOTORCYCLE_LARGE = "motorcycle_large"
    UNKNOWN = "unknown"


class FuelTypeLabelEnum(str, Enum):
    """
    Fuel types for personal vehicle commute emissions tracking.

    Differentiates emission factors by propulsion technology and
    fuel type. Critical for accurate per-km emission calculations
    as fuel carbon content varies significantly between types.

    Values:
        GASOLINE: Conventional petrol / gasoline engine
        DIESEL: Conventional diesel engine
        HYBRID: Non-plug-in hybrid electric (HEV)
        PLUGIN_HYBRID: Plug-in hybrid electric (PHEV)
        ELECTRIC: Battery electric vehicle (BEV, zero tailpipe)
        CNG: Compressed natural gas vehicle
        LPG: Liquefied petroleum gas vehicle
        HYDROGEN: Hydrogen fuel cell vehicle (FCEV)
        E85: Flex-fuel ethanol (85%) blend
        UNKNOWN: Unspecified or unreported fuel type
    """
    GASOLINE = "gasoline"
    DIESEL = "diesel"
    HYBRID = "hybrid"
    PLUGIN_HYBRID = "plugin_hybrid"
    ELECTRIC = "electric"
    CNG = "cng"
    LPG = "lpg"
    HYDROGEN = "hydrogen"
    E85 = "e85"
    UNKNOWN = "unknown"


class TransitTypeLabelEnum(str, Enum):
    """
    Public transit sub-types for commute emissions tracking.

    Provides granularity beyond the top-level commute mode for
    public transit options. Different transit sub-types have
    materially different per-passenger-km emission factors due
    to differences in vehicle capacity, occupancy, and energy
    sources.

    Values:
        LOCAL_BUS: City / local route bus
        EXPRESS_BUS: Express / limited-stop bus
        COACH: Long-distance / intercity coach bus
        COMMUTER_RAIL: Suburban / commuter heavy rail
        SUBWAY_METRO: Underground metro / rapid transit
        LIGHT_RAIL: Surface or elevated light rail
        TRAM_STREETCAR: Tram / streetcar (shared right-of-way)
        FERRY_BOAT: Commuter ferry / passenger vessel
        WATER_TAXI: Small water taxi / river bus
        UNKNOWN: Unspecified transit sub-type
    """
    LOCAL_BUS = "local_bus"
    EXPRESS_BUS = "express_bus"
    COACH = "coach"
    COMMUTER_RAIL = "commuter_rail"
    SUBWAY_METRO = "subway_metro"
    LIGHT_RAIL = "light_rail"
    TRAM_STREETCAR = "tram_streetcar"
    FERRY_BOAT = "ferry_boat"
    WATER_TAXI = "water_taxi"
    UNKNOWN = "unknown"


class TeleworkCategoryLabelEnum(str, Enum):
    """
    Telework frequency categories for remote work emissions tracking.

    Categorizes employees by telework frequency to enable accurate
    modelling of avoided commute emissions versus incremental home
    office energy emissions. GHG Protocol Technical Guidance notes
    that telework patterns materially affect Category 7 totals.

    Values:
        FULL_REMOTE: 5 days/week remote (no commute)
        HYBRID_1DAY: 1 day/week remote, 4 days office
        HYBRID_2DAY: 2 days/week remote, 3 days office
        HYBRID_3DAY: 3 days/week remote, 2 days office
        HYBRID_4DAY: 4 days/week remote, 1 day office
        OFFICE_BASED: 5 days/week office (no telework)
        UNKNOWN: Unspecified telework pattern
    """
    FULL_REMOTE = "full_remote"
    HYBRID_1DAY = "hybrid_1day"
    HYBRID_2DAY = "hybrid_2day"
    HYBRID_3DAY = "hybrid_3day"
    HYBRID_4DAY = "hybrid_4day"
    OFFICE_BASED = "office_based"
    UNKNOWN = "unknown"


class CalculationMethodLabelEnum(str, Enum):
    """
    Calculation methods for employee commuting emissions.

    GHG Protocol Scope 3 Category 7 supports several approaches
    depending on data availability and the level of accuracy required:
        - Distance-based: commute distance x mode-specific EF x days
        - Fuel-based: fuel consumed x fuel emission factor
        - Average-data: national/regional average per employee
        - Survey-based: company-specific survey data aggregation
        - Spend-based: transit spend x spend-based EF per mode

    Values:
        DISTANCE_BASED: Per-trip distance x mode EF x working days
        FUEL_BASED: Fuel consumed x fuel-specific EF (personal vehicles)
        AVERAGE_DATA: National/regional average per employee FTE
        SURVEY_BASED: Company survey data, extrapolated to population
        SPEND_BASED: Transit spend x spend-based EF per mode
        UNKNOWN: Unspecified calculation method
    """
    DISTANCE_BASED = "distance_based"
    FUEL_BASED = "fuel_based"
    AVERAGE_DATA = "average_data"
    SURVEY_BASED = "survey_based"
    SPEND_BASED = "spend_based"
    UNKNOWN = "unknown"


class ComplianceFrameworkLabelEnum(str, Enum):
    """
    Compliance frameworks for employee commuting reporting.

    Tracks validation against regulatory and voluntary reporting
    standards applicable to Scope 3 Category 7 emissions.

    Values:
        GHG_PROTOCOL: GHG Protocol Corporate Value Chain (Scope 3) Standard
        CSRD_E1: EU CSRD / ESRS E1 Climate Change
        CDP: CDP Climate Change Questionnaire
        SBTI: Science Based Targets initiative (SBTi)
        SB253: California SB 253 Climate Corporate Data Accountability Act
        GRI_305: GRI 305: Emissions standard
        ISO_14064: ISO 14064-1 GHG inventories
        UNKNOWN: Unspecified compliance framework
    """
    GHG_PROTOCOL = "ghg_protocol"
    CSRD_E1 = "csrd_e1"
    CDP = "cdp"
    SBTI = "sbti"
    SB253 = "sb253"
    GRI_305 = "gri_305"
    ISO_14064 = "iso_14064"
    UNKNOWN = "unknown"


class DataQualityTierLabelEnum(str, Enum):
    """
    Data quality tiers for employee commuting calculation inputs.

    Classifies the quality of input data used in commuting emission
    calculations. GHG Protocol recommends companies improve data
    quality over time, progressing from Tier 3 (estimates) toward
    Tier 1 (primary, measured data).

    Values:
        TIER1: Primary data -- employee-specific survey, fuel records,
               GPS tracking, or transit card data
        TIER2: Secondary data -- regional averages, proxy data from
               similar organizations, partial survey with extrapolation
        TIER3: Tertiary data -- national averages, industry benchmarks,
               default assumptions, spend-based estimates
        UNKNOWN: Unspecified data quality tier
    """
    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"
    UNKNOWN = "unknown"


class PipelineStageLabelEnum(str, Enum):
    """
    Pipeline stages for the Employee Commuting calculation pipeline.

    Tracks per-stage execution timing for performance monitoring and
    bottleneck identification. Each stage in the pipeline has distinct
    performance characteristics and failure modes.

    Values:
        VALIDATE: Input data validation and schema conformance
        CLASSIFY: Commute mode classification and categorization
        NORMALIZE: Unit normalization (distance, currency, fuel volume)
        RESOLVE_EFS: Emission factor resolution and lookup
        CALCULATE_COMMUTE: Transport commute emission calculation
        CALCULATE_TELEWORK: Telework/remote work emission calculation
        COMPLIANCE: Regulatory compliance checks across frameworks
        AGGREGATE: Result aggregation across employees and modes
        SEAL: Provenance hashing and audit trail sealing
        UNKNOWN: Unspecified pipeline stage
    """
    VALIDATE = "validate"
    CLASSIFY = "classify"
    NORMALIZE = "normalize"
    RESOLVE_EFS = "resolve_efs"
    CALCULATE_COMMUTE = "calculate_commute"
    CALCULATE_TELEWORK = "calculate_telework"
    COMPLIANCE = "compliance"
    AGGREGATE = "aggregate"
    SEAL = "seal"
    UNKNOWN = "unknown"


class StatusLabelEnum(str, Enum):
    """
    General operation status labels for employee commuting metrics.

    Provides a bounded set of status values used across multiple
    metrics to indicate operation outcomes.

    Values:
        SUCCESS: Operation completed successfully
        FAILURE: Operation failed with a known/expected error
        ERROR: Operation failed with an unexpected error
        TIMEOUT: Operation exceeded its time budget
        UNKNOWN: Indeterminate status
    """
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class TenantLabelEnum(str, Enum):
    """
    Tenant label placeholder for multi-tenant metrics isolation.

    In a multi-tenant deployment the tenant_id label is dynamic
    but bounded by truncation to 32 characters. This enum provides
    a sentinel value for cases where tenant_id is missing or
    unavailable.

    Note: Unlike other label enums, tenant_id is validated by
    truncation rather than strict enum membership, because the
    set of valid tenant identifiers is dynamic. The enum here
    provides the fallback/default value only.

    Values:
        UNKNOWN: Unspecified or missing tenant identifier
    """
    UNKNOWN = "unknown"


# ===========================================================================
# EmployeeCommutingMetrics -- Thread-safe Singleton
# ===========================================================================


class EmployeeCommutingMetrics:
    """
    Thread-safe singleton metrics collector for Employee Commuting Agent (MRV-020).

    Provides 14 Prometheus metrics for tracking Scope 3 Category 7
    employee commuting emissions calculations, including distance-based,
    fuel-based, average-data, survey-based, and spend-based methods
    across all commute modes and telework offsets.

    All metrics use the ``gl_ec_`` prefix to ensure namespace isolation
    within the GreenLang Prometheus ecosystem.

    Scope 3 Category 7 Sub-Categories Tracked:
        A. Personal vehicle commuting (car, motorcycle, EV)
        B. Public transit (bus, rail, subway, tram, light rail, ferry)
        C. Shared transport (carpool, vanpool, shuttle)
        D. Micro-mobility (e-bike, e-scooter)
        E. Active transport (cycling, walking) -- zero-emission baseline
        F. Telework/remote work (home office energy)
        G. Multi-modal commuting (combined modes per trip)

    Calculation Methods Supported:
        - Distance-based: commute distance x mode-specific EF x working days
        - Fuel-based: fuel consumed x fuel emission factor
        - Average-data: national/regional average commute emissions per employee
        - Survey-based: company-specific survey data aggregation
        - Spend-based: transit spend x spend-based EF per mode

    Prometheus Metrics (14):
        1.  gl_ec_calculations_total               (Counter)
        2.  gl_ec_calculation_duration_seconds      (Histogram)
        3.  gl_ec_emissions_kg_co2e                (Counter)
        4.  gl_ec_batch_size                       (Histogram)
        5.  gl_ec_batch_duration_seconds           (Histogram)
        6.  gl_ec_ef_lookups_total                 (Counter)
        7.  gl_ec_ef_cache_hits_total              (Counter)
        8.  gl_ec_ef_cache_misses_total            (Counter)
        9.  gl_ec_compliance_checks_total          (Counter)
        10. gl_ec_pipeline_stage_duration_seconds   (Histogram)
        11. gl_ec_data_quality_score               (Histogram)
        12. gl_ec_survey_response_rate             (Histogram)
        13. gl_ec_telework_fraction                (Histogram)
        14. gl_ec_active_calculations              (Gauge)

    Attributes:
        calculations_total: Counter for total calculation operations
        calculation_duration_seconds: Histogram for calculation latency
        emissions_kg_co2e: Counter for total emissions in kgCO2e
        batch_size: Histogram for batch sizes
        batch_duration_seconds: Histogram for batch processing time
        ef_lookups_total: Counter for EF lookup count
        ef_cache_hits_total: Counter for cache hits
        ef_cache_misses_total: Counter for cache misses
        compliance_checks_total: Counter for compliance checks
        pipeline_stage_duration_seconds: Histogram for per-stage timing
        data_quality_score: Histogram for DQI scores
        survey_response_rate: Histogram for survey response rates
        telework_fraction: Histogram for telework fraction distribution
        active_calculations: Gauge for currently active calculations

    Example:
        >>> metrics = EmployeeCommutingMetrics()
        >>> metrics.record_calculation(
        ...     mode="personal_car",
        ...     method="distance_based",
        ...     status="success",
        ...     duration=0.025,
        ...     co2e=1845.2,
        ...     tenant_id="tenant_abc"
        ... )
        >>> summary = metrics.get_metrics_summary()
        >>> assert summary['calculations'] >= 1
    """

    _instance: Optional["EmployeeCommutingMetrics"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "EmployeeCommutingMetrics":
        """
        Thread-safe singleton instantiation using double-checked locking.

        Returns:
            The singleton EmployeeCommutingMetrics instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize metrics (only once due to singleton pattern).

        Sets up in-memory statistics counters, initializes all 14
        Prometheus metrics with the gl_ec_ prefix, and logs the
        initialization status including Prometheus availability.
        """
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._start_time: datetime = datetime.utcnow()
        self._stats_lock: threading.Lock = threading.Lock()
        self._in_memory_stats: Dict[str, Any] = {
            "calculations": 0,
            "emissions_kg_co2e": 0.0,
            "batches": 0,
            "ef_lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "compliance_checks": 0,
            "pipeline_stages": 0,
            "data_quality_records": 0,
            "survey_responses": 0,
            "telework_records": 0,
            "errors": 0,
        }

        # Initialize all 14 Prometheus metrics
        self._init_metrics()

        logger.info(
            "EmployeeCommutingMetrics initialized with 14 metrics "
            "(Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable",
        )

    # ======================================================================
    # Metric initialization
    # ======================================================================

    def _init_metrics(self) -> None:
        """
        Initialize all 14 Prometheus metrics with gl_ec_ prefix.

        Each metric is documented with its purpose, label set, and the
        Scope 3 Category 7 aspect it supports.

        When Prometheus is available, metric registration may fail if the
        metrics were previously registered (e.g., after a reset() call in
        tests). In that case we unregister from the default registry and
        re-register to obtain fresh collector objects.
        """
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import REGISTRY

            def _safe_create(
                metric_cls: type, name: str, *args: Any, **kwargs: Any
            ) -> Any:
                """Create a metric, unregistering any prior collector on conflict."""
                try:
                    return metric_cls(name, *args, **kwargs)
                except ValueError:
                    try:
                        REGISTRY.unregister(
                            REGISTRY._names_to_collectors.get(name)
                        )
                    except Exception:
                        for collector in list(
                            REGISTRY._names_to_collectors.values()
                        ):
                            try:
                                REGISTRY.unregister(collector)
                            except Exception:
                                pass
                    return metric_cls(name, *args, **kwargs)
        else:

            def _safe_create(
                metric_cls: type, name: str, *args: Any, **kwargs: Any
            ) -> Any:
                """No-op stub creation (Prometheus not available)."""
                return metric_cls(name, *args, **kwargs)

        # ------------------------------------------------------------------
        # 1. gl_ec_calculations_total (Counter)
        #    Total employee commuting emission calculations performed.
        #    Labels:
        #      - mode: personal_car, motorcycle, bus, rail, subway, tram,
        #              light_rail, ferry, cycling, walking, e_bike,
        #              e_scooter, carpool, vanpool, shuttle, telework,
        #              multi_modal, unknown
        #      - method: distance_based, fuel_based, average_data,
        #                survey_based, spend_based, unknown
        #      - status: success, failure, error, timeout, unknown
        #      - tenant_id: tenant identifier (truncated to 32 chars)
        #    Primary throughput counter for all calculation operations
        #    across methods and commute modes.
        # ------------------------------------------------------------------
        self.calculations_total = _safe_create(
            Counter,
            "gl_ec_calculations_total",
            "Total employee commuting emission calculations performed",
            ["mode", "method", "status", "tenant_id"],
        )

        # ------------------------------------------------------------------
        # 2. gl_ec_calculation_duration_seconds (Histogram)
        #    Duration of individual calculation operations in seconds.
        #    Labels:
        #      - mode: Commute mode being calculated
        #      - method: Calculation method used
        #    Buckets tuned for typical employee commuting calculation
        #    latencies:
        #      - 5-10ms for cached factor lookups and simple distance x EF
        #      - 25-100ms for survey aggregation per employee
        #      - 250ms-1s for multi-mode commute calculations
        #      - 1-10s for full-company batch or extrapolation calculations
        # ------------------------------------------------------------------
        self.calculation_duration_seconds = _safe_create(
            Histogram,
            "gl_ec_calculation_duration_seconds",
            "Duration of employee commuting calculation operations in seconds",
            ["mode", "method"],
            buckets=(
                0.005, 0.01, 0.025, 0.05, 0.1,
                0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ),
        )

        # ------------------------------------------------------------------
        # 3. gl_ec_emissions_kg_co2e (Counter)
        #    Total emissions calculated in kilograms CO2-equivalent.
        #    Labels:
        #      - mode: Commute mode generating emissions
        #      - method: Calculation method used
        #      - tenant_id: Tenant identifier
        #    Tracks cumulative emissions output for rate calculation and
        #    modal breakdown. Uses kgCO2e as individual commute trips
        #    may produce modest amounts that would round to zero in tCO2e.
        # ------------------------------------------------------------------
        self.emissions_kg_co2e = _safe_create(
            Counter,
            "gl_ec_emissions_kg_co2e",
            "Total employee commuting emissions calculated in kgCO2e",
            ["mode", "method", "tenant_id"],
        )

        # ------------------------------------------------------------------
        # 4. gl_ec_batch_size (Histogram)
        #    Size of batch calculation operations (number of employee
        #    records). Labels:
        #      - method: Calculation method used for the batch
        #    Buckets cover typical batch sizes from a single employee to
        #    large-scale company-wide calculations spanning tens of
        #    thousands of employees.
        # ------------------------------------------------------------------
        self.batch_size = _safe_create(
            Histogram,
            "gl_ec_batch_size",
            "Batch calculation size for employee commuting operations",
            ["method"],
            buckets=(
                1, 5, 10, 25, 50, 100, 250, 500,
                1000, 2500, 5000, 10000,
            ),
        )

        # ------------------------------------------------------------------
        # 5. gl_ec_batch_duration_seconds (Histogram)
        #    Duration of batch processing operations in seconds.
        #    Labels:
        #      - method: Calculation method used for the batch
        #    Buckets cover typical batch durations from sub-second (small
        #    batches with cached EFs) to minutes (full company-wide
        #    calculations with complex multi-modal survey data).
        # ------------------------------------------------------------------
        self.batch_duration_seconds = _safe_create(
            Histogram,
            "gl_ec_batch_duration_seconds",
            "Duration of batch processing for employee commuting operations",
            ["method"],
            buckets=(
                0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
                10.0, 30.0, 60.0, 120.0, 300.0,
            ),
        )

        # ------------------------------------------------------------------
        # 6. gl_ec_ef_lookups_total (Counter)
        #    Total emission factor lookups performed.
        #    Labels:
        #      - mode: Commute mode for which the EF was looked up
        #      - source: EF database source (defra, epa, ghg_protocol, etc.)
        #      - status: success, failure, error, timeout, unknown
        #    Tracks the frequency and source distribution of EF retrievals
        #    for cache optimization and database coverage monitoring.
        # ------------------------------------------------------------------
        self.ef_lookups_total = _safe_create(
            Counter,
            "gl_ec_ef_lookups_total",
            "Total emission factor lookups for employee commuting",
            ["mode", "source", "status"],
        )

        # ------------------------------------------------------------------
        # 7. gl_ec_ef_cache_hits_total (Counter)
        #    Total emission factor cache hits.
        #    Labels:
        #      - mode: Commute mode for the cached EF
        #    Tracks cache effectiveness for emission factor lookups.
        #    High hit rates indicate good cache warming and reduce
        #    database load during batch calculations.
        # ------------------------------------------------------------------
        self.ef_cache_hits_total = _safe_create(
            Counter,
            "gl_ec_ef_cache_hits_total",
            "Total emission factor cache hits for employee commuting",
            ["mode"],
        )

        # ------------------------------------------------------------------
        # 8. gl_ec_ef_cache_misses_total (Counter)
        #    Total emission factor cache misses.
        #    Labels:
        #      - mode: Commute mode for the missed EF lookup
        #    Tracks cache misses to monitor cache warming effectiveness
        #    and identify modes with poor cache coverage that may benefit
        #    from pre-warming or larger cache sizes.
        # ------------------------------------------------------------------
        self.ef_cache_misses_total = _safe_create(
            Counter,
            "gl_ec_ef_cache_misses_total",
            "Total emission factor cache misses for employee commuting",
            ["mode"],
        )

        # ------------------------------------------------------------------
        # 9. gl_ec_compliance_checks_total (Counter)
        #    Total compliance checks performed.
        #    Labels:
        #      - framework: ghg_protocol, csrd_e1, cdp, sbti, sb253,
        #                   gri_305, iso_14064, unknown
        #      - status: success, failure, error, timeout, unknown
        #    Tracks regulatory compliance validation for Scope 3 Cat 7
        #    reporting across applicable frameworks.
        # ------------------------------------------------------------------
        self.compliance_checks_total = _safe_create(
            Counter,
            "gl_ec_compliance_checks_total",
            "Total employee commuting compliance checks performed",
            ["framework", "status"],
        )

        # ------------------------------------------------------------------
        # 10. gl_ec_pipeline_stage_duration_seconds (Histogram)
        #     Duration of individual pipeline stages in seconds.
        #     Labels:
        #       - stage: validate, classify, normalize, resolve_efs,
        #                calculate_commute, calculate_telework, compliance,
        #                aggregate, seal, unknown
        #     Enables per-stage performance monitoring and bottleneck
        #     identification within the employee commuting pipeline.
        # ------------------------------------------------------------------
        self.pipeline_stage_duration_seconds = _safe_create(
            Histogram,
            "gl_ec_pipeline_stage_duration_seconds",
            "Duration of pipeline stages for employee commuting",
            ["stage"],
            buckets=(
                0.001, 0.005, 0.01, 0.025, 0.05,
                0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
            ),
        )

        # ------------------------------------------------------------------
        # 11. gl_ec_data_quality_score (Histogram)
        #     Distribution of data quality indicator (DQI) scores.
        #     Labels:
        #       - method: Calculation method used
        #       - tier: Data quality tier (tier1, tier2, tier3, unknown)
        #     Scores range from 0.0 (worst) to 1.0 (best). Enables
        #     monitoring of input data quality trends over time and
        #     across calculation methods.
        # ------------------------------------------------------------------
        self.data_quality_score = _safe_create(
            Histogram,
            "gl_ec_data_quality_score",
            "Data quality indicator scores for employee commuting inputs",
            ["method", "tier"],
            buckets=(
                0.0, 0.1, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            ),
        )

        # ------------------------------------------------------------------
        # 12. gl_ec_survey_response_rate (Histogram)
        #     Distribution of survey response rates.
        #     Labels:
        #       - tenant_id: Tenant identifier
        #     Rates range from 0.0 (0%) to 1.0 (100%). GHG Protocol
        #     guidance recommends at least 80% response rates for
        #     accurate extrapolation. This histogram enables monitoring
        #     response rate trends and alerting on low rates.
        # ------------------------------------------------------------------
        self.survey_response_rate = _safe_create(
            Histogram,
            "gl_ec_survey_response_rate",
            "Survey response rates for employee commuting data collection",
            ["tenant_id"],
            buckets=(
                0.0, 0.1, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            ),
        )

        # ------------------------------------------------------------------
        # 13. gl_ec_telework_fraction (Histogram)
        #     Distribution of telework fractions across the employee
        #     population.
        #     Labels:
        #       - category: full_remote, hybrid_1day, hybrid_2day,
        #                   hybrid_3day, hybrid_4day, office_based, unknown
        #     Fractions range from 0.0 (fully office-based) to 1.0
        #     (fully remote). Tracks the distribution of telework
        #     patterns for net emission impact modelling.
        # ------------------------------------------------------------------
        self.telework_fraction = _safe_create(
            Histogram,
            "gl_ec_telework_fraction",
            "Telework fraction distribution for employee commuting",
            ["category"],
            buckets=(
                0.0, 0.1, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            ),
        )

        # ------------------------------------------------------------------
        # 14. gl_ec_active_calculations (Gauge)
        #     Number of currently active (in-flight) calculations.
        #     Labels:
        #       - method: Calculation method of the active calculation
        #     Tracks concurrency by method for capacity monitoring
        #     and resource allocation.
        # ------------------------------------------------------------------
        self.active_calculations = _safe_create(
            Gauge,
            "gl_ec_active_calculations",
            "Number of currently active employee commuting calculations",
            ["method"],
        )

        # ------------------------------------------------------------------
        # Agent info metric (non-numeric metadata)
        # ------------------------------------------------------------------
        self.agent_info = _safe_create(
            Info,
            "gl_ec_agent",
            "Employee Commuting Agent metadata",
        )
        if PROMETHEUS_AVAILABLE:
            try:
                self.agent_info.info({
                    "agent_id": "GL-MRV-S3-007",
                    "version": "1.0.0",
                    "scope": "scope_3_category_7",
                    "description": "Employee Commuting emissions calculator",
                    "metrics_count": "14",
                    "prefix": "gl_ec_",
                })
            except Exception:
                pass

    # ======================================================================
    # Convenience recording methods
    # ======================================================================

    def record_calculation(
        self,
        mode: str,
        method: str,
        status: str,
        duration: float,
        co2e: float,
        tenant_id: str = "unknown",
    ) -> None:
        """
        Record an employee commuting emission calculation operation.

        This is the primary recording method that increments the calculations
        counter, observes the duration histogram, and tracks emissions output.
        It covers all calculation methods and commute modes for Scope 3
        Category 7.

        Args:
            mode: Commute mode (personal_car/motorcycle/bus/rail/subway/tram/
                   light_rail/ferry/cycling/walking/e_bike/e_scooter/carpool/
                   vanpool/shuttle/telework/multi_modal/unknown)
            method: Calculation method (distance_based/fuel_based/average_data/
                     survey_based/spend_based/unknown)
            status: Calculation status (success/failure/error/timeout/unknown)
            duration: Operation duration in seconds
            co2e: Emissions calculated in kgCO2e
            tenant_id: Tenant identifier (default: "unknown")

        Example:
            >>> metrics.record_calculation(
            ...     mode="personal_car",
            ...     method="distance_based",
            ...     status="success",
            ...     duration=0.025,
            ...     co2e=1845.2,
            ...     tenant_id="tenant_abc"
            ... )
        """
        try:
            mode = self._validate_enum_value(
                mode, CommuteModeLabelEnum,
                CommuteModeLabelEnum.UNKNOWN.value,
            )
            method = self._validate_enum_value(
                method, CalculationMethodLabelEnum,
                CalculationMethodLabelEnum.UNKNOWN.value,
            )
            status = self._validate_enum_value(
                status, StatusLabelEnum,
                StatusLabelEnum.ERROR.value,
            )
            tenant_id = self._sanitize_tenant_id(tenant_id)

            # 1. Increment calculation counter
            self.calculations_total.labels(
                mode=mode,
                method=method,
                status=status,
                tenant_id=tenant_id,
            ).inc()

            # 2. Observe duration
            if duration is not None and duration > 0:
                self.calculation_duration_seconds.labels(
                    mode=mode,
                    method=method,
                ).observe(duration)

            # 3. Record emissions
            if co2e is not None and co2e > 0:
                self.emissions_kg_co2e.labels(
                    mode=mode,
                    method=method,
                    tenant_id=tenant_id,
                ).inc(co2e)

                with self._stats_lock:
                    self._in_memory_stats["emissions_kg_co2e"] += co2e

            # 4. Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats["calculations"] += 1
                if status in (
                    StatusLabelEnum.ERROR.value,
                    StatusLabelEnum.FAILURE.value,
                ):
                    self._in_memory_stats["errors"] += 1

            logger.debug(
                "Recorded calculation: mode=%s, method=%s, status=%s, "
                "duration=%.3fs, co2e=%.2f kgCO2e, tenant=%s",
                mode, method, status,
                duration if duration else 0.0,
                co2e if co2e else 0.0,
                tenant_id,
            )

        except Exception as e:
            logger.error(
                "Failed to record calculation metrics: %s", e, exc_info=True
            )

    def record_batch(
        self,
        method: str,
        size: int,
        duration: float,
    ) -> None:
        """
        Record a batch processing operation.

        Tracks batch sizes and durations for capacity planning and
        performance monitoring. Employee commuting calculations frequently
        run in batch mode across the full employee population during
        annual inventory periods.

        Args:
            method: Calculation method used for the batch
                     (distance_based/fuel_based/average_data/survey_based/
                      spend_based/unknown)
            size: Number of employee records in the batch
            duration: Batch processing duration in seconds

        Example:
            >>> metrics.record_batch(
            ...     method="survey_based",
            ...     size=5000,
            ...     duration=12.5
            ... )
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethodLabelEnum,
                CalculationMethodLabelEnum.UNKNOWN.value,
            )

            # Record batch size
            if size is not None and size > 0:
                self.batch_size.labels(method=method).observe(size)

            # Record batch duration
            if duration is not None and duration > 0:
                self.batch_duration_seconds.labels(method=method).observe(
                    duration
                )

            with self._stats_lock:
                self._in_memory_stats["batches"] += 1

            logger.debug(
                "Recorded batch: method=%s, size=%d, duration=%.3fs",
                method,
                size if size else 0,
                duration if duration else 0.0,
            )

        except Exception as e:
            logger.error(
                "Failed to record batch metrics: %s", e, exc_info=True
            )

    def record_ef_lookup(
        self,
        mode: str,
        source: str,
        status: str,
    ) -> None:
        """
        Record an emission factor lookup operation.

        Tracks the frequency, source distribution, and success rate of
        emission factor retrievals for cache optimization and database
        coverage monitoring.

        Args:
            mode: Commute mode for which the EF was looked up
            source: EF database source identifier (e.g., "defra", "epa",
                     "ghg_protocol"). This is a free-form string validated
                     for length only (max 32 chars).
            status: Lookup status (success/failure/error/timeout/unknown)

        Example:
            >>> metrics.record_ef_lookup(
            ...     mode="personal_car",
            ...     source="defra",
            ...     status="success"
            ... )
        """
        try:
            mode = self._validate_enum_value(
                mode, CommuteModeLabelEnum,
                CommuteModeLabelEnum.UNKNOWN.value,
            )
            status = self._validate_enum_value(
                status, StatusLabelEnum,
                StatusLabelEnum.ERROR.value,
            )
            # Source is free-form but bounded by truncation
            source = self._sanitize_label(source, max_len=32, default="unknown")

            self.ef_lookups_total.labels(
                mode=mode,
                source=source,
                status=status,
            ).inc()

            with self._stats_lock:
                self._in_memory_stats["ef_lookups"] += 1

            logger.debug(
                "Recorded EF lookup: mode=%s, source=%s, status=%s",
                mode, source, status,
            )

        except Exception as e:
            logger.error(
                "Failed to record EF lookup metrics: %s", e, exc_info=True
            )

    def record_cache_hit(self, mode: str) -> None:
        """
        Record an emission factor cache hit.

        Increments the cache hit counter for the given commute mode.
        Used together with record_cache_miss to calculate cache hit
        rate: hits / (hits + misses).

        Args:
            mode: Commute mode for the cached EF

        Example:
            >>> metrics.record_cache_hit("personal_car")
        """
        try:
            mode = self._validate_enum_value(
                mode, CommuteModeLabelEnum,
                CommuteModeLabelEnum.UNKNOWN.value,
            )

            self.ef_cache_hits_total.labels(mode=mode).inc()

            with self._stats_lock:
                self._in_memory_stats["cache_hits"] += 1

            logger.debug("Recorded cache hit: mode=%s", mode)

        except Exception as e:
            logger.error(
                "Failed to record cache hit metrics: %s", e, exc_info=True
            )

    def record_cache_miss(self, mode: str) -> None:
        """
        Record an emission factor cache miss.

        Increments the cache miss counter for the given commute mode.
        Used together with record_cache_hit to calculate cache hit
        rate: hits / (hits + misses).

        Args:
            mode: Commute mode for the missed EF lookup

        Example:
            >>> metrics.record_cache_miss("rail")
        """
        try:
            mode = self._validate_enum_value(
                mode, CommuteModeLabelEnum,
                CommuteModeLabelEnum.UNKNOWN.value,
            )

            self.ef_cache_misses_total.labels(mode=mode).inc()

            with self._stats_lock:
                self._in_memory_stats["cache_misses"] += 1

            logger.debug("Recorded cache miss: mode=%s", mode)

        except Exception as e:
            logger.error(
                "Failed to record cache miss metrics: %s", e, exc_info=True
            )

    def record_compliance_check(
        self,
        framework: str,
        status: str,
    ) -> None:
        """
        Record a compliance check result.

        Tracks regulatory compliance validation for Scope 3 Category 7
        reporting across applicable frameworks. Each framework check is
        counted independently to enable per-framework pass rate monitoring.

        Args:
            framework: Compliance framework (ghg_protocol/csrd_e1/cdp/sbti/
                        sb253/gri_305/iso_14064/unknown)
            status: Check result (success/failure/error/timeout/unknown)

        Example:
            >>> metrics.record_compliance_check(
            ...     framework="ghg_protocol",
            ...     status="success"
            ... )
        """
        try:
            framework = self._validate_enum_value(
                framework, ComplianceFrameworkLabelEnum,
                ComplianceFrameworkLabelEnum.UNKNOWN.value,
            )
            status = self._validate_enum_value(
                status, StatusLabelEnum,
                StatusLabelEnum.ERROR.value,
            )

            self.compliance_checks_total.labels(
                framework=framework,
                status=status,
            ).inc()

            with self._stats_lock:
                self._in_memory_stats["compliance_checks"] += 1

            logger.debug(
                "Recorded compliance check: framework=%s, status=%s",
                framework, status,
            )

        except Exception as e:
            logger.error(
                "Failed to record compliance check metrics: %s",
                e, exc_info=True,
            )

    def record_pipeline_stage(
        self,
        stage: str,
        duration: float,
    ) -> None:
        """
        Record the execution duration of a pipeline stage.

        Enables per-stage performance monitoring and bottleneck
        identification within the employee commuting calculation
        pipeline. Each stage records its wall-clock duration.

        Args:
            stage: Pipeline stage name (validate/classify/normalize/
                    resolve_efs/calculate_commute/calculate_telework/
                    compliance/aggregate/seal/unknown)
            duration: Stage execution duration in seconds

        Example:
            >>> metrics.record_pipeline_stage(
            ...     stage="resolve_efs",
            ...     duration=0.045
            ... )
        """
        try:
            stage = self._validate_enum_value(
                stage, PipelineStageLabelEnum,
                PipelineStageLabelEnum.UNKNOWN.value,
            )

            if duration is not None and duration > 0:
                self.pipeline_stage_duration_seconds.labels(
                    stage=stage
                ).observe(duration)

            with self._stats_lock:
                self._in_memory_stats["pipeline_stages"] += 1

            logger.debug(
                "Recorded pipeline stage: stage=%s, duration=%.3fs",
                stage,
                duration if duration else 0.0,
            )

        except Exception as e:
            logger.error(
                "Failed to record pipeline stage metrics: %s",
                e, exc_info=True,
            )

    def record_data_quality(
        self,
        method: str,
        tier: str,
        score: float,
    ) -> None:
        """
        Record a data quality indicator (DQI) score.

        Tracks the distribution of input data quality scores across
        calculation methods and quality tiers. GHG Protocol recommends
        companies improve data quality over time from Tier 3 toward
        Tier 1.

        Args:
            method: Calculation method (distance_based/fuel_based/
                     average_data/survey_based/spend_based/unknown)
            tier: Data quality tier (tier1/tier2/tier3/unknown)
            score: DQI score (0.0 = worst, 1.0 = best)

        Example:
            >>> metrics.record_data_quality(
            ...     method="survey_based",
            ...     tier="tier1",
            ...     score=0.87
            ... )
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethodLabelEnum,
                CalculationMethodLabelEnum.UNKNOWN.value,
            )
            tier = self._validate_enum_value(
                tier, DataQualityTierLabelEnum,
                DataQualityTierLabelEnum.UNKNOWN.value,
            )

            # Clamp score to [0.0, 1.0]
            if score is not None:
                score = max(0.0, min(1.0, float(score)))
                self.data_quality_score.labels(
                    method=method,
                    tier=tier,
                ).observe(score)

            with self._stats_lock:
                self._in_memory_stats["data_quality_records"] += 1

            logger.debug(
                "Recorded data quality: method=%s, tier=%s, score=%.2f",
                method, tier,
                score if score is not None else 0.0,
            )

        except Exception as e:
            logger.error(
                "Failed to record data quality metrics: %s", e, exc_info=True
            )

    def record_survey_response_rate(
        self,
        rate: float,
        tenant_id: str = "unknown",
    ) -> None:
        """
        Record a survey response rate observation.

        Tracks the distribution of survey response rates across tenants.
        GHG Protocol guidance recommends at least 80% response rates for
        accurate extrapolation of employee commuting data. This histogram
        enables monitoring response rate trends and alerting on low rates.

        Args:
            rate: Survey response rate (0.0 to 1.0, where 1.0 = 100%)
            tenant_id: Tenant identifier (default: "unknown")

        Example:
            >>> metrics.record_survey_response_rate(0.85, tenant_id="tenant_abc")
        """
        try:
            tenant_id = self._sanitize_tenant_id(tenant_id)

            if rate is not None:
                # Clamp to [0.0, 1.0]
                rate = max(0.0, min(1.0, float(rate)))
                self.survey_response_rate.labels(
                    tenant_id=tenant_id
                ).observe(rate)

            with self._stats_lock:
                self._in_memory_stats["survey_responses"] += 1

            logger.debug(
                "Recorded survey response rate: rate=%.2f, tenant=%s",
                rate if rate is not None else 0.0,
                tenant_id,
            )

        except Exception as e:
            logger.error(
                "Failed to record survey response rate metrics: %s",
                e, exc_info=True,
            )

    def record_telework_fraction(
        self,
        category: str,
        fraction: float,
    ) -> None:
        """
        Record a telework fraction observation.

        Tracks the distribution of telework fractions for net emission
        impact modelling. A fraction of 0.0 means fully office-based;
        1.0 means fully remote.

        Args:
            category: Telework category (full_remote/hybrid_1day/
                       hybrid_2day/hybrid_3day/hybrid_4day/office_based/
                       unknown)
            fraction: Telework fraction (0.0 to 1.0)

        Example:
            >>> metrics.record_telework_fraction(
            ...     category="hybrid_3day",
            ...     fraction=0.6
            ... )
        """
        try:
            category = self._validate_enum_value(
                category, TeleworkCategoryLabelEnum,
                TeleworkCategoryLabelEnum.UNKNOWN.value,
            )

            if fraction is not None:
                # Clamp to [0.0, 1.0]
                fraction = max(0.0, min(1.0, float(fraction)))
                self.telework_fraction.labels(
                    category=category
                ).observe(fraction)

            with self._stats_lock:
                self._in_memory_stats["telework_records"] += 1

            logger.debug(
                "Recorded telework fraction: category=%s, fraction=%.2f",
                category,
                fraction if fraction is not None else 0.0,
            )

        except Exception as e:
            logger.error(
                "Failed to record telework fraction metrics: %s",
                e, exc_info=True,
            )

    def start_calculation(self, method: str) -> None:
        """
        Increment the active calculations gauge when a calculation starts.

        Should be paired with end_calculation() when the calculation
        completes, to keep the gauge accurate. For automatic lifecycle
        management, use the track_calculation() context manager instead.

        Args:
            method: Calculation method being started

        Example:
            >>> metrics.start_calculation("distance_based")
            >>> try:
            ...     result = perform_calculation(data)
            ... finally:
            ...     metrics.end_calculation("distance_based")
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethodLabelEnum,
                CalculationMethodLabelEnum.UNKNOWN.value,
            )
            self.active_calculations.labels(method=method).inc()

            logger.debug("Started calculation: method=%s", method)

        except Exception as e:
            logger.error(
                "Failed to increment active calculations: %s",
                e, exc_info=True,
            )

    def end_calculation(self, method: str) -> None:
        """
        Decrement the active calculations gauge when a calculation ends.

        Should be paired with start_calculation() and called in a
        finally block to ensure the gauge is always decremented.
        For automatic lifecycle management, use the track_calculation()
        context manager instead.

        Args:
            method: Calculation method that completed

        Example:
            >>> metrics.start_calculation("distance_based")
            >>> try:
            ...     result = perform_calculation(data)
            ... finally:
            ...     metrics.end_calculation("distance_based")
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethodLabelEnum,
                CalculationMethodLabelEnum.UNKNOWN.value,
            )
            self.active_calculations.labels(method=method).dec()

            logger.debug("Ended calculation: method=%s", method)

        except Exception as e:
            logger.error(
                "Failed to decrement active calculations: %s",
                e, exc_info=True,
            )

    # ======================================================================
    # Summary and lifecycle
    # ======================================================================

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of metrics collected since initialization.

        Returns a dictionary with all in-memory counters, uptime information,
        and calculated rates (per-hour throughput). Useful for health checks,
        admin dashboards, and operational monitoring outside of Prometheus.

        Returns:
            Dictionary with metrics summary including counts, uptime, rates,
            cache performance, and operational breakdown.

        Example:
            >>> summary = metrics.get_metrics_summary()
            >>> print(summary['calculations'])
            142
            >>> print(summary['rates']['calculations_per_hour'])
            28.4
        """
        try:
            uptime_seconds = (
                datetime.utcnow() - self._start_time
            ).total_seconds()
            uptime_hours = uptime_seconds / 3600 if uptime_seconds > 0 else 1.0

            with self._stats_lock:
                stats_snapshot = dict(self._in_memory_stats)

            # Calculate cache hit rate
            total_cache_ops = (
                stats_snapshot["cache_hits"] + stats_snapshot["cache_misses"]
            )
            cache_hit_rate = (
                stats_snapshot["cache_hits"] / total_cache_ops
                if total_cache_ops > 0
                else 0.0
            )

            summary: Dict[str, Any] = {
                "prometheus_available": PROMETHEUS_AVAILABLE,
                "agent": "employee_commuting",
                "agent_id": "GL-MRV-S3-007",
                "prefix": "gl_ec_",
                "scope": "Scope 3 Category 7",
                "description": "Employee Commuting",
                "metrics_count": 14,
                "uptime_seconds": uptime_seconds,
                "uptime_hours": uptime_seconds / 3600,
                "start_time": self._start_time.isoformat(),
                "current_time": datetime.utcnow().isoformat(),
                **stats_snapshot,
                "rates": {
                    "calculations_per_hour": (
                        stats_snapshot["calculations"] / uptime_hours
                    ),
                    "emissions_kg_co2e_per_hour": (
                        stats_snapshot["emissions_kg_co2e"] / uptime_hours
                    ),
                    "batches_per_hour": (
                        stats_snapshot["batches"] / uptime_hours
                    ),
                    "ef_lookups_per_hour": (
                        stats_snapshot["ef_lookups"] / uptime_hours
                    ),
                    "compliance_checks_per_hour": (
                        stats_snapshot["compliance_checks"] / uptime_hours
                    ),
                    "errors_per_hour": (
                        stats_snapshot["errors"] / uptime_hours
                    ),
                },
                "cache_performance": {
                    "hits": stats_snapshot["cache_hits"],
                    "misses": stats_snapshot["cache_misses"],
                    "total_lookups": total_cache_ops,
                    "hit_rate": cache_hit_rate,
                },
                "operational": {
                    "ef_lookups": stats_snapshot["ef_lookups"],
                    "compliance_checks": stats_snapshot["compliance_checks"],
                    "pipeline_stages": stats_snapshot["pipeline_stages"],
                    "data_quality_records": stats_snapshot["data_quality_records"],
                    "survey_responses": stats_snapshot["survey_responses"],
                    "telework_records": stats_snapshot["telework_records"],
                    "batches": stats_snapshot["batches"],
                    "errors": stats_snapshot["errors"],
                },
            }

            logger.debug(
                "Generated metrics summary: %d calculations tracked",
                stats_snapshot["calculations"],
            )
            return summary

        except Exception as e:
            logger.error(
                "Failed to generate metrics summary: %s", e, exc_info=True
            )
            return {
                "error": str(e),
                "prometheus_available": PROMETHEUS_AVAILABLE,
                "agent": "employee_commuting",
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
            >>> EmployeeCommutingMetrics.reset()
            >>> metrics = EmployeeCommutingMetrics()  # Fresh instance
        """
        with cls._lock:
            if cls._instance is not None:
                if hasattr(cls._instance, "_initialized"):
                    del cls._instance._initialized
                cls._instance = None

                global _metrics_instance
                _metrics_instance = None

                logger.info("EmployeeCommutingMetrics singleton reset")

    def reset_stats(self) -> None:
        """
        Reset in-memory statistics (not Prometheus metrics).

        Note: This only resets the in-memory counters used for
        get_metrics_summary(). Prometheus metrics are cumulative and
        cannot be reset without restarting the process. Also resets
        start_time for rate calculations.

        Example:
            >>> metrics.reset_stats()
            >>> summary = metrics.get_metrics_summary()
            >>> assert summary['calculations'] == 0
        """
        try:
            with self._stats_lock:
                self._in_memory_stats = {
                    "calculations": 0,
                    "emissions_kg_co2e": 0.0,
                    "batches": 0,
                    "ef_lookups": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "compliance_checks": 0,
                    "pipeline_stages": 0,
                    "data_quality_records": 0,
                    "survey_responses": 0,
                    "telework_records": 0,
                    "errors": 0,
                }
            self._start_time = datetime.utcnow()

            logger.info(
                "Reset in-memory statistics for EmployeeCommutingMetrics"
            )

        except Exception as e:
            logger.error("Failed to reset statistics: %s", e, exc_info=True)

    # ======================================================================
    # Internal helpers
    # ======================================================================

    @staticmethod
    def _validate_enum_value(
        value: Optional[str],
        enum_class: type,
        default: str,
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
            >>> EmployeeCommutingMetrics._validate_enum_value(
            ...     "personal_car", CommuteModeLabelEnum, "unknown"
            ... )
            'personal_car'
            >>> EmployeeCommutingMetrics._validate_enum_value(
            ...     "hoverboard", CommuteModeLabelEnum, "unknown"
            ... )
            'unknown'
        """
        if value is None:
            return default

        valid_values = [m.value for m in enum_class]
        if value not in valid_values:
            logger.warning(
                "Invalid %s value '%s', using default '%s'",
                enum_class.__name__, value, default,
            )
            return default

        return value

    @staticmethod
    def _sanitize_tenant_id(
        tenant_id: Optional[str],
        max_len: int = 32,
    ) -> str:
        """
        Sanitize a tenant ID for use as a Prometheus label value.

        Ensures bounded cardinality by truncating to max_len characters,
        stripping whitespace, and providing a default for None values.

        Args:
            tenant_id: Raw tenant identifier
            max_len: Maximum label length (default: 32)

        Returns:
            Sanitized tenant ID string

        Example:
            >>> EmployeeCommutingMetrics._sanitize_tenant_id("tenant_abc")
            'tenant_abc'
            >>> EmployeeCommutingMetrics._sanitize_tenant_id(None)
            'unknown'
            >>> EmployeeCommutingMetrics._sanitize_tenant_id("a" * 50)
            'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        """
        if tenant_id is None:
            return TenantLabelEnum.UNKNOWN.value

        tenant_id = str(tenant_id).strip()
        if not tenant_id:
            return TenantLabelEnum.UNKNOWN.value

        return tenant_id[:max_len]

    @staticmethod
    def _sanitize_label(
        value: Optional[str],
        max_len: int = 32,
        default: str = "unknown",
    ) -> str:
        """
        Sanitize a free-form string for use as a Prometheus label value.

        Ensures bounded cardinality by truncating to max_len characters,
        converting to lowercase, stripping whitespace, and providing a
        default for None/empty values.

        Args:
            value: Raw label value
            max_len: Maximum label length (default: 32)
            default: Default value for None/empty inputs (default: "unknown")

        Returns:
            Sanitized label value string

        Example:
            >>> EmployeeCommutingMetrics._sanitize_label("DEFRA")
            'defra'
            >>> EmployeeCommutingMetrics._sanitize_label(None)
            'unknown'
            >>> EmployeeCommutingMetrics._sanitize_label("")
            'unknown'
        """
        if value is None:
            return default

        value = str(value).strip().lower()
        if not value:
            return default

        return value[:max_len]


# ===========================================================================
# Module-level singleton accessor
# ===========================================================================

_metrics_instance: Optional[EmployeeCommutingMetrics] = None
_metrics_lock: threading.Lock = threading.Lock()


def get_metrics() -> EmployeeCommutingMetrics:
    """
    Get the singleton EmployeeCommutingMetrics instance.

    Thread-safe accessor for the global metrics instance. Prefer this
    function over direct instantiation for consistency across the
    employee commuting agent codebase.

    Returns:
        EmployeeCommutingMetrics singleton instance

    Example:
        >>> from greenlang.employee_commuting.metrics import get_metrics
        >>> metrics = get_metrics()
        >>> metrics.record_calculation(
        ...     mode="personal_car",
        ...     method="distance_based",
        ...     status="success",
        ...     duration=0.025,
        ...     co2e=1845.2,
        ...     tenant_id="tenant_abc"
        ... )
    """
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = EmployeeCommutingMetrics()

    return _metrics_instance


def reset_metrics() -> None:
    """
    Reset the singleton metrics instance for testing purposes.

    Convenience function that delegates to EmployeeCommutingMetrics.reset().
    Should only be called in test teardown.

    Example:
        >>> from greenlang.employee_commuting.metrics import reset_metrics
        >>> reset_metrics()
    """
    EmployeeCommutingMetrics.reset()


# ===========================================================================
# Context manager helpers
# ===========================================================================


@contextmanager
def track_calculation(
    method: str = "distance_based",
    mode: str = "personal_car",
    tenant_id: str = "unknown",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a calculation's lifecycle.

    Automatically increments/decrements the active calculation gauge,
    measures wall-clock duration, and records the calculation when the
    context exits. The caller can set ``context['co2e']`` inside the
    block to include emissions in the recorded metric.

    Args:
        method: Calculation method (default: "distance_based")
        mode: Commute mode (default: "personal_car")
        tenant_id: Tenant identifier (default: "unknown")

    Yields:
        Mutable context dict. Set ``context['co2e']`` to record emissions.

    Example:
        >>> with track_calculation(method="distance_based", mode="rail") as ctx:
        ...     result = calculate_rail_commute_emissions(employee)
        ...     ctx['co2e'] = result.co2e_kg
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        "co2e": 0.0,
        "status": "success",
    }

    metrics.start_calculation(method)
    start = time.monotonic()

    try:
        yield context
    except Exception:
        context["status"] = "error"
        raise
    finally:
        duration = time.monotonic() - start
        metrics.end_calculation(method)
        metrics.record_calculation(
            mode=mode,
            method=method,
            status=context["status"],
            duration=duration,
            co2e=context.get("co2e", 0.0),
            tenant_id=tenant_id,
        )


@contextmanager
def track_batch(
    method: str = "distance_based",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a batch job's lifecycle.

    Automatically measures duration and records the batch outcome when
    the context exits. The caller should set ``context['size']`` inside
    the block to record the batch size.

    Args:
        method: Calculation method used for the batch (default: "distance_based")

    Yields:
        Mutable context dict. Set ``context['size']`` to record batch size.

    Example:
        >>> with track_batch(method="survey_based") as ctx:
        ...     results = process_employee_commuting_batch(employees)
        ...     ctx['size'] = len(employees)
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        "size": 0,
        "status": "success",
    }

    start = time.monotonic()

    try:
        yield context
    except Exception:
        context["status"] = "error"
        raise
    finally:
        duration = time.monotonic() - start
        metrics.record_batch(
            method=method,
            size=context.get("size", 0),
            duration=duration,
        )


@contextmanager
def track_pipeline_stage(
    stage: str,
) -> Generator[None, None, None]:
    """
    Context manager that tracks a pipeline stage's execution duration.

    Automatically measures wall-clock duration and records the stage
    timing when the context exits.

    Args:
        stage: Pipeline stage name (validate/classify/normalize/
                resolve_efs/calculate_commute/calculate_telework/
                compliance/aggregate/seal/unknown)

    Yields:
        None

    Example:
        >>> with track_pipeline_stage("resolve_efs"):
        ...     emission_factors = resolve_all_factors(inputs)
    """
    metrics = get_metrics()
    start = time.monotonic()

    try:
        yield
    finally:
        duration = time.monotonic() - start
        metrics.record_pipeline_stage(stage=stage, duration=duration)


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    # Availability flag
    "PROMETHEUS_AVAILABLE",
    # Label Enums (11)
    "CommuteModeLabelEnum",
    "VehicleTypeLabelEnum",
    "FuelTypeLabelEnum",
    "TransitTypeLabelEnum",
    "TeleworkCategoryLabelEnum",
    "CalculationMethodLabelEnum",
    "ComplianceFrameworkLabelEnum",
    "DataQualityTierLabelEnum",
    "PipelineStageLabelEnum",
    "StatusLabelEnum",
    "TenantLabelEnum",
    # Singleton class
    "EmployeeCommutingMetrics",
    # Module-level accessors
    "get_metrics",
    "reset_metrics",
    # Context managers
    "track_calculation",
    "track_batch",
    "track_pipeline_stage",
]
