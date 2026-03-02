# -*- coding: utf-8 -*-
"""
Business Travel Prometheus Metrics - AGENT-MRV-019

12 Prometheus metrics with gl_bt_ prefix for monitoring the
GL-MRV-S3-006 Business Travel Agent.

This module provides Prometheus metrics tracking for business travel
emissions calculations (Scope 3, Category 6) including distance-based,
spend-based, supplier-specific, and average-data calculation methods
across all transport modes (air, rail, road, bus, taxi, ferry,
motorcycle) and hotel accommodation.

Thread-safe singleton pattern with graceful fallback if Prometheus
unavailable.

Metrics prefix: gl_bt_

12 Prometheus Metrics:
    1.  gl_bt_calculations_total              - Counter: total calculations performed
    2.  gl_bt_emissions_kg_co2e_total         - Counter: total emissions in kgCO2e
    3.  gl_bt_flights_total                   - Counter: total flights calculated
    4.  gl_bt_ground_trips_total              - Counter: total ground trips
    5.  gl_bt_hotel_nights_total              - Counter: total hotel nights
    6.  gl_bt_factor_selections_total         - Counter: EF selections
    7.  gl_bt_compliance_checks_total         - Counter: compliance checks
    8.  gl_bt_batch_jobs_total                - Counter: batch jobs
    9.  gl_bt_calculation_duration_seconds    - Histogram: calculation duration
    10. gl_bt_batch_size                      - Histogram: batch sizes
    11. gl_bt_active_calculations             - Gauge: active calculations
    12. gl_bt_distance_km_total               - Counter: total distance in km

GHG Protocol Scope 3 Category 6 covers business travel:
    A. Transportation of employees for business-related activities
       in vehicles not owned or operated by the reporting company.
    B. Includes air, rail, road (rental car, personal vehicle),
       bus, taxi, ferry, and motorcycle travel.
    C. Hotel accommodation emissions during business travel.
    D. Excludes employee commuting (Category 7) and upstream
       transportation (Category 4).

Calculation methods defined by GHG Protocol:
    - Distance-based: distance x mode-specific EF (with/without RF)
    - Spend-based: spend x spend-based EF per mode
    - Supplier-specific: primary data from airlines/hotels/agencies
    - Average-data: total spend or distance x blended average EF

Example:
    >>> metrics = BusinessTravelMetrics()
    >>> metrics.record_calculation(
    ...     method="distance_based",
    ...     mode="air",
    ...     status="success",
    ...     duration=0.035,
    ...     co2e=245.8,
    ...     rf_option="with_rf"
    ... )

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-006
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
        """No-op metric stub for environments without prometheus_client."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, *args: Any, **kwargs: Any) -> "_NoOpMetric":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def dec(self, amount: float = 1) -> None:
            pass

        def set(self, value: float = 0) -> None:
            pass

        def observe(self, amount: float = 0) -> None:
            pass

        def info(self, data: Optional[Dict[str, str]] = None) -> None:
            pass

    Counter = _NoOpMetric  # type: ignore[misc,assignment]
    Histogram = _NoOpMetric  # type: ignore[misc,assignment]
    Gauge = _NoOpMetric  # type: ignore[misc,assignment]
    Info = _NoOpMetric  # type: ignore[misc,assignment]


# ===========================================================================
# Enumerations -- Business Travel domain-specific label value sets
# ===========================================================================


class CalculationMethodLabel(str, Enum):
    """
    Calculation methods for business travel emissions.

    GHG Protocol Scope 3 Category 6 supports several approaches depending
    on data availability and the level of accuracy required:
        - Supplier-specific: primary data from airlines, hotels, agencies
        - Distance-based: distance x mode-specific EF (with/without RF)
        - Average-data: total spend or distance x blended average EF
        - Spend-based: spend x spend-based EF per mode
    """
    SUPPLIER_SPECIFIC = "supplier_specific"
    DISTANCE_BASED = "distance_based"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"


class TransportModeLabel(str, Enum):
    """
    Transport modes tracked for business travel emissions.

    Covers the primary transport modes defined in the GHG Protocol
    Technical Guidance for Calculating Scope 3 Emissions, Category 6.
    Hotel accommodation is included as a pseudo-mode for unified tracking.
    """
    AIR = "air"
    RAIL = "rail"
    ROAD = "road"
    BUS = "bus"
    TAXI = "taxi"
    FERRY = "ferry"
    MOTORCYCLE = "motorcycle"
    HOTEL = "hotel"


class CalculationStatusLabel(str, Enum):
    """Calculation operation status for business travel calculations."""
    SUCCESS = "success"
    ERROR = "error"
    VALIDATION_ERROR = "validation_error"


class DistanceBandLabel(str, Enum):
    """
    Flight distance bands for aviation emissions tracking.

    DEFRA and GHG Protocol differentiate emission factors by flight
    distance band due to the disproportionate fuel burn during takeoff
    and landing phases relative to cruise.
    """
    DOMESTIC = "domestic"
    SHORT_HAUL = "short_haul"
    LONG_HAUL = "long_haul"


class CabinClassLabel(str, Enum):
    """
    Aircraft cabin classes for aviation emissions allocation.

    Emissions per passenger-km vary by cabin class due to seat pitch,
    weight allocation, and floor space. Business and first class have
    higher per-passenger emissions because fewer seats fit in the same
    aircraft space.
    """
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"


class RFOptionLabel(str, Enum):
    """
    Radiative forcing options for aviation emissions.

    The IPCC recommends applying a radiative forcing (RF) multiplier to
    account for the non-CO2 climate effects of aviation at altitude
    (contrails, NOx, water vapour). Common RF multipliers range from
    1.7 to 2.7, with 1.9 being the DEFRA default.
    """
    WITHOUT_RF = "without_rf"
    WITH_RF = "with_rf"


class EFSourceLabel(str, Enum):
    """
    Emission factor sources for business travel calculations.

    Tracks the origin of emission factors used in business travel
    calculations, enabling monitoring of factor database coverage
    and source distribution.
    """
    DEFRA = "defra"
    EPA = "epa"
    ICAO = "icao"
    IEA = "iea"
    ECOINVENT = "ecoinvent"
    GHG_PROTOCOL = "ghg_protocol"
    SUPPLIER_SPECIFIC = "supplier_specific"
    CUSTOM = "custom"
    OTHER = "other"


class FrameworkLabel(str, Enum):
    """
    Compliance frameworks for business travel reporting.

    Tracks validation against regulatory and voluntary reporting
    standards applicable to Scope 3 Category 6 emissions.
    """
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD = "csrd"
    CDP = "cdp"
    SBTI = "sbti"
    SB253 = "sb253"
    GRI = "gri"
    SEC_CLIMATE = "sec_climate"


class ComplianceStatusLabel(str, Enum):
    """Compliance check result status for business travel calculations."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class BatchStatusLabel(str, Enum):
    """Batch processing job status for business travel calculations."""
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


class VehicleTypeLabel(str, Enum):
    """
    Ground vehicle types for road transport emissions tracking.

    Differentiates emission factors by vehicle size and fuel type
    for rental cars and personal vehicles.
    """
    SMALL_CAR = "small_car"
    MEDIUM_CAR = "medium_car"
    LARGE_CAR = "large_car"
    SUV = "suv"
    HYBRID = "hybrid"
    ELECTRIC = "electric"
    VAN = "van"
    MOTORCYCLE = "motorcycle"
    BUS_LOCAL = "bus_local"
    BUS_COACH = "bus_coach"
    TAXI_STANDARD = "taxi_standard"
    TAXI_BLACK_CAB = "taxi_black_cab"
    OTHER = "other"


# ===========================================================================
# BusinessTravelMetrics -- Thread-safe Singleton
# ===========================================================================


class BusinessTravelMetrics:
    """
    Thread-safe singleton metrics collector for Business Travel Agent (MRV-019).

    Provides 12 Prometheus metrics for tracking Scope 3 Category 6
    business travel emissions calculations, including distance-based,
    spend-based, supplier-specific, and average-data methods across
    all transport modes and hotel accommodation.

    All metrics use the ``gl_bt_`` prefix to ensure namespace isolation
    within the GreenLang Prometheus ecosystem.

    Scope 3 Category 6 Sub-Categories Tracked:
        A. Air travel (domestic, short-haul, long-haul by cabin class)
        B. Ground transport (rail, road, bus, taxi, ferry, motorcycle)
        C. Hotel accommodation (by country)

    Calculation Methods Supported:
        - Distance-based: distance x mode-specific EF (with/without RF)
        - Spend-based: spend x spend-based EF per mode
        - Supplier-specific: primary data from airlines/hotels/agencies
        - Average-data: total spend or distance x blended average EF

    Attributes:
        calculations_total: Counter for total calculation operations
        emissions_kg_co2e_total: Counter for total emissions in kgCO2e
        flights_total: Counter for total flights calculated
        ground_trips_total: Counter for total ground trips
        hotel_nights_total: Counter for total hotel nights
        factor_selections_total: Counter for EF selections
        compliance_checks_total: Counter for compliance checks
        batch_jobs_total: Counter for batch jobs
        calculation_duration_seconds: Histogram for operation durations
        batch_size: Histogram for batch calculation sizes
        active_calculations: Gauge for active calculations
        distance_km_total: Counter for total distance tracked

    Example:
        >>> metrics = BusinessTravelMetrics()
        >>> metrics.record_calculation(
        ...     method="distance_based",
        ...     mode="air",
        ...     status="success",
        ...     duration=0.035,
        ...     co2e=245.8,
        ...     rf_option="with_rf"
        ... )
        >>> summary = metrics.get_stats()
        >>> assert summary['calculations'] >= 1
    """

    _instance: Optional["BusinessTravelMetrics"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "BusinessTravelMetrics":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize metrics (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._start_time: datetime = datetime.utcnow()
        self._stats_lock: threading.Lock = threading.Lock()
        self._in_memory_stats: Dict[str, Any] = {
            "calculations": 0,
            "emissions_kg_co2e": 0.0,
            "flights": 0,
            "ground_trips": 0,
            "hotel_nights": 0,
            "factor_selections": 0,
            "compliance_checks": 0,
            "batch_jobs": 0,
            "distance_km": 0.0,
            "errors": 0,
        }

        # Initialize Prometheus metrics
        self._init_metrics()

        logger.info(
            "BusinessTravelMetrics initialized (Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable",
        )

    def _init_metrics(self) -> None:
        """
        Initialize all 12 Prometheus metrics with gl_bt_ prefix.

        Each metric is documented with its purpose, label set, and the
        Scope 3 Category 6 aspect it supports.

        When Prometheus is available, metric registration may fail if the
        metrics were previously registered (e.g., after a reset() call in
        tests). In that case we unregister from the default registry and
        re-register to obtain fresh collector objects.
        """
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import REGISTRY

            def _safe_create(metric_cls: type, name: str, *args: Any, **kwargs: Any) -> Any:
                """Create a metric, unregistering any prior collector on conflict."""
                try:
                    return metric_cls(name, *args, **kwargs)
                except ValueError:
                    try:
                        REGISTRY.unregister(REGISTRY._names_to_collectors.get(name))
                    except Exception:
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
        # 1. gl_bt_calculations_total (Counter)
        #    Total business travel emission calculations performed.
        #    Labels:
        #      - method: supplier_specific, distance_based, average_data,
        #                spend_based
        #      - mode: air, rail, road, bus, taxi, ferry, motorcycle, hotel
        #      - status: success, error, validation_error
        #    Primary throughput counter for all calculation operations
        #    across methods and transport modes.
        # ------------------------------------------------------------------
        self.calculations_total = _safe_create(
            Counter,
            "gl_bt_calculations_total",
            "Total business travel emission calculations performed",
            ["method", "mode", "status"],
        )

        # ------------------------------------------------------------------
        # 2. gl_bt_emissions_kg_co2e_total (Counter)
        #    Total emissions calculated in kilograms CO2-equivalent.
        #    Labels:
        #      - mode: Transport mode generating emissions
        #      - rf_option: without_rf, with_rf (aviation radiative forcing)
        #    Tracks cumulative emissions output for rate calculation and
        #    modal breakdown. Uses kgCO2e as individual trips may produce
        #    modest amounts that would round to zero in tCO2e.
        # ------------------------------------------------------------------
        self.emissions_kg_co2e_total = _safe_create(
            Counter,
            "gl_bt_emissions_kg_co2e_total",
            "Total business travel emissions calculated in kgCO2e",
            ["mode", "rf_option"],
        )

        # ------------------------------------------------------------------
        # 3. gl_bt_flights_total (Counter)
        #    Total flights calculated.
        #    Labels:
        #      - distance_band: domestic, short_haul, long_haul
        #      - cabin_class: economy, premium_economy, business, first
        #    Tracks flight volume by distance band and cabin class for
        #    aviation-specific analysis. DEFRA and GHG Protocol define
        #    different EFs per distance band and cabin class.
        # ------------------------------------------------------------------
        self.flights_total = _safe_create(
            Counter,
            "gl_bt_flights_total",
            "Total flights calculated for business travel",
            ["distance_band", "cabin_class"],
        )

        # ------------------------------------------------------------------
        # 4. gl_bt_ground_trips_total (Counter)
        #    Total ground transport trips calculated.
        #    Labels:
        #      - mode: rail, road, bus, taxi, ferry, motorcycle
        #      - vehicle_type: small_car, medium_car, large_car, suv,
        #                      hybrid, electric, bus_local, bus_coach, etc.
        #    Tracks non-aviation trip volume by mode and vehicle type for
        #    ground transport emissions analysis and fleet policy monitoring.
        # ------------------------------------------------------------------
        self.ground_trips_total = _safe_create(
            Counter,
            "gl_bt_ground_trips_total",
            "Total ground transport trips calculated",
            ["mode", "vehicle_type"],
        )

        # ------------------------------------------------------------------
        # 5. gl_bt_hotel_nights_total (Counter)
        #    Total hotel nights calculated.
        #    Labels:
        #      - country: ISO 3166-1 alpha-2 country code
        #    Tracks accommodation volume by country. Hotel emissions vary
        #    significantly by country due to differing grid emission factors,
        #    heating fuel mixes, and building efficiency standards.
        # ------------------------------------------------------------------
        self.hotel_nights_total = _safe_create(
            Counter,
            "gl_bt_hotel_nights_total",
            "Total hotel nights calculated for business travel",
            ["country"],
        )

        # ------------------------------------------------------------------
        # 6. gl_bt_factor_selections_total (Counter)
        #    Total emission factor selections/lookups performed.
        #    Labels:
        #      - source: defra, epa, icao, iea, ecoinvent, ghg_protocol,
        #                supplier_specific, custom, other
        #      - mode: Transport mode for which the EF was selected
        #    Tracks the frequency and source distribution of EF retrievals
        #    for cache optimization and database coverage monitoring.
        # ------------------------------------------------------------------
        self.factor_selections_total = _safe_create(
            Counter,
            "gl_bt_factor_selections_total",
            "Total emission factor selections for business travel",
            ["source", "mode"],
        )

        # ------------------------------------------------------------------
        # 7. gl_bt_compliance_checks_total (Counter)
        #    Total compliance checks performed.
        #    Labels:
        #      - framework: ghg_protocol, iso_14064, csrd, cdp, sbti,
        #                   sb253, gri, sec_climate
        #      - status: compliant, partially_compliant, non_compliant,
        #                warning, not_applicable
        #    Tracks regulatory compliance validation for Scope 3 Cat 6
        #    reporting across applicable standards.
        # ------------------------------------------------------------------
        self.compliance_checks_total = _safe_create(
            Counter,
            "gl_bt_compliance_checks_total",
            "Total business travel compliance checks performed",
            ["framework", "status"],
        )

        # ------------------------------------------------------------------
        # 8. gl_bt_batch_jobs_total (Counter)
        #    Total batch processing jobs completed.
        #    Labels:
        #      - status: completed, failed, partial, timeout
        #    Tracks batch job outcomes for reliability monitoring.
        # ------------------------------------------------------------------
        self.batch_jobs_total = _safe_create(
            Counter,
            "gl_bt_batch_jobs_total",
            "Total business travel batch processing jobs",
            ["status"],
        )

        # ------------------------------------------------------------------
        # 9. gl_bt_calculation_duration_seconds (Histogram)
        #    Duration of calculation operations in seconds.
        #    Labels:
        #      - method: Calculation method used
        #      - mode: Transport mode
        #    Buckets tuned for typical business travel calculation latencies:
        #      - 5-10ms for cached factor lookups and simple distance x EF
        #      - 25-100ms for multi-leg itinerary calculations
        #      - 250ms-1s for spend-based with currency conversion
        #      - 1-10s for complex batch or full-company calculations
        # ------------------------------------------------------------------
        self.calculation_duration_seconds = _safe_create(
            Histogram,
            "gl_bt_calculation_duration_seconds",
            "Duration of business travel calculation operations",
            ["method", "mode"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        # ------------------------------------------------------------------
        # 10. gl_bt_batch_size (Histogram)
        #     Size of batch calculation operations (number of trip records).
        #     Buckets cover typical batch sizes from a single trip to
        #     large-scale company-wide calculations.
        # ------------------------------------------------------------------
        self.batch_size = _safe_create(
            Histogram,
            "gl_bt_batch_size",
            "Batch calculation size for business travel operations",
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
        )

        # ------------------------------------------------------------------
        # 11. gl_bt_active_calculations (Gauge)
        #     Number of currently active (in-flight) calculations.
        #     No labels -- single gauge for concurrency monitoring.
        # ------------------------------------------------------------------
        self.active_calculations = _safe_create(
            Gauge,
            "gl_bt_active_calculations",
            "Number of currently active business travel calculations",
        )

        # ------------------------------------------------------------------
        # 12. gl_bt_distance_km_total (Counter)
        #     Total distance tracked across all transport modes.
        #     Labels:
        #       - mode: Transport mode for which distance was tracked
        #     Tracks cumulative distance by mode for modal split analysis
        #     and intensity metric calculation (kgCO2e/km).
        # ------------------------------------------------------------------
        self.distance_km_total = _safe_create(
            Counter,
            "gl_bt_distance_km_total",
            "Total distance tracked in km for business travel",
            ["mode"],
        )

        # ------------------------------------------------------------------
        # Agent info metric (non-numeric metadata)
        # ------------------------------------------------------------------
        self.agent_info = _safe_create(
            Info,
            "gl_bt_agent",
            "Business Travel Agent metadata",
        )
        if PROMETHEUS_AVAILABLE:
            try:
                self.agent_info.info({
                    "agent_id": "GL-MRV-S3-006",
                    "version": "1.0.0",
                    "scope": "scope_3_category_6",
                    "description": "Business Travel emissions calculator",
                })
            except Exception:
                pass

    # ======================================================================
    # Primary recording methods
    # ======================================================================

    def record_calculation(
        self,
        method: str,
        mode: str,
        status: str,
        duration: float,
        co2e: float,
        rf_option: str = "without_rf",
    ) -> None:
        """
        Record a business travel emission calculation operation.

        This is the primary recording method that increments the calculations
        counter, observes the duration histogram, and tracks emissions output.
        It covers all calculation methods and transport modes for Scope 3
        Category 6.

        Args:
            method: Calculation method (supplier_specific/distance_based/
                     average_data/spend_based)
            mode: Transport mode (air/rail/road/bus/taxi/ferry/motorcycle/hotel)
            status: Calculation status (success/error/validation_error)
            duration: Operation duration in seconds
            co2e: Emissions calculated in kgCO2e
            rf_option: Radiative forcing option (without_rf/with_rf)

        Example:
            >>> metrics.record_calculation(
            ...     method="distance_based",
            ...     mode="air",
            ...     status="success",
            ...     duration=0.035,
            ...     co2e=245.8,
            ...     rf_option="with_rf"
            ... )
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethodLabel, CalculationMethodLabel.DISTANCE_BASED.value
            )
            mode = self._validate_enum_value(
                mode, TransportModeLabel, TransportModeLabel.AIR.value
            )
            status = self._validate_enum_value(
                status, CalculationStatusLabel, CalculationStatusLabel.ERROR.value
            )
            rf_option = self._validate_enum_value(
                rf_option, RFOptionLabel, RFOptionLabel.WITHOUT_RF.value
            )

            # 1. Increment calculation counter
            self.calculations_total.labels(
                method=method, mode=mode, status=status
            ).inc()

            # 2. Observe duration
            if duration is not None and duration > 0:
                self.calculation_duration_seconds.labels(
                    method=method, mode=mode
                ).observe(duration)

            # 3. Record emissions
            if co2e is not None and co2e > 0:
                self.emissions_kg_co2e_total.labels(
                    mode=mode, rf_option=rf_option
                ).inc(co2e)

                with self._stats_lock:
                    self._in_memory_stats["emissions_kg_co2e"] += co2e

            # 4. Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats["calculations"] += 1
                if status == CalculationStatusLabel.ERROR.value:
                    self._in_memory_stats["errors"] += 1

            logger.debug(
                "Recorded calculation: method=%s, mode=%s, status=%s, "
                "duration=%.3fs, co2e=%.2f kgCO2e, rf=%s",
                method, mode, status,
                duration if duration else 0.0,
                co2e if co2e else 0.0,
                rf_option,
            )

        except Exception as e:
            logger.error("Failed to record calculation metrics: %s", e, exc_info=True)

    def record_flight(
        self,
        distance_band: str,
        cabin_class: str,
        distance_km: float,
    ) -> None:
        """
        Record a flight calculation.

        Tracks flight volume by distance band and cabin class, and
        accumulates distance for modal split analysis.

        Args:
            distance_band: Flight distance band (domestic/short_haul/long_haul)
            cabin_class: Cabin class (economy/premium_economy/business/first)
            distance_km: One-way flight distance in kilometres

        Example:
            >>> metrics.record_flight(
            ...     distance_band="long_haul",
            ...     cabin_class="business",
            ...     distance_km=8500.0
            ... )
        """
        try:
            distance_band = self._validate_enum_value(
                distance_band, DistanceBandLabel, DistanceBandLabel.DOMESTIC.value
            )
            cabin_class = self._validate_enum_value(
                cabin_class, CabinClassLabel, CabinClassLabel.ECONOMY.value
            )

            # Increment flight counter
            self.flights_total.labels(
                distance_band=distance_band, cabin_class=cabin_class
            ).inc()

            # Record distance
            if distance_km is not None and distance_km > 0:
                self.distance_km_total.labels(
                    mode=TransportModeLabel.AIR.value
                ).inc(distance_km)

                with self._stats_lock:
                    self._in_memory_stats["distance_km"] += distance_km

            with self._stats_lock:
                self._in_memory_stats["flights"] += 1

            logger.debug(
                "Recorded flight: band=%s, class=%s, distance=%.1f km",
                distance_band, cabin_class,
                distance_km if distance_km else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record flight metrics: %s", e, exc_info=True)

    def record_ground_trip(
        self,
        mode: str,
        vehicle_type: str,
        distance_km: float,
    ) -> None:
        """
        Record a ground transport trip calculation.

        Tracks non-aviation trip volume by mode and vehicle type, and
        accumulates distance for modal split analysis.

        Args:
            mode: Transport mode (rail/road/bus/taxi/ferry/motorcycle)
            vehicle_type: Vehicle type (small_car/medium_car/large_car/
                           suv/hybrid/electric/bus_local/bus_coach/etc.)
            distance_km: Trip distance in kilometres

        Example:
            >>> metrics.record_ground_trip(
            ...     mode="rail",
            ...     vehicle_type="other",
            ...     distance_km=320.0
            ... )
        """
        try:
            mode = self._validate_enum_value(
                mode, TransportModeLabel, TransportModeLabel.ROAD.value
            )
            vehicle_type = self._validate_enum_value(
                vehicle_type, VehicleTypeLabel, VehicleTypeLabel.OTHER.value
            )

            # Increment ground trip counter
            self.ground_trips_total.labels(
                mode=mode, vehicle_type=vehicle_type
            ).inc()

            # Record distance
            if distance_km is not None and distance_km > 0:
                self.distance_km_total.labels(mode=mode).inc(distance_km)

                with self._stats_lock:
                    self._in_memory_stats["distance_km"] += distance_km

            with self._stats_lock:
                self._in_memory_stats["ground_trips"] += 1

            logger.debug(
                "Recorded ground trip: mode=%s, type=%s, distance=%.1f km",
                mode, vehicle_type,
                distance_km if distance_km else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record ground trip metrics: %s", e, exc_info=True)

    def record_hotel(
        self,
        country: str,
        nights: int,
    ) -> None:
        """
        Record hotel accommodation calculation.

        Tracks hotel night volume by country. Hotel emissions vary by
        country due to differing grid emission factors, heating fuel mixes,
        and building efficiency standards.

        Args:
            country: ISO 3166-1 alpha-2 country code (e.g., "US", "GB", "DE")
            nights: Number of hotel nights

        Example:
            >>> metrics.record_hotel(country="US", nights=3)
        """
        try:
            # Sanitize country: uppercase, max 3 chars
            if country is None:
                country = "UNKNOWN"
            else:
                country = country.upper().strip()[:3]

            if nights is not None and nights > 0:
                self.hotel_nights_total.labels(country=country).inc(nights)

                with self._stats_lock:
                    self._in_memory_stats["hotel_nights"] += nights

            logger.debug(
                "Recorded hotel: country=%s, nights=%d",
                country, nights if nights else 0,
            )

        except Exception as e:
            logger.error("Failed to record hotel metrics: %s", e, exc_info=True)

    def record_factor_selection(
        self,
        source: str,
        mode: str,
    ) -> None:
        """
        Record an emission factor selection/lookup.

        Tracks the source and mode of emission factor retrievals for
        monitoring database coverage and source distribution.

        Args:
            source: EF source (defra/epa/icao/iea/ecoinvent/ghg_protocol/
                     supplier_specific/custom/other)
            mode: Transport mode for which the EF was selected

        Example:
            >>> metrics.record_factor_selection(source="defra", mode="air")
        """
        try:
            source = self._validate_enum_value(
                source, EFSourceLabel, EFSourceLabel.OTHER.value
            )
            mode = self._validate_enum_value(
                mode, TransportModeLabel, TransportModeLabel.AIR.value
            )

            self.factor_selections_total.labels(source=source, mode=mode).inc()

            with self._stats_lock:
                self._in_memory_stats["factor_selections"] += 1

            logger.debug(
                "Recorded factor selection: source=%s, mode=%s",
                source, mode,
            )

        except Exception as e:
            logger.error("Failed to record factor selection metrics: %s", e, exc_info=True)

    def record_compliance_check(
        self,
        framework: str,
        status: str,
    ) -> None:
        """
        Record a compliance check result.

        Tracks regulatory compliance validation for Scope 3 Category 6
        reporting across applicable frameworks.

        Args:
            framework: Compliance framework (ghg_protocol/iso_14064/csrd/
                        cdp/sbti/sb253/gri/sec_climate)
            status: Check result (compliant/partially_compliant/non_compliant/
                     warning/not_applicable)

        Example:
            >>> metrics.record_compliance_check(
            ...     framework="ghg_protocol",
            ...     status="compliant"
            ... )
        """
        try:
            framework = self._validate_enum_value(
                framework, FrameworkLabel, FrameworkLabel.GHG_PROTOCOL.value
            )
            status = self._validate_enum_value(
                status, ComplianceStatusLabel, ComplianceStatusLabel.WARNING.value
            )

            self.compliance_checks_total.labels(
                framework=framework, status=status
            ).inc()

            with self._stats_lock:
                self._in_memory_stats["compliance_checks"] += 1

            logger.debug(
                "Recorded compliance check: framework=%s, status=%s",
                framework, status,
            )

        except Exception as e:
            logger.error("Failed to record compliance check metrics: %s", e, exc_info=True)

    def record_batch(
        self,
        status: str,
        size: int,
    ) -> None:
        """
        Record a batch processing job outcome.

        Tracks batch job completions and batch size distribution for
        capacity planning and performance monitoring.

        Args:
            status: Batch job status (completed/failed/partial/timeout)
            size: Number of trip records in the batch

        Example:
            >>> metrics.record_batch(status="completed", size=250)
        """
        try:
            status = self._validate_enum_value(
                status, BatchStatusLabel, BatchStatusLabel.FAILED.value
            )

            self.batch_jobs_total.labels(status=status).inc()

            if size is not None and size > 0:
                self.batch_size.observe(size)

            with self._stats_lock:
                self._in_memory_stats["batch_jobs"] += 1

            logger.debug(
                "Recorded batch job: status=%s, size=%d",
                status, size if size else 0,
            )

        except Exception as e:
            logger.error("Failed to record batch metrics: %s", e, exc_info=True)

    # ======================================================================
    # Summary and lifecycle
    # ======================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get a summary of metrics collected since initialization.

        Returns a dictionary with all in-memory counters, uptime information,
        and calculated rates (per-hour throughput).

        Returns:
            Dictionary with metrics summary including counts, uptime, rates,
            and modal breakdown.

        Example:
            >>> stats = metrics.get_stats()
            >>> print(stats['calculations'])
            142
            >>> print(stats['rates']['calculations_per_hour'])
            28.4
        """
        try:
            uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()
            uptime_hours = uptime_seconds / 3600 if uptime_seconds > 0 else 1.0

            with self._stats_lock:
                stats_snapshot = dict(self._in_memory_stats)

            summary: Dict[str, Any] = {
                "prometheus_available": PROMETHEUS_AVAILABLE,
                "agent": "business_travel",
                "agent_id": "GL-MRV-S3-006",
                "prefix": "gl_bt_",
                "scope": "Scope 3 Category 6",
                "description": "Business Travel",
                "metrics_count": 12,
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
                    "flights_per_hour": (
                        stats_snapshot["flights"] / uptime_hours
                    ),
                    "ground_trips_per_hour": (
                        stats_snapshot["ground_trips"] / uptime_hours
                    ),
                    "hotel_nights_per_hour": (
                        stats_snapshot["hotel_nights"] / uptime_hours
                    ),
                    "distance_km_per_hour": (
                        stats_snapshot["distance_km"] / uptime_hours
                    ),
                    "compliance_checks_per_hour": (
                        stats_snapshot["compliance_checks"] / uptime_hours
                    ),
                    "batch_jobs_per_hour": (
                        stats_snapshot["batch_jobs"] / uptime_hours
                    ),
                    "errors_per_hour": (
                        stats_snapshot["errors"] / uptime_hours
                    ),
                },
                "modal_breakdown": {
                    "flights": stats_snapshot["flights"],
                    "ground_trips": stats_snapshot["ground_trips"],
                    "hotel_nights": stats_snapshot["hotel_nights"],
                },
                "operational": {
                    "factor_selections": stats_snapshot["factor_selections"],
                    "compliance_checks": stats_snapshot["compliance_checks"],
                    "batch_jobs": stats_snapshot["batch_jobs"],
                    "errors": stats_snapshot["errors"],
                },
            }

            logger.debug(
                "Generated metrics summary: %d calculations tracked",
                stats_snapshot["calculations"],
            )
            return summary

        except Exception as e:
            logger.error("Failed to generate metrics summary: %s", e, exc_info=True)
            return {
                "error": str(e),
                "prometheus_available": PROMETHEUS_AVAILABLE,
                "agent": "business_travel",
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
            >>> BusinessTravelMetrics.reset()
            >>> metrics = BusinessTravelMetrics()  # Fresh instance
        """
        with cls._lock:
            if cls._instance is not None:
                if hasattr(cls._instance, "_initialized"):
                    del cls._instance._initialized
                cls._instance = None

                global _metrics_instance
                _metrics_instance = None

                logger.info("BusinessTravelMetrics singleton reset")

    def reset_stats(self) -> None:
        """
        Reset in-memory statistics (not Prometheus metrics).

        Note: This only resets the in-memory counters used for get_stats().
        Prometheus metrics are cumulative and cannot be reset without
        restarting the process. Also resets start_time for rate calculations.

        Example:
            >>> metrics.reset_stats()
            >>> stats = metrics.get_stats()
            >>> assert stats['calculations'] == 0
        """
        try:
            with self._stats_lock:
                self._in_memory_stats = {
                    "calculations": 0,
                    "emissions_kg_co2e": 0.0,
                    "flights": 0,
                    "ground_trips": 0,
                    "hotel_nights": 0,
                    "factor_selections": 0,
                    "compliance_checks": 0,
                    "batch_jobs": 0,
                    "distance_km": 0.0,
                    "errors": 0,
                }
            self._start_time = datetime.utcnow()

            logger.info("Reset in-memory statistics for BusinessTravelMetrics")

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
            >>> BusinessTravelMetrics._validate_enum_value(
            ...     "air", TransportModeLabel, "road"
            ... )
            'air'
            >>> BusinessTravelMetrics._validate_enum_value(
            ...     "spaceship", TransportModeLabel, "road"
            ... )
            'road'
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


# ===========================================================================
# Module-level singleton accessor
# ===========================================================================

_metrics_instance: Optional[BusinessTravelMetrics] = None
_metrics_lock: threading.Lock = threading.Lock()


def get_metrics() -> BusinessTravelMetrics:
    """
    Get the singleton BusinessTravelMetrics instance.

    Thread-safe accessor for the global metrics instance. Prefer this
    function over direct instantiation for consistency across the
    business travel agent codebase.

    Returns:
        BusinessTravelMetrics singleton instance

    Example:
        >>> from greenlang.business_travel.metrics import get_metrics
        >>> metrics = get_metrics()
        >>> metrics.record_calculation(
        ...     method="distance_based",
        ...     mode="air",
        ...     status="success",
        ...     duration=0.035,
        ...     co2e=245.8,
        ...     rf_option="with_rf"
        ... )
    """
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = BusinessTravelMetrics()

    return _metrics_instance


def reset_metrics() -> None:
    """
    Reset the singleton metrics instance for testing purposes.

    Convenience function that delegates to BusinessTravelMetrics.reset().
    Should only be called in test teardown.

    Example:
        >>> from greenlang.business_travel.metrics import reset_metrics
        >>> reset_metrics()
    """
    BusinessTravelMetrics.reset()


# ===========================================================================
# Context manager helpers
# ===========================================================================


@contextmanager
def track_calculation(
    method: str = "distance_based",
    mode: str = "air",
    rf_option: str = "without_rf",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a calculation's lifecycle.

    Automatically increments/decrements the active calculation gauge,
    measures wall-clock duration, and records the calculation when the
    context exits. The caller can set ``context['co2e']`` inside the
    block to include emissions in the recorded metric.

    Args:
        method: Calculation method (default: "distance_based")
        mode: Transport mode (default: "air")
        rf_option: Radiative forcing option (default: "without_rf")

    Yields:
        Mutable context dict. Set ``context['co2e']`` to record emissions.

    Example:
        >>> with track_calculation(method="distance_based", mode="rail") as ctx:
        ...     result = calculate_rail_emissions(trip)
        ...     ctx['co2e'] = result.co2e_kg
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        "co2e": 0.0,
        "status": "success",
    }

    metrics.active_calculations.inc()
    start = time.monotonic()

    try:
        yield context
    except Exception:
        context["status"] = "error"
        raise
    finally:
        duration = time.monotonic() - start
        metrics.active_calculations.dec()
        metrics.record_calculation(
            method=method,
            mode=mode,
            status=context["status"],
            duration=duration,
            co2e=context.get("co2e", 0.0),
            rf_option=rf_option,
        )


@contextmanager
def track_batch() -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a batch job's lifecycle.

    Automatically measures duration and records the batch outcome when
    the context exits. The caller should set ``context['size']`` inside
    the block to record the batch size.

    Yields:
        Mutable context dict. Set ``context['size']`` to record batch size.

    Example:
        >>> with track_batch() as ctx:
        ...     results = process_batch(trips)
        ...     ctx['size'] = len(trips)
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        "size": 0,
        "status": "completed",
    }

    start = time.monotonic()

    try:
        yield context
    except Exception:
        context["status"] = "failed"
        raise
    finally:
        metrics.record_batch(
            status=context["status"],
            size=context.get("size", 0),
        )


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    # Availability flag
    "PROMETHEUS_AVAILABLE",
    # Enums
    "CalculationMethodLabel",
    "TransportModeLabel",
    "CalculationStatusLabel",
    "DistanceBandLabel",
    "CabinClassLabel",
    "RFOptionLabel",
    "EFSourceLabel",
    "FrameworkLabel",
    "ComplianceStatusLabel",
    "BatchStatusLabel",
    "VehicleTypeLabel",
    # Singleton class
    "BusinessTravelMetrics",
    # Module-level accessors
    "get_metrics",
    "reset_metrics",
    # Context managers
    "track_calculation",
    "track_batch",
]
