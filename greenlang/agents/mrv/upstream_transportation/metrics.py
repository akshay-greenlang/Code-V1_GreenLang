# -*- coding: utf-8 -*-
"""
Upstream Transportation & Distribution Prometheus Metrics - AGENT-MRV-017

12 Prometheus metrics with gl_uto_ prefix for monitoring the
GL-MRV-S3-004 Upstream Transportation & Distribution Agent.

This module provides Prometheus metrics tracking for upstream transportation
and distribution emissions calculations (Scope 3, Category 4) including
distance-based, fuel-based, spend-based, supplier-specific, and hybrid
calculation methods across all transport modes (road, rail, maritime, air,
pipeline, and intermodal).

Thread-safe singleton pattern with graceful fallback if Prometheus unavailable.

Metrics prefix: gl_uto_

12 Prometheus Metrics:
    1.  gl_uto_calculations_total              - Counter: total calculations performed
    2.  gl_uto_emissions_tco2e_total           - Counter: total emissions in tCO2e
    3.  gl_uto_transport_lookups_total         - Counter: transport database lookups
    4.  gl_uto_distance_calculations_total     - Counter: distance-based calculations
    5.  gl_uto_fuel_calculations_total         - Counter: fuel-based calculations
    6.  gl_uto_spend_calculations_total        - Counter: spend-based calculations
    7.  gl_uto_multi_leg_calculations_total    - Counter: multi-leg chain calculations
    8.  gl_uto_compliance_checks_total         - Counter: compliance checks performed
    9.  gl_uto_calculation_duration_seconds    - Histogram: calculation durations
    10. gl_uto_batch_size                      - Histogram: batch calculation sizes
    11. gl_uto_active_calculations             - Gauge: currently active calculations
    12. gl_uto_emission_factors_loaded         - Gauge: loaded emission factors count

GHG Protocol Scope 3 Category 4 covers upstream transportation and
distribution of products purchased by the reporting company:
    A. Transportation of purchased goods between Tier 1 suppliers and the
       reporting company's own operations (in vehicles not owned or operated
       by the reporting company).
    B. Third-party transportation and distribution services purchased by the
       reporting company (e.g., outbound logistics paid for by the company).
    C. Transportation of purchased goods between Tier 1 and further upstream
       suppliers (where known).

Calculation methods defined by GHG Protocol:
    - Distance-based: activity data (tonne-km or vehicle-km) x emission factor
    - Fuel-based: fuel consumption x fuel-specific emission factor
    - Spend-based: expenditure x EEIO emission factor
    - Supplier-specific: primary data from logistics providers
    - Hybrid: combination of the above methods for different legs/modes

Transport modes tracked:
    - Road (truck, van, LCV, HGV, refrigerated)
    - Rail (freight, intermodal, bulk)
    - Maritime (container, bulk carrier, tanker, ro-ro, barge)
    - Air (freight, belly cargo, express, charter)
    - Pipeline (natural gas, liquid, slurry)
    - Intermodal (combined transport across modes)

Example:
    >>> metrics = UpstreamTransportationMetrics()
    >>> metrics.record_calculation(
    ...     method="distance_based",
    ...     transport_mode="road",
    ...     tenant_id="tenant-001",
    ...     status="success",
    ...     emissions_tco2e=12.85,
    ...     duration_s=0.45
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
# Enumerations -- Upstream Transportation domain-specific label value sets
# ===========================================================================

class CalculationMethod(str, Enum):
    """
    Calculation methods for upstream transportation & distribution emissions.

    GHG Protocol Scope 3 Category 4 supports several approaches depending
    on data availability and the level of accuracy required:
        - Distance-based: Uses tonne-km or vehicle-km with mode-specific EFs
        - Fuel-based: Uses actual or estimated fuel consumption data
        - Spend-based: Uses expenditure data with EEIO emission factors
        - Supplier-specific: Uses primary data from logistics providers
        - Hybrid: Combines multiple methods for different shipment legs
    """
    DISTANCE_BASED = "distance_based"
    FUEL_BASED = "fuel_based"
    SPEND_BASED = "spend_based"
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"


class TransportMode(str, Enum):
    """
    Transport modes for upstream transportation emissions tracking.

    Covers the primary freight transport modes defined in the GHG Protocol
    Technical Guidance for Calculating Scope 3 Emissions, Category 4,
    including road, rail, maritime, air, pipeline, and intermodal services.
    """
    ROAD = "road"
    RAIL = "rail"
    MARITIME = "maritime"
    AIR = "air"
    PIPELINE = "pipeline"
    INTERMODAL = "intermodal"
    BARGE = "barge"
    OTHER = "other"


class CalculationStatus(str, Enum):
    """Calculation operation status for upstream transportation calculations."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    INSUFFICIENT_DATA = "insufficient_data"
    SKIPPED = "skipped"


class EmissionFactorScope(str, Enum):
    """
    Emission factor scope for transportation calculations.

    Transportation emission factors can be reported as:
        - TTW (Tank-to-Wheel): Direct combustion emissions from the vehicle
        - WTT (Well-to-Tank): Upstream fuel production and distribution
        - WTW (Well-to-Wheel): Total lifecycle = TTW + WTT
    """
    TTW = "ttw"
    WTT = "wtt"
    WTW = "wtw"


class LookupType(str, Enum):
    """
    Types of transport database lookups performed by the agent.

    The agent maintains reference databases for vehicle emission factors,
    vessel specifications, aircraft performance, fuel properties, EEIO
    factors, and hub/port distance matrices.
    """
    VEHICLE = "vehicle"
    VESSEL = "vessel"
    AIRCRAFT = "aircraft"
    FUEL = "fuel"
    EEIO = "eeio"
    HUB = "hub"
    DISTANCE_MATRIX = "distance_matrix"
    PORT = "port"
    AIRPORT = "airport"
    ROUTE = "route"
    LOAD_FACTOR = "load_factor"
    OTHER = "other"


class VehicleType(str, Enum):
    """
    Vehicle types for distance-based road transport calculations.

    Aligned with DEFRA, EPA SmartWay, and GLEC Framework vehicle
    classifications used in freight transport emission factor databases.
    """
    ARTICULATED_TRUCK = "articulated_truck"
    RIGID_TRUCK = "rigid_truck"
    LIGHT_COMMERCIAL = "light_commercial"
    VAN = "van"
    HEAVY_GOODS_VEHICLE = "heavy_goods_vehicle"
    MEDIUM_GOODS_VEHICLE = "medium_goods_vehicle"
    REFRIGERATED_TRUCK = "refrigerated_truck"
    TANKER = "tanker"
    FLATBED = "flatbed"
    CONTAINER_TRUCK = "container_truck"
    FREIGHT_TRAIN = "freight_train"
    INTERMODAL_TRAIN = "intermodal_train"
    BULK_TRAIN = "bulk_train"
    CONTAINER_SHIP = "container_ship"
    BULK_CARRIER = "bulk_carrier"
    OIL_TANKER = "oil_tanker"
    CHEMICAL_TANKER = "chemical_tanker"
    RORO_VESSEL = "roro_vessel"
    GENERAL_CARGO = "general_cargo"
    INLAND_BARGE = "inland_barge"
    FREIGHT_AIRCRAFT = "freight_aircraft"
    BELLY_CARGO = "belly_cargo"
    EXPRESS_AIRCRAFT = "express_aircraft"
    CHARTER_AIRCRAFT = "charter_aircraft"
    GAS_PIPELINE = "gas_pipeline"
    LIQUID_PIPELINE = "liquid_pipeline"
    SLURRY_PIPELINE = "slurry_pipeline"
    OTHER = "other"


class DistanceMethod(str, Enum):
    """
    Methods for determining transport distances.

    Distance can be obtained through various means with decreasing accuracy:
        - Actual: Measured distance from GPS/telematics or logistics records
        - SFD (Shortest Feasible Distance): Network-based routing calculation
        - GCD (Great Circle Distance): Geodesic straight-line distance
        - Estimated: Based on origin/destination country/region averages
    """
    ACTUAL = "actual"
    SFD = "sfd"
    GCD = "gcd"
    ESTIMATED = "estimated"
    ROUTE_PLANNER = "route_planner"
    CARRIER_REPORTED = "carrier_reported"
    OTHER = "other"


class FuelType(str, Enum):
    """
    Fuel types tracked for fuel-based transportation emission calculations.

    Covers primary transport fuels across all modes including conventional
    fossil fuels, biofuel blends, LNG for maritime, and aviation fuels.
    """
    DIESEL = "diesel"
    PETROL = "petrol"
    BIODIESEL_B20 = "biodiesel_b20"
    BIODIESEL_B100 = "biodiesel_b100"
    HVO = "hvo"
    CNG = "cng"
    LNG = "lng"
    LPG = "lpg"
    MARINE_FUEL_OIL = "marine_fuel_oil"
    MARINE_GAS_OIL = "marine_gas_oil"
    MARINE_DIESEL_OIL = "marine_diesel_oil"
    VLSFO = "vlsfo"
    METHANOL = "methanol"
    AMMONIA = "ammonia"
    JET_A1 = "jet_a1"
    JET_A = "jet_a"
    AVGAS = "avgas"
    SAF = "saf"
    ELECTRICITY = "electricity"
    HYDROGEN = "hydrogen"
    ETHANOL_E85 = "ethanol_e85"
    NATURAL_GAS = "natural_gas"
    PROPANE = "propane"
    OTHER = "other"


class EEIOSource(str, Enum):
    """
    Environmentally-Extended Input-Output (EEIO) model sources for
    spend-based transportation emission calculations.

    EEIO models provide emission factors per unit of economic expenditure,
    enabling emissions estimation when only financial data is available.
    """
    USEEIO = "useeio"
    EXIOBASE = "exiobase"
    DEFRA = "defra"
    GTAP = "gtap"
    WIOD = "wiod"
    OECD_ICIO = "oecd_icio"
    INDUSTRY_AVERAGE = "industry_average"
    CUSTOM = "custom"


class Currency(str, Enum):
    """
    Currencies for spend-based emission calculations.

    Spend-based methods require currency normalization to match the
    EEIO model's base currency. The top reporting currencies are tracked
    as labels for Prometheus metrics cardinality control.
    """
    USD = "usd"
    EUR = "eur"
    GBP = "gbp"
    JPY = "jpy"
    CNY = "cny"
    INR = "inr"
    AUD = "aud"
    CAD = "cad"
    CHF = "chf"
    SEK = "sek"
    NOK = "nok"
    DKK = "dkk"
    KRW = "krw"
    BRL = "brl"
    MXN = "mxn"
    ZAR = "zar"
    SGD = "sgd"
    AED = "aed"
    OTHER = "other"


class NumLegsBucket(str, Enum):
    """
    Bucketed ranges for number of legs in multi-leg transport chains.

    Multi-leg calculations involve routing through multiple transport
    segments (e.g., truck to port, ship across ocean, truck to warehouse).
    Buckets provide bounded cardinality for Prometheus labels.
    """
    ONE = "1"
    TWO_TO_THREE = "2-3"
    FOUR_TO_FIVE = "4-5"
    SIX_PLUS = "6+"


class NumModesBucket(str, Enum):
    """
    Bucketed ranges for number of distinct transport modes in a chain.

    A shipment may traverse multiple modes (e.g., road + maritime + road).
    This label captures the modal complexity of multi-leg calculations.
    """
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR_PLUS = "4+"


class Framework(str, Enum):
    """
    Compliance frameworks for upstream transportation reporting.

    Tracks validation against regulatory and voluntary reporting
    standards applicable to Scope 3 Category 4 emissions.
    """
    GHG_PROTOCOL = "ghg_protocol"
    GLEC_FRAMEWORK = "glec_framework"
    ISO_14083 = "iso_14083"
    ISO_14064 = "iso_14064"
    EN_16258 = "en_16258"
    CSRD = "csrd"
    ESRS = "esrs"
    TCFD = "tcfd"
    CDP = "cdp"
    SBTI = "sbti"
    SMART_FREIGHT = "smart_freight"
    CLEAN_CARGO = "clean_cargo"
    SEC_CLIMATE = "sec_climate"
    EU_ETS_MARITIME = "eu_ets_maritime"
    IMO_DCS = "imo_dcs"
    CORSIA = "corsia"


class ComplianceStatus(str, Enum):
    """Compliance check result status for upstream transportation calculations."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    NEEDS_REVIEW = "needs_review"


class EmissionFactorType(str, Enum):
    """
    Types of emission factors loaded into the agent's reference database.

    Tracks the composition of the loaded emission factor set by transport
    mode and source type, enabling monitoring of factor database coverage
    and completeness.
    """
    ROAD = "road"
    RAIL = "rail"
    MARITIME = "maritime"
    AIR = "air"
    PIPELINE = "pipeline"
    FUEL = "fuel"
    EEIO = "eeio"
    HUB = "hub"
    INTERMODAL = "intermodal"
    OTHER = "other"


class ErrorType(str, Enum):
    """Error types for detailed tracking of transportation calculation failures."""
    VALIDATION_ERROR = "validation_error"
    CALCULATION_ERROR = "calculation_error"
    DATABASE_ERROR = "database_error"
    API_ERROR = "api_error"
    TIMEOUT_ERROR = "timeout_error"
    CONFIGURATION_ERROR = "configuration_error"
    DATA_NOT_FOUND = "data_not_found"
    EMISSION_FACTOR_UNAVAILABLE = "emission_factor_unavailable"
    DISTANCE_CALCULATION_ERROR = "distance_calculation_error"
    FUEL_CONVERSION_ERROR = "fuel_conversion_error"
    CURRENCY_CONVERSION_ERROR = "currency_conversion_error"
    EEIO_FACTOR_UNAVAILABLE = "eeio_factor_unavailable"
    VEHICLE_TYPE_UNKNOWN = "vehicle_type_unknown"
    ROUTE_NOT_FOUND = "route_not_found"
    MULTI_LEG_ASSEMBLY_ERROR = "multi_leg_assembly_error"
    LOAD_FACTOR_UNAVAILABLE = "load_factor_unavailable"
    CARRIER_DATA_UNAVAILABLE = "carrier_data_unavailable"
    WEIGHT_UNIT_CONVERSION_ERROR = "weight_unit_conversion_error"


class BatchStatus(str, Enum):
    """Batch processing job status for upstream transportation calculations."""
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


# ===========================================================================
# UpstreamTransportationMetrics -- Thread-safe Singleton
# ===========================================================================

class UpstreamTransportationMetrics:
    """
    Thread-safe singleton metrics collector for Upstream Transportation Agent (MRV-017).

    Provides 12 Prometheus metrics for tracking Scope 3 Category 4
    upstream transportation and distribution emissions calculations,
    including distance-based, fuel-based, spend-based, supplier-specific,
    and hybrid methods across all transport modes.

    All metrics use the ``gl_uto_`` prefix to ensure namespace isolation
    within the GreenLang Prometheus ecosystem.

    Scope 3 Category 4 Sub-Categories Tracked:
        A. Transportation of purchased goods (Tier 1 suppliers to reporting company)
        B. Third-party logistics services purchased by reporting company
        C. Upstream supplier-to-supplier transportation (where known)

    Calculation Methods Supported:
        - Distance-based: tonne-km / vehicle-km x emission factor
        - Fuel-based: litres / kg fuel x fuel emission factor
        - Spend-based: expenditure x EEIO factor
        - Supplier-specific: primary data from logistics providers
        - Hybrid: multi-method combination for complex supply chains

    Transport Modes:
        - Road (truck, van, HGV, refrigerated)
        - Rail (freight, intermodal, bulk)
        - Maritime (container, bulk, tanker, barge)
        - Air (freight, belly cargo, express)
        - Pipeline (gas, liquid, slurry)
        - Intermodal (combined transport)

    Attributes:
        calculations_total: Counter for total calculation operations
        emissions_tco2e_total: Counter for total emissions in tCO2e
        transport_lookups_total: Counter for transport database lookups
        distance_calculations_total: Counter for distance-based calculations
        fuel_calculations_total: Counter for fuel-based calculations
        spend_calculations_total: Counter for spend-based calculations
        multi_leg_calculations_total: Counter for multi-leg chain calculations
        compliance_checks_total: Counter for compliance checks
        calculation_duration_seconds: Histogram for operation durations
        batch_size: Histogram for batch calculation sizes
        active_calculations: Gauge for currently active calculations
        emission_factors_loaded: Gauge for loaded emission factor counts

    Example:
        >>> metrics = UpstreamTransportationMetrics()
        >>> metrics.record_calculation(
        ...     method="distance_based",
        ...     transport_mode="road",
        ...     tenant_id="tenant-001",
        ...     status="success",
        ...     emissions_tco2e=12.85,
        ...     duration_s=0.45
        ... )
        >>> summary = metrics.get_metrics_summary()
        >>> assert summary['calculations'] == 1
    """

    _instance: Optional["UpstreamTransportationMetrics"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "UpstreamTransportationMetrics":
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
            'emissions_tco2e': 0.0,
            'transport_lookups': 0,
            'distance_calculations': 0,
            'fuel_calculations': 0,
            'spend_calculations': 0,
            'multi_leg_calculations': 0,
            'compliance_checks': 0,
            'batch_jobs': 0,
            'errors': 0,
        }

        # Initialize Prometheus metrics
        self._init_metrics()

        logger.info(
            "UpstreamTransportationMetrics initialized (Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable"
        )

    def _init_metrics(self) -> None:
        """
        Initialize all 12 Prometheus metrics with gl_uto_ prefix.

        Each metric is documented with its purpose, label set, and the
        Scope 3 Category 4 aspect it supports.

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
        # 1. gl_uto_calculations_total (Counter)
        #    Total upstream transportation emission calculations performed.
        #    Labels:
        #      - method: distance_based, fuel_based, spend_based,
        #                supplier_specific, hybrid
        #      - transport_mode: road, rail, maritime, air, pipeline,
        #                        intermodal, barge, other
        #      - tenant_id: Tenant identifier for multi-tenant isolation
        #      - status: success, error, partial, insufficient_data, skipped
        #    Primary throughput counter for all calculation operations
        #    across methods, modes, and tenants.
        # ------------------------------------------------------------------
        self.calculations_total = _safe_create(Counter,
            'gl_uto_calculations_total',
            'Total upstream transportation emission calculations performed',
            ['method', 'transport_mode', 'tenant_id', 'status']
        )

        # ------------------------------------------------------------------
        # 2. gl_uto_emissions_tco2e_total (Counter)
        #    Total emissions calculated in tonnes CO2-equivalent (tCO2e).
        #    Labels:
        #      - transport_mode: Mode of transport generating emissions
        #      - ef_scope: ttw (tank-to-wheel), wtt (well-to-tank),
        #                  wtw (well-to-wheel / total lifecycle)
        #      - tenant_id: Tenant identifier
        #    Tracks cumulative emissions output enabling rate calculation,
        #    modal split analysis, and TTW/WTT breakdown reporting.
        #    Uses tCO2e (not kgCO2e) as the standard unit for Scope 3
        #    Category 4 reporting aligned with GHG Protocol guidance.
        # ------------------------------------------------------------------
        self.emissions_tco2e_total = _safe_create(Counter,
            'gl_uto_emissions_tco2e_total',
            'Total upstream transportation emissions calculated in tCO2e',
            ['transport_mode', 'ef_scope', 'tenant_id']
        )

        # ------------------------------------------------------------------
        # 3. gl_uto_transport_lookups_total (Counter)
        #    Total transport database lookups performed.
        #    Labels:
        #      - lookup_type: vehicle, vessel, aircraft, fuel, eeio, hub,
        #                     distance_matrix, port, airport, route,
        #                     load_factor, other
        #      - transport_mode: Mode for which the lookup was performed
        #    Tracks the frequency and type distribution of reference data
        #    retrievals, enabling cache hit ratio optimization and database
        #    coverage monitoring.
        # ------------------------------------------------------------------
        self.transport_lookups_total = _safe_create(Counter,
            'gl_uto_transport_lookups_total',
            'Total transport reference database lookups performed',
            ['lookup_type', 'transport_mode']
        )

        # ------------------------------------------------------------------
        # 4. gl_uto_distance_calculations_total (Counter)
        #    Total distance-based emission calculations performed.
        #    Labels:
        #      - transport_mode: road, rail, maritime, air, pipeline, etc.
        #      - vehicle_type: Specific vehicle/vessel classification
        #      - distance_method: actual, sfd, gcd, estimated, route_planner,
        #                         carrier_reported, other
        #    Distance-based is the most common method for Scope 3 Cat 4.
        #    Emissions = mass x distance x emission factor (per tonne-km).
        #    Tracks method distribution and vehicle type coverage.
        # ------------------------------------------------------------------
        self.distance_calculations_total = _safe_create(Counter,
            'gl_uto_distance_calculations_total',
            'Total distance-based upstream transportation calculations',
            ['transport_mode', 'vehicle_type', 'distance_method']
        )

        # ------------------------------------------------------------------
        # 5. gl_uto_fuel_calculations_total (Counter)
        #    Total fuel-based emission calculations performed.
        #    Labels:
        #      - fuel_type: diesel, lng, marine_fuel_oil, jet_a1, etc.
        #      - ef_scope: ttw, wtt, wtw
        #    Fuel-based method uses actual fuel consumption:
        #    Emissions = fuel quantity x fuel emission factor.
        #    Higher accuracy than distance-based when fuel data available.
        # ------------------------------------------------------------------
        self.fuel_calculations_total = _safe_create(Counter,
            'gl_uto_fuel_calculations_total',
            'Total fuel-based upstream transportation calculations',
            ['fuel_type', 'ef_scope']
        )

        # ------------------------------------------------------------------
        # 6. gl_uto_spend_calculations_total (Counter)
        #    Total spend-based emission calculations performed.
        #    Labels:
        #      - eeio_source: useeio, exiobase, defra, gtap, wiod, etc.
        #      - currency: usd, eur, gbp, etc.
        #    Spend-based is the least accurate but most data-available method:
        #    Emissions = expenditure x EEIO emission factor.
        #    Used as fallback when distance/fuel data unavailable.
        # ------------------------------------------------------------------
        self.spend_calculations_total = _safe_create(Counter,
            'gl_uto_spend_calculations_total',
            'Total spend-based upstream transportation calculations',
            ['eeio_source', 'currency']
        )

        # ------------------------------------------------------------------
        # 7. gl_uto_multi_leg_calculations_total (Counter)
        #    Total multi-leg transport chain calculations performed.
        #    Labels:
        #      - num_legs_bucket: 1, 2-3, 4-5, 6+ (bucketed for cardinality)
        #      - num_modes: 1, 2, 3, 4+ (distinct modes in the chain)
        #    Multi-leg calculations assemble emissions from multiple
        #    transport segments (e.g., truck -> port -> ship -> port -> truck).
        #    Tracks complexity distribution for performance optimization.
        # ------------------------------------------------------------------
        self.multi_leg_calculations_total = _safe_create(Counter,
            'gl_uto_multi_leg_calculations_total',
            'Total multi-leg transport chain calculations performed',
            ['num_legs_bucket', 'num_modes']
        )

        # ------------------------------------------------------------------
        # 8. gl_uto_compliance_checks_total (Counter)
        #    Total compliance checks performed for upstream transportation.
        #    Labels:
        #      - framework: ghg_protocol, glec_framework, iso_14083,
        #                   csrd, cdp, sbti, corsia, imo_dcs, etc.
        #      - status: compliant, partially_compliant, non_compliant,
        #                warning, not_applicable, needs_review
        #    Tracks regulatory compliance validation for Scope 3 Cat 4
        #    reporting across GHG Protocol, GLEC, ISO 14083, and other
        #    transport-specific standards.
        # ------------------------------------------------------------------
        self.compliance_checks_total = _safe_create(Counter,
            'gl_uto_compliance_checks_total',
            'Total upstream transportation compliance checks performed',
            ['framework', 'status']
        )

        # ------------------------------------------------------------------
        # 9. gl_uto_calculation_duration_seconds (Histogram)
        #    Duration of calculation operations in seconds.
        #    Labels:
        #      - method: Calculation method used
        #      - transport_mode: Mode of transport
        #    Buckets tuned for typical upstream transportation calculation
        #    latencies:
        #      - 10ms for cached factor lookups and simple calculations
        #      - 50-100ms for single-leg distance-based calculations
        #      - 250ms-1s for multi-leg or fuel-based with conversions
        #      - 1-10s for complex hybrid or batch operations
        # ------------------------------------------------------------------
        self.calculation_duration_seconds = _safe_create(Histogram,
            'gl_uto_calculation_duration_seconds',
            'Duration of upstream transportation calculation operations',
            ['method', 'transport_mode'],
            buckets=[
                0.01, 0.05, 0.1, 0.25, 0.5,
                1.0, 2.5, 5.0, 10.0
            ]
        )

        # ------------------------------------------------------------------
        # 10. gl_uto_batch_size (Histogram)
        #     Size of batch calculation operations (number of shipments).
        #     Labels:
        #       - method: Calculation method used for the batch
        #     Buckets cover typical batch sizes from single shipment
        #     to large-scale freight portfolio calculations.
        #     Enables monitoring of batch size distribution for capacity
        #     planning and performance tuning.
        # ------------------------------------------------------------------
        self.batch_size = _safe_create(Histogram,
            'gl_uto_batch_size',
            'Batch calculation size for upstream transportation operations',
            ['method'],
            buckets=[
                1, 5, 10, 25, 50, 100, 250, 500
            ]
        )

        # ------------------------------------------------------------------
        # 11. gl_uto_active_calculations (Gauge)
        #     Number of currently active calculations in progress.
        #     Labels:
        #       - method: Calculation method currently executing
        #     Provides real-time concurrency visibility per method for
        #     capacity planning and load monitoring. Enables alerting
        #     when active calculations exceed expected thresholds.
        # ------------------------------------------------------------------
        self.active_calculations = _safe_create(Gauge,
            'gl_uto_active_calculations',
            'Number of currently active upstream transportation calculations',
            ['method']
        )

        # ------------------------------------------------------------------
        # 12. gl_uto_emission_factors_loaded (Gauge)
        #     Number of emission factors currently loaded in memory.
        #     Labels:
        #       - factor_type: road, rail, maritime, air, pipeline, fuel,
        #                      eeio, hub, intermodal, other
        #     Tracks the completeness and composition of the emission
        #     factor reference database. Enables alerting when factor
        #     counts drop below expected thresholds (e.g., after a failed
        #     database refresh).
        # ------------------------------------------------------------------
        self.emission_factors_loaded = _safe_create(Gauge,
            'gl_uto_emission_factors_loaded',
            'Number of emission factors loaded for upstream transportation',
            ['factor_type']
        )

    # ======================================================================
    # Primary recording methods
    # ======================================================================

    def record_calculation(
        self,
        method: str,
        transport_mode: str,
        tenant_id: str,
        status: str,
        emissions_tco2e: Optional[float] = None,
        ef_scope: Optional[str] = None,
        duration_s: Optional[float] = None
    ) -> None:
        """
        Record an upstream transportation emission calculation operation.

        This is the primary recording method that increments the calculations
        counter, optionally observes the duration histogram, and optionally
        tracks emissions output. It covers all calculation methods and
        transport modes for Scope 3 Category 4.

        Args:
            method: Calculation method (distance_based/fuel_based/spend_based/
                     supplier_specific/hybrid)
            transport_mode: Transport mode (road/rail/maritime/air/pipeline/
                            intermodal/barge/other)
            tenant_id: Tenant identifier for multi-tenant isolation
            status: Calculation status (success/error/partial/
                     insufficient_data/skipped)
            emissions_tco2e: Emissions calculated in tCO2e (optional)
            ef_scope: Emission factor scope (ttw/wtt/wtw) (optional)
            duration_s: Operation duration in seconds (optional)

        Example:
            >>> metrics.record_calculation(
            ...     method="distance_based",
            ...     transport_mode="road",
            ...     tenant_id="tenant-001",
            ...     status="success",
            ...     emissions_tco2e=12.85,
            ...     ef_scope="wtw",
            ...     duration_s=0.45
            ... )
        """
        try:
            # Validate and normalize inputs
            method = self._validate_enum_value(
                method, CalculationMethod, CalculationMethod.DISTANCE_BASED.value
            )
            transport_mode = self._validate_enum_value(
                transport_mode, TransportMode, TransportMode.OTHER.value
            )
            status = self._validate_enum_value(
                status, CalculationStatus, CalculationStatus.ERROR.value
            )

            # Sanitize tenant_id: truncate long values to prevent label explosion
            if tenant_id is None:
                tenant_id = "unknown"
            elif len(tenant_id) > 64:
                tenant_id = tenant_id[:64]

            # 1. Increment calculation counter
            self.calculations_total.labels(
                method=method,
                transport_mode=transport_mode,
                tenant_id=tenant_id,
                status=status
            ).inc()

            # 2. Observe duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    method=method,
                    transport_mode=transport_mode
                ).observe(duration_s)

            # 3. Record emissions if provided
            if emissions_tco2e is not None and emissions_tco2e > 0:
                ef_scope_val = self._validate_enum_value(
                    ef_scope or EmissionFactorScope.WTW.value,
                    EmissionFactorScope,
                    EmissionFactorScope.WTW.value
                )

                self.emissions_tco2e_total.labels(
                    transport_mode=transport_mode,
                    ef_scope=ef_scope_val,
                    tenant_id=tenant_id
                ).inc(emissions_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_tco2e'] += emissions_tco2e

            # 4. Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['calculations'] += 1

            logger.debug(
                "Recorded calculation: method=%s, mode=%s, tenant=%s, "
                "status=%s, duration=%.3fs, emissions=%.4f tCO2e",
                method, transport_mode, tenant_id, status,
                duration_s if duration_s else 0.0,
                emissions_tco2e if emissions_tco2e else 0.0
            )

        except Exception as e:
            logger.error("Failed to record calculation metrics: %s", e, exc_info=True)

    def record_transport_lookup(
        self,
        lookup_type: str,
        transport_mode: str,
        count: int = 1,
        duration_s: Optional[float] = None
    ) -> None:
        """
        Record a transport reference database lookup operation.

        Tracks lookups to the vehicle, vessel, aircraft, fuel, EEIO,
        hub distance, port, airport, route, and load factor reference
        databases used for emission calculations.

        Args:
            lookup_type: Type of lookup (vehicle/vessel/aircraft/fuel/eeio/
                          hub/distance_matrix/port/airport/route/load_factor/other)
            transport_mode: Transport mode for which the lookup was performed
            count: Number of lookups in this operation (default: 1)
            duration_s: Lookup duration in seconds (optional)

        Example:
            >>> metrics.record_transport_lookup(
            ...     lookup_type="vehicle",
            ...     transport_mode="road",
            ...     count=1,
            ...     duration_s=0.003
            ... )
        """
        try:
            # Validate and normalize inputs
            lookup_type = self._validate_enum_value(
                lookup_type, LookupType, LookupType.OTHER.value
            )
            transport_mode = self._validate_enum_value(
                transport_mode, TransportMode, TransportMode.OTHER.value
            )

            # Clamp count to non-negative
            count = max(0, count)

            # Increment transport lookup counter
            self.transport_lookups_total.labels(
                lookup_type=lookup_type,
                transport_mode=transport_mode
            ).inc(count)

            # Record lookup duration if provided (use distance_based as default method)
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    method=CalculationMethod.DISTANCE_BASED.value,
                    transport_mode=transport_mode
                ).observe(duration_s)

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['transport_lookups'] += count

            logger.debug(
                "Recorded transport lookup: type=%s, mode=%s, count=%d",
                lookup_type, transport_mode, count
            )

        except Exception as e:
            logger.error(
                "Failed to record transport lookup metrics: %s",
                e, exc_info=True
            )

    def record_distance_calculation(
        self,
        transport_mode: str,
        vehicle_type: str,
        distance_method: str,
        distance_km: Optional[float] = None,
        weight_tonnes: Optional[float] = None,
        tonne_km: Optional[float] = None,
        emissions_tco2e: Optional[float] = None,
        ef_scope: Optional[str] = None,
        duration_s: Optional[float] = None,
        tenant_id: Optional[str] = None
    ) -> None:
        """
        Record a distance-based emission calculation.

        Distance-based is the most common method for Scope 3 Category 4:
            Emissions = mass (tonnes) x distance (km) x EF (per tonne-km)
        or equivalently:
            Emissions = tonne-km x EF (per tonne-km)

        Also supports vehicle-km based calculations for less-than-truckload
        or when load factor data is unavailable.

        Args:
            transport_mode: Transport mode (road/rail/maritime/air/pipeline/etc)
            vehicle_type: Specific vehicle/vessel type for EF selection
            distance_method: How distance was determined (actual/sfd/gcd/estimated/etc)
            distance_km: Total transport distance in kilometres (optional)
            weight_tonnes: Cargo weight in metric tonnes (optional)
            tonne_km: Tonne-kilometres (mass x distance) (optional)
            emissions_tco2e: Emissions in tCO2e if already calculated (optional)
            ef_scope: Emission factor scope (ttw/wtt/wtw) (optional)
            duration_s: Calculation duration in seconds (optional)
            tenant_id: Tenant identifier (optional)

        Example:
            >>> metrics.record_distance_calculation(
            ...     transport_mode="road",
            ...     vehicle_type="articulated_truck",
            ...     distance_method="actual",
            ...     distance_km=450.0,
            ...     weight_tonnes=18.5,
            ...     tonne_km=8325.0,
            ...     emissions_tco2e=0.562,
            ...     ef_scope="wtw"
            ... )
        """
        try:
            # Validate and normalize inputs
            transport_mode = self._validate_enum_value(
                transport_mode, TransportMode, TransportMode.OTHER.value
            )
            vehicle_type = self._validate_enum_value(
                vehicle_type, VehicleType, VehicleType.OTHER.value
            )
            distance_method = self._validate_enum_value(
                distance_method, DistanceMethod, DistanceMethod.OTHER.value
            )

            # Increment distance calculation counter
            self.distance_calculations_total.labels(
                transport_mode=transport_mode,
                vehicle_type=vehicle_type,
                distance_method=distance_method
            ).inc()

            # Record associated emissions if provided
            if emissions_tco2e is not None and emissions_tco2e > 0:
                ef_scope_val = self._validate_enum_value(
                    ef_scope or EmissionFactorScope.WTW.value,
                    EmissionFactorScope,
                    EmissionFactorScope.WTW.value
                )
                tenant_id_val = tenant_id or "unknown"
                if len(tenant_id_val) > 64:
                    tenant_id_val = tenant_id_val[:64]

                self.emissions_tco2e_total.labels(
                    transport_mode=transport_mode,
                    ef_scope=ef_scope_val,
                    tenant_id=tenant_id_val
                ).inc(emissions_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_tco2e'] += emissions_tco2e

            # Record calculation in the primary counter
            if tenant_id is not None:
                tenant_id_val = tenant_id
                if len(tenant_id_val) > 64:
                    tenant_id_val = tenant_id_val[:64]
                self.calculations_total.labels(
                    method=CalculationMethod.DISTANCE_BASED.value,
                    transport_mode=transport_mode,
                    tenant_id=tenant_id_val,
                    status=CalculationStatus.SUCCESS.value
                ).inc()

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    method=CalculationMethod.DISTANCE_BASED.value,
                    transport_mode=transport_mode
                ).observe(duration_s)

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['distance_calculations'] += 1

            logger.debug(
                "Recorded distance calculation: mode=%s, vehicle=%s, "
                "method=%s, distance=%.1f km, weight=%.2f t, "
                "tonne_km=%.1f, emissions=%.4f tCO2e",
                transport_mode, vehicle_type, distance_method,
                distance_km if distance_km else 0.0,
                weight_tonnes if weight_tonnes else 0.0,
                tonne_km if tonne_km else 0.0,
                emissions_tco2e if emissions_tco2e else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record distance calculation metrics: %s",
                e, exc_info=True
            )

    def record_fuel_calculation(
        self,
        fuel_type: str,
        ef_scope: str,
        fuel_quantity: Optional[float] = None,
        fuel_unit: Optional[str] = None,
        emissions_tco2e: Optional[float] = None,
        transport_mode: Optional[str] = None,
        duration_s: Optional[float] = None,
        tenant_id: Optional[str] = None
    ) -> None:
        """
        Record a fuel-based emission calculation.

        Fuel-based method uses actual or estimated fuel consumption:
            Emissions = fuel quantity x fuel emission factor
        This approach provides higher accuracy than distance-based when
        actual fuel consumption data is available from carriers.

        Args:
            fuel_type: Fuel type (diesel/lng/marine_fuel_oil/jet_a1/etc)
            ef_scope: Emission factor scope (ttw/wtt/wtw)
            fuel_quantity: Quantity of fuel consumed (optional, for logging)
            fuel_unit: Unit of fuel quantity (litres/kg/mj/etc) (optional)
            emissions_tco2e: Emissions in tCO2e if already calculated (optional)
            transport_mode: Transport mode (optional, for cross-reference)
            duration_s: Calculation duration in seconds (optional)
            tenant_id: Tenant identifier (optional)

        Example:
            >>> metrics.record_fuel_calculation(
            ...     fuel_type="diesel",
            ...     ef_scope="wtw",
            ...     fuel_quantity=125.0,
            ...     fuel_unit="litres",
            ...     emissions_tco2e=0.332,
            ...     transport_mode="road"
            ... )
        """
        try:
            # Validate and normalize inputs
            fuel_type = self._validate_enum_value(
                fuel_type, FuelType, FuelType.OTHER.value
            )
            ef_scope = self._validate_enum_value(
                ef_scope, EmissionFactorScope, EmissionFactorScope.WTW.value
            )

            # Increment fuel calculation counter
            self.fuel_calculations_total.labels(
                fuel_type=fuel_type,
                ef_scope=ef_scope
            ).inc()

            # Record associated emissions if provided
            if emissions_tco2e is not None and emissions_tco2e > 0:
                mode = self._validate_enum_value(
                    transport_mode or TransportMode.OTHER.value,
                    TransportMode,
                    TransportMode.OTHER.value
                )
                tenant_id_val = tenant_id or "unknown"
                if len(tenant_id_val) > 64:
                    tenant_id_val = tenant_id_val[:64]

                self.emissions_tco2e_total.labels(
                    transport_mode=mode,
                    ef_scope=ef_scope,
                    tenant_id=tenant_id_val
                ).inc(emissions_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_tco2e'] += emissions_tco2e

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                mode = self._validate_enum_value(
                    transport_mode or TransportMode.OTHER.value,
                    TransportMode,
                    TransportMode.OTHER.value
                )
                self.calculation_duration_seconds.labels(
                    method=CalculationMethod.FUEL_BASED.value,
                    transport_mode=mode
                ).observe(duration_s)

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['fuel_calculations'] += 1

            logger.debug(
                "Recorded fuel calculation: fuel=%s, scope=%s, "
                "quantity=%.2f %s, emissions=%.4f tCO2e",
                fuel_type, ef_scope,
                fuel_quantity if fuel_quantity else 0.0,
                fuel_unit if fuel_unit else "N/A",
                emissions_tco2e if emissions_tco2e else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record fuel calculation metrics: %s",
                e, exc_info=True
            )

    def record_spend_calculation(
        self,
        eeio_source: str,
        currency: str,
        spend_amount: Optional[float] = None,
        emissions_tco2e: Optional[float] = None,
        transport_mode: Optional[str] = None,
        duration_s: Optional[float] = None,
        tenant_id: Optional[str] = None
    ) -> None:
        """
        Record a spend-based emission calculation.

        Spend-based method uses expenditure data with EEIO factors:
            Emissions = expenditure (in base currency) x EEIO factor
        This is the least accurate method but most broadly applicable
        when physical activity data (distance, fuel) is unavailable.

        Args:
            eeio_source: EEIO model source (useeio/exiobase/defra/etc)
            currency: Currency of the expenditure (usd/eur/gbp/etc)
            spend_amount: Expenditure amount in original currency (optional)
            emissions_tco2e: Emissions in tCO2e if already calculated (optional)
            transport_mode: Transport mode (optional, for cross-reference)
            duration_s: Calculation duration in seconds (optional)
            tenant_id: Tenant identifier (optional)

        Example:
            >>> metrics.record_spend_calculation(
            ...     eeio_source="defra",
            ...     currency="gbp",
            ...     spend_amount=15000.0,
            ...     emissions_tco2e=4.25,
            ...     transport_mode="road"
            ... )
        """
        try:
            # Validate and normalize inputs
            eeio_source = self._validate_enum_value(
                eeio_source, EEIOSource, EEIOSource.CUSTOM.value
            )
            currency = self._validate_enum_value(
                currency, Currency, Currency.OTHER.value
            )

            # Increment spend calculation counter
            self.spend_calculations_total.labels(
                eeio_source=eeio_source,
                currency=currency
            ).inc()

            # Record associated emissions if provided
            if emissions_tco2e is not None and emissions_tco2e > 0:
                mode = self._validate_enum_value(
                    transport_mode or TransportMode.OTHER.value,
                    TransportMode,
                    TransportMode.OTHER.value
                )
                tenant_id_val = tenant_id or "unknown"
                if len(tenant_id_val) > 64:
                    tenant_id_val = tenant_id_val[:64]

                self.emissions_tco2e_total.labels(
                    transport_mode=mode,
                    ef_scope=EmissionFactorScope.WTW.value,
                    tenant_id=tenant_id_val
                ).inc(emissions_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_tco2e'] += emissions_tco2e

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                mode = self._validate_enum_value(
                    transport_mode or TransportMode.OTHER.value,
                    TransportMode,
                    TransportMode.OTHER.value
                )
                self.calculation_duration_seconds.labels(
                    method=CalculationMethod.SPEND_BASED.value,
                    transport_mode=mode
                ).observe(duration_s)

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['spend_calculations'] += 1

            logger.debug(
                "Recorded spend calculation: source=%s, currency=%s, "
                "amount=%.2f, emissions=%.4f tCO2e",
                eeio_source, currency,
                spend_amount if spend_amount else 0.0,
                emissions_tco2e if emissions_tco2e else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record spend calculation metrics: %s",
                e, exc_info=True
            )

    def record_multi_leg_calculation(
        self,
        num_legs: int,
        num_modes: int,
        total_emissions_tco2e: Optional[float] = None,
        total_distance_km: Optional[float] = None,
        duration_s: Optional[float] = None,
        tenant_id: Optional[str] = None
    ) -> None:
        """
        Record a multi-leg transport chain calculation.

        Multi-leg calculations assemble emissions from multiple transport
        segments, potentially crossing different modes. For example:
            Leg 1: Truck (factory to port) - 50 km
            Leg 2: Container ship (port to port) - 8000 km
            Leg 3: Truck (port to warehouse) - 120 km

        The number of legs and modes are bucketed to control Prometheus
        label cardinality.

        Args:
            num_legs: Number of transport legs in the chain
            num_modes: Number of distinct transport modes used
            total_emissions_tco2e: Total emissions across all legs in tCO2e (optional)
            total_distance_km: Total distance across all legs in km (optional)
            duration_s: Total calculation duration in seconds (optional)
            tenant_id: Tenant identifier (optional)

        Example:
            >>> metrics.record_multi_leg_calculation(
            ...     num_legs=3,
            ...     num_modes=2,
            ...     total_emissions_tco2e=8.45,
            ...     total_distance_km=8170.0,
            ...     duration_s=1.2
            ... )
        """
        try:
            # Bucket the number of legs
            legs_bucket = self._bucket_num_legs(num_legs)

            # Bucket the number of modes
            modes_bucket = self._bucket_num_modes(num_modes)

            # Increment multi-leg calculation counter
            self.multi_leg_calculations_total.labels(
                num_legs_bucket=legs_bucket,
                num_modes=modes_bucket
            ).inc()

            # Record total emissions if provided
            if total_emissions_tco2e is not None and total_emissions_tco2e > 0:
                tenant_id_val = tenant_id or "unknown"
                if len(tenant_id_val) > 64:
                    tenant_id_val = tenant_id_val[:64]

                self.emissions_tco2e_total.labels(
                    transport_mode=TransportMode.INTERMODAL.value,
                    ef_scope=EmissionFactorScope.WTW.value,
                    tenant_id=tenant_id_val
                ).inc(total_emissions_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_tco2e'] += total_emissions_tco2e

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    method=CalculationMethod.HYBRID.value,
                    transport_mode=TransportMode.INTERMODAL.value
                ).observe(duration_s)

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['multi_leg_calculations'] += 1

            logger.debug(
                "Recorded multi-leg calculation: legs=%d (bucket=%s), "
                "modes=%d (bucket=%s), distance=%.1f km, "
                "emissions=%.4f tCO2e",
                num_legs, legs_bucket, num_modes, modes_bucket,
                total_distance_km if total_distance_km else 0.0,
                total_emissions_tco2e if total_emissions_tco2e else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record multi-leg calculation metrics: %s",
                e, exc_info=True
            )

    def record_compliance_check(
        self,
        framework: str,
        status: str,
        duration_s: Optional[float] = None,
        details: Optional[str] = None
    ) -> None:
        """
        Record a compliance check operation for upstream transportation.

        Tracks validation of Scope 3 Category 4 calculations against
        regulatory and voluntary reporting frameworks including GHG Protocol,
        GLEC Framework, ISO 14083, CSRD, CDP, SBTi, CORSIA, and IMO DCS.

        Args:
            framework: Compliance framework (ghg_protocol/glec_framework/
                        iso_14083/csrd/cdp/sbti/corsia/imo_dcs/etc)
            status: Compliance check result (compliant/partially_compliant/
                     non_compliant/warning/not_applicable/needs_review)
            duration_s: Check duration in seconds (optional)
            details: Additional compliance details for logging (optional)

        Example:
            >>> metrics.record_compliance_check(
            ...     framework="glec_framework",
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
                    method=CalculationMethod.DISTANCE_BASED.value,
                    transport_mode=TransportMode.OTHER.value
                ).observe(duration_s)

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['compliance_checks'] += 1

            logger.debug(
                "Recorded compliance check: framework=%s, status=%s, details=%s",
                framework, status,
                details if details else "N/A"
            )

        except Exception as e:
            logger.error(
                "Failed to record compliance check metrics: %s",
                e, exc_info=True
            )

    def record_batch(
        self,
        method: str,
        size: int,
        successful: Optional[int] = None,
        failed: Optional[int] = None,
        duration_s: Optional[float] = None,
        total_emissions_tco2e: Optional[float] = None,
        tenant_id: Optional[str] = None
    ) -> None:
        """
        Record a batch calculation operation.

        Batch operations process multiple shipments or transport records
        in a single operation, typically for portfolio-level calculations
        covering a full reporting period or freight network.

        Args:
            method: Calculation method used for the batch
            size: Number of items (shipments) in the batch
            successful: Number of successful calculations (optional)
            failed: Number of failed calculations (optional)
            duration_s: Total batch duration in seconds (optional)
            total_emissions_tco2e: Total emissions for the batch in tCO2e (optional)
            tenant_id: Tenant identifier (optional)

        Example:
            >>> metrics.record_batch(
            ...     method="distance_based",
            ...     size=250,
            ...     successful=248,
            ...     failed=2,
            ...     duration_s=15.3,
            ...     total_emissions_tco2e=875.0
            ... )
        """
        try:
            # Validate and normalize method
            method = self._validate_enum_value(
                method, CalculationMethod, CalculationMethod.DISTANCE_BASED.value
            )

            # Clamp size to non-negative
            size = max(0, size)

            # Observe batch size in histogram
            if size > 0:
                self.batch_size.labels(
                    method=method
                ).observe(size)

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    method=method,
                    transport_mode=TransportMode.OTHER.value
                ).observe(duration_s)

            # Record successful calculations in primary counter
            if successful is not None and successful > 0:
                tenant_id_val = tenant_id or "unknown"
                if len(tenant_id_val) > 64:
                    tenant_id_val = tenant_id_val[:64]
                self.calculations_total.labels(
                    method=method,
                    transport_mode=TransportMode.OTHER.value,
                    tenant_id=tenant_id_val,
                    status=CalculationStatus.SUCCESS.value
                ).inc(successful)

            # Record failed calculations in primary counter
            if failed is not None and failed > 0:
                tenant_id_val = tenant_id or "unknown"
                if len(tenant_id_val) > 64:
                    tenant_id_val = tenant_id_val[:64]
                self.calculations_total.labels(
                    method=method,
                    transport_mode=TransportMode.OTHER.value,
                    tenant_id=tenant_id_val,
                    status=CalculationStatus.ERROR.value
                ).inc(failed)

            # Record aggregate emissions if provided
            if total_emissions_tco2e is not None and total_emissions_tco2e > 0:
                tenant_id_val = tenant_id or "unknown"
                if len(tenant_id_val) > 64:
                    tenant_id_val = tenant_id_val[:64]
                self.emissions_tco2e_total.labels(
                    transport_mode=TransportMode.OTHER.value,
                    ef_scope=EmissionFactorScope.WTW.value,
                    tenant_id=tenant_id_val
                ).inc(total_emissions_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_tco2e'] += total_emissions_tco2e

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['batch_jobs'] += 1
                if size > 0:
                    self._in_memory_stats['calculations'] += size

            logger.info(
                "Recorded batch: method=%s, size=%d, successful=%s, "
                "failed=%s, duration=%.2fs, emissions=%.4f tCO2e",
                method, size,
                successful if successful is not None else "N/A",
                failed if failed is not None else "N/A",
                duration_s if duration_s else 0.0,
                total_emissions_tco2e if total_emissions_tco2e else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record batch metrics: %s",
                e, exc_info=True
            )

    def record_error(
        self,
        error_type: str,
        operation: Optional[str] = None,
        transport_mode: Optional[str] = None,
        method: Optional[str] = None
    ) -> None:
        """
        Record a calculation error occurrence.

        Tracks error frequency by type for diagnostics, alerting, and
        root cause analysis. Error types specific to Scope 3 Category 4
        include emission factor unavailable, distance calculation error,
        currency conversion error, EEIO factor unavailable, multi-leg
        assembly error, and carrier data unavailable.

        Note: Errors are tracked via the calculations_total counter with
        status='error', but this method provides a lightweight way to
        increment just the in-memory error count and log the event without
        requiring all calculation parameters.

        Args:
            error_type: Type of error (validation_error/emission_factor_unavailable/
                         distance_calculation_error/etc)
            operation: Operation where the error occurred (optional, for logging)
            transport_mode: Transport mode related to error (optional)
            method: Calculation method related to error (optional)

        Example:
            >>> metrics.record_error(
            ...     error_type="emission_factor_unavailable",
            ...     operation="calculate_road_emissions",
            ...     transport_mode="road"
            ... )
        """
        try:
            # Validate error type
            error_type = self._validate_enum_value(
                error_type, ErrorType, ErrorType.VALIDATION_ERROR.value
            )

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['errors'] += 1

            logger.debug(
                "Recorded error: type=%s, operation=%s, mode=%s, method=%s",
                error_type,
                operation if operation else "N/A",
                transport_mode if transport_mode else "N/A",
                method if method else "N/A"
            )

        except Exception as e:
            logger.error("Failed to record error metrics: %s", e, exc_info=True)

    # ======================================================================
    # Gauge methods (active calculations, emission factors)
    # ======================================================================

    def set_active_calculations(self, method: str, count: float) -> None:
        """
        Set the active calculations gauge for a specific method.

        Directly sets the gauge value rather than incrementing/decrementing.
        Useful when the caller knows the exact number of active calculations
        (e.g., from a task queue depth query).

        Args:
            method: Calculation method (distance_based/fuel_based/spend_based/etc)
            count: Number of active calculations

        Example:
            >>> metrics.set_active_calculations("distance_based", 5)
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethod, CalculationMethod.DISTANCE_BASED.value
            )
            self.active_calculations.labels(
                method=method
            ).set(count)

            logger.debug("Set active calculations: method=%s, count=%.0f", method, count)

        except Exception as e:
            logger.error(
                "Failed to set active calculations gauge: %s",
                e, exc_info=True
            )

    def inc_active(self, method: str = "distance_based", amount: float = 1) -> None:
        """
        Increment the active calculations gauge for a specific method.

        Call this when a new calculation begins. Pair with dec_active()
        when the calculation completes (use track_calculation context
        manager for automatic inc/dec).

        Args:
            method: Calculation method (default: distance_based)
            amount: Amount to increment by (default: 1)

        Example:
            >>> metrics.inc_active("fuel_based")
            >>> try:
            ...     result = perform_fuel_calculation()
            ... finally:
            ...     metrics.dec_active("fuel_based")
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethod, CalculationMethod.DISTANCE_BASED.value
            )
            self.active_calculations.labels(
                method=method
            ).inc(amount)

            logger.debug("Active calculations incremented: method=%s, amount=%.0f", method, amount)

        except Exception as e:
            logger.error(
                "Failed to increment active calculations: %s",
                e, exc_info=True
            )

    def dec_active(self, method: str = "distance_based", amount: float = 1) -> None:
        """
        Decrement the active calculations gauge for a specific method.

        Call this when a calculation completes (successfully or not).

        Args:
            method: Calculation method (default: distance_based)
            amount: Amount to decrement by (default: 1)

        Example:
            >>> metrics.dec_active("fuel_based")
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethod, CalculationMethod.DISTANCE_BASED.value
            )
            self.active_calculations.labels(
                method=method
            ).dec(amount)

            logger.debug("Active calculations decremented: method=%s, amount=%.0f", method, amount)

        except Exception as e:
            logger.error(
                "Failed to decrement active calculations: %s",
                e, exc_info=True
            )

    def set_emission_factors_loaded(self, factor_type: str, count: float) -> None:
        """
        Set the number of loaded emission factors for a specific type.

        Tracks the completeness and composition of the emission factor
        reference database. Should be called after factor database
        initialization or refresh operations.

        Args:
            factor_type: Type of emission factors (road/rail/maritime/air/
                          pipeline/fuel/eeio/hub/intermodal/other)
            count: Number of emission factors loaded

        Example:
            >>> metrics.set_emission_factors_loaded("road", 156)
            >>> metrics.set_emission_factors_loaded("maritime", 89)
            >>> metrics.set_emission_factors_loaded("eeio", 45)
        """
        try:
            factor_type = self._validate_enum_value(
                factor_type, EmissionFactorType, EmissionFactorType.OTHER.value
            )
            self.emission_factors_loaded.labels(
                factor_type=factor_type
            ).set(count)

            logger.debug(
                "Set emission factors loaded: type=%s, count=%.0f",
                factor_type, count
            )

        except Exception as e:
            logger.error(
                "Failed to set emission factors loaded gauge: %s",
                e, exc_info=True
            )

    # ======================================================================
    # Summary and reset
    # ======================================================================

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of metrics collected since initialization.

        Returns a dictionary with all in-memory counters, uptime information,
        calculated rates (per-hour throughput), method breakdown, and
        transport mode distribution.

        Returns:
            Dictionary with metrics summary including counts, uptime, rates,
            method breakdown, and modal split analysis.

        Example:
            >>> summary = metrics.get_metrics_summary()
            >>> print(summary['calculations'])
            5432
            >>> print(summary['rates']['calculations_per_hour'])
            271.6
            >>> print(summary['emissions_tco2e'])
            12500.0
        """
        try:
            uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()
            uptime_hours = uptime_seconds / 3600 if uptime_seconds > 0 else 1.0

            with self._stats_lock:
                stats_snapshot = dict(self._in_memory_stats)

            summary: Dict[str, Any] = {
                'prometheus_available': PROMETHEUS_AVAILABLE,
                'agent': 'upstream_transportation',
                'agent_id': 'GL-MRV-S3-004',
                'prefix': 'gl_uto_',
                'scope': 'Scope 3 Category 4',
                'description': 'Upstream Transportation & Distribution',
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
                    'emissions_tco2e_per_hour': (
                        stats_snapshot['emissions_tco2e'] / uptime_hours
                    ),
                    'distance_calculations_per_hour': (
                        stats_snapshot['distance_calculations'] / uptime_hours
                    ),
                    'fuel_calculations_per_hour': (
                        stats_snapshot['fuel_calculations'] / uptime_hours
                    ),
                    'spend_calculations_per_hour': (
                        stats_snapshot['spend_calculations'] / uptime_hours
                    ),
                    'multi_leg_calculations_per_hour': (
                        stats_snapshot['multi_leg_calculations'] / uptime_hours
                    ),
                    'transport_lookups_per_hour': (
                        stats_snapshot['transport_lookups'] / uptime_hours
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
                'method_breakdown': {
                    'distance_based': {
                        'calculations': stats_snapshot['distance_calculations'],
                    },
                    'fuel_based': {
                        'calculations': stats_snapshot['fuel_calculations'],
                    },
                    'spend_based': {
                        'calculations': stats_snapshot['spend_calculations'],
                    },
                    'multi_leg': {
                        'calculations': stats_snapshot['multi_leg_calculations'],
                    },
                },
                'operational': {
                    'transport_lookups': stats_snapshot['transport_lookups'],
                    'compliance_checks': stats_snapshot['compliance_checks'],
                    'batch_jobs': stats_snapshot['batch_jobs'],
                    'errors': stats_snapshot['errors'],
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
                'agent': 'upstream_transportation',
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
            >>> UpstreamTransportationMetrics.reset()
            >>> metrics = UpstreamTransportationMetrics()  # Fresh instance
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

                logger.info("UpstreamTransportationMetrics singleton reset")

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
                    'emissions_tco2e': 0.0,
                    'transport_lookups': 0,
                    'distance_calculations': 0,
                    'fuel_calculations': 0,
                    'spend_calculations': 0,
                    'multi_leg_calculations': 0,
                    'compliance_checks': 0,
                    'batch_jobs': 0,
                    'errors': 0,
                }
            self._start_time = datetime.utcnow()

            logger.info("Reset in-memory statistics for UpstreamTransportationMetrics")

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
            >>> UpstreamTransportationMetrics._validate_enum_value(
            ...     "road", TransportMode, "other"
            ... )
            'road'
            >>> UpstreamTransportationMetrics._validate_enum_value(
            ...     "invalid_mode", TransportMode, "other"
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

    @staticmethod
    def _bucket_num_legs(num_legs: int) -> str:
        """
        Bucket the number of transport legs for label cardinality control.

        Maps an integer leg count to one of four predefined buckets:
            1 -> "1"
            2-3 -> "2-3"
            4-5 -> "4-5"
            6+ -> "6+"

        Args:
            num_legs: Number of transport legs in the chain

        Returns:
            Bucketed string value

        Example:
            >>> UpstreamTransportationMetrics._bucket_num_legs(1)
            '1'
            >>> UpstreamTransportationMetrics._bucket_num_legs(3)
            '2-3'
            >>> UpstreamTransportationMetrics._bucket_num_legs(7)
            '6+'
        """
        if num_legs <= 1:
            return NumLegsBucket.ONE.value
        elif num_legs <= 3:
            return NumLegsBucket.TWO_TO_THREE.value
        elif num_legs <= 5:
            return NumLegsBucket.FOUR_TO_FIVE.value
        else:
            return NumLegsBucket.SIX_PLUS.value

    @staticmethod
    def _bucket_num_modes(num_modes: int) -> str:
        """
        Bucket the number of distinct transport modes for label cardinality control.

        Maps an integer mode count to one of four predefined buckets:
            1 -> "1"
            2 -> "2"
            3 -> "3"
            4+ -> "4+"

        Args:
            num_modes: Number of distinct transport modes in the chain

        Returns:
            Bucketed string value

        Example:
            >>> UpstreamTransportationMetrics._bucket_num_modes(1)
            '1'
            >>> UpstreamTransportationMetrics._bucket_num_modes(2)
            '2'
            >>> UpstreamTransportationMetrics._bucket_num_modes(5)
            '4+'
        """
        if num_modes <= 1:
            return NumModesBucket.ONE.value
        elif num_modes == 2:
            return NumModesBucket.TWO.value
        elif num_modes == 3:
            return NumModesBucket.THREE.value
        else:
            return NumModesBucket.FOUR_PLUS.value


# ===========================================================================
# Module-level singleton accessor
# ===========================================================================

_metrics_instance: Optional[UpstreamTransportationMetrics] = None
_metrics_lock: threading.Lock = threading.Lock()


def get_metrics() -> UpstreamTransportationMetrics:
    """
    Get the singleton UpstreamTransportationMetrics instance.

    Thread-safe accessor for the global metrics instance. Prefer this
    function over direct instantiation for consistency across the
    upstream transportation agent codebase.

    Returns:
        UpstreamTransportationMetrics singleton instance

    Example:
        >>> from greenlang.agents.mrv.upstream_transportation.metrics import get_metrics
        >>> metrics = get_metrics()
        >>> metrics.record_calculation(
        ...     method="distance_based",
        ...     transport_mode="road",
        ...     tenant_id="tenant-001",
        ...     status="success",
        ...     emissions_tco2e=12.85,
        ...     duration_s=0.45
        ... )
    """
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = UpstreamTransportationMetrics()

    return _metrics_instance


def reset_metrics() -> None:
    """
    Reset the singleton metrics instance for testing purposes.

    Convenience function that delegates to UpstreamTransportationMetrics.reset().
    Should only be called in test teardown.

    Example:
        >>> from greenlang.agents.mrv.upstream_transportation.metrics import reset_metrics
        >>> reset_metrics()
    """
    UpstreamTransportationMetrics.reset()


# ===========================================================================
# Context manager helpers
# ===========================================================================

@contextmanager
def track_calculation(
    method: str = "distance_based",
    transport_mode: str = "road",
    tenant_id: str = "unknown"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a calculation's lifecycle.

    Automatically increments/decrements the active calculations gauge
    and records duration when the context exits. The caller can set
    ``context['emissions_tco2e']``, ``context['status']``, ``context['ef_scope']``
    before exiting to record those values.

    Args:
        method: Calculation method being used
        transport_mode: Transport mode being calculated
        tenant_id: Tenant identifier

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_calculation("distance_based", "road", "tenant-001") as ctx:
        ...     result = perform_distance_calculation()
        ...     ctx['emissions_tco2e'] = result.total_tco2e
        ...     ctx['ef_scope'] = "wtw"
        ...     ctx['status'] = "success"
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'method': method,
        'transport_mode': transport_mode,
        'tenant_id': tenant_id,
        'status': 'success',
        'emissions_tco2e': None,
        'ef_scope': None,
        'start_time': time.monotonic(),
    }

    # Increment active gauge
    metrics.inc_active(method)

    try:
        yield context

    except Exception as exc:
        context['status'] = 'error'
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
        metrics.dec_active(context['method'])

        # Record the calculation
        metrics.record_calculation(
            method=context['method'],
            transport_mode=context['transport_mode'],
            tenant_id=context['tenant_id'],
            status=context['status'],
            emissions_tco2e=context.get('emissions_tco2e'),
            ef_scope=context.get('ef_scope'),
            duration_s=duration_s
        )


@contextmanager
def track_transport_lookup(
    lookup_type: str = "vehicle",
    transport_mode: str = "road"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a transport database lookup lifecycle.

    Args:
        lookup_type: Type of lookup being performed
        transport_mode: Transport mode for the lookup

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_transport_lookup("vehicle", "road") as ctx:
        ...     factor = vehicle_db.lookup(vehicle_type="articulated_truck")
        ...     ctx['count'] = 1
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'lookup_type': lookup_type,
        'transport_mode': transport_mode,
        'count': 1,
        'start_time': time.monotonic(),
    }

    try:
        yield context

    except Exception as exc:
        context['error'] = str(exc)
        metrics.record_error(
            error_type=ErrorType.DATA_NOT_FOUND.value,
            operation="transport_lookup",
            transport_mode=transport_mode
        )
        logger.error(
            "Transport lookup failed: type=%s, mode=%s, error=%s",
            lookup_type, transport_mode, exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']

        if 'error' not in context:
            metrics.record_transport_lookup(
                lookup_type=context['lookup_type'],
                transport_mode=context['transport_mode'],
                count=context.get('count', 1),
                duration_s=duration_s
            )


@contextmanager
def track_distance_calculation(
    transport_mode: str = "road",
    vehicle_type: str = "articulated_truck",
    distance_method: str = "actual"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a distance-based calculation lifecycle.

    Automatically records the distance-based calculation metrics when the
    context exits. The caller should populate ``context['emissions_tco2e']``,
    ``context['distance_km']``, ``context['weight_tonnes']``, and
    ``context['tonne_km']`` before exiting.

    Corresponds to Scope 3 Category 4 distance-based method:
        Emissions = mass x distance x EF (per tonne-km)

    Args:
        transport_mode: Mode of transport being calculated
        vehicle_type: Specific vehicle/vessel classification
        distance_method: How distance was determined

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_distance_calculation("road", "articulated_truck", "actual") as ctx:
        ...     result = distance_engine.calculate(shipment_data)
        ...     ctx['emissions_tco2e'] = result.total_tco2e
        ...     ctx['distance_km'] = result.distance_km
        ...     ctx['weight_tonnes'] = result.weight_tonnes
        ...     ctx['tonne_km'] = result.tonne_km
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'transport_mode': transport_mode,
        'vehicle_type': vehicle_type,
        'distance_method': distance_method,
        'status': 'success',
        'emissions_tco2e': None,
        'ef_scope': None,
        'distance_km': None,
        'weight_tonnes': None,
        'tonne_km': None,
        'tenant_id': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active(CalculationMethod.DISTANCE_BASED.value)

    try:
        yield context

    except Exception as exc:
        context['status'] = 'error'
        context['error'] = str(exc)
        logger.error(
            "Distance-based calculation failed: mode=%s, vehicle=%s, error=%s",
            transport_mode, vehicle_type, exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active(CalculationMethod.DISTANCE_BASED.value)

        # Record the distance calculation
        metrics.record_distance_calculation(
            transport_mode=context['transport_mode'],
            vehicle_type=context['vehicle_type'],
            distance_method=context['distance_method'],
            distance_km=context.get('distance_km'),
            weight_tonnes=context.get('weight_tonnes'),
            tonne_km=context.get('tonne_km'),
            emissions_tco2e=context.get('emissions_tco2e'),
            ef_scope=context.get('ef_scope'),
            duration_s=duration_s,
            tenant_id=context.get('tenant_id')
        )


@contextmanager
def track_fuel_calculation(
    fuel_type: str = "diesel",
    ef_scope: str = "wtw"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a fuel-based calculation lifecycle.

    Automatically records the fuel-based calculation metrics when the
    context exits. The caller should populate ``context['emissions_tco2e']``,
    ``context['fuel_quantity']``, and ``context['fuel_unit']`` before exiting.

    Corresponds to Scope 3 Category 4 fuel-based method:
        Emissions = fuel quantity x fuel emission factor

    Args:
        fuel_type: Type of fuel consumed
        ef_scope: Emission factor scope (ttw/wtt/wtw)

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_fuel_calculation("diesel", "wtw") as ctx:
        ...     result = fuel_engine.calculate(fuel_data)
        ...     ctx['emissions_tco2e'] = result.total_tco2e
        ...     ctx['fuel_quantity'] = result.fuel_litres
        ...     ctx['fuel_unit'] = "litres"
        ...     ctx['transport_mode'] = "road"
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'fuel_type': fuel_type,
        'ef_scope': ef_scope,
        'status': 'success',
        'emissions_tco2e': None,
        'fuel_quantity': None,
        'fuel_unit': None,
        'transport_mode': None,
        'tenant_id': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active(CalculationMethod.FUEL_BASED.value)

    try:
        yield context

    except Exception as exc:
        context['status'] = 'error'
        context['error'] = str(exc)
        logger.error(
            "Fuel-based calculation failed: fuel=%s, scope=%s, error=%s",
            fuel_type, ef_scope, exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active(CalculationMethod.FUEL_BASED.value)

        # Record the fuel calculation
        metrics.record_fuel_calculation(
            fuel_type=context['fuel_type'],
            ef_scope=context['ef_scope'],
            fuel_quantity=context.get('fuel_quantity'),
            fuel_unit=context.get('fuel_unit'),
            emissions_tco2e=context.get('emissions_tco2e'),
            transport_mode=context.get('transport_mode'),
            duration_s=duration_s,
            tenant_id=context.get('tenant_id')
        )


@contextmanager
def track_spend_calculation(
    eeio_source: str = "defra",
    currency: str = "usd"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a spend-based calculation lifecycle.

    Automatically records the spend-based calculation metrics when the
    context exits. The caller should populate ``context['emissions_tco2e']``
    and ``context['spend_amount']`` before exiting.

    Corresponds to Scope 3 Category 4 spend-based method:
        Emissions = expenditure (base currency) x EEIO factor

    Args:
        eeio_source: EEIO model source for spend-based factors
        currency: Currency of the expenditure

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_spend_calculation("defra", "gbp") as ctx:
        ...     result = spend_engine.calculate(spend_data)
        ...     ctx['emissions_tco2e'] = result.total_tco2e
        ...     ctx['spend_amount'] = result.normalized_spend
        ...     ctx['transport_mode'] = "road"
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'eeio_source': eeio_source,
        'currency': currency,
        'status': 'success',
        'emissions_tco2e': None,
        'spend_amount': None,
        'transport_mode': None,
        'tenant_id': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active(CalculationMethod.SPEND_BASED.value)

    try:
        yield context

    except Exception as exc:
        context['status'] = 'error'
        context['error'] = str(exc)
        logger.error(
            "Spend-based calculation failed: source=%s, currency=%s, error=%s",
            eeio_source, currency, exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active(CalculationMethod.SPEND_BASED.value)

        # Record the spend calculation
        metrics.record_spend_calculation(
            eeio_source=context['eeio_source'],
            currency=context['currency'],
            spend_amount=context.get('spend_amount'),
            emissions_tco2e=context.get('emissions_tco2e'),
            transport_mode=context.get('transport_mode'),
            duration_s=duration_s,
            tenant_id=context.get('tenant_id')
        )


@contextmanager
def track_multi_leg_calculation(
    num_legs: int = 1,
    num_modes: int = 1
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a multi-leg chain calculation lifecycle.

    Automatically records the multi-leg calculation metrics when the
    context exits. The caller should populate ``context['total_emissions_tco2e']``
    and ``context['total_distance_km']`` before exiting.

    Multi-leg calculations assemble emissions from multiple transport
    segments (e.g., factory -> port -> ship -> port -> warehouse).

    Args:
        num_legs: Number of transport legs in the chain
        num_modes: Number of distinct transport modes used

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_multi_leg_calculation(3, 2) as ctx:
        ...     result = multi_leg_engine.calculate(chain_data)
        ...     ctx['total_emissions_tco2e'] = result.total_tco2e
        ...     ctx['total_distance_km'] = result.total_distance_km
        ...     ctx['num_legs'] = len(result.legs)
        ...     ctx['num_modes'] = len(result.modes)
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'num_legs': num_legs,
        'num_modes': num_modes,
        'status': 'success',
        'total_emissions_tco2e': None,
        'total_distance_km': None,
        'tenant_id': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active(CalculationMethod.HYBRID.value)

    try:
        yield context

    except Exception as exc:
        context['status'] = 'error'
        context['error'] = str(exc)
        logger.error(
            "Multi-leg calculation failed: legs=%d, modes=%d, error=%s",
            num_legs, num_modes, exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active(CalculationMethod.HYBRID.value)

        # Use potentially updated leg/mode counts from context
        final_legs = context.get('num_legs', num_legs)
        final_modes = context.get('num_modes', num_modes)

        # Record the multi-leg calculation
        metrics.record_multi_leg_calculation(
            num_legs=final_legs,
            num_modes=final_modes,
            total_emissions_tco2e=context.get('total_emissions_tco2e'),
            total_distance_km=context.get('total_distance_km'),
            duration_s=duration_s,
            tenant_id=context.get('tenant_id')
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
        >>> with track_compliance_check("glec_framework") as ctx:
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
            "Compliance check failed: framework=%s, error=%s",
            framework, exc, exc_info=True
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
    method: str = "distance_based"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a batch calculation lifecycle.

    Args:
        method: Primary calculation method for the batch

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_batch("distance_based") as ctx:
        ...     results = batch_engine.process(shipment_records)
        ...     ctx['size'] = len(shipment_records)
        ...     ctx['successful'] = sum(1 for r in results if r.ok)
        ...     ctx['failed'] = sum(1 for r in results if not r.ok)
        ...     ctx['total_emissions_tco2e'] = sum(r.tco2e for r in results if r.ok)
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'method': method,
        'status': 'completed',
        'size': 0,
        'successful': None,
        'failed': None,
        'total_emissions_tco2e': None,
        'tenant_id': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active(method)

    try:
        yield context

    except Exception as exc:
        context['status'] = 'failed'
        context['error'] = str(exc)
        logger.error(
            "Batch calculation failed: method=%s, error=%s",
            method, exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active(context['method'])

        size = context.get('size', 0)
        if size > 0:
            metrics.record_batch(
                method=context['method'],
                size=size,
                successful=context.get('successful'),
                failed=context.get('failed'),
                duration_s=duration_s,
                total_emissions_tco2e=context.get('total_emissions_tco2e'),
                tenant_id=context.get('tenant_id')
            )


@contextmanager
def track_duration(
    method: str = "distance_based",
    transport_mode: str = "road"
) -> Generator[None, None, None]:
    """
    Context manager that tracks the duration of an arbitrary operation.

    Records the elapsed time in the calculation_duration_seconds histogram
    when the context exits. Lightweight alternative to track_calculation
    when only duration tracking is needed.

    Args:
        method: Calculation method label for the duration histogram
        transport_mode: Transport mode label for the duration histogram

    Yields:
        None

    Example:
        >>> with track_duration("distance_based", "maritime"):
        ...     factors = load_maritime_emission_factors()
    """
    metrics = get_metrics()
    start = time.monotonic()

    try:
        yield

    finally:
        duration_s = time.monotonic() - start

        try:
            metrics.calculation_duration_seconds.labels(
                method=method,
                transport_mode=transport_mode
            ).observe(duration_s)

            logger.debug(
                "Tracked duration: method=%s, mode=%s, duration=%.4fs",
                method, transport_mode, duration_s
            )

        except Exception as e:
            logger.error(
                "Failed to record duration for method=%s, mode=%s: %s",
                method, transport_mode, e, exc_info=True
            )


# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    # Main class and accessors
    'UpstreamTransportationMetrics',
    'get_metrics',
    'reset_metrics',

    # Context managers
    'track_calculation',
    'track_transport_lookup',
    'track_distance_calculation',
    'track_fuel_calculation',
    'track_spend_calculation',
    'track_multi_leg_calculation',
    'track_compliance_check',
    'track_batch',
    'track_duration',

    # Enumerations
    'CalculationMethod',
    'TransportMode',
    'CalculationStatus',
    'EmissionFactorScope',
    'LookupType',
    'VehicleType',
    'DistanceMethod',
    'FuelType',
    'EEIOSource',
    'Currency',
    'NumLegsBucket',
    'NumModesBucket',
    'Framework',
    'ComplianceStatus',
    'EmissionFactorType',
    'ErrorType',
    'BatchStatus',

    # Availability flag
    'PROMETHEUS_AVAILABLE',
]
