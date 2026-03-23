"""
Metrics Collection for Capital Goods Agent (AGENT-MRV-015, GL-MRV-S3-002).

This module provides Prometheus metrics tracking for capital goods emissions
calculations (Scope 3, Category 2) including spend-based EEIO, average-data,
supplier-specific, and hybrid calculation methods, as well as asset lookups,
compliance checks, and batch processing.

Thread-safe singleton pattern with graceful fallback if Prometheus unavailable.

Metrics prefix: gl_cg_

12 Prometheus Metrics:
    1.  gl_cg_calculations_total           - Counter: total calculations performed
    2.  gl_cg_emissions_tco2e_total        - Counter: total emissions in tCO2e
    3.  gl_cg_asset_lookups_total          - Counter: asset/factor lookups
    4.  gl_cg_spend_based_calculations_total - Counter: spend-based EEIO calcs
    5.  gl_cg_average_data_calculations_total - Counter: average-data calcs
    6.  gl_cg_supplier_specific_calculations_total - Counter: supplier-specific calcs
    7.  gl_cg_hybrid_calculations_total    - Counter: hybrid aggregation calcs
    8.  gl_cg_compliance_checks_total      - Counter: compliance checks
    9.  gl_cg_calculation_duration_seconds  - Histogram: operation durations
    10. gl_cg_batch_size                   - Histogram: batch request sizes
    11. gl_cg_active_calculations          - Gauge: currently active calculations
    12. gl_cg_emission_factors_loaded      - Gauge: emission factors loaded

Example:
    >>> metrics = CapitalGoodsMetrics()
    >>> metrics.record_calculation(
    ...     method="spend_based",
    ...     category="machinery",
    ...     status="success",
    ...     duration_s=1.8,
    ...     emissions_tco2e=42.5
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
# Enumerations -- Capital Goods domain-specific label value sets
# ===========================================================================

class CalculationMethod(str, Enum):
    """Calculation methods for capital goods emissions (GHG Protocol Cat 2)."""
    SPEND_BASED = "spend_based"
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"
    AVERAGE_DATA = "average_data"
    EEIO = "eeio"  # Environmentally-Extended Input-Output
    LCA = "lca"  # Life Cycle Assessment
    DIRECT_MEASUREMENT = "direct_measurement"


class CalculationStatus(str, Enum):
    """Calculation operation status."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    INSUFFICIENT_DATA = "insufficient_data"
    SKIPPED = "skipped"


class AssetCategory(str, Enum):
    """Capital goods asset categories per GHG Protocol Scope 3 Category 2."""
    MACHINERY = "machinery"
    VEHICLES = "vehicles"
    BUILDINGS = "buildings"
    IT_EQUIPMENT = "it_equipment"
    FURNITURE = "furniture"
    PLANT_EQUIPMENT = "plant_equipment"
    LAND_IMPROVEMENTS = "land_improvements"
    INFRASTRUCTURE = "infrastructure"
    HVAC_SYSTEMS = "hvac_systems"
    ELECTRICAL_SYSTEMS = "electrical_systems"
    MANUFACTURING_EQUIPMENT = "manufacturing_equipment"
    LABORATORY_EQUIPMENT = "laboratory_equipment"
    MEDICAL_EQUIPMENT = "medical_equipment"
    TELECOMMUNICATIONS = "telecommunications"
    RENEWABLE_ENERGY_ASSETS = "renewable_energy_assets"
    OTHER = "other"


class EmissionGas(str, Enum):
    """Greenhouse gas types tracked in capital goods calculations."""
    CO2 = "co2"
    CH4 = "ch4"
    N2O = "n2o"
    HFCS = "hfcs"
    PFCS = "pfcs"
    SF6 = "sf6"
    NF3 = "nf3"
    CO2E = "co2e"  # Aggregate CO2-equivalent


class EEIODatabase(str, Enum):
    """EEIO databases used for spend-based capital goods calculations."""
    USEEIO = "useeio"
    EXIOBASE = "exiobase"
    GTAP = "gtap"
    WIOD = "wiod"
    DEFRA = "defra"
    ECOINVENT = "ecoinvent"
    OECD_ICIO = "oecd_icio"
    CUSTOM = "custom"


class Currency(str, Enum):
    """Currencies for spend-based calculations."""
    USD = "usd"
    EUR = "eur"
    GBP = "gbp"
    JPY = "jpy"
    CNY = "cny"
    AUD = "aud"
    CAD = "cad"
    CHF = "chf"
    INR = "inr"
    BRL = "brl"
    OTHER = "other"


class EFSource(str, Enum):
    """Emission factor sources for average-data calculations."""
    IPCC = "ipcc"
    EPA = "epa"
    DEFRA = "defra"
    ECOINVENT = "ecoinvent"
    GHG_PROTOCOL = "ghg_protocol"
    IEA = "iea"
    INDUSTRY_AVERAGE = "industry_average"
    CUSTOM = "custom"


class SupplierDataSource(str, Enum):
    """Data sources for supplier-specific calculations."""
    CDP_QUESTIONNAIRE = "cdp_questionnaire"
    SUPPLIER_SURVEY = "supplier_survey"
    EPD = "epd"  # Environmental Product Declaration
    LCA_REPORT = "lca_report"
    VERIFIED_INVENTORY = "verified_inventory"
    ERP_SYSTEM = "erp_system"
    THIRD_PARTY_DATABASE = "third_party_database"
    ECOVADIS = "ecovadis"
    DIRECT_REPORT = "direct_report"


class VerificationStatus(str, Enum):
    """Supplier data verification status."""
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    SELF_DECLARED = "self_declared"
    THIRD_PARTY_VERIFIED = "third_party_verified"
    LIMITED_ASSURANCE = "limited_assurance"
    REASONABLE_ASSURANCE = "reasonable_assurance"
    PENDING = "pending"


class AssetLookupSource(str, Enum):
    """Sources for asset/factor lookup operations."""
    DATABASE = "database"
    CACHE = "cache"
    API = "api"
    CALCULATED = "calculated"
    DEFAULT = "default"
    FALLBACK = "fallback"


class Framework(str, Enum):
    """Compliance frameworks for capital goods reporting."""
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
    """Error types for detailed tracking."""
    VALIDATION_ERROR = "validation_error"
    CALCULATION_ERROR = "calculation_error"
    DATABASE_ERROR = "database_error"
    API_ERROR = "api_error"
    TIMEOUT_ERROR = "timeout_error"
    CONFIGURATION_ERROR = "configuration_error"
    DATA_NOT_FOUND = "data_not_found"
    SUPPLIER_DATA_UNAVAILABLE = "supplier_data_unavailable"
    EF_LOOKUP_FAILED = "ef_lookup_failed"
    CURRENCY_CONVERSION_ERROR = "currency_conversion_error"
    ASSET_CLASSIFICATION_ERROR = "asset_classification_error"
    DEPRECIATION_ERROR = "depreciation_error"


# ===========================================================================
# CapitalGoodsMetrics -- Thread-safe Singleton
# ===========================================================================

class CapitalGoodsMetrics:
    """
    Thread-safe singleton metrics collector for Capital Goods Agent (MRV-015).

    Provides 12 Prometheus metrics for tracking capital goods emissions
    calculations across spend-based, average-data, supplier-specific, and
    hybrid methods, along with asset lookups, compliance checks, batch
    processing, and active calculation gauges.

    All metrics use the ``gl_cg_`` prefix to ensure namespace isolation
    within the GreenLang Prometheus ecosystem.

    Attributes:
        calculations_total: Counter for total calculation operations
        emissions_tco2e_total: Counter for total emissions calculated (tCO2e)
        asset_lookups_total: Counter for asset/factor lookups
        spend_based_calculations_total: Counter for spend-based EEIO calcs
        average_data_calculations_total: Counter for average-data calcs
        supplier_specific_calculations_total: Counter for supplier-specific calcs
        hybrid_calculations_total: Counter for hybrid aggregation calcs
        compliance_checks_total: Counter for compliance checks
        calculation_duration_seconds: Histogram for operation durations
        batch_size: Histogram for batch request sizes
        active_calculations: Gauge for currently active calculations
        emission_factors_loaded: Gauge for loaded emission factors

    Example:
        >>> metrics = CapitalGoodsMetrics()
        >>> metrics.record_calculation(
        ...     method="spend_based",
        ...     category="machinery",
        ...     status="success",
        ...     duration_s=1.8,
        ...     emissions_tco2e=42.5
        ... )
        >>> summary = metrics.get_metrics_summary()
        >>> assert summary['calculations'] == 1
    """

    _instance: Optional["CapitalGoodsMetrics"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "CapitalGoodsMetrics":
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
            'asset_lookups': 0,
            'spend_based_calculations': 0,
            'average_data_calculations': 0,
            'supplier_specific_calculations': 0,
            'hybrid_calculations': 0,
            'compliance_checks': 0,
            'batch_jobs': 0,
            'errors': 0,
        }

        # Initialize Prometheus metrics
        self._init_metrics()

        logger.info(
            "CapitalGoodsMetrics initialized (Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable"
        )

    def _init_metrics(self) -> None:
        """Initialize all 12 Prometheus metrics with gl_cg_ prefix."""

        # ------------------------------------------------------------------
        # 1. gl_cg_calculations_total (Counter)
        #    Total capital goods emission calculations performed.
        # ------------------------------------------------------------------
        self.calculations_total = Counter(
            'gl_cg_calculations_total',
            'Total capital goods emission calculations performed',
            ['calculation_method', 'asset_category', 'status']
        )

        # ------------------------------------------------------------------
        # 2. gl_cg_emissions_tco2e_total (Counter)
        #    Total emissions calculated in tonnes CO2e.
        # ------------------------------------------------------------------
        self.emissions_tco2e_total = Counter(
            'gl_cg_emissions_tco2e_total',
            'Total emissions calculated in tonnes CO2e',
            ['asset_category', 'emission_gas', 'calculation_method']
        )

        # ------------------------------------------------------------------
        # 3. gl_cg_asset_lookups_total (Counter)
        #    Total asset/factor lookups performed.
        # ------------------------------------------------------------------
        self.asset_lookups_total = Counter(
            'gl_cg_asset_lookups_total',
            'Total asset/factor lookups performed',
            ['source', 'asset_category']
        )

        # ------------------------------------------------------------------
        # 4. gl_cg_spend_based_calculations_total (Counter)
        #    Total spend-based EEIO calculations.
        # ------------------------------------------------------------------
        self.spend_based_calculations_total = Counter(
            'gl_cg_spend_based_calculations_total',
            'Total spend-based EEIO calculations',
            ['eeio_database', 'currency', 'status']
        )

        # ------------------------------------------------------------------
        # 5. gl_cg_average_data_calculations_total (Counter)
        #    Total average-data calculations.
        # ------------------------------------------------------------------
        self.average_data_calculations_total = Counter(
            'gl_cg_average_data_calculations_total',
            'Total average-data calculations',
            ['ef_source', 'asset_category', 'status']
        )

        # ------------------------------------------------------------------
        # 6. gl_cg_supplier_specific_calculations_total (Counter)
        #    Total supplier-specific calculations.
        # ------------------------------------------------------------------
        self.supplier_specific_calculations_total = Counter(
            'gl_cg_supplier_specific_calculations_total',
            'Total supplier-specific calculations',
            ['data_source', 'verification_status', 'status']
        )

        # ------------------------------------------------------------------
        # 7. gl_cg_hybrid_calculations_total (Counter)
        #    Total hybrid aggregation calculations.
        # ------------------------------------------------------------------
        self.hybrid_calculations_total = Counter(
            'gl_cg_hybrid_calculations_total',
            'Total hybrid aggregation calculations',
            ['primary_method', 'status']
        )

        # ------------------------------------------------------------------
        # 8. gl_cg_compliance_checks_total (Counter)
        #    Total compliance checks performed.
        # ------------------------------------------------------------------
        self.compliance_checks_total = Counter(
            'gl_cg_compliance_checks_total',
            'Total compliance checks performed',
            ['framework', 'compliance_status']
        )

        # ------------------------------------------------------------------
        # 9. gl_cg_calculation_duration_seconds (Histogram)
        #    Duration of calculation operations.
        # ------------------------------------------------------------------
        self.calculation_duration_seconds = Histogram(
            'gl_cg_calculation_duration_seconds',
            'Duration of calculation operations',
            ['operation', 'calculation_method'],
            buckets=[
                0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
                1.0, 2.5, 5.0, 10.0, 30.0, 60.0
            ]
        )

        # ------------------------------------------------------------------
        # 10. gl_cg_batch_size (Histogram)
        #     Size of batch calculation requests.
        # ------------------------------------------------------------------
        self.batch_size = Histogram(
            'gl_cg_batch_size',
            'Size of batch calculation requests',
            ['calculation_method'],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500]
        )

        # ------------------------------------------------------------------
        # 11. gl_cg_active_calculations (Gauge)
        #     Number of currently active calculations.
        # ------------------------------------------------------------------
        self.active_calculations = Gauge(
            'gl_cg_active_calculations',
            'Number of currently active calculations'
        )

        # ------------------------------------------------------------------
        # 12. gl_cg_emission_factors_loaded (Gauge)
        #     Number of emission factors currently loaded.
        # ------------------------------------------------------------------
        self.emission_factors_loaded = Gauge(
            'gl_cg_emission_factors_loaded',
            'Number of emission factors currently loaded',
            ['source', 'asset_category']
        )

    # ======================================================================
    # Recording methods
    # ======================================================================

    def record_calculation(
        self,
        method: str,
        category: str,
        status: str,
        duration_s: float,
        emissions_tco2e: Optional[float] = None,
        emission_gas: Optional[str] = None,
        operation: Optional[str] = None
    ) -> None:
        """
        Record a capital goods emission calculation operation.

        This is the primary recording method that increments the calculations
        counter, observes the duration histogram, and optionally tracks
        emissions output.

        Args:
            method: Calculation method (spend_based/supplier_specific/hybrid/etc)
            category: Asset category (machinery/vehicles/buildings/etc)
            status: Calculation status (success/failed/partial/insufficient_data)
            duration_s: Operation duration in seconds
            emissions_tco2e: Emissions calculated in tonnes CO2e (optional)
            emission_gas: Gas type for emissions (co2/ch4/n2o/co2e) (optional)
            operation: Operation name for duration histogram (optional)

        Example:
            >>> metrics.record_calculation(
            ...     method="spend_based",
            ...     category="machinery",
            ...     status="success",
            ...     duration_s=1.8,
            ...     emissions_tco2e=42.5,
            ...     emission_gas="co2e",
            ...     operation="calculate_emissions"
            ... )
        """
        try:
            # Validate and normalize method
            method = self._validate_enum_value(
                method, CalculationMethod, CalculationMethod.SPEND_BASED.value
            )

            # Validate and normalize category
            category = self._validate_enum_value(
                category, AssetCategory, AssetCategory.OTHER.value
            )

            # Validate and normalize status
            status = self._validate_enum_value(
                status, CalculationStatus, CalculationStatus.FAILED.value
            )

            # Validate and normalize operation
            if operation is None:
                operation = "calculate_emissions"

            # 1. Increment calculation counter
            self.calculations_total.labels(
                calculation_method=method,
                asset_category=category,
                status=status
            ).inc()

            # 2. Observe duration
            self.calculation_duration_seconds.labels(
                operation=operation,
                calculation_method=method
            ).observe(duration_s)

            # 3. Record emissions if provided
            if emissions_tco2e is not None and emissions_tco2e > 0:
                emission_gas = self._validate_enum_value(
                    emission_gas or EmissionGas.CO2E.value,
                    EmissionGas,
                    EmissionGas.CO2E.value
                )

                self.emissions_tco2e_total.labels(
                    asset_category=category,
                    emission_gas=emission_gas,
                    calculation_method=method
                ).inc(emissions_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_tco2e'] += emissions_tco2e

            # 4. Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['calculations'] += 1

            logger.debug(
                "Recorded calculation: method=%s, category=%s, status=%s, "
                "duration=%.3fs, emissions=%.4f tCO2e",
                method, category, status, duration_s,
                emissions_tco2e if emissions_tco2e else 0.0
            )

        except Exception as e:
            logger.error("Failed to record calculation metrics: %s", e, exc_info=True)

    def record_spend_based(
        self,
        database: str,
        currency: str,
        status: str,
        duration_s: Optional[float] = None,
        spend_amount: Optional[float] = None,
        emissions_tco2e: Optional[float] = None
    ) -> None:
        """
        Record a spend-based EEIO calculation.

        Args:
            database: EEIO database used (useeio/exiobase/gtap/defra/etc)
            currency: Transaction currency (usd/eur/gbp/etc)
            status: Calculation status (success/failed/partial)
            duration_s: Operation duration in seconds (optional)
            spend_amount: Spend amount in original currency (optional)
            emissions_tco2e: Calculated emissions in tCO2e (optional)

        Example:
            >>> metrics.record_spend_based(
            ...     database="useeio",
            ...     currency="usd",
            ...     status="success",
            ...     duration_s=0.45,
            ...     spend_amount=250000.0,
            ...     emissions_tco2e=87.3
            ... )
        """
        try:
            # Validate and normalize inputs
            database = self._validate_enum_value(
                database, EEIODatabase, EEIODatabase.USEEIO.value
            )
            currency = self._validate_enum_value(
                currency, Currency, Currency.USD.value
            )
            status = self._validate_enum_value(
                status, CalculationStatus, CalculationStatus.FAILED.value
            )

            # Increment spend-based counter
            self.spend_based_calculations_total.labels(
                eeio_database=database,
                currency=currency,
                status=status
            ).inc()

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    operation="spend_based_calculation",
                    calculation_method=CalculationMethod.SPEND_BASED.value
                ).observe(duration_s)

            # Record emissions if provided
            if emissions_tco2e is not None and emissions_tco2e > 0:
                self.emissions_tco2e_total.labels(
                    asset_category=AssetCategory.OTHER.value,
                    emission_gas=EmissionGas.CO2E.value,
                    calculation_method=CalculationMethod.SPEND_BASED.value
                ).inc(emissions_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_tco2e'] += emissions_tco2e

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['spend_based_calculations'] += 1

            logger.debug(
                "Recorded spend-based calc: db=%s, currency=%s, status=%s, "
                "spend=%.2f, emissions=%.4f tCO2e",
                database, currency, status,
                spend_amount if spend_amount else 0.0,
                emissions_tco2e if emissions_tco2e else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record spend-based calculation metrics: %s",
                e, exc_info=True
            )

    def record_average_data(
        self,
        source: str,
        category: str,
        status: str,
        duration_s: Optional[float] = None,
        emissions_tco2e: Optional[float] = None
    ) -> None:
        """
        Record an average-data calculation.

        Average-data calculations use industry-average emission factors
        per unit of goods/assets purchased (e.g., tCO2e per tonne of steel).

        Args:
            source: Emission factor source (ipcc/epa/defra/ecoinvent/etc)
            category: Asset category (machinery/vehicles/buildings/etc)
            status: Calculation status (success/failed/partial)
            duration_s: Operation duration in seconds (optional)
            emissions_tco2e: Calculated emissions in tCO2e (optional)

        Example:
            >>> metrics.record_average_data(
            ...     source="ecoinvent",
            ...     category="machinery",
            ...     status="success",
            ...     duration_s=0.32,
            ...     emissions_tco2e=18.7
            ... )
        """
        try:
            # Validate and normalize inputs
            source = self._validate_enum_value(
                source, EFSource, EFSource.INDUSTRY_AVERAGE.value
            )
            category = self._validate_enum_value(
                category, AssetCategory, AssetCategory.OTHER.value
            )
            status = self._validate_enum_value(
                status, CalculationStatus, CalculationStatus.FAILED.value
            )

            # Increment average-data counter
            self.average_data_calculations_total.labels(
                ef_source=source,
                asset_category=category,
                status=status
            ).inc()

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    operation="average_data_calculation",
                    calculation_method=CalculationMethod.AVERAGE_DATA.value
                ).observe(duration_s)

            # Record emissions if provided
            if emissions_tco2e is not None and emissions_tco2e > 0:
                self.emissions_tco2e_total.labels(
                    asset_category=category,
                    emission_gas=EmissionGas.CO2E.value,
                    calculation_method=CalculationMethod.AVERAGE_DATA.value
                ).inc(emissions_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_tco2e'] += emissions_tco2e

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['average_data_calculations'] += 1

            logger.debug(
                "Recorded average-data calc: source=%s, category=%s, "
                "status=%s, emissions=%.4f tCO2e",
                source, category, status,
                emissions_tco2e if emissions_tco2e else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record average-data calculation metrics: %s",
                e, exc_info=True
            )

    def record_supplier_specific(
        self,
        source: str,
        verification: str,
        status: str,
        duration_s: Optional[float] = None,
        emissions_tco2e: Optional[float] = None,
        category: Optional[str] = None
    ) -> None:
        """
        Record a supplier-specific calculation.

        Supplier-specific calculations use primary data from suppliers
        (e.g., EPDs, LCA reports, verified carbon inventories).

        Args:
            source: Supplier data source (cdp_questionnaire/epd/lca_report/etc)
            verification: Data verification status (verified/unverified/etc)
            status: Calculation status (success/failed/partial)
            duration_s: Operation duration in seconds (optional)
            emissions_tco2e: Calculated emissions in tCO2e (optional)
            category: Asset category (optional, defaults to "other")

        Example:
            >>> metrics.record_supplier_specific(
            ...     source="epd",
            ...     verification="third_party_verified",
            ...     status="success",
            ...     duration_s=0.88,
            ...     emissions_tco2e=12.4,
            ...     category="machinery"
            ... )
        """
        try:
            # Validate and normalize inputs
            source = self._validate_enum_value(
                source, SupplierDataSource, SupplierDataSource.SUPPLIER_SURVEY.value
            )
            verification = self._validate_enum_value(
                verification, VerificationStatus, VerificationStatus.UNVERIFIED.value
            )
            status = self._validate_enum_value(
                status, CalculationStatus, CalculationStatus.FAILED.value
            )

            # Increment supplier-specific counter
            self.supplier_specific_calculations_total.labels(
                data_source=source,
                verification_status=verification,
                status=status
            ).inc()

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    operation="supplier_specific_calculation",
                    calculation_method=CalculationMethod.SUPPLIER_SPECIFIC.value
                ).observe(duration_s)

            # Record emissions if provided
            if emissions_tco2e is not None and emissions_tco2e > 0:
                asset_cat = self._validate_enum_value(
                    category or AssetCategory.OTHER.value,
                    AssetCategory,
                    AssetCategory.OTHER.value
                )
                self.emissions_tco2e_total.labels(
                    asset_category=asset_cat,
                    emission_gas=EmissionGas.CO2E.value,
                    calculation_method=CalculationMethod.SUPPLIER_SPECIFIC.value
                ).inc(emissions_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_tco2e'] += emissions_tco2e

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['supplier_specific_calculations'] += 1

            logger.debug(
                "Recorded supplier-specific calc: source=%s, verification=%s, "
                "status=%s, emissions=%.4f tCO2e",
                source, verification, status,
                emissions_tco2e if emissions_tco2e else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record supplier-specific calculation metrics: %s",
                e, exc_info=True
            )

    def record_hybrid(
        self,
        primary_method: str,
        status: str,
        duration_s: Optional[float] = None,
        emissions_tco2e: Optional[float] = None,
        methods_combined: Optional[int] = None
    ) -> None:
        """
        Record a hybrid aggregation calculation.

        Hybrid calculations combine multiple methods (e.g., supplier-specific
        for top suppliers + spend-based for remainder).

        Args:
            primary_method: The primary calculation method used in the hybrid
            status: Calculation status (success/failed/partial)
            duration_s: Operation duration in seconds (optional)
            emissions_tco2e: Total combined emissions in tCO2e (optional)
            methods_combined: Number of methods combined (optional)

        Example:
            >>> metrics.record_hybrid(
            ...     primary_method="supplier_specific",
            ...     status="success",
            ...     duration_s=3.2,
            ...     emissions_tco2e=156.8,
            ...     methods_combined=3
            ... )
        """
        try:
            # Validate and normalize inputs
            primary_method = self._validate_enum_value(
                primary_method, CalculationMethod, CalculationMethod.HYBRID.value
            )
            status = self._validate_enum_value(
                status, CalculationStatus, CalculationStatus.FAILED.value
            )

            # Increment hybrid counter
            self.hybrid_calculations_total.labels(
                primary_method=primary_method,
                status=status
            ).inc()

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    operation="hybrid_calculation",
                    calculation_method=CalculationMethod.HYBRID.value
                ).observe(duration_s)

            # Record emissions if provided
            if emissions_tco2e is not None and emissions_tco2e > 0:
                self.emissions_tco2e_total.labels(
                    asset_category=AssetCategory.OTHER.value,
                    emission_gas=EmissionGas.CO2E.value,
                    calculation_method=CalculationMethod.HYBRID.value
                ).inc(emissions_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_tco2e'] += emissions_tco2e

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['hybrid_calculations'] += 1

            logger.debug(
                "Recorded hybrid calc: primary=%s, status=%s, "
                "methods_combined=%s, emissions=%.4f tCO2e",
                primary_method, status,
                methods_combined if methods_combined else "N/A",
                emissions_tco2e if emissions_tco2e else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record hybrid calculation metrics: %s",
                e, exc_info=True
            )

    def record_asset_lookup(
        self,
        source: str,
        category: str,
        count: int = 1,
        duration_s: Optional[float] = None
    ) -> None:
        """
        Record an asset/factor lookup operation.

        Tracks how many lookups are performed against databases, caches,
        or API endpoints to retrieve emission factors for capital goods.

        Args:
            source: Lookup source (database/cache/api/calculated/default/fallback)
            category: Asset category being looked up
            count: Number of lookups (default: 1)
            duration_s: Lookup duration in seconds (optional)

        Example:
            >>> metrics.record_asset_lookup(
            ...     source="cache",
            ...     category="machinery",
            ...     count=5,
            ...     duration_s=0.003
            ... )
        """
        try:
            # Validate and normalize inputs
            source = self._validate_enum_value(
                source, AssetLookupSource, AssetLookupSource.DATABASE.value
            )
            category = self._validate_enum_value(
                category, AssetCategory, AssetCategory.OTHER.value
            )

            # Increment asset lookup counter
            self.asset_lookups_total.labels(
                source=source,
                asset_category=category
            ).inc(count)

            # Record lookup duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    operation="asset_lookup",
                    calculation_method="lookup"
                ).observe(duration_s)

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['asset_lookups'] += count

            logger.debug(
                "Recorded asset lookup: source=%s, category=%s, count=%d",
                source, category, count
            )

        except Exception as e:
            logger.error(
                "Failed to record asset lookup metrics: %s",
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
                compliance_status=status
            ).inc()

            # Record check duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    operation="compliance_check",
                    calculation_method="compliance"
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

    def record_batch(
        self,
        method: str,
        size: int,
        successful: Optional[int] = None,
        failed: Optional[int] = None,
        duration_s: Optional[float] = None,
        total_emissions_tco2e: Optional[float] = None
    ) -> None:
        """
        Record a batch calculation request.

        Observes the batch size histogram and optionally records aggregate
        results for the batch.

        Args:
            method: Calculation method used for the batch
            size: Number of items in the batch
            successful: Number of successful calculations (optional)
            failed: Number of failed calculations (optional)
            duration_s: Total batch duration in seconds (optional)
            total_emissions_tco2e: Total emissions for the batch (optional)

        Example:
            >>> metrics.record_batch(
            ...     method="spend_based",
            ...     size=100,
            ...     successful=95,
            ...     failed=5,
            ...     duration_s=12.5,
            ...     total_emissions_tco2e=4500.0
            ... )
        """
        try:
            # Validate and normalize method
            method = self._validate_enum_value(
                method, CalculationMethod, CalculationMethod.SPEND_BASED.value
            )

            # Observe batch size
            self.batch_size.labels(
                calculation_method=method
            ).observe(size)

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    operation="batch_calculation",
                    calculation_method=method
                ).observe(duration_s)

            # Record successful calculations
            if successful is not None and successful > 0:
                self.calculations_total.labels(
                    calculation_method=method,
                    asset_category=AssetCategory.OTHER.value,
                    status=CalculationStatus.SUCCESS.value
                ).inc(successful)

            # Record failed calculations
            if failed is not None and failed > 0:
                self.calculations_total.labels(
                    calculation_method=method,
                    asset_category=AssetCategory.OTHER.value,
                    status=CalculationStatus.FAILED.value
                ).inc(failed)

            # Record aggregate emissions if provided
            if total_emissions_tco2e is not None and total_emissions_tco2e > 0:
                self.emissions_tco2e_total.labels(
                    asset_category=AssetCategory.OTHER.value,
                    emission_gas=EmissionGas.CO2E.value,
                    calculation_method=method
                ).inc(total_emissions_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_tco2e'] += total_emissions_tco2e

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['batch_jobs'] += 1
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
        operation: str
    ) -> None:
        """
        Record an error occurrence.

        Args:
            error_type: Type of error (validation_error/calculation_error/etc)
            operation: Operation where the error occurred

        Example:
            >>> metrics.record_error(
            ...     error_type="ef_lookup_failed",
            ...     operation="calculate_spend_based"
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

            logger.debug("Recorded error: type=%s, operation=%s", error_type, operation)

        except Exception as e:
            logger.error("Failed to record error metrics: %s", e, exc_info=True)

    # ======================================================================
    # Gauge methods (active calculations, emission factors loaded)
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
            ...     result = perform_calculation()
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

    def set_factors_loaded(
        self,
        source: str,
        category: str,
        count: int
    ) -> None:
        """
        Set the number of emission factors currently loaded.

        This gauge tracks how many emission factors are available in memory
        for a given source/category combination.

        Args:
            source: Factor source (ipcc/epa/defra/ecoinvent/etc)
            category: Asset category
            count: Number of factors loaded

        Example:
            >>> metrics.set_factors_loaded(
            ...     source="ecoinvent",
            ...     category="machinery",
            ...     count=245
            ... )
        """
        try:
            # Validate and normalize inputs
            source = self._validate_enum_value(
                source, EFSource, EFSource.CUSTOM.value
            )
            category = self._validate_enum_value(
                category, AssetCategory, AssetCategory.OTHER.value
            )

            # Set gauge value
            self.emission_factors_loaded.labels(
                source=source,
                asset_category=category
            ).set(count)

            logger.debug(
                "Set factors loaded: source=%s, category=%s, count=%d",
                source, category, count
            )

        except Exception as e:
            logger.error(
                "Failed to set emission factors loaded: %s",
                e, exc_info=True
            )

    # ======================================================================
    # Summary and reset
    # ======================================================================

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of metrics collected since initialization.

        Returns a dictionary with all in-memory counters, uptime information,
        and calculated rates (per-hour throughput).

        Returns:
            Dictionary with metrics summary including counts, uptime, and rates

        Example:
            >>> summary = metrics.get_metrics_summary()
            >>> print(summary['calculations'])
            5432
            >>> print(summary['rates']['calculations_per_hour'])
            271.6
        """
        try:
            uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()
            uptime_hours = uptime_seconds / 3600 if uptime_seconds > 0 else 1.0

            with self._stats_lock:
                stats_snapshot = dict(self._in_memory_stats)

            summary: Dict[str, Any] = {
                'prometheus_available': PROMETHEUS_AVAILABLE,
                'agent': 'capital_goods',
                'agent_id': 'GL-MRV-S3-002',
                'prefix': 'gl_cg_',
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
                    'asset_lookups_per_hour': (
                        stats_snapshot['asset_lookups'] / uptime_hours
                    ),
                    'spend_based_per_hour': (
                        stats_snapshot['spend_based_calculations'] / uptime_hours
                    ),
                    'average_data_per_hour': (
                        stats_snapshot['average_data_calculations'] / uptime_hours
                    ),
                    'supplier_specific_per_hour': (
                        stats_snapshot['supplier_specific_calculations'] / uptime_hours
                    ),
                    'hybrid_per_hour': (
                        stats_snapshot['hybrid_calculations'] / uptime_hours
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
                    'spend_based': stats_snapshot['spend_based_calculations'],
                    'average_data': stats_snapshot['average_data_calculations'],
                    'supplier_specific': stats_snapshot['supplier_specific_calculations'],
                    'hybrid': stats_snapshot['hybrid_calculations'],
                },
            }

            logger.debug("Generated metrics summary: %d calculations tracked", stats_snapshot['calculations'])
            return summary

        except Exception as e:
            logger.error("Failed to generate metrics summary: %s", e, exc_info=True)
            return {
                'error': str(e),
                'prometheus_available': PROMETHEUS_AVAILABLE,
                'agent': 'capital_goods',
            }

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance for testing purposes.

        This destroys the existing singleton so that a fresh instance
        will be created on next access. Primarily used in unit tests.

        Example:
            >>> CapitalGoodsMetrics.reset()
            >>> metrics = CapitalGoodsMetrics()  # Fresh instance
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

                logger.info("CapitalGoodsMetrics singleton reset")

    def reset_stats(self) -> None:
        """
        Reset in-memory statistics (not Prometheus metrics).

        Note: This only resets the in-memory counters used for get_metrics_summary().
        Prometheus metrics are cumulative and cannot be reset without restarting.

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
                    'asset_lookups': 0,
                    'spend_based_calculations': 0,
                    'average_data_calculations': 0,
                    'supplier_specific_calculations': 0,
                    'hybrid_calculations': 0,
                    'compliance_checks': 0,
                    'batch_jobs': 0,
                    'errors': 0,
                }
            self._start_time = datetime.utcnow()

            logger.info("Reset in-memory statistics for CapitalGoodsMetrics")

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
        warning and returns the default.

        Args:
            value: The string value to validate
            enum_class: The Enum class to validate against
            default: The default value if validation fails

        Returns:
            Validated value or default
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

_metrics_instance: Optional[CapitalGoodsMetrics] = None
_metrics_lock: threading.Lock = threading.Lock()


def get_metrics() -> CapitalGoodsMetrics:
    """
    Get the singleton CapitalGoodsMetrics instance.

    Thread-safe accessor for the global metrics instance.  Prefer this
    function over direct instantiation for consistency.

    Returns:
        CapitalGoodsMetrics singleton instance

    Example:
        >>> from greenlang.agents.mrv.capital_goods.metrics import get_metrics
        >>> metrics = get_metrics()
        >>> metrics.record_calculation(
        ...     method="spend_based",
        ...     category="machinery",
        ...     status="success",
        ...     duration_s=1.5,
        ...     emissions_tco2e=35.2
        ... )
    """
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = CapitalGoodsMetrics()

    return _metrics_instance


# ===========================================================================
# Context manager helpers
# ===========================================================================

@contextmanager
def track_calculation(
    method: str = "spend_based",
    category: str = "other",
    operation: str = "calculate_emissions"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a calculation's lifecycle.

    Automatically increments/decrements the active calculations gauge
    and records duration when the context exits. The caller can set
    ``context['emissions_tco2e']`` and ``context['status']`` before
    exiting to record those values.

    Args:
        method: Calculation method being used
        category: Asset category being calculated
        operation: Operation name for the duration histogram

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_calculation("spend_based", "machinery") as ctx:
        ...     result = perform_calculation()
        ...     ctx['emissions_tco2e'] = result.emissions
        ...     ctx['status'] = "success"
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'method': method,
        'category': category,
        'operation': operation,
        'status': 'success',
        'emissions_tco2e': None,
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
            method=context['method'],
            category=context['category'],
            status=context['status'],
            duration_s=duration_s,
            emissions_tco2e=context.get('emissions_tco2e'),
            operation=context.get('operation')
        )


@contextmanager
def track_duration(
    operation: str,
    method: str = "spend_based"
) -> Generator[None, None, None]:
    """
    Context manager that tracks the duration of an arbitrary operation.

    Records the elapsed time in the calculation_duration_seconds histogram
    when the context exits.

    Args:
        operation: Name of the operation being timed
        method: Calculation method label (default: "spend_based")

    Yields:
        None

    Example:
        >>> with track_duration("asset_lookup", "average_data"):
        ...     factors = load_emission_factors("machinery")
    """
    metrics = get_metrics()
    start = time.monotonic()

    try:
        yield

    finally:
        duration_s = time.monotonic() - start

        try:
            metrics.calculation_duration_seconds.labels(
                operation=operation,
                calculation_method=method
            ).observe(duration_s)

            logger.debug(
                "Tracked duration: operation=%s, method=%s, duration=%.4fs",
                operation, method, duration_s
            )

        except Exception as e:
            logger.error(
                "Failed to record duration for operation=%s: %s",
                operation, e, exc_info=True
            )


@contextmanager
def track_spend_based(
    database: str = "useeio",
    currency: str = "usd"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a spend-based EEIO calculation lifecycle.

    Automatically records the spend-based calculation metrics when the
    context exits.  The caller should populate ``context['emissions_tco2e']``
    and optionally ``context['spend_amount']`` before exiting.

    Args:
        database: EEIO database being used
        currency: Transaction currency

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_spend_based("useeio", "usd") as ctx:
        ...     result = eeio_calculator.calculate(spend_data)
        ...     ctx['emissions_tco2e'] = result.total_tco2e
        ...     ctx['spend_amount'] = result.total_spend
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'database': database,
        'currency': currency,
        'status': 'success',
        'emissions_tco2e': None,
        'spend_amount': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active()

    try:
        yield context

    except Exception as exc:
        context['status'] = 'failed'
        context['error'] = str(exc)
        logger.error(
            "Spend-based calculation failed: %s",
            exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active()

        metrics.record_spend_based(
            database=context['database'],
            currency=context['currency'],
            status=context['status'],
            duration_s=duration_s,
            spend_amount=context.get('spend_amount'),
            emissions_tco2e=context.get('emissions_tco2e')
        )


@contextmanager
def track_supplier_specific(
    source: str = "supplier_survey",
    verification: str = "unverified"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a supplier-specific calculation lifecycle.

    Args:
        source: Supplier data source
        verification: Data verification status

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_supplier_specific("epd", "third_party_verified") as ctx:
        ...     result = calculate_from_epd(epd_data)
        ...     ctx['emissions_tco2e'] = result.total_tco2e
        ...     ctx['category'] = "machinery"
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'source': source,
        'verification': verification,
        'status': 'success',
        'emissions_tco2e': None,
        'category': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active()

    try:
        yield context

    except Exception as exc:
        context['status'] = 'failed'
        context['error'] = str(exc)
        logger.error(
            "Supplier-specific calculation failed: %s",
            exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active()

        metrics.record_supplier_specific(
            source=context['source'],
            verification=context['verification'],
            status=context['status'],
            duration_s=duration_s,
            emissions_tco2e=context.get('emissions_tco2e'),
            category=context.get('category')
        )


@contextmanager
def track_average_data(
    source: str = "industry_average",
    category: str = "other"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks an average-data calculation lifecycle.

    Args:
        source: Emission factor source
        category: Asset category

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_average_data("ecoinvent", "vehicles") as ctx:
        ...     result = avg_calculator.calculate(asset_data)
        ...     ctx['emissions_tco2e'] = result.total_tco2e
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'source': source,
        'category': category,
        'status': 'success',
        'emissions_tco2e': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active()

    try:
        yield context

    except Exception as exc:
        context['status'] = 'failed'
        context['error'] = str(exc)
        logger.error(
            "Average-data calculation failed: %s",
            exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active()

        metrics.record_average_data(
            source=context['source'],
            category=context['category'],
            status=context['status'],
            duration_s=duration_s,
            emissions_tco2e=context.get('emissions_tco2e')
        )


@contextmanager
def track_hybrid(
    primary_method: str = "supplier_specific"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a hybrid calculation lifecycle.

    Args:
        primary_method: The primary method used in the hybrid approach

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_hybrid("supplier_specific") as ctx:
        ...     result = hybrid_engine.calculate(mixed_data)
        ...     ctx['emissions_tco2e'] = result.total_tco2e
        ...     ctx['methods_combined'] = 3
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'primary_method': primary_method,
        'status': 'success',
        'emissions_tco2e': None,
        'methods_combined': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active()

    try:
        yield context

    except Exception as exc:
        context['status'] = 'failed'
        context['error'] = str(exc)
        logger.error(
            "Hybrid calculation failed: %s",
            exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active()

        metrics.record_hybrid(
            primary_method=context['primary_method'],
            status=context['status'],
            duration_s=duration_s,
            emissions_tco2e=context.get('emissions_tco2e'),
            methods_combined=context.get('methods_combined')
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
    method: str = "spend_based"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a batch calculation lifecycle.

    Args:
        method: Calculation method used for the batch

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_batch("spend_based") as ctx:
        ...     results = batch_engine.process(items)
        ...     ctx['size'] = len(items)
        ...     ctx['successful'] = sum(1 for r in results if r.ok)
        ...     ctx['failed'] = sum(1 for r in results if not r.ok)
        ...     ctx['total_emissions_tco2e'] = sum(r.tco2e for r in results if r.ok)
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'method': method,
        'size': 0,
        'successful': None,
        'failed': None,
        'total_emissions_tco2e': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active()

    try:
        yield context

    except Exception as exc:
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
            metrics.record_batch(
                method=context['method'],
                size=size,
                successful=context.get('successful'),
                failed=context.get('failed'),
                duration_s=duration_s,
                total_emissions_tco2e=context.get('total_emissions_tco2e')
            )


# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    # Main class and accessor
    'CapitalGoodsMetrics',
    'get_metrics',

    # Context managers
    'track_calculation',
    'track_duration',
    'track_spend_based',
    'track_supplier_specific',
    'track_average_data',
    'track_hybrid',
    'track_compliance_check',
    'track_batch',

    # Enumerations
    'CalculationMethod',
    'CalculationStatus',
    'AssetCategory',
    'EmissionGas',
    'EEIODatabase',
    'Currency',
    'EFSource',
    'SupplierDataSource',
    'VerificationStatus',
    'AssetLookupSource',
    'Framework',
    'ComplianceStatus',
    'BatchStatus',
    'ErrorType',

    # Availability flag
    'PROMETHEUS_AVAILABLE',
]
