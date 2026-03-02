# -*- coding: utf-8 -*-
"""
Prometheus Metrics - Upstream Leased Assets Agent (AGENT-MRV-021)

Thread-safe singleton Prometheus metrics collection with graceful fallback
when prometheus_client is not installed. All metrics use the gl_ula_ prefix.

This module provides Prometheus metrics tracking for upstream leased assets
emissions calculations (Scope 3, Category 8) including asset-specific,
building, vehicle, equipment, and IT asset calculation methods across all
leased asset types and allocation methods.

Thread-safe singleton pattern with graceful fallback if Prometheus
unavailable.

Metrics prefix: gl_ula_

14 Prometheus Metrics:
    1.  gl_ula_calculations_total                - Counter: total calculations performed
    2.  gl_ula_calculation_duration_seconds       - Histogram: calculation latency
    3.  gl_ula_emissions_kg_co2e                 - Counter: total emissions in kgCO2e
    4.  gl_ula_batch_size                        - Histogram: batch sizes
    5.  gl_ula_batch_duration_seconds            - Histogram: batch processing time
    6.  gl_ula_ef_lookups_total                  - Counter: EF lookup count
    7.  gl_ula_ef_cache_hits_total               - Counter: cache hits
    8.  gl_ula_ef_cache_misses_total             - Counter: cache misses
    9.  gl_ula_compliance_checks_total           - Counter: compliance check count
    10. gl_ula_pipeline_stage_duration_seconds   - Histogram: per-stage timing
    11. gl_ula_data_quality_score                - Histogram: DQI scores
    12. gl_ula_allocation_factor                 - Histogram: allocation factor dist
    13. gl_ula_floor_area_sqm                    - Histogram: floor area distribution
    14. gl_ula_active_calculations               - Gauge: currently active calcs

GHG Protocol Scope 3 Category 8 covers upstream leased assets:
    A. Emissions from the operation of assets leased by the reporting
       company in the reporting year and not already included in the
       reporting company's Scope 1 or Scope 2 inventories.
    B. Applicable only when the reporting company is a lessee and the
       leased assets are not included in Scope 1 and 2 (operating
       lease approach).
    C. Asset types: buildings, vehicles, equipment, IT assets.
    D. Requires lease classification (operating vs finance) per
       IFRS 16 / ASC 842.
    E. Allocation based on floor area, headcount, or other basis.

Calculation methods defined by GHG Protocol:
    - Asset-specific: Actual energy/fuel data per leased asset
    - Average-data: Building EUI benchmarks x floor area x grid EF
    - Spend-based: Lease payments x EEIO emission factors
    - Lessor-specific: EFs provided by lessor/landlord

Example:
    >>> metrics = UpstreamLeasedAssetsMetrics()
    >>> metrics.record_calculation(
    ...     asset_type="building",
    ...     method="asset_specific",
    ...     status="success",
    ...     duration=0.035,
    ...     co2e=24500.0,
    ...     tenant_id="tenant_abc"
    ... )

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-021 Upstream Leased Assets (GL-MRV-S3-008)
Status: Production Ready
Version: 1.0.0
Agent: GL-MRV-S3-008
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

        This allows the Upstream Leased Assets Agent to operate correctly in
        environments where prometheus_client is not installed (e.g.,
        lightweight test environments, CLI tools, or edge deployments).

        Example:
            >>> metric = _NoOpMetric("test_metric", "A test metric")
            >>> metric.labels(asset_type="building").inc()  # Silent no-op
            >>> metric.observe(0.5)                         # Silent no-op
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
# Enumerations -- Upstream Leased Assets domain-specific label value sets
# ===========================================================================
# 9 bounded-cardinality enums for Prometheus label values.
# Each enum constrains the label values that can be written to a metric,
# preventing cardinality explosion in Prometheus TSDB.
# ===========================================================================


class AssetTypeLabelEnum(str, Enum):
    """
    Leased asset types tracked for upstream leased assets emissions.

    Covers the primary asset categories defined in the GHG Protocol
    Technical Guidance for Calculating Scope 3 Emissions, Category 8.

    Values:
        BUILDING: Leased commercial buildings (office, warehouse, retail)
        VEHICLE: Leased vehicles (fleet cars, vans, trucks)
        EQUIPMENT: Leased equipment (generators, compressors, forklifts)
        IT_ASSET: Leased IT assets (servers, storage, networking)
        LAND: Leased land parcels
        UNKNOWN: Unclassified or unreported asset type
    """
    BUILDING = "building"
    VEHICLE = "vehicle"
    EQUIPMENT = "equipment"
    IT_ASSET = "it_asset"
    LAND = "land"
    UNKNOWN = "unknown"


class BuildingTypeLabelEnum(str, Enum):
    """
    Building types for leased building emissions tracking.

    Values:
        OFFICE: Commercial office space
        WAREHOUSE: Storage and distribution facility
        RETAIL: Retail store or shopping centre
        DATA_CENTER: Server room or co-location facility
        MANUFACTURING: Light manufacturing facility
        MIXED_USE: Multi-purpose building
        LABORATORY: Research and laboratory facility
        HEALTHCARE: Clinic or medical office
        UNKNOWN: Unspecified building type
    """
    OFFICE = "office"
    WAREHOUSE = "warehouse"
    RETAIL = "retail"
    DATA_CENTER = "data_center"
    MANUFACTURING = "manufacturing"
    MIXED_USE = "mixed_use"
    LABORATORY = "laboratory"
    HEALTHCARE = "healthcare"
    UNKNOWN = "unknown"


class CalculationMethodLabelEnum(str, Enum):
    """
    Calculation methods for upstream leased assets emissions.

    GHG Protocol Scope 3 Category 8 supports several approaches
    depending on data availability and the level of accuracy required.

    Values:
        ASSET_SPECIFIC: Actual energy/fuel consumption data per asset
        AVERAGE_DATA: Building EUI benchmarks or industry averages
        SPEND_BASED: Lease payments x EEIO emission factors
        LESSOR_SPECIFIC: EFs provided by lessor/landlord
        UNKNOWN: Unspecified calculation method
    """
    ASSET_SPECIFIC = "asset_specific"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"
    LESSOR_SPECIFIC = "lessor_specific"
    UNKNOWN = "unknown"


class ComplianceFrameworkLabelEnum(str, Enum):
    """
    Compliance frameworks for upstream leased assets reporting.

    Values:
        GHG_PROTOCOL: GHG Protocol Corporate Value Chain (Scope 3) Standard
        CSRD_ESRS: EU CSRD / ESRS E1 Climate Change
        CDP: CDP Climate Change Questionnaire
        SBTI: Science Based Targets initiative (SBTi)
        SB253: California SB 253
        GRI_305: GRI 305: Emissions standard
        ISO_14064: ISO 14064-1 GHG inventories
        UNKNOWN: Unspecified compliance framework
    """
    GHG_PROTOCOL = "ghg_protocol"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    SB253 = "sb253"
    GRI_305 = "gri_305"
    ISO_14064 = "iso_14064"
    UNKNOWN = "unknown"


class DataQualityTierLabelEnum(str, Enum):
    """
    Data quality tiers for upstream leased assets calculation inputs.

    Values:
        TIER1: Primary data -- asset-specific metered energy/fuel data
        TIER2: Secondary data -- building benchmarks, fleet averages
        TIER3: Tertiary data -- EEIO, national averages, spend-based
        UNKNOWN: Unspecified data quality tier
    """
    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"
    UNKNOWN = "unknown"


class PipelineStageLabelEnum(str, Enum):
    """
    Pipeline stages for the Upstream Leased Assets calculation pipeline.

    Values:
        VALIDATE: Input data validation and schema conformance
        CLASSIFY: Lease classification (operating vs finance)
        NORMALIZE: Unit normalization (area, distance, currency)
        RESOLVE_EFS: Emission factor resolution and lookup
        CALCULATE: Asset-level emission calculation
        ALLOCATE: Allocation of shared asset emissions
        AGGREGATE: Result aggregation across assets
        COMPLIANCE: Regulatory compliance checks
        PROVENANCE: Provenance hash computation
        SEAL: Final sealing and verification
        UNKNOWN: Unspecified pipeline stage
    """
    VALIDATE = "validate"
    CLASSIFY = "classify"
    NORMALIZE = "normalize"
    RESOLVE_EFS = "resolve_efs"
    CALCULATE = "calculate"
    ALLOCATE = "allocate"
    AGGREGATE = "aggregate"
    COMPLIANCE = "compliance"
    PROVENANCE = "provenance"
    SEAL = "seal"
    UNKNOWN = "unknown"


class StatusLabelEnum(str, Enum):
    """
    General operation status labels for upstream leased assets metrics.

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


class LeaseTypeLabelEnum(str, Enum):
    """
    Lease classification labels for tracking lease type distribution.

    Values:
        OPERATING: Operating lease (reported in Scope 3 Cat 8)
        FINANCE: Finance lease (reported in Scope 1/2)
        SHORT_TERM: Short-term lease exemption (< 12 months)
        LOW_VALUE: Low-value asset exemption
        UNKNOWN: Unclassified lease
    """
    OPERATING = "operating"
    FINANCE = "finance"
    SHORT_TERM = "short_term"
    LOW_VALUE = "low_value"
    UNKNOWN = "unknown"


class TenantLabelEnum(str, Enum):
    """
    Tenant label placeholder for multi-tenant metrics isolation.

    Values:
        UNKNOWN: Unspecified or missing tenant identifier
    """
    UNKNOWN = "unknown"


# ===========================================================================
# UpstreamLeasedAssetsMetrics -- Thread-safe Singleton
# ===========================================================================


class UpstreamLeasedAssetsMetrics:
    """
    Thread-safe singleton metrics collector for Upstream Leased Assets Agent (MRV-021).

    Provides 14 Prometheus metrics for tracking Scope 3 Category 8
    upstream leased assets emissions calculations, including asset-specific,
    average-data, spend-based, and lessor-specific methods across all
    leased asset types and allocation methods.

    All metrics use the ``gl_ula_`` prefix to ensure namespace isolation
    within the GreenLang Prometheus ecosystem.

    Scope 3 Category 8 Sub-Categories Tracked:
        A. Building emissions (energy consumption, refrigerants)
        B. Vehicle fleet emissions (fuel combustion, WTT)
        C. Equipment emissions (fuel, electricity)
        D. IT asset emissions (electricity, PUE, embodied)
        E. Allocation tracking (floor area, headcount, FTE, revenue)

    Calculation Methods Supported:
        - Asset-specific: Actual metered energy/fuel data per leased asset
        - Average-data: Building EUI benchmarks x floor area x grid EF
        - Spend-based: Lease payments x EEIO emission factors
        - Lessor-specific: Emission factors provided by asset lessors

    Prometheus Metrics (14):
        1.  gl_ula_calculations_total               (Counter)
        2.  gl_ula_calculation_duration_seconds      (Histogram)
        3.  gl_ula_emissions_kg_co2e                (Counter)
        4.  gl_ula_batch_size                       (Histogram)
        5.  gl_ula_batch_duration_seconds           (Histogram)
        6.  gl_ula_ef_lookups_total                 (Counter)
        7.  gl_ula_ef_cache_hits_total              (Counter)
        8.  gl_ula_ef_cache_misses_total            (Counter)
        9.  gl_ula_compliance_checks_total          (Counter)
        10. gl_ula_pipeline_stage_duration_seconds   (Histogram)
        11. gl_ula_data_quality_score               (Histogram)
        12. gl_ula_allocation_factor                (Histogram)
        13. gl_ula_floor_area_sqm                   (Histogram)
        14. gl_ula_active_calculations              (Gauge)

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
        allocation_factor: Histogram for allocation factor distribution
        floor_area_sqm: Histogram for floor area distribution
        active_calculations: Gauge for currently active calculations

    Example:
        >>> metrics = UpstreamLeasedAssetsMetrics()
        >>> metrics.record_calculation(
        ...     asset_type="building",
        ...     method="asset_specific",
        ...     status="success",
        ...     duration=0.035,
        ...     co2e=24500.0,
        ...     tenant_id="tenant_abc"
        ... )
        >>> summary = metrics.get_metrics_summary()
        >>> assert summary['calculations'] >= 1
    """

    _instance: Optional["UpstreamLeasedAssetsMetrics"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "UpstreamLeasedAssetsMetrics":
        """
        Thread-safe singleton instantiation using double-checked locking.

        Returns:
            The singleton UpstreamLeasedAssetsMetrics instance.
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
        Prometheus metrics with the gl_ula_ prefix, and logs the
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
            "allocation_records": 0,
            "floor_area_records": 0,
            "errors": 0,
        }

        # Initialize all 14 Prometheus metrics
        self._init_metrics()

        logger.info(
            "UpstreamLeasedAssetsMetrics initialized with 14 metrics "
            "(Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable",
        )

    # ======================================================================
    # Metric initialization
    # ======================================================================

    def _init_metrics(self) -> None:
        """
        Initialize all 14 Prometheus metrics with gl_ula_ prefix.

        Each metric is documented with its purpose, label set, and the
        Scope 3 Category 8 aspect it supports.
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
        # 1. gl_ula_calculations_total (Counter)
        # ------------------------------------------------------------------
        self.calculations_total = _safe_create(
            Counter,
            "gl_ula_calculations_total",
            "Total upstream leased assets emission calculations performed",
            ["asset_type", "method", "status", "tenant_id"],
        )

        # ------------------------------------------------------------------
        # 2. gl_ula_calculation_duration_seconds (Histogram)
        # ------------------------------------------------------------------
        self.calculation_duration_seconds = _safe_create(
            Histogram,
            "gl_ula_calculation_duration_seconds",
            "Duration of upstream leased assets calculation operations in seconds",
            ["asset_type", "method"],
            buckets=(
                0.005, 0.01, 0.025, 0.05, 0.1,
                0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ),
        )

        # ------------------------------------------------------------------
        # 3. gl_ula_emissions_kg_co2e (Counter)
        # ------------------------------------------------------------------
        self.emissions_kg_co2e = _safe_create(
            Counter,
            "gl_ula_emissions_kg_co2e",
            "Total upstream leased assets emissions calculated in kgCO2e",
            ["asset_type", "method", "tenant_id"],
        )

        # ------------------------------------------------------------------
        # 4. gl_ula_batch_size (Histogram)
        # ------------------------------------------------------------------
        self.batch_size = _safe_create(
            Histogram,
            "gl_ula_batch_size",
            "Batch calculation size for upstream leased assets operations",
            ["method"],
            buckets=(
                1, 5, 10, 25, 50, 100, 250, 500,
                1000, 2500, 5000, 10000,
            ),
        )

        # ------------------------------------------------------------------
        # 5. gl_ula_batch_duration_seconds (Histogram)
        # ------------------------------------------------------------------
        self.batch_duration_seconds = _safe_create(
            Histogram,
            "gl_ula_batch_duration_seconds",
            "Duration of batch processing for upstream leased assets operations",
            ["method"],
            buckets=(
                0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
                10.0, 30.0, 60.0, 120.0, 300.0,
            ),
        )

        # ------------------------------------------------------------------
        # 6. gl_ula_ef_lookups_total (Counter)
        # ------------------------------------------------------------------
        self.ef_lookups_total = _safe_create(
            Counter,
            "gl_ula_ef_lookups_total",
            "Total emission factor lookups for upstream leased assets",
            ["asset_type", "source", "status"],
        )

        # ------------------------------------------------------------------
        # 7. gl_ula_ef_cache_hits_total (Counter)
        # ------------------------------------------------------------------
        self.ef_cache_hits_total = _safe_create(
            Counter,
            "gl_ula_ef_cache_hits_total",
            "Total emission factor cache hits for upstream leased assets",
            ["asset_type"],
        )

        # ------------------------------------------------------------------
        # 8. gl_ula_ef_cache_misses_total (Counter)
        # ------------------------------------------------------------------
        self.ef_cache_misses_total = _safe_create(
            Counter,
            "gl_ula_ef_cache_misses_total",
            "Total emission factor cache misses for upstream leased assets",
            ["asset_type"],
        )

        # ------------------------------------------------------------------
        # 9. gl_ula_compliance_checks_total (Counter)
        # ------------------------------------------------------------------
        self.compliance_checks_total = _safe_create(
            Counter,
            "gl_ula_compliance_checks_total",
            "Total upstream leased assets compliance checks performed",
            ["framework", "status"],
        )

        # ------------------------------------------------------------------
        # 10. gl_ula_pipeline_stage_duration_seconds (Histogram)
        # ------------------------------------------------------------------
        self.pipeline_stage_duration_seconds = _safe_create(
            Histogram,
            "gl_ula_pipeline_stage_duration_seconds",
            "Duration of pipeline stages for upstream leased assets",
            ["stage"],
            buckets=(
                0.001, 0.005, 0.01, 0.025, 0.05,
                0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
            ),
        )

        # ------------------------------------------------------------------
        # 11. gl_ula_data_quality_score (Histogram)
        # ------------------------------------------------------------------
        self.data_quality_score = _safe_create(
            Histogram,
            "gl_ula_data_quality_score",
            "Data quality indicator scores for upstream leased assets inputs",
            ["method", "tier"],
            buckets=(
                0.0, 0.1, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            ),
        )

        # ------------------------------------------------------------------
        # 12. gl_ula_allocation_factor (Histogram)
        # ------------------------------------------------------------------
        self.allocation_factor = _safe_create(
            Histogram,
            "gl_ula_allocation_factor",
            "Allocation factor distribution for upstream leased assets",
            ["asset_type", "method"],
            buckets=(
                0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            ),
        )

        # ------------------------------------------------------------------
        # 13. gl_ula_floor_area_sqm (Histogram)
        # ------------------------------------------------------------------
        self.floor_area_sqm = _safe_create(
            Histogram,
            "gl_ula_floor_area_sqm",
            "Floor area distribution for leased buildings in square metres",
            ["building_type"],
            buckets=(
                50, 100, 250, 500, 1000, 2500, 5000,
                10000, 25000, 50000, 100000,
            ),
        )

        # ------------------------------------------------------------------
        # 14. gl_ula_active_calculations (Gauge)
        # ------------------------------------------------------------------
        self.active_calculations = _safe_create(
            Gauge,
            "gl_ula_active_calculations",
            "Number of currently active upstream leased assets calculations",
            ["method"],
        )

        # ------------------------------------------------------------------
        # Agent info metric (non-numeric metadata)
        # ------------------------------------------------------------------
        self.agent_info = _safe_create(
            Info,
            "gl_ula_agent",
            "Upstream Leased Assets Agent metadata",
        )
        if PROMETHEUS_AVAILABLE:
            try:
                self.agent_info.info({
                    "agent_id": "GL-MRV-S3-008",
                    "version": "1.0.0",
                    "scope": "scope_3_category_8",
                    "description": "Upstream Leased Assets emissions calculator",
                    "metrics_count": "14",
                    "prefix": "gl_ula_",
                })
            except Exception:
                pass

    # ======================================================================
    # Convenience recording methods
    # ======================================================================

    def record_calculation(
        self,
        asset_type: str,
        method: str,
        status: str,
        duration: float,
        co2e: float,
        tenant_id: str = "unknown",
    ) -> None:
        """
        Record an upstream leased assets emission calculation operation.

        This is the primary recording method that increments the calculations
        counter, observes the duration histogram, and tracks emissions output.

        Args:
            asset_type: Leased asset type (building/vehicle/equipment/it_asset/land/unknown)
            method: Calculation method (asset_specific/average_data/spend_based/
                     lessor_specific/unknown)
            status: Calculation status (success/failure/error/timeout/unknown)
            duration: Operation duration in seconds
            co2e: Emissions calculated in kgCO2e
            tenant_id: Tenant identifier (default: "unknown")

        Example:
            >>> metrics.record_calculation(
            ...     asset_type="building",
            ...     method="asset_specific",
            ...     status="success",
            ...     duration=0.035,
            ...     co2e=24500.0,
            ...     tenant_id="tenant_abc"
            ... )
        """
        try:
            asset_type = self._validate_enum_value(
                asset_type, AssetTypeLabelEnum,
                AssetTypeLabelEnum.UNKNOWN.value,
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
                asset_type=asset_type,
                method=method,
                status=status,
                tenant_id=tenant_id,
            ).inc()

            # 2. Observe duration
            if duration is not None and duration > 0:
                self.calculation_duration_seconds.labels(
                    asset_type=asset_type,
                    method=method,
                ).observe(duration)

            # 3. Record emissions
            if co2e is not None and co2e > 0:
                self.emissions_kg_co2e.labels(
                    asset_type=asset_type,
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
                "Recorded calculation: asset_type=%s, method=%s, status=%s, "
                "duration=%.3fs, co2e=%.2f kgCO2e, tenant=%s",
                asset_type, method, status,
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

        Args:
            method: Calculation method used for the batch
            size: Number of asset records in the batch
            duration: Batch processing duration in seconds

        Example:
            >>> metrics.record_batch(
            ...     method="average_data",
            ...     size=500,
            ...     duration=8.5
            ... )
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethodLabelEnum,
                CalculationMethodLabelEnum.UNKNOWN.value,
            )

            if size is not None and size > 0:
                self.batch_size.labels(method=method).observe(size)

            if duration is not None and duration > 0:
                self.batch_duration_seconds.labels(method=method).observe(duration)

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
        asset_type: str,
        source: str,
        status: str,
    ) -> None:
        """
        Record an emission factor lookup operation.

        Args:
            asset_type: Leased asset type for which the EF was looked up
            source: EF database source identifier
            status: Lookup status (success/failure/error/timeout/unknown)

        Example:
            >>> metrics.record_ef_lookup(
            ...     asset_type="building",
            ...     source="defra",
            ...     status="success"
            ... )
        """
        try:
            asset_type = self._validate_enum_value(
                asset_type, AssetTypeLabelEnum,
                AssetTypeLabelEnum.UNKNOWN.value,
            )
            status = self._validate_enum_value(
                status, StatusLabelEnum,
                StatusLabelEnum.ERROR.value,
            )
            source = self._sanitize_label(source, max_len=32, default="unknown")

            self.ef_lookups_total.labels(
                asset_type=asset_type,
                source=source,
                status=status,
            ).inc()

            with self._stats_lock:
                self._in_memory_stats["ef_lookups"] += 1

            logger.debug(
                "Recorded EF lookup: asset_type=%s, source=%s, status=%s",
                asset_type, source, status,
            )

        except Exception as e:
            logger.error(
                "Failed to record EF lookup metrics: %s", e, exc_info=True
            )

    def record_cache_hit(self, asset_type: str) -> None:
        """
        Record an emission factor cache hit.

        Args:
            asset_type: Leased asset type for the cached EF

        Example:
            >>> metrics.record_cache_hit("building")
        """
        try:
            asset_type = self._validate_enum_value(
                asset_type, AssetTypeLabelEnum,
                AssetTypeLabelEnum.UNKNOWN.value,
            )

            self.ef_cache_hits_total.labels(asset_type=asset_type).inc()

            with self._stats_lock:
                self._in_memory_stats["cache_hits"] += 1

            logger.debug("Recorded cache hit: asset_type=%s", asset_type)

        except Exception as e:
            logger.error(
                "Failed to record cache hit metrics: %s", e, exc_info=True
            )

    def record_cache_miss(self, asset_type: str) -> None:
        """
        Record an emission factor cache miss.

        Args:
            asset_type: Leased asset type for the missed EF lookup

        Example:
            >>> metrics.record_cache_miss("vehicle")
        """
        try:
            asset_type = self._validate_enum_value(
                asset_type, AssetTypeLabelEnum,
                AssetTypeLabelEnum.UNKNOWN.value,
            )

            self.ef_cache_misses_total.labels(asset_type=asset_type).inc()

            with self._stats_lock:
                self._in_memory_stats["cache_misses"] += 1

            logger.debug("Recorded cache miss: asset_type=%s", asset_type)

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

        Args:
            framework: Compliance framework (ghg_protocol/csrd_esrs/cdp/sbti/
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

        Args:
            stage: Pipeline stage name
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

        Args:
            method: Calculation method
            tier: Data quality tier (tier1/tier2/tier3/unknown)
            score: DQI score (0.0 = worst, 1.0 = best)

        Example:
            >>> metrics.record_data_quality(
            ...     method="asset_specific",
            ...     tier="tier1",
            ...     score=0.92
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

    def record_allocation_factor(
        self,
        asset_type: str,
        method: str,
        factor: float,
    ) -> None:
        """
        Record an allocation factor observation.

        Args:
            asset_type: Leased asset type
            method: Allocation method (floor_area/headcount/fte/revenue/custom)
            factor: Allocation factor (0.0 to 1.0)

        Example:
            >>> metrics.record_allocation_factor(
            ...     asset_type="building",
            ...     method="floor_area",
            ...     factor=0.35
            ... )
        """
        try:
            asset_type = self._validate_enum_value(
                asset_type, AssetTypeLabelEnum,
                AssetTypeLabelEnum.UNKNOWN.value,
            )
            method = self._sanitize_label(method, max_len=32, default="unknown")

            if factor is not None:
                factor = max(0.0, min(1.0, float(factor)))
                self.allocation_factor.labels(
                    asset_type=asset_type,
                    method=method,
                ).observe(factor)

            with self._stats_lock:
                self._in_memory_stats["allocation_records"] += 1

            logger.debug(
                "Recorded allocation factor: asset_type=%s, method=%s, factor=%.4f",
                asset_type, method,
                factor if factor is not None else 0.0,
            )

        except Exception as e:
            logger.error(
                "Failed to record allocation factor metrics: %s",
                e, exc_info=True,
            )

    def record_floor_area(
        self,
        building_type: str,
        area_sqm: float,
    ) -> None:
        """
        Record a floor area observation for a leased building.

        Args:
            building_type: Building type (office/warehouse/retail/data_center/etc.)
            area_sqm: Floor area in square metres

        Example:
            >>> metrics.record_floor_area(
            ...     building_type="office",
            ...     area_sqm=2500.0
            ... )
        """
        try:
            building_type = self._validate_enum_value(
                building_type, BuildingTypeLabelEnum,
                BuildingTypeLabelEnum.UNKNOWN.value,
            )

            if area_sqm is not None and area_sqm > 0:
                self.floor_area_sqm.labels(
                    building_type=building_type
                ).observe(area_sqm)

            with self._stats_lock:
                self._in_memory_stats["floor_area_records"] += 1

            logger.debug(
                "Recorded floor area: building_type=%s, area_sqm=%.1f",
                building_type,
                area_sqm if area_sqm is not None else 0.0,
            )

        except Exception as e:
            logger.error(
                "Failed to record floor area metrics: %s", e, exc_info=True
            )

    def start_calculation(self, method: str) -> None:
        """
        Increment the active calculations gauge when a calculation starts.

        Args:
            method: Calculation method being started

        Example:
            >>> metrics.start_calculation("asset_specific")
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

        Args:
            method: Calculation method that completed

        Example:
            >>> metrics.end_calculation("asset_specific")
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

        Returns:
            Dictionary with metrics summary including counts, uptime, rates,
            cache performance, and operational breakdown.

        Example:
            >>> summary = metrics.get_metrics_summary()
            >>> print(summary['calculations'])
            142
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
                "agent": "upstream_leased_assets",
                "agent_id": "GL-MRV-S3-008",
                "prefix": "gl_ula_",
                "scope": "Scope 3 Category 8",
                "description": "Upstream Leased Assets",
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
                    "allocation_records": stats_snapshot["allocation_records"],
                    "floor_area_records": stats_snapshot["floor_area_records"],
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
                "agent": "upstream_leased_assets",
            }

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance for testing purposes.

        Example:
            >>> UpstreamLeasedAssetsMetrics.reset()
            >>> metrics = UpstreamLeasedAssetsMetrics()  # Fresh instance
        """
        with cls._lock:
            if cls._instance is not None:
                if hasattr(cls._instance, "_initialized"):
                    del cls._instance._initialized
                cls._instance = None

                global _metrics_instance
                _metrics_instance = None

                logger.info("UpstreamLeasedAssetsMetrics singleton reset")

    def reset_stats(self) -> None:
        """
        Reset in-memory statistics (not Prometheus metrics).

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
                    "allocation_records": 0,
                    "floor_area_records": 0,
                    "errors": 0,
                }
            self._start_time = datetime.utcnow()

            logger.info(
                "Reset in-memory statistics for UpstreamLeasedAssetsMetrics"
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

        Args:
            tenant_id: Raw tenant identifier
            max_len: Maximum label length (default: 32)

        Returns:
            Sanitized tenant ID string
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

        Args:
            value: Raw label value
            max_len: Maximum label length (default: 32)
            default: Default value for None/empty inputs (default: "unknown")

        Returns:
            Sanitized label value string
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

_metrics_instance: Optional[UpstreamLeasedAssetsMetrics] = None
_metrics_lock: threading.Lock = threading.Lock()


def get_metrics() -> UpstreamLeasedAssetsMetrics:
    """
    Get the singleton UpstreamLeasedAssetsMetrics instance.

    Thread-safe accessor for the global metrics instance.

    Returns:
        UpstreamLeasedAssetsMetrics singleton instance

    Example:
        >>> from greenlang.upstream_leased_assets.metrics import get_metrics
        >>> metrics = get_metrics()
        >>> metrics.record_calculation(
        ...     asset_type="building",
        ...     method="asset_specific",
        ...     status="success",
        ...     duration=0.035,
        ...     co2e=24500.0,
        ...     tenant_id="tenant_abc"
        ... )
    """
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = UpstreamLeasedAssetsMetrics()

    return _metrics_instance


def reset_metrics() -> None:
    """
    Reset the singleton metrics instance for testing purposes.

    Example:
        >>> from greenlang.upstream_leased_assets.metrics import reset_metrics
        >>> reset_metrics()
    """
    UpstreamLeasedAssetsMetrics.reset()


# ===========================================================================
# Context manager helpers
# ===========================================================================


@contextmanager
def track_calculation(
    method: str = "asset_specific",
    asset_type: str = "building",
    tenant_id: str = "unknown",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a calculation's lifecycle.

    Automatically increments/decrements the active calculation gauge,
    measures wall-clock duration, and records the calculation when the
    context exits.

    Args:
        method: Calculation method (default: "asset_specific")
        asset_type: Leased asset type (default: "building")
        tenant_id: Tenant identifier (default: "unknown")

    Yields:
        Mutable context dict. Set ``context['co2e']`` to record emissions.

    Example:
        >>> with track_calculation(method="asset_specific", asset_type="building") as ctx:
        ...     result = calculate_building_emissions(asset)
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
            asset_type=asset_type,
            method=method,
            status=context["status"],
            duration=duration,
            co2e=context.get("co2e", 0.0),
            tenant_id=tenant_id,
        )


@contextmanager
def track_batch(
    method: str = "asset_specific",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a batch job's lifecycle.

    Args:
        method: Calculation method used for the batch (default: "asset_specific")

    Yields:
        Mutable context dict. Set ``context['size']`` to record batch size.

    Example:
        >>> with track_batch(method="average_data") as ctx:
        ...     results = process_leased_assets_batch(assets)
        ...     ctx['size'] = len(assets)
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

    Args:
        stage: Pipeline stage name

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
    # Label Enums (9)
    "AssetTypeLabelEnum",
    "BuildingTypeLabelEnum",
    "CalculationMethodLabelEnum",
    "ComplianceFrameworkLabelEnum",
    "DataQualityTierLabelEnum",
    "PipelineStageLabelEnum",
    "StatusLabelEnum",
    "LeaseTypeLabelEnum",
    "TenantLabelEnum",
    # Singleton class
    "UpstreamLeasedAssetsMetrics",
    # Module-level accessors
    "get_metrics",
    "reset_metrics",
    # Context managers
    "track_calculation",
    "track_batch",
    "track_pipeline_stage",
]
