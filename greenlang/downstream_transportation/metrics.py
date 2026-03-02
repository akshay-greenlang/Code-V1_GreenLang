# -*- coding: utf-8 -*-
"""
Downstream Transportation & Distribution Prometheus Metrics - AGENT-MRV-022

14 Prometheus metrics with gl_dto_ prefix for monitoring the
GL-MRV-S3-009 Downstream Transportation & Distribution Agent.

This module provides Prometheus metrics tracking for downstream transportation
and distribution emissions calculations (Scope 3, Category 9) including
distance-based, spend-based, average-data, and supplier-specific methods
across all transport modes (road, rail, maritime, air, pipeline, intermodal),
warehouse/distribution centre operations, last-mile delivery, cold chain,
and return logistics.

Thread-safe singleton pattern with graceful fallback if Prometheus
unavailable.

Metrics prefix: gl_dto_

14 Prometheus Metrics:
    1.  gl_dto_calculations_total              - Counter: total calculations
    2.  gl_dto_calculation_duration_seconds     - Histogram: calculation duration
    3.  gl_dto_emissions_kg_total              - Counter: total emissions kgCO2e
    4.  gl_dto_shipments_processed             - Counter: shipments by mode/regime
    5.  gl_dto_warehouse_emissions_total       - Counter: warehouse emissions
    6.  gl_dto_last_mile_deliveries            - Counter: last-mile deliveries
    7.  gl_dto_tonne_km_total                  - Counter: total tonne-km
    8.  gl_dto_cold_chain_uplift_total         - Counter: cold chain uplifts
    9.  gl_dto_return_emissions_total          - Counter: return logistics emissions
    10. gl_dto_compliance_checks               - Counter: compliance checks
    11. gl_dto_provenance_chains               - Counter: provenance chains created
    12. gl_dto_errors_total                    - Counter: errors by engine/type
    13. gl_dto_eeio_spend_usd                  - Counter: EEIO spend processed
    14. gl_dto_allocation_count                - Counter: allocation operations

GHG Protocol Scope 3 Category 9 covers downstream transportation:
    A. Transportation of sold products between reporting company operations
       and the end consumer (not paid for by reporting company).
    B. Distribution centre and warehouse operations.
    C. Third-party retail storage energy consumption.
    D. Last-mile delivery to end consumer.
    E. Excludes upstream transportation (Category 4) based on Incoterm split.

Calculation methods defined by GHG Protocol:
    - Distance-based: tonne-km x mode-specific EF (with WTT option)
    - Spend-based: spend x EEIO EF (with CPI deflation, margin removal)
    - Average-data: industry average by distribution channel
    - Supplier-specific: primary data from carriers/3PLs

Example:
    >>> metrics = DownstreamTransportMetrics()
    >>> metrics.record_calculation(
    ...     method="distance_based",
    ...     mode="road",
    ...     status="success",
    ...     duration=0.042,
    ...     co2e=1250.5,
    ... )

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-009
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
# Enumerations -- Downstream Transportation domain label value sets
# ===========================================================================


class CalculationMethodLabel(str, Enum):
    """
    Calculation methods for downstream transportation emissions.

    GHG Protocol Scope 3 Category 9 supports several approaches:
        - Distance-based: tonne-km x mode-specific EF
        - Spend-based: spend x EEIO EF
        - Average-data: industry channel averages
        - Supplier-specific: primary data from carriers
    """
    DISTANCE_BASED = "distance_based"
    SPEND_BASED = "spend_based"
    AVERAGE_DATA = "average_data"
    SUPPLIER_SPECIFIC = "supplier_specific"


class TransportModeLabel(str, Enum):
    """
    Transport modes for downstream distribution emissions.

    Covers primary freight transport modes per GHG Protocol Technical
    Guidance for Scope 3, Category 9.
    """
    ROAD = "road"
    RAIL = "rail"
    MARITIME = "maritime"
    AIR = "air"
    PIPELINE = "pipeline"
    INTERMODAL = "intermodal"


class CalculationStatusLabel(str, Enum):
    """Calculation operation status."""
    SUCCESS = "success"
    ERROR = "error"
    VALIDATION_ERROR = "validation_error"


class TemperatureRegimeLabel(str, Enum):
    """Temperature regime for cold chain tracking."""
    AMBIENT = "ambient"
    CHILLED = "chilled"
    FROZEN = "frozen"
    DEEP_FROZEN = "deep_frozen"
    PHARMA = "pharma"


class WarehouseTypeLabel(str, Enum):
    """Warehouse types for distribution emissions."""
    AMBIENT = "ambient"
    COLD_STORAGE = "cold_storage"
    FROZEN = "frozen"
    RETAIL = "retail"
    CROSS_DOCK = "cross_dock"


class LastMileTypeLabel(str, Enum):
    """Last-mile delivery vehicle types."""
    VAN_DIESEL = "van_diesel"
    VAN_ELECTRIC = "van_electric"
    CARGO_BIKE = "cargo_bike"
    DRONE = "drone"
    MOTORCYCLE = "motorcycle"
    CAR_PETROL = "car_petrol"
    CAR_ELECTRIC = "car_electric"
    AVERAGE = "average"


class DeliveryAreaLabel(str, Enum):
    """Delivery area classifications."""
    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"
    REMOTE = "remote"


class EmissionComponentLabel(str, Enum):
    """Emission components tracked separately."""
    TRANSPORT = "transport"
    WTT = "wtt"
    WAREHOUSE = "warehouse"
    LAST_MILE = "last_mile"
    COLD_CHAIN = "cold_chain"
    RETURN = "return"
    HUB = "hub"


class FrameworkLabel(str, Enum):
    """Compliance frameworks for downstream transport reporting."""
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD = "csrd"
    CDP = "cdp"
    SBTI = "sbti"
    GRI = "gri"
    GLEC = "glec"


class ComplianceStatusLabel(str, Enum):
    """Compliance check result status."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class ReturnTypeLabel(str, Enum):
    """Return logistics emission types."""
    CONSUMER_RETURN = "consumer_return"
    CONSOLIDATION = "consolidation"
    REPACKAGING = "repackaging"


class EngineLabel(str, Enum):
    """Engine identifiers for error tracking."""
    DATABASE = "database"
    DISTANCE_CALC = "distance_calc"
    SPEND_CALC = "spend_calc"
    AVERAGE_DATA = "average_data"
    WAREHOUSE = "warehouse"
    COMPLIANCE = "compliance"
    PIPELINE = "pipeline"


class ErrorTypeLabel(str, Enum):
    """Error type classifications."""
    VALIDATION = "validation"
    CALCULATION = "calculation"
    DATABASE = "database"
    INTEGRATION = "integration"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class AllocationMethodLabel(str, Enum):
    """Allocation methods for shared emissions."""
    MASS = "mass"
    VOLUME = "volume"
    REVENUE = "revenue"
    UNIT_COUNT = "unit_count"
    TONNE_KM = "tonne_km"


# ===========================================================================
# DownstreamTransportMetrics -- Thread-safe Singleton
# ===========================================================================


class DownstreamTransportMetrics:
    """
    Thread-safe singleton metrics collector for Downstream Transportation
    Agent (MRV-022).

    Provides 14 Prometheus metrics for tracking Scope 3 Category 9
    downstream transportation and distribution emissions calculations.

    All metrics use the ``gl_dto_`` prefix for namespace isolation within
    the GreenLang Prometheus ecosystem.

    Attributes:
        calculations_total: Counter for calculation operations
        calculation_duration_seconds: Histogram for calculation latency
        emissions_kg_total: Counter for emissions in kgCO2e
        shipments_processed: Counter for shipments processed
        warehouse_emissions_total: Counter for warehouse emissions
        last_mile_deliveries: Counter for last-mile deliveries
        tonne_km_total: Counter for tonne-km
        cold_chain_uplift_total: Counter for cold chain uplifts
        return_emissions_total: Counter for return logistics
        compliance_checks: Counter for compliance checks
        provenance_chains: Counter for provenance chains
        errors_total: Counter for errors
        eeio_spend_usd: Counter for EEIO spend
        allocation_count: Counter for allocation operations

    Example:
        >>> metrics = DownstreamTransportMetrics()
        >>> metrics.record_calculation(
        ...     method="distance_based",
        ...     mode="road",
        ...     status="success",
        ...     duration=0.042,
        ...     co2e=1250.5,
        ... )
    """

    _instance: Optional["DownstreamTransportMetrics"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "DownstreamTransportMetrics":
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
            "shipments": 0,
            "warehouse_emissions": 0.0,
            "last_mile_deliveries": 0,
            "tonne_km": 0.0,
            "cold_chain_uplifts": 0,
            "return_emissions": 0.0,
            "compliance_checks": 0,
            "provenance_chains": 0,
            "errors": 0,
            "eeio_spend_usd": 0.0,
            "allocations": 0,
        }

        self._init_metrics()

        logger.info(
            "DownstreamTransportMetrics initialized (Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable",
        )

    def _init_metrics(self) -> None:
        """
        Initialize all 14 Prometheus metrics with gl_dto_ prefix.

        When Prometheus is available, metric registration may fail if the
        metrics were previously registered (e.g., after reset in tests).
        We unregister and re-register to obtain fresh collector objects.
        """
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import REGISTRY

            def _safe_create(metric_cls: type, name: str, *args: Any, **kwargs: Any) -> Any:
                """Create a metric, unregistering prior collector on conflict."""
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
                """No-op stub creation."""
                return metric_cls(name, *args, **kwargs)

        # ------------------------------------------------------------------
        # 1. gl_dto_calculations_total (Counter)
        #    Total downstream transportation emission calculations.
        #    Labels: method, mode, status
        # ------------------------------------------------------------------
        self.calculations_total = _safe_create(
            Counter,
            "gl_dto_calculations_total",
            "Total downstream transportation emission calculations",
            ["method", "mode", "status"],
        )

        # ------------------------------------------------------------------
        # 2. gl_dto_calculation_duration_seconds (Histogram)
        #    Duration of calculation operations in seconds.
        #    Labels: method, mode
        #    Buckets tuned for freight emissions calculation latencies.
        # ------------------------------------------------------------------
        self.calculation_duration_seconds = _safe_create(
            Histogram,
            "gl_dto_calculation_duration_seconds",
            "Duration of downstream transport calculation operations",
            ["method", "mode"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        # ------------------------------------------------------------------
        # 3. gl_dto_emissions_kg_total (Counter)
        #    Total emissions calculated in kgCO2e.
        #    Labels: mode, component
        # ------------------------------------------------------------------
        self.emissions_kg_total = _safe_create(
            Counter,
            "gl_dto_emissions_kg_total",
            "Total downstream transport emissions in kgCO2e",
            ["mode", "component"],
        )

        # ------------------------------------------------------------------
        # 4. gl_dto_shipments_processed (Counter)
        #    Total shipments processed.
        #    Labels: mode, temperature_regime
        # ------------------------------------------------------------------
        self.shipments_processed = _safe_create(
            Counter,
            "gl_dto_shipments_processed",
            "Total downstream shipments processed",
            ["mode", "temperature_regime"],
        )

        # ------------------------------------------------------------------
        # 5. gl_dto_warehouse_emissions_total (Counter)
        #    Total warehouse/DC emissions in kgCO2e.
        #    Labels: warehouse_type
        # ------------------------------------------------------------------
        self.warehouse_emissions_total = _safe_create(
            Counter,
            "gl_dto_warehouse_emissions_total",
            "Total warehouse and distribution centre emissions kgCO2e",
            ["warehouse_type"],
        )

        # ------------------------------------------------------------------
        # 6. gl_dto_last_mile_deliveries (Counter)
        #    Total last-mile deliveries processed.
        #    Labels: delivery_type, area
        # ------------------------------------------------------------------
        self.last_mile_deliveries = _safe_create(
            Counter,
            "gl_dto_last_mile_deliveries",
            "Total last-mile deliveries processed",
            ["delivery_type", "area"],
        )

        # ------------------------------------------------------------------
        # 7. gl_dto_tonne_km_total (Counter)
        #    Total tonne-kilometres tracked.
        #    Labels: mode
        # ------------------------------------------------------------------
        self.tonne_km_total = _safe_create(
            Counter,
            "gl_dto_tonne_km_total",
            "Total tonne-km tracked for downstream transport",
            ["mode"],
        )

        # ------------------------------------------------------------------
        # 8. gl_dto_cold_chain_uplift_total (Counter)
        #    Total cold chain uplift applications.
        #    No labels -- simple event counter.
        # ------------------------------------------------------------------
        self.cold_chain_uplift_total = _safe_create(
            Counter,
            "gl_dto_cold_chain_uplift_total",
            "Total cold chain emission uplifts applied",
        )

        # ------------------------------------------------------------------
        # 9. gl_dto_return_emissions_total (Counter)
        #    Total return logistics emissions in kgCO2e.
        #    Labels: return_type
        # ------------------------------------------------------------------
        self.return_emissions_total = _safe_create(
            Counter,
            "gl_dto_return_emissions_total",
            "Total return logistics emissions kgCO2e",
            ["return_type"],
        )

        # ------------------------------------------------------------------
        # 10. gl_dto_compliance_checks (Counter)
        #     Total compliance checks performed.
        #     Labels: framework, status
        # ------------------------------------------------------------------
        self.compliance_checks = _safe_create(
            Counter,
            "gl_dto_compliance_checks",
            "Total downstream transport compliance checks",
            ["framework", "status"],
        )

        # ------------------------------------------------------------------
        # 11. gl_dto_provenance_chains (Counter)
        #     Total provenance chains created.
        #     No labels -- simple event counter.
        # ------------------------------------------------------------------
        self.provenance_chains = _safe_create(
            Counter,
            "gl_dto_provenance_chains",
            "Total provenance chains created",
        )

        # ------------------------------------------------------------------
        # 12. gl_dto_errors_total (Counter)
        #     Total errors by engine and error type.
        #     Labels: engine, error_type
        # ------------------------------------------------------------------
        self.errors_total = _safe_create(
            Counter,
            "gl_dto_errors_total",
            "Total downstream transport errors",
            ["engine", "error_type"],
        )

        # ------------------------------------------------------------------
        # 13. gl_dto_eeio_spend_usd (Counter)
        #     Total EEIO spend processed in USD.
        #     Labels: naics_code
        # ------------------------------------------------------------------
        self.eeio_spend_usd = _safe_create(
            Counter,
            "gl_dto_eeio_spend_usd",
            "Total EEIO spend processed in USD",
            ["naics_code"],
        )

        # ------------------------------------------------------------------
        # 14. gl_dto_allocation_count (Counter)
        #     Total allocation operations performed.
        #     Labels: method
        # ------------------------------------------------------------------
        self.allocation_count = _safe_create(
            Counter,
            "gl_dto_allocation_count",
            "Total allocation operations performed",
            ["method"],
        )

        # ------------------------------------------------------------------
        # Active calculations gauge (supplementary)
        # ------------------------------------------------------------------
        self.active_calculations = _safe_create(
            Gauge,
            "gl_dto_active_calculations",
            "Number of currently active downstream transport calculations",
        )

        # Agent info metric
        self.agent_info = _safe_create(
            Info,
            "gl_dto_agent",
            "Downstream Transport Agent metadata",
        )
        if PROMETHEUS_AVAILABLE:
            try:
                self.agent_info.info({
                    "agent_id": "GL-MRV-S3-009",
                    "version": "1.0.0",
                    "scope": "scope_3_category_9",
                    "description": "Downstream Transportation and Distribution",
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
        component: str = "transport",
    ) -> None:
        """
        Record a downstream transportation emission calculation.

        Args:
            method: Calculation method (distance_based/spend_based/average_data/supplier_specific)
            mode: Transport mode (road/rail/maritime/air/pipeline/intermodal)
            status: Calculation status (success/error/validation_error)
            duration: Operation duration in seconds
            co2e: Emissions calculated in kgCO2e
            component: Emission component (transport/wtt/warehouse/last_mile/cold_chain/return/hub)

        Example:
            >>> metrics.record_calculation(
            ...     method="distance_based", mode="road", status="success",
            ...     duration=0.042, co2e=1250.5,
            ... )
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethodLabel, CalculationMethodLabel.DISTANCE_BASED.value
            )
            mode = self._validate_enum_value(
                mode, TransportModeLabel, TransportModeLabel.ROAD.value
            )
            status = self._validate_enum_value(
                status, CalculationStatusLabel, CalculationStatusLabel.ERROR.value
            )
            component = self._validate_enum_value(
                component, EmissionComponentLabel, EmissionComponentLabel.TRANSPORT.value
            )

            self.calculations_total.labels(
                method=method, mode=mode, status=status
            ).inc()

            if duration is not None and duration > 0:
                self.calculation_duration_seconds.labels(
                    method=method, mode=mode
                ).observe(duration)

            if co2e is not None and co2e > 0:
                self.emissions_kg_total.labels(
                    mode=mode, component=component
                ).inc(co2e)
                with self._stats_lock:
                    self._in_memory_stats["emissions_kg_co2e"] += co2e

            with self._stats_lock:
                self._in_memory_stats["calculations"] += 1
                if status == CalculationStatusLabel.ERROR.value:
                    self._in_memory_stats["errors"] += 1

            logger.debug(
                "Recorded calculation: method=%s, mode=%s, status=%s, "
                "duration=%.3fs, co2e=%.2f kgCO2e",
                method, mode, status,
                duration if duration else 0.0,
                co2e if co2e else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record calculation metrics: %s", e, exc_info=True)

    def record_shipment(
        self,
        mode: str,
        temperature_regime: str = "ambient",
        tonne_km: float = 0.0,
    ) -> None:
        """
        Record a shipment processed.

        Args:
            mode: Transport mode
            temperature_regime: Temperature regime (ambient/chilled/frozen/deep_frozen/pharma)
            tonne_km: Tonne-kilometres for this shipment

        Example:
            >>> metrics.record_shipment(mode="road", temperature_regime="chilled", tonne_km=5200.0)
        """
        try:
            mode = self._validate_enum_value(
                mode, TransportModeLabel, TransportModeLabel.ROAD.value
            )
            temperature_regime = self._validate_enum_value(
                temperature_regime, TemperatureRegimeLabel, TemperatureRegimeLabel.AMBIENT.value
            )

            self.shipments_processed.labels(
                mode=mode, temperature_regime=temperature_regime
            ).inc()

            if tonne_km is not None and tonne_km > 0:
                self.tonne_km_total.labels(mode=mode).inc(tonne_km)
                with self._stats_lock:
                    self._in_memory_stats["tonne_km"] += tonne_km

            with self._stats_lock:
                self._in_memory_stats["shipments"] += 1

            logger.debug(
                "Recorded shipment: mode=%s, regime=%s, tonne_km=%.1f",
                mode, temperature_regime, tonne_km if tonne_km else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record shipment metrics: %s", e, exc_info=True)

    def record_warehouse_emission(
        self,
        warehouse_type: str,
        co2e: float,
    ) -> None:
        """
        Record warehouse/DC emission calculation.

        Args:
            warehouse_type: Type of warehouse (ambient/cold_storage/frozen/retail/cross_dock)
            co2e: Emissions in kgCO2e

        Example:
            >>> metrics.record_warehouse_emission(warehouse_type="cold_storage", co2e=850.0)
        """
        try:
            warehouse_type = self._validate_enum_value(
                warehouse_type, WarehouseTypeLabel, WarehouseTypeLabel.AMBIENT.value
            )

            if co2e is not None and co2e > 0:
                self.warehouse_emissions_total.labels(
                    warehouse_type=warehouse_type
                ).inc(co2e)
                with self._stats_lock:
                    self._in_memory_stats["warehouse_emissions"] += co2e

            logger.debug(
                "Recorded warehouse emission: type=%s, co2e=%.2f kgCO2e",
                warehouse_type, co2e if co2e else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record warehouse metrics: %s", e, exc_info=True)

    def record_last_mile_delivery(
        self,
        delivery_type: str,
        area: str,
    ) -> None:
        """
        Record a last-mile delivery.

        Args:
            delivery_type: Delivery vehicle type
            area: Delivery area classification (urban/suburban/rural/remote)

        Example:
            >>> metrics.record_last_mile_delivery(delivery_type="van_electric", area="urban")
        """
        try:
            delivery_type = self._validate_enum_value(
                delivery_type, LastMileTypeLabel, LastMileTypeLabel.AVERAGE.value
            )
            area = self._validate_enum_value(
                area, DeliveryAreaLabel, DeliveryAreaLabel.URBAN.value
            )

            self.last_mile_deliveries.labels(
                delivery_type=delivery_type, area=area
            ).inc()

            with self._stats_lock:
                self._in_memory_stats["last_mile_deliveries"] += 1

            logger.debug(
                "Recorded last-mile delivery: type=%s, area=%s",
                delivery_type, area,
            )

        except Exception as e:
            logger.error("Failed to record last-mile metrics: %s", e, exc_info=True)

    def record_cold_chain_uplift(self) -> None:
        """
        Record a cold chain emission uplift application.

        Example:
            >>> metrics.record_cold_chain_uplift()
        """
        try:
            self.cold_chain_uplift_total.inc()
            with self._stats_lock:
                self._in_memory_stats["cold_chain_uplifts"] += 1
        except Exception as e:
            logger.error("Failed to record cold chain metrics: %s", e, exc_info=True)

    def record_return_emission(
        self,
        return_type: str,
        co2e: float,
    ) -> None:
        """
        Record return logistics emission.

        Args:
            return_type: Return type (consumer_return/consolidation/repackaging)
            co2e: Emissions in kgCO2e

        Example:
            >>> metrics.record_return_emission(return_type="consumer_return", co2e=12.5)
        """
        try:
            return_type = self._validate_enum_value(
                return_type, ReturnTypeLabel, ReturnTypeLabel.CONSUMER_RETURN.value
            )

            if co2e is not None and co2e > 0:
                self.return_emissions_total.labels(
                    return_type=return_type
                ).inc(co2e)
                with self._stats_lock:
                    self._in_memory_stats["return_emissions"] += co2e

            logger.debug(
                "Recorded return emission: type=%s, co2e=%.2f",
                return_type, co2e if co2e else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record return metrics: %s", e, exc_info=True)

    def record_compliance_check(
        self,
        framework: str,
        status: str,
    ) -> None:
        """
        Record a compliance check result.

        Args:
            framework: Compliance framework
            status: Check result

        Example:
            >>> metrics.record_compliance_check(framework="ghg_protocol", status="compliant")
        """
        try:
            framework = self._validate_enum_value(
                framework, FrameworkLabel, FrameworkLabel.GHG_PROTOCOL.value
            )
            status = self._validate_enum_value(
                status, ComplianceStatusLabel, ComplianceStatusLabel.WARNING.value
            )

            self.compliance_checks.labels(
                framework=framework, status=status
            ).inc()

            with self._stats_lock:
                self._in_memory_stats["compliance_checks"] += 1

            logger.debug(
                "Recorded compliance check: framework=%s, status=%s",
                framework, status,
            )

        except Exception as e:
            logger.error("Failed to record compliance metrics: %s", e, exc_info=True)

    def record_provenance_chain(self) -> None:
        """
        Record provenance chain creation.

        Example:
            >>> metrics.record_provenance_chain()
        """
        try:
            self.provenance_chains.inc()
            with self._stats_lock:
                self._in_memory_stats["provenance_chains"] += 1
        except Exception as e:
            logger.error("Failed to record provenance metrics: %s", e, exc_info=True)

    def record_error(
        self,
        engine: str,
        error_type: str,
    ) -> None:
        """
        Record an error occurrence.

        Args:
            engine: Engine that generated the error
            error_type: Error classification

        Example:
            >>> metrics.record_error(engine="distance_calc", error_type="validation")
        """
        try:
            engine = self._validate_enum_value(
                engine, EngineLabel, EngineLabel.PIPELINE.value
            )
            error_type = self._validate_enum_value(
                error_type, ErrorTypeLabel, ErrorTypeLabel.UNKNOWN.value
            )

            self.errors_total.labels(
                engine=engine, error_type=error_type
            ).inc()

            with self._stats_lock:
                self._in_memory_stats["errors"] += 1

            logger.debug(
                "Recorded error: engine=%s, type=%s", engine, error_type,
            )

        except Exception as e:
            logger.error("Failed to record error metrics: %s", e, exc_info=True)

    def record_eeio_spend(
        self,
        naics_code: str,
        amount_usd: float,
    ) -> None:
        """
        Record EEIO spend-based calculation.

        Args:
            naics_code: NAICS sector code
            amount_usd: Spend amount in USD

        Example:
            >>> metrics.record_eeio_spend(naics_code="484000", amount_usd=50000.0)
        """
        try:
            if naics_code is None:
                naics_code = "unknown"
            naics_code = naics_code.strip()[:10]

            if amount_usd is not None and amount_usd > 0:
                self.eeio_spend_usd.labels(naics_code=naics_code).inc(amount_usd)
                with self._stats_lock:
                    self._in_memory_stats["eeio_spend_usd"] += amount_usd

            logger.debug(
                "Recorded EEIO spend: naics=%s, usd=%.2f",
                naics_code, amount_usd if amount_usd else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record EEIO spend metrics: %s", e, exc_info=True)

    def record_allocation(
        self,
        method: str,
    ) -> None:
        """
        Record an allocation operation.

        Args:
            method: Allocation method (mass/volume/revenue/unit_count/tonne_km)

        Example:
            >>> metrics.record_allocation(method="mass")
        """
        try:
            method = self._validate_enum_value(
                method, AllocationMethodLabel, AllocationMethodLabel.MASS.value
            )

            self.allocation_count.labels(method=method).inc()

            with self._stats_lock:
                self._in_memory_stats["allocations"] += 1

            logger.debug("Recorded allocation: method=%s", method)

        except Exception as e:
            logger.error("Failed to record allocation metrics: %s", e, exc_info=True)

    # ======================================================================
    # Summary and lifecycle
    # ======================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get a summary of metrics collected since initialization.

        Returns:
            Dictionary with metrics summary including counts, uptime, and rates.
        """
        try:
            uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()
            uptime_hours = uptime_seconds / 3600 if uptime_seconds > 0 else 1.0

            with self._stats_lock:
                stats_snapshot = dict(self._in_memory_stats)

            summary: Dict[str, Any] = {
                "prometheus_available": PROMETHEUS_AVAILABLE,
                "agent": "downstream_transportation",
                "agent_id": "GL-MRV-S3-009",
                "prefix": "gl_dto_",
                "scope": "Scope 3 Category 9",
                "description": "Downstream Transportation and Distribution",
                "metrics_count": 14,
                "uptime_seconds": uptime_seconds,
                "uptime_hours": uptime_seconds / 3600,
                "start_time": self._start_time.isoformat(),
                "current_time": datetime.utcnow().isoformat(),
                **stats_snapshot,
                "rates": {
                    "calculations_per_hour": stats_snapshot["calculations"] / uptime_hours,
                    "emissions_kg_per_hour": stats_snapshot["emissions_kg_co2e"] / uptime_hours,
                    "shipments_per_hour": stats_snapshot["shipments"] / uptime_hours,
                    "tonne_km_per_hour": stats_snapshot["tonne_km"] / uptime_hours,
                    "errors_per_hour": stats_snapshot["errors"] / uptime_hours,
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
                "agent": "downstream_transportation",
            }

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance for testing purposes.

        WARNING: Not safe for concurrent use. Only call in test teardown.
        """
        with cls._lock:
            if cls._instance is not None:
                if hasattr(cls._instance, "_initialized"):
                    del cls._instance._initialized
                cls._instance = None

                global _metrics_instance
                _metrics_instance = None

                logger.info("DownstreamTransportMetrics singleton reset")

    def reset_stats(self) -> None:
        """
        Reset in-memory statistics (not Prometheus metrics).

        Prometheus metrics are cumulative and cannot be reset.
        """
        try:
            with self._stats_lock:
                self._in_memory_stats = {
                    "calculations": 0,
                    "emissions_kg_co2e": 0.0,
                    "shipments": 0,
                    "warehouse_emissions": 0.0,
                    "last_mile_deliveries": 0,
                    "tonne_km": 0.0,
                    "cold_chain_uplifts": 0,
                    "return_emissions": 0.0,
                    "compliance_checks": 0,
                    "provenance_chains": 0,
                    "errors": 0,
                    "eeio_spend_usd": 0.0,
                    "allocations": 0,
                }
            self._start_time = datetime.utcnow()
            logger.info("Reset in-memory statistics for DownstreamTransportMetrics")
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


# ===========================================================================
# Module-level singleton accessor
# ===========================================================================

_metrics_instance: Optional[DownstreamTransportMetrics] = None
_metrics_lock: threading.Lock = threading.Lock()


def get_metrics() -> DownstreamTransportMetrics:
    """
    Get the singleton DownstreamTransportMetrics instance.

    Thread-safe accessor.

    Returns:
        DownstreamTransportMetrics singleton instance

    Example:
        >>> metrics = get_metrics()
        >>> metrics.record_calculation(
        ...     method="distance_based", mode="road", status="success",
        ...     duration=0.042, co2e=1250.5,
        ... )
    """
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = DownstreamTransportMetrics()

    return _metrics_instance


def reset_metrics() -> None:
    """
    Reset the singleton metrics instance for testing purposes.

    Example:
        >>> reset_metrics()
    """
    DownstreamTransportMetrics.reset()


# ===========================================================================
# Context manager helpers
# ===========================================================================


@contextmanager
def track_calculation(
    method: str = "distance_based",
    mode: str = "road",
    component: str = "transport",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a calculation's lifecycle.

    Automatically increments/decrements the active calculation gauge,
    measures wall-clock duration, and records the calculation on exit.
    Set ``context['co2e']`` inside the block to include emissions.

    Args:
        method: Calculation method (default: "distance_based")
        mode: Transport mode (default: "road")
        component: Emission component (default: "transport")

    Yields:
        Mutable context dict. Set ``context['co2e']`` to record emissions.

    Example:
        >>> with track_calculation(method="distance_based", mode="maritime") as ctx:
        ...     result = calculate_maritime_emissions(shipment)
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
            component=component,
        )


@contextmanager
def track_batch() -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for tracking batch job lifecycle.

    Yields:
        Mutable context dict. Set ``context['size']`` to record batch size.

    Example:
        >>> with track_batch() as ctx:
        ...     results = process_batch(shipments)
        ...     ctx['size'] = len(shipments)
    """
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
        pass  # Batch metrics recorded by caller


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
    "TemperatureRegimeLabel",
    "WarehouseTypeLabel",
    "LastMileTypeLabel",
    "DeliveryAreaLabel",
    "EmissionComponentLabel",
    "FrameworkLabel",
    "ComplianceStatusLabel",
    "ReturnTypeLabel",
    "EngineLabel",
    "ErrorTypeLabel",
    "AllocationMethodLabel",
    # Singleton class
    "DownstreamTransportMetrics",
    # Module-level accessors
    "get_metrics",
    "reset_metrics",
    # Context managers
    "track_calculation",
    "track_batch",
]
