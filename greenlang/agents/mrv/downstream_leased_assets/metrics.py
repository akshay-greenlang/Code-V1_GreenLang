# -*- coding: utf-8 -*-
"""
Downstream Leased Assets Prometheus Metrics - AGENT-MRV-026

14 Prometheus metrics with gl_dla_ prefix for monitoring the
GL-MRV-S3-013 Downstream Leased Assets Agent.

This module provides Prometheus metrics tracking for downstream leased assets
emissions calculations (Scope 3, Category 13) including asset-specific,
average-data, spend-based, and hybrid calculation methods across all asset
categories (buildings, vehicles, equipment, IT assets).

Thread-safe singleton pattern with graceful fallback if Prometheus unavailable.

Metrics prefix: gl_dla_

14 Prometheus Metrics:
    1.  gl_dla_calculations_total                - Counter: total calculations
    2.  gl_dla_calculation_duration_seconds       - Histogram: calculation duration
    3.  gl_dla_emissions_kg_total                - Counter: total emissions in kgCO2e
    4.  gl_dla_assets_processed_total            - Counter: assets processed
    5.  gl_dla_building_emissions_total          - Counter: building emissions
    6.  gl_dla_vehicle_emissions_total           - Counter: vehicle emissions
    7.  gl_dla_equipment_emissions_total         - Counter: equipment emissions
    8.  gl_dla_it_emissions_total                - Counter: IT asset emissions
    9.  gl_dla_compliance_checks_total           - Counter: compliance checks
    10. gl_dla_dc_rule_triggers_total            - Counter: double-counting rule triggers
    11. gl_dla_pipeline_stage_duration_seconds   - Histogram: pipeline stage duration
    12. gl_dla_allocation_operations_total       - Counter: allocation operations
    13. gl_dla_vacancy_adjustments_total         - Counter: vacancy adjustments
    14. gl_dla_active_calculations               - Gauge: active calculations

GHG Protocol Scope 3 Category 13 covers downstream leased assets:
    A. Emissions from assets owned by the reporting company and leased
       to other entities (operating leases where lessor does not include
       in Scope 1/2).
    B. Includes buildings, vehicles, equipment, and IT assets.
    C. Calculation methods: asset-specific, average-data, spend-based.
    D. Requires allocation between lessor and lessee.

Example:
    >>> metrics = get_metrics_collector()
    >>> metrics.record_calculation(
    ...     method="asset_specific",
    ...     asset_category="building",
    ...     status="success",
    ...     duration=0.045,
    ...     co2e=1250.3
    ... )

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-013
"""

import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
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
# Enumerations -- Downstream Leased Assets domain-specific label value sets
# ===========================================================================


class CalculationMethodLabel(str, Enum):
    """
    Calculation methods for downstream leased assets emissions.

    GHG Protocol Scope 3 Category 13 supports several approaches depending
    on data availability:
        - Asset-specific: direct metering or sub-metering data
        - Average-data: benchmarks and regional defaults
        - Spend-based: EEIO factors applied to lease revenue/costs
    """
    ASSET_SPECIFIC = "asset_specific"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"
    HYBRID = "hybrid"


class AssetCategoryLabel(str, Enum):
    """
    Asset categories for downstream leased assets.

    GHG Protocol Category 13 covers all assets owned by the reporting
    company and leased to other entities.
    """
    BUILDING = "building"
    VEHICLE = "vehicle"
    EQUIPMENT = "equipment"
    IT_ASSET = "it_asset"


class CalculationStatusLabel(str, Enum):
    """Calculation operation status."""
    SUCCESS = "success"
    ERROR = "error"
    VALIDATION_ERROR = "validation_error"


class BuildingTypeLabel(str, Enum):
    """Building types for downstream leased building assets."""
    OFFICE = "office"
    RETAIL = "retail"
    WAREHOUSE = "warehouse"
    INDUSTRIAL = "industrial"
    RESIDENTIAL = "residential"
    DATA_CENTER = "data_center"
    MIXED_USE = "mixed_use"
    OTHER = "other"


class VehicleTypeLabel(str, Enum):
    """Vehicle types for downstream leased vehicle assets."""
    PASSENGER_CAR = "passenger_car"
    LIGHT_TRUCK = "light_truck"
    HEAVY_TRUCK = "heavy_truck"
    BUS = "bus"
    VAN = "van"
    MOTORCYCLE = "motorcycle"
    SPECIAL_PURPOSE = "special_purpose"
    OTHER = "other"


class EquipmentTypeLabel(str, Enum):
    """Equipment types for downstream leased equipment assets."""
    GENERATOR = "generator"
    COMPRESSOR = "compressor"
    HVAC = "hvac"
    REFRIGERATION = "refrigeration"
    MACHINERY = "machinery"
    OTHER = "other"


class ITAssetTypeLabel(str, Enum):
    """IT asset types for downstream leased IT assets."""
    SERVER = "server"
    STORAGE = "storage"
    NETWORK_EQUIPMENT = "network_equipment"
    DESKTOP = "desktop"
    LAPTOP = "laptop"
    PRINTER = "printer"
    OTHER = "other"


class FrameworkLabel(str, Enum):
    """Compliance frameworks for downstream leased assets reporting."""
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD = "csrd"
    CDP = "cdp"
    SBTI = "sbti"
    GRI = "gri"
    DEFRA_BEIS = "defra_beis"


class ComplianceStatusLabel(str, Enum):
    """Compliance check result status."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class PipelineStageLabel(str, Enum):
    """Pipeline stages for duration tracking."""
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


class AllocationMethodLabel(str, Enum):
    """Allocation methods for multi-tenant assets."""
    FLOOR_AREA = "floor_area"
    HEADCOUNT = "headcount"
    FTE = "fte"
    REVENUE = "revenue"
    ENERGY_CONSUMPTION = "energy_consumption"
    EQUAL = "equal"
    CUSTOM = "custom"


# ===========================================================================
# MetricsCollector -- Thread-safe Singleton
# ===========================================================================


class MetricsCollector:
    """
    Thread-safe singleton metrics collector for Downstream Leased Assets Agent.

    Provides 14 Prometheus metrics for tracking Scope 3 Category 13
    downstream leased assets emissions calculations across all asset
    categories and calculation methods.

    All metrics use the ``gl_dla_`` prefix to ensure namespace isolation
    within the GreenLang Prometheus ecosystem.

    Asset Categories Tracked:
        A. Buildings (office, retail, warehouse, industrial, residential,
           data center, mixed-use, other)
        B. Vehicles (passenger car, light truck, heavy truck, bus, van,
           motorcycle, special purpose, other)
        C. Equipment (generator, compressor, HVAC, refrigeration,
           machinery, other)
        D. IT Assets (server, storage, network, desktop, laptop,
           printer, other)

    Attributes:
        calculations_total: Counter for total calculation operations
        calculation_duration_seconds: Histogram for calculation durations
        emissions_kg_total: Counter for total emissions in kgCO2e
        assets_processed_total: Counter for assets processed
        building_emissions_total: Counter for building emissions
        vehicle_emissions_total: Counter for vehicle emissions
        equipment_emissions_total: Counter for equipment emissions
        it_emissions_total: Counter for IT asset emissions
        compliance_checks_total: Counter for compliance checks
        dc_rule_triggers_total: Counter for double-counting rule triggers
        pipeline_stage_duration_seconds: Histogram for pipeline stage durations
        allocation_operations_total: Counter for allocation operations
        vacancy_adjustments_total: Counter for vacancy adjustments
        active_calculations: Gauge for active calculations

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_calculation(
        ...     method="asset_specific",
        ...     asset_category="building",
        ...     status="success",
        ...     duration=0.045,
        ...     co2e=1250.3
        ... )
    """

    _instance: Optional["MetricsCollector"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
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
        self._start_time: datetime = datetime.now(timezone.utc)
        self._stats_lock: threading.Lock = threading.Lock()
        self._in_memory_stats: Dict[str, Any] = {
            "calculations": 0,
            "emissions_kg_co2e": 0.0,
            "assets_processed": 0,
            "building_emissions": 0.0,
            "vehicle_emissions": 0.0,
            "equipment_emissions": 0.0,
            "it_emissions": 0.0,
            "compliance_checks": 0,
            "dc_rule_triggers": 0,
            "allocation_operations": 0,
            "vacancy_adjustments": 0,
            "errors": 0,
        }

        # Initialize Prometheus metrics
        self._init_metrics()

        logger.info(
            "MetricsCollector initialized (Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable",
        )

    def _init_metrics(self) -> None:
        """
        Initialize all 14 Prometheus metrics with gl_dla_ prefix.

        When Prometheus is available, metric registration may fail if the
        metrics were previously registered. In that case we unregister
        from the default registry and re-register.
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
        # 1. gl_dla_calculations_total (Counter)
        #    Total downstream leased asset emission calculations.
        #    Labels: method, asset_category, status
        # ------------------------------------------------------------------
        self.calculations_total = _safe_create(
            Counter,
            "gl_dla_calculations_total",
            "Total downstream leased asset emission calculations performed",
            ["method", "asset_category", "status"],
        )

        # ------------------------------------------------------------------
        # 2. gl_dla_calculation_duration_seconds (Histogram)
        #    Duration of calculation operations in seconds.
        #    Labels: method, asset_category
        # ------------------------------------------------------------------
        self.calculation_duration_seconds = _safe_create(
            Histogram,
            "gl_dla_calculation_duration_seconds",
            "Duration of downstream leased asset calculation operations",
            ["method", "asset_category"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        # ------------------------------------------------------------------
        # 3. gl_dla_emissions_kg_total (Counter)
        #    Total emissions calculated in kilograms CO2-equivalent.
        #    Labels: asset_category
        # ------------------------------------------------------------------
        self.emissions_kg_total = _safe_create(
            Counter,
            "gl_dla_emissions_kg_total",
            "Total downstream leased asset emissions calculated in kgCO2e",
            ["asset_category"],
        )

        # ------------------------------------------------------------------
        # 4. gl_dla_assets_processed_total (Counter)
        #    Total assets processed across all categories.
        #    Labels: asset_category, asset_type
        # ------------------------------------------------------------------
        self.assets_processed_total = _safe_create(
            Counter,
            "gl_dla_assets_processed_total",
            "Total downstream leased assets processed",
            ["asset_category", "asset_type"],
        )

        # ------------------------------------------------------------------
        # 5. gl_dla_building_emissions_total (Counter)
        #    Total building emissions by building type.
        #    Labels: building_type
        # ------------------------------------------------------------------
        self.building_emissions_total = _safe_create(
            Counter,
            "gl_dla_building_emissions_total",
            "Total building emissions from downstream leased assets in kgCO2e",
            ["building_type"],
        )

        # ------------------------------------------------------------------
        # 6. gl_dla_vehicle_emissions_total (Counter)
        #    Total vehicle emissions by vehicle type.
        #    Labels: vehicle_type
        # ------------------------------------------------------------------
        self.vehicle_emissions_total = _safe_create(
            Counter,
            "gl_dla_vehicle_emissions_total",
            "Total vehicle emissions from downstream leased assets in kgCO2e",
            ["vehicle_type"],
        )

        # ------------------------------------------------------------------
        # 7. gl_dla_equipment_emissions_total (Counter)
        #    Total equipment emissions by equipment type.
        #    Labels: equipment_type
        # ------------------------------------------------------------------
        self.equipment_emissions_total = _safe_create(
            Counter,
            "gl_dla_equipment_emissions_total",
            "Total equipment emissions from downstream leased assets in kgCO2e",
            ["equipment_type"],
        )

        # ------------------------------------------------------------------
        # 8. gl_dla_it_emissions_total (Counter)
        #    Total IT asset emissions by IT asset type.
        #    Labels: it_asset_type
        # ------------------------------------------------------------------
        self.it_emissions_total = _safe_create(
            Counter,
            "gl_dla_it_emissions_total",
            "Total IT asset emissions from downstream leased assets in kgCO2e",
            ["it_asset_type"],
        )

        # ------------------------------------------------------------------
        # 9. gl_dla_compliance_checks_total (Counter)
        #    Total compliance checks performed.
        #    Labels: framework, status
        # ------------------------------------------------------------------
        self.compliance_checks_total = _safe_create(
            Counter,
            "gl_dla_compliance_checks_total",
            "Total downstream leased asset compliance checks performed",
            ["framework", "status"],
        )

        # ------------------------------------------------------------------
        # 10. gl_dla_dc_rule_triggers_total (Counter)
        #     Total double-counting rule triggers.
        #     Labels: rule_id
        # ------------------------------------------------------------------
        self.dc_rule_triggers_total = _safe_create(
            Counter,
            "gl_dla_dc_rule_triggers_total",
            "Total double-counting rule triggers for downstream leased assets",
            ["rule_id"],
        )

        # ------------------------------------------------------------------
        # 11. gl_dla_pipeline_stage_duration_seconds (Histogram)
        #     Duration of each pipeline stage in seconds.
        #     Labels: stage
        # ------------------------------------------------------------------
        self.pipeline_stage_duration_seconds = _safe_create(
            Histogram,
            "gl_dla_pipeline_stage_duration_seconds",
            "Duration of pipeline stages for downstream leased assets",
            ["stage"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0),
        )

        # ------------------------------------------------------------------
        # 12. gl_dla_allocation_operations_total (Counter)
        #     Total allocation operations by method.
        #     Labels: method
        # ------------------------------------------------------------------
        self.allocation_operations_total = _safe_create(
            Counter,
            "gl_dla_allocation_operations_total",
            "Total allocation operations for downstream leased assets",
            ["method"],
        )

        # ------------------------------------------------------------------
        # 13. gl_dla_vacancy_adjustments_total (Counter)
        #     Total vacancy adjustments applied. No labels.
        # ------------------------------------------------------------------
        self.vacancy_adjustments_total = _safe_create(
            Counter,
            "gl_dla_vacancy_adjustments_total",
            "Total vacancy adjustments applied to downstream leased assets",
        )

        # ------------------------------------------------------------------
        # 14. gl_dla_active_calculations (Gauge)
        #     Number of currently active calculations. No labels.
        # ------------------------------------------------------------------
        self.active_calculations = _safe_create(
            Gauge,
            "gl_dla_active_calculations",
            "Number of currently active downstream leased asset calculations",
        )

        # Agent info metric (non-numeric metadata)
        self.agent_info = _safe_create(
            Info,
            "gl_dla_agent",
            "Downstream Leased Assets Agent metadata",
        )
        if PROMETHEUS_AVAILABLE:
            try:
                self.agent_info.info({
                    "agent_id": "GL-MRV-S3-013",
                    "version": "1.0.0",
                    "scope": "scope_3_category_13",
                    "description": "Downstream Leased Assets emissions calculator",
                })
            except Exception:
                pass

    # ======================================================================
    # Primary recording methods
    # ======================================================================

    def record_calculation(
        self,
        method: str,
        asset_category: str,
        status: str,
        duration: float,
        co2e: float,
    ) -> None:
        """
        Record a downstream leased asset emission calculation operation.

        Args:
            method: Calculation method (asset_specific/average_data/spend_based/hybrid)
            asset_category: Asset category (building/vehicle/equipment/it_asset)
            status: Calculation status (success/error/validation_error)
            duration: Operation duration in seconds
            co2e: Emissions calculated in kgCO2e

        Example:
            >>> collector.record_calculation(
            ...     method="asset_specific",
            ...     asset_category="building",
            ...     status="success",
            ...     duration=0.045,
            ...     co2e=1250.3
            ... )
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethodLabel, CalculationMethodLabel.AVERAGE_DATA.value
            )
            asset_category = self._validate_enum_value(
                asset_category, AssetCategoryLabel, AssetCategoryLabel.BUILDING.value
            )
            status = self._validate_enum_value(
                status, CalculationStatusLabel, CalculationStatusLabel.ERROR.value
            )

            self.calculations_total.labels(
                method=method, asset_category=asset_category, status=status
            ).inc()

            if duration is not None and duration > 0:
                self.calculation_duration_seconds.labels(
                    method=method, asset_category=asset_category
                ).observe(duration)

            if co2e is not None and co2e > 0:
                self.emissions_kg_total.labels(
                    asset_category=asset_category
                ).inc(co2e)

                with self._stats_lock:
                    self._in_memory_stats["emissions_kg_co2e"] += co2e

            with self._stats_lock:
                self._in_memory_stats["calculations"] += 1
                if status == CalculationStatusLabel.ERROR.value:
                    self._in_memory_stats["errors"] += 1

            logger.debug(
                "Recorded calculation: method=%s, category=%s, status=%s, "
                "duration=%.3fs, co2e=%.2f kgCO2e",
                method, asset_category, status,
                duration if duration else 0.0,
                co2e if co2e else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record calculation metrics: %s", e, exc_info=True)

    def record_asset_processed(
        self,
        asset_category: str,
        asset_type: str,
    ) -> None:
        """
        Record an asset processing event.

        Args:
            asset_category: Asset category (building/vehicle/equipment/it_asset)
            asset_type: Specific asset type within the category

        Example:
            >>> collector.record_asset_processed(
            ...     asset_category="building",
            ...     asset_type="office"
            ... )
        """
        try:
            asset_category = self._validate_enum_value(
                asset_category, AssetCategoryLabel, AssetCategoryLabel.BUILDING.value
            )

            self.assets_processed_total.labels(
                asset_category=asset_category, asset_type=asset_type or "unknown"
            ).inc()

            with self._stats_lock:
                self._in_memory_stats["assets_processed"] += 1

            logger.debug(
                "Recorded asset processed: category=%s, type=%s",
                asset_category, asset_type,
            )

        except Exception as e:
            logger.error("Failed to record asset processed metrics: %s", e, exc_info=True)

    def record_building_emissions(
        self,
        building_type: str,
        co2e: float,
    ) -> None:
        """
        Record building emissions.

        Args:
            building_type: Building type (office/retail/warehouse/etc.)
            co2e: Emissions in kgCO2e

        Example:
            >>> collector.record_building_emissions(
            ...     building_type="office",
            ...     co2e=5200.0
            ... )
        """
        try:
            building_type = self._validate_enum_value(
                building_type, BuildingTypeLabel, BuildingTypeLabel.OTHER.value
            )

            if co2e is not None and co2e > 0:
                self.building_emissions_total.labels(
                    building_type=building_type
                ).inc(co2e)

                with self._stats_lock:
                    self._in_memory_stats["building_emissions"] += co2e

            logger.debug(
                "Recorded building emissions: type=%s, co2e=%.2f kgCO2e",
                building_type, co2e if co2e else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record building emissions metrics: %s", e, exc_info=True)

    def record_vehicle_emissions(
        self,
        vehicle_type: str,
        co2e: float,
    ) -> None:
        """
        Record vehicle emissions.

        Args:
            vehicle_type: Vehicle type (passenger_car/light_truck/etc.)
            co2e: Emissions in kgCO2e

        Example:
            >>> collector.record_vehicle_emissions(
            ...     vehicle_type="passenger_car",
            ...     co2e=850.0
            ... )
        """
        try:
            vehicle_type = self._validate_enum_value(
                vehicle_type, VehicleTypeLabel, VehicleTypeLabel.OTHER.value
            )

            if co2e is not None and co2e > 0:
                self.vehicle_emissions_total.labels(
                    vehicle_type=vehicle_type
                ).inc(co2e)

                with self._stats_lock:
                    self._in_memory_stats["vehicle_emissions"] += co2e

            logger.debug(
                "Recorded vehicle emissions: type=%s, co2e=%.2f kgCO2e",
                vehicle_type, co2e if co2e else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record vehicle emissions metrics: %s", e, exc_info=True)

    def record_equipment_emissions(
        self,
        equipment_type: str,
        co2e: float,
    ) -> None:
        """
        Record equipment emissions.

        Args:
            equipment_type: Equipment type (generator/compressor/etc.)
            co2e: Emissions in kgCO2e

        Example:
            >>> collector.record_equipment_emissions(
            ...     equipment_type="generator",
            ...     co2e=3200.0
            ... )
        """
        try:
            equipment_type = self._validate_enum_value(
                equipment_type, EquipmentTypeLabel, EquipmentTypeLabel.OTHER.value
            )

            if co2e is not None and co2e > 0:
                self.equipment_emissions_total.labels(
                    equipment_type=equipment_type
                ).inc(co2e)

                with self._stats_lock:
                    self._in_memory_stats["equipment_emissions"] += co2e

            logger.debug(
                "Recorded equipment emissions: type=%s, co2e=%.2f kgCO2e",
                equipment_type, co2e if co2e else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record equipment emissions metrics: %s", e, exc_info=True)

    def record_it_emissions(
        self,
        it_asset_type: str,
        co2e: float,
    ) -> None:
        """
        Record IT asset emissions.

        Args:
            it_asset_type: IT asset type (server/storage/etc.)
            co2e: Emissions in kgCO2e

        Example:
            >>> collector.record_it_emissions(
            ...     it_asset_type="server",
            ...     co2e=920.0
            ... )
        """
        try:
            it_asset_type = self._validate_enum_value(
                it_asset_type, ITAssetTypeLabel, ITAssetTypeLabel.OTHER.value
            )

            if co2e is not None and co2e > 0:
                self.it_emissions_total.labels(
                    it_asset_type=it_asset_type
                ).inc(co2e)

                with self._stats_lock:
                    self._in_memory_stats["it_emissions"] += co2e

            logger.debug(
                "Recorded IT asset emissions: type=%s, co2e=%.2f kgCO2e",
                it_asset_type, co2e if co2e else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record IT emissions metrics: %s", e, exc_info=True)

    def record_compliance_check(
        self,
        framework: str,
        status: str,
    ) -> None:
        """
        Record a compliance check result.

        Args:
            framework: Compliance framework (ghg_protocol/iso_14064/csrd/etc.)
            status: Check result (compliant/partially_compliant/non_compliant/etc.)

        Example:
            >>> collector.record_compliance_check(
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

    def record_dc_rule_trigger(self, rule_id: str) -> None:
        """
        Record a double-counting rule trigger.

        Args:
            rule_id: Rule identifier (e.g., "DC-CAT13-SCOPE1", "DC-CAT13-CAT8")

        Example:
            >>> collector.record_dc_rule_trigger("DC-CAT13-SCOPE1")
        """
        try:
            if not rule_id:
                rule_id = "unknown"

            self.dc_rule_triggers_total.labels(rule_id=rule_id).inc()

            with self._stats_lock:
                self._in_memory_stats["dc_rule_triggers"] += 1

            logger.debug("Recorded DC rule trigger: rule_id=%s", rule_id)

        except Exception as e:
            logger.error("Failed to record DC rule trigger metrics: %s", e, exc_info=True)

    def record_pipeline_stage(
        self,
        stage: str,
        duration: float,
    ) -> None:
        """
        Record a pipeline stage duration.

        Args:
            stage: Pipeline stage name (validate/classify/normalize/etc.)
            duration: Stage duration in seconds

        Example:
            >>> collector.record_pipeline_stage(
            ...     stage="calculate",
            ...     duration=0.125
            ... )
        """
        try:
            stage = self._validate_enum_value(
                stage, PipelineStageLabel, PipelineStageLabel.CALCULATE.value
            )

            if duration is not None and duration > 0:
                self.pipeline_stage_duration_seconds.labels(
                    stage=stage
                ).observe(duration)

            logger.debug(
                "Recorded pipeline stage: stage=%s, duration=%.3fs",
                stage, duration if duration else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record pipeline stage metrics: %s", e, exc_info=True)

    def record_allocation(self, method: str) -> None:
        """
        Record an allocation operation.

        Args:
            method: Allocation method (floor_area/headcount/fte/revenue/etc.)

        Example:
            >>> collector.record_allocation("floor_area")
        """
        try:
            method = self._validate_enum_value(
                method, AllocationMethodLabel, AllocationMethodLabel.FLOOR_AREA.value
            )

            self.allocation_operations_total.labels(method=method).inc()

            with self._stats_lock:
                self._in_memory_stats["allocation_operations"] += 1

            logger.debug("Recorded allocation operation: method=%s", method)

        except Exception as e:
            logger.error("Failed to record allocation metrics: %s", e, exc_info=True)

    def record_vacancy_adjustment(self) -> None:
        """
        Record a vacancy adjustment.

        Example:
            >>> collector.record_vacancy_adjustment()
        """
        try:
            self.vacancy_adjustments_total.inc()

            with self._stats_lock:
                self._in_memory_stats["vacancy_adjustments"] += 1

            logger.debug("Recorded vacancy adjustment")

        except Exception as e:
            logger.error("Failed to record vacancy adjustment metrics: %s", e, exc_info=True)

    # ======================================================================
    # Summary and lifecycle
    # ======================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get a summary of metrics collected since initialization.

        Returns:
            Dictionary with metrics summary including counts, uptime, rates.

        Example:
            >>> stats = collector.get_stats()
            >>> print(stats['calculations'])
            142
        """
        try:
            uptime_seconds = (
                datetime.now(timezone.utc) - self._start_time
            ).total_seconds()
            uptime_hours = uptime_seconds / 3600 if uptime_seconds > 0 else 1.0

            with self._stats_lock:
                stats_snapshot = dict(self._in_memory_stats)

            summary: Dict[str, Any] = {
                "prometheus_available": PROMETHEUS_AVAILABLE,
                "agent": "downstream_leased_assets",
                "agent_id": "GL-MRV-S3-013",
                "prefix": "gl_dla_",
                "scope": "Scope 3 Category 13",
                "description": "Downstream Leased Assets",
                "metrics_count": 14,
                "uptime_seconds": uptime_seconds,
                "uptime_hours": uptime_seconds / 3600,
                "start_time": self._start_time.isoformat(),
                "current_time": datetime.now(timezone.utc).isoformat(),
                **stats_snapshot,
                "rates": {
                    "calculations_per_hour": (
                        stats_snapshot["calculations"] / uptime_hours
                    ),
                    "emissions_kg_co2e_per_hour": (
                        stats_snapshot["emissions_kg_co2e"] / uptime_hours
                    ),
                    "assets_per_hour": (
                        stats_snapshot["assets_processed"] / uptime_hours
                    ),
                    "compliance_checks_per_hour": (
                        stats_snapshot["compliance_checks"] / uptime_hours
                    ),
                    "errors_per_hour": (
                        stats_snapshot["errors"] / uptime_hours
                    ),
                },
                "category_breakdown": {
                    "building_emissions": stats_snapshot["building_emissions"],
                    "vehicle_emissions": stats_snapshot["vehicle_emissions"],
                    "equipment_emissions": stats_snapshot["equipment_emissions"],
                    "it_emissions": stats_snapshot["it_emissions"],
                },
                "operational": {
                    "compliance_checks": stats_snapshot["compliance_checks"],
                    "dc_rule_triggers": stats_snapshot["dc_rule_triggers"],
                    "allocation_operations": stats_snapshot["allocation_operations"],
                    "vacancy_adjustments": stats_snapshot["vacancy_adjustments"],
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
                "agent": "downstream_leased_assets",
            }

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance for testing purposes.

        WARNING: Not safe for concurrent use. Call only in test teardown.
        """
        with cls._lock:
            if cls._instance is not None:
                if hasattr(cls._instance, "_initialized"):
                    del cls._instance._initialized
                cls._instance = None

                global _metrics_instance
                _metrics_instance = None

                logger.info("MetricsCollector singleton reset")

    def reset_stats(self) -> None:
        """
        Reset in-memory statistics (not Prometheus metrics).

        Note: Prometheus metrics are cumulative and cannot be reset without
        restarting the process.
        """
        try:
            with self._stats_lock:
                self._in_memory_stats = {
                    "calculations": 0,
                    "emissions_kg_co2e": 0.0,
                    "assets_processed": 0,
                    "building_emissions": 0.0,
                    "vehicle_emissions": 0.0,
                    "equipment_emissions": 0.0,
                    "it_emissions": 0.0,
                    "compliance_checks": 0,
                    "dc_rule_triggers": 0,
                    "allocation_operations": 0,
                    "vacancy_adjustments": 0,
                    "errors": 0,
                }
            self._start_time = datetime.now(timezone.utc)

            logger.info("Reset in-memory statistics for MetricsCollector")

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

        If the value is None or not a valid member, returns the default
        to ensure bounded label cardinality.

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

_metrics_instance: Optional[MetricsCollector] = None
_metrics_lock: threading.Lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """
    Get the singleton MetricsCollector instance.

    Thread-safe accessor for the global metrics instance.

    Returns:
        MetricsCollector singleton instance

    Example:
        >>> from greenlang.agents.mrv.downstream_leased_assets.metrics import get_metrics_collector
        >>> collector = get_metrics_collector()
        >>> collector.record_calculation(
        ...     method="asset_specific",
        ...     asset_category="building",
        ...     status="success",
        ...     duration=0.045,
        ...     co2e=1250.3
        ... )
    """
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = MetricsCollector()

    return _metrics_instance


def reset_metrics_collector() -> None:
    """
    Reset the singleton metrics instance for testing purposes.

    Convenience function that delegates to MetricsCollector.reset().
    Should only be called in test teardown.
    """
    MetricsCollector.reset()


# ===========================================================================
# Context manager helpers
# ===========================================================================


@contextmanager
def track_calculation(
    method: str = "asset_specific",
    asset_category: str = "building",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a calculation's lifecycle.

    Automatically increments/decrements the active calculation gauge,
    measures wall-clock duration, and records the calculation when the
    context exits.

    Args:
        method: Calculation method (default: "asset_specific")
        asset_category: Asset category (default: "building")

    Yields:
        Mutable context dict. Set ``context['co2e']`` to record emissions.

    Example:
        >>> with track_calculation(method="asset_specific", asset_category="building") as ctx:
        ...     result = calculate_building_emissions(asset)
        ...     ctx['co2e'] = result.co2e_kg
    """
    collector = get_metrics_collector()
    context: Dict[str, Any] = {
        "co2e": 0.0,
        "status": "success",
    }

    collector.active_calculations.inc()
    start = time.monotonic()

    try:
        yield context
    except Exception:
        context["status"] = "error"
        raise
    finally:
        duration = time.monotonic() - start
        collector.active_calculations.dec()
        collector.record_calculation(
            method=method,
            asset_category=asset_category,
            status=context["status"],
            duration=duration,
            co2e=context.get("co2e", 0.0),
        )


@contextmanager
def track_pipeline_stage(
    stage: str = "calculate",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a pipeline stage's lifecycle.

    Automatically measures wall-clock duration and records the stage
    when the context exits.

    Args:
        stage: Pipeline stage name (default: "calculate")

    Yields:
        Mutable context dict for additional metadata.

    Example:
        >>> with track_pipeline_stage("validate") as ctx:
        ...     validated_data = validate_input(data)
        ...     ctx['records'] = len(validated_data)
    """
    collector = get_metrics_collector()
    context: Dict[str, Any] = {}

    start = time.monotonic()

    try:
        yield context
    finally:
        duration = time.monotonic() - start
        collector.record_pipeline_stage(stage=stage, duration=duration)


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    # Availability flag
    "PROMETHEUS_AVAILABLE",
    # Enums
    "CalculationMethodLabel",
    "AssetCategoryLabel",
    "CalculationStatusLabel",
    "BuildingTypeLabel",
    "VehicleTypeLabel",
    "EquipmentTypeLabel",
    "ITAssetTypeLabel",
    "FrameworkLabel",
    "ComplianceStatusLabel",
    "PipelineStageLabel",
    "AllocationMethodLabel",
    # Singleton class
    "MetricsCollector",
    # Module-level accessors
    "get_metrics_collector",
    "reset_metrics_collector",
    # Context managers
    "track_calculation",
    "track_pipeline_stage",
]
