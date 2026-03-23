# -*- coding: utf-8 -*-
"""
End-of-Life Treatment of Sold Products Prometheus Metrics - AGENT-MRV-025

14 Prometheus metrics with gl_eol_ prefix for monitoring the
GL-MRV-S3-012 End-of-Life Treatment of Sold Products Agent.

This module provides Prometheus metrics tracking for end-of-life treatment
emissions calculations (Scope 3, Category 12) including waste-type-specific,
average-data, producer-specific, and hybrid calculation methods across all
end-of-life treatment pathways (landfill, incineration, recycling, composting,
anaerobic digestion, and open burning).

Thread-safe singleton pattern with graceful fallback if Prometheus unavailable.

Metrics prefix: gl_eol_

14 Prometheus Metrics:
    1.  gl_eol_calculations_total              - Counter: total calculations performed
    2.  gl_eol_calculation_duration_seconds     - Histogram: calculation durations
    3.  gl_eol_emissions_kg_total              - Counter: total emissions in kg CO2e
    4.  gl_eol_products_processed_total        - Counter: total products processed
    5.  gl_eol_landfill_emissions_total        - Counter: landfill emissions by material
    6.  gl_eol_incineration_emissions_total    - Counter: incineration emissions by material
    7.  gl_eol_recycling_emissions_total       - Counter: recycling emissions by material
    8.  gl_eol_avoided_emissions_total         - Counter: avoided emissions (recycling/energy)
    9.  gl_eol_compliance_checks_total         - Counter: compliance checks by framework
    10. gl_eol_dc_rule_triggers_total          - Counter: double-counting rule triggers
    11. gl_eol_pipeline_stage_duration_seconds - Histogram: pipeline stage durations
    12. gl_eol_circularity_score               - Gauge: circularity index score
    13. gl_eol_diversion_rate                  - Gauge: waste diversion rate
    14. gl_eol_active_calculations             - Gauge: currently active calculations

GHG Protocol Scope 3 Category 12 covers end-of-life treatment of sold products:
    A. Emissions from waste disposal and treatment of products sold by the
       reporting company (in the reporting year) at the end of their life.
    B. Treatment methods: landfill, incineration (with/without energy recovery),
       recycling, composting, anaerobic digestion, open burning.
    C. Includes the expected end-of-life emissions based on product material
       composition and regional waste treatment statistics.
    D. Excludes use-phase emissions (Category 11) and downstream
       transportation (Category 9).

Calculation methods defined by GHG Protocol:
    - Waste-type-specific: product mass by material x waste-type-specific EF
    - Average-data: product mass x average regional waste treatment EF
    - Producer-specific: EPD or take-back programme primary data
    - Hybrid: waterfall combining producer-specific, waste-type, average-data

Example:
    >>> collector = get_metrics_collector()
    >>> collector.record_calculation(
    ...     method="waste_type_specific",
    ...     category="electronics",
    ...     status="success",
    ...     duration=0.045,
    ...     emissions_kg=12.5
    ... )

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-012
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

        def observe(self, amount: float) -> None:
            pass

        def info(self, data: Dict[str, str]) -> None:
            pass

    Counter = _NoOpMetric  # type: ignore[misc, assignment]
    Histogram = _NoOpMetric  # type: ignore[misc, assignment]
    Gauge = _NoOpMetric  # type: ignore[misc, assignment]
    Info = _NoOpMetric  # type: ignore[misc, assignment]


# ===========================================================================
# Enumerations -- End-of-Life domain-specific label value sets
# ===========================================================================


class CalculationMethod(str, Enum):
    """
    Calculation methods for end-of-life treatment emissions.

    GHG Protocol Scope 3 Category 12 supports several approaches:
        - Waste-type-specific: product material composition x treatment EFs
        - Average-data: product mass x regional average EOL EFs
        - Producer-specific: EPD, take-back, or product stewardship data
        - Hybrid: waterfall combining multiple data sources
    """
    WASTE_TYPE_SPECIFIC = "waste_type_specific"
    AVERAGE_DATA = "average_data"
    PRODUCER_SPECIFIC = "producer_specific"
    HYBRID = "hybrid"


class TreatmentMethod(str, Enum):
    """
    End-of-life treatment methods for sold products.

    Covers the primary treatment pathways for products after consumer use,
    aligned with regional waste management statistics and IPCC waste sector
    methodology.
    """
    LANDFILL = "landfill"
    INCINERATION = "incineration"
    INCINERATION_ENERGY_RECOVERY = "incineration_energy_recovery"
    RECYCLING = "recycling"
    COMPOSTING = "composting"
    ANAEROBIC_DIGESTION = "anaerobic_digestion"
    OPEN_BURNING = "open_burning"
    REUSE = "reuse"
    OTHER = "other"


class ProductCategory(str, Enum):
    """
    Product categories for end-of-life emissions tracking.

    Categories aligned with typical product material compositions used
    in Scope 3 Category 12 calculations.
    """
    ELECTRONICS = "electronics"
    PACKAGING = "packaging"
    TEXTILES = "textiles"
    FURNITURE = "furniture"
    VEHICLES = "vehicles"
    APPLIANCES = "appliances"
    FOOD_PRODUCTS = "food_products"
    BEVERAGES = "beverages"
    PAPER_PRODUCTS = "paper_products"
    PLASTIC_PRODUCTS = "plastic_products"
    METAL_PRODUCTS = "metal_products"
    GLASS_PRODUCTS = "glass_products"
    CHEMICALS = "chemicals"
    CONSTRUCTION_MATERIALS = "construction_materials"
    MEDICAL_DEVICES = "medical_devices"
    BATTERIES = "batteries"
    MIXED = "mixed"
    OTHER = "other"


class MaterialType(str, Enum):
    """
    Material types for treatment-pathway-specific emissions tracking.

    Tracks emissions by material composition of products reaching
    end-of-life, aligned with waste classification systems.
    """
    PAPER_CARDBOARD = "paper_cardboard"
    PLASTIC_PET = "plastic_pet"
    PLASTIC_HDPE = "plastic_hdpe"
    PLASTIC_PVC = "plastic_pvc"
    PLASTIC_LDPE = "plastic_ldpe"
    PLASTIC_PP = "plastic_pp"
    PLASTIC_PS = "plastic_ps"
    PLASTIC_OTHER = "plastic_other"
    GLASS = "glass"
    STEEL = "steel"
    ALUMINUM = "aluminum"
    COPPER = "copper"
    WOOD = "wood"
    TEXTILE_NATURAL = "textile_natural"
    TEXTILE_SYNTHETIC = "textile_synthetic"
    RUBBER = "rubber"
    FOOD_WASTE = "food_waste"
    GARDEN_WASTE = "garden_waste"
    ELECTRONIC_WASTE = "electronic_waste"
    MIXED = "mixed"
    OTHER = "other"


class AvoidedEmissionType(str, Enum):
    """Types of avoided emissions from end-of-life treatment."""
    RECYCLING = "recycling"
    ENERGY_RECOVERY = "energy_recovery"
    COMPOSTING_SUBSTITUTE = "composting_substitute"
    REUSE = "reuse"


class CalculationStatus(str, Enum):
    """Calculation operation status."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class ComplianceFramework(str, Enum):
    """Compliance frameworks for end-of-life treatment reporting."""
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    EU_WASTE_FRAMEWORK = "eu_waste_framework"
    EPA = "epa"


class ComplianceStatus(str, Enum):
    """Compliance check result status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class PipelineStage(str, Enum):
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


# ===========================================================================
# MetricsCollector -- Thread-safe Singleton
# ===========================================================================


class MetricsCollector:
    """
    Thread-safe singleton metrics collector for End-of-Life Treatment Agent.

    Provides 14 Prometheus metrics for tracking Scope 3 Category 12
    end-of-life treatment of sold products emissions calculations.

    All metrics use the ``gl_eol_`` prefix for namespace isolation within
    the GreenLang Prometheus ecosystem.

    Attributes:
        calculations_total: Counter for total calculation operations
        calculation_duration_seconds: Histogram for calculation durations
        emissions_kg_total: Counter for total emissions in kg CO2e
        products_processed_total: Counter for products processed
        landfill_emissions_total: Counter for landfill emissions
        incineration_emissions_total: Counter for incineration emissions
        recycling_emissions_total: Counter for recycling emissions
        avoided_emissions_total: Counter for avoided emissions
        compliance_checks_total: Counter for compliance checks
        dc_rule_triggers_total: Counter for double-counting rule triggers
        pipeline_stage_duration_seconds: Histogram for pipeline stages
        circularity_score: Gauge for circularity index
        diversion_rate: Gauge for waste diversion rate
        active_calculations: Gauge for active calculations

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_calculation(
        ...     method="waste_type_specific",
        ...     category="electronics",
        ...     status="success",
        ...     duration=0.045,
        ...     emissions_kg=12.5
        ... )
        >>> summary = collector.get_metrics_summary()
        >>> assert summary['calculations'] >= 1
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
            "errors": 0,
            "emissions_kg": 0.0,
            "products_processed": 0,
            "landfill_emissions_kg": 0.0,
            "incineration_emissions_kg": 0.0,
            "recycling_emissions_kg": 0.0,
            "avoided_emissions_kg": 0.0,
            "compliance_checks": 0,
            "dc_triggers": 0,
        }

        # Initialize Prometheus metrics
        self._init_metrics()

        logger.info(
            "MetricsCollector initialized (Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable",
        )

    def _init_metrics(self) -> None:
        """
        Initialize all 14 Prometheus metrics with gl_eol_ prefix.

        Handles re-registration when metrics were previously registered
        (e.g., after reset in tests).
        """
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import REGISTRY

            def _safe_create(metric_cls: type, name: str, *args: Any, **kwargs: Any) -> Any:
                """Create metric, unregistering prior collector on conflict."""
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
        # 1. gl_eol_calculations_total (Counter)
        #    Total end-of-life treatment emission calculations performed.
        #    Labels: method, category, status
        # ------------------------------------------------------------------
        self.calculations_total = _safe_create(
            Counter,
            "gl_eol_calculations_total",
            "Total end-of-life treatment emission calculations performed",
            ["method", "category", "status"],
        )

        # ------------------------------------------------------------------
        # 2. gl_eol_calculation_duration_seconds (Histogram)
        #    Duration of end-of-life calculation operations.
        #    Labels: method, category
        # ------------------------------------------------------------------
        self.calculation_duration_seconds = _safe_create(
            Histogram,
            "gl_eol_calculation_duration_seconds",
            "Duration of end-of-life treatment calculation operations",
            ["method", "category"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        # ------------------------------------------------------------------
        # 3. gl_eol_emissions_kg_total (Counter)
        #    Total end-of-life emissions in kg CO2e.
        #    Labels: treatment_method
        # ------------------------------------------------------------------
        self.emissions_kg_total = _safe_create(
            Counter,
            "gl_eol_emissions_kg_total",
            "Total end-of-life treatment emissions in kg CO2e",
            ["treatment_method"],
        )

        # ------------------------------------------------------------------
        # 4. gl_eol_products_processed_total (Counter)
        #    Total number of sold products processed for EOL calculations.
        #    Labels: category
        # ------------------------------------------------------------------
        self.products_processed_total = _safe_create(
            Counter,
            "gl_eol_products_processed_total",
            "Total sold products processed for end-of-life calculations",
            ["category"],
        )

        # ------------------------------------------------------------------
        # 5. gl_eol_landfill_emissions_total (Counter)
        #    Landfill emissions from end-of-life treatment in kg CO2e.
        #    Labels: material_type
        # ------------------------------------------------------------------
        self.landfill_emissions_total = _safe_create(
            Counter,
            "gl_eol_landfill_emissions_total",
            "Landfill emissions from end-of-life treatment in kg CO2e",
            ["material_type"],
        )

        # ------------------------------------------------------------------
        # 6. gl_eol_incineration_emissions_total (Counter)
        #    Incineration emissions from end-of-life treatment in kg CO2e.
        #    Labels: material_type
        # ------------------------------------------------------------------
        self.incineration_emissions_total = _safe_create(
            Counter,
            "gl_eol_incineration_emissions_total",
            "Incineration emissions from end-of-life treatment in kg CO2e",
            ["material_type"],
        )

        # ------------------------------------------------------------------
        # 7. gl_eol_recycling_emissions_total (Counter)
        #    Recycling process emissions from end-of-life treatment in kg CO2e.
        #    Labels: material_type
        # ------------------------------------------------------------------
        self.recycling_emissions_total = _safe_create(
            Counter,
            "gl_eol_recycling_emissions_total",
            "Recycling process emissions from end-of-life treatment in kg CO2e",
            ["material_type"],
        )

        # ------------------------------------------------------------------
        # 8. gl_eol_avoided_emissions_total (Counter)
        #    Avoided emissions from recycling/energy recovery in kg CO2e.
        #    Labels: type (recycling, energy_recovery, composting_substitute, reuse)
        # ------------------------------------------------------------------
        self.avoided_emissions_total = _safe_create(
            Counter,
            "gl_eol_avoided_emissions_total",
            "Avoided emissions from end-of-life treatment in kg CO2e",
            ["type"],
        )

        # ------------------------------------------------------------------
        # 9. gl_eol_compliance_checks_total (Counter)
        #    Total compliance checks performed.
        #    Labels: framework, status
        # ------------------------------------------------------------------
        self.compliance_checks_total = _safe_create(
            Counter,
            "gl_eol_compliance_checks_total",
            "Total end-of-life treatment compliance checks performed",
            ["framework", "status"],
        )

        # ------------------------------------------------------------------
        # 10. gl_eol_dc_rule_triggers_total (Counter)
        #     Double-counting rule triggers.
        #     Labels: rule_id
        # ------------------------------------------------------------------
        self.dc_rule_triggers_total = _safe_create(
            Counter,
            "gl_eol_dc_rule_triggers_total",
            "Double-counting rule triggers for end-of-life treatment",
            ["rule_id"],
        )

        # ------------------------------------------------------------------
        # 11. gl_eol_pipeline_stage_duration_seconds (Histogram)
        #     Duration of individual pipeline stages.
        #     Labels: stage
        # ------------------------------------------------------------------
        self.pipeline_stage_duration_seconds = _safe_create(
            Histogram,
            "gl_eol_pipeline_stage_duration_seconds",
            "Duration of end-of-life treatment pipeline stages",
            ["stage"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
        )

        # ------------------------------------------------------------------
        # 12. gl_eol_circularity_score (Gauge)
        #     Material Circularity Index score (0.0-1.0).
        #     No labels (single global gauge).
        # ------------------------------------------------------------------
        self.circularity_score = _safe_create(
            Gauge,
            "gl_eol_circularity_score",
            "Material Circularity Index score (0.0-1.0)",
            [],
        )

        # ------------------------------------------------------------------
        # 13. gl_eol_diversion_rate (Gauge)
        #     Waste diversion rate from landfill (0.0-1.0).
        #     No labels (single global gauge).
        # ------------------------------------------------------------------
        self.diversion_rate = _safe_create(
            Gauge,
            "gl_eol_diversion_rate",
            "End-of-life waste diversion rate from landfill (0.0-1.0)",
            [],
        )

        # ------------------------------------------------------------------
        # 14. gl_eol_active_calculations (Gauge)
        #     Currently active (in-progress) calculations.
        #     No labels (single global gauge).
        # ------------------------------------------------------------------
        self.active_calculations = _safe_create(
            Gauge,
            "gl_eol_active_calculations",
            "Currently active end-of-life treatment calculations",
            [],
        )

    # ======================================================================
    # Enum validation helper
    # ======================================================================

    @staticmethod
    def _validate_enum_value(value: str, enum_cls: type, fallback: str) -> str:
        """
        Validate a string against an enum, falling back to a default.

        Args:
            value: Value to validate
            enum_cls: Enum class to validate against
            fallback: Fallback value if validation fails

        Returns:
            Validated value string
        """
        if value is None:
            return fallback
        try:
            enum_cls(value)
            return value
        except ValueError:
            return fallback

    # ======================================================================
    # Primary recording methods
    # ======================================================================

    def record_calculation(
        self,
        method: str,
        category: str,
        status: str = "success",
        duration: Optional[float] = None,
        emissions_kg: Optional[float] = None,
    ) -> None:
        """
        Record an end-of-life treatment emission calculation.

        Args:
            method: Calculation method (waste_type_specific/average_data/
                     producer_specific/hybrid)
            category: Product category (electronics/packaging/textiles/etc)
            status: Calculation status (success/error/partial/skipped)
            duration: Calculation duration in seconds (optional)
            emissions_kg: Emissions in kg CO2e (optional)

        Example:
            >>> collector.record_calculation(
            ...     method="waste_type_specific",
            ...     category="electronics",
            ...     status="success",
            ...     duration=0.045,
            ...     emissions_kg=12.5
            ... )
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethod, CalculationMethod.AVERAGE_DATA.value
            )
            category = self._validate_enum_value(
                category, ProductCategory, ProductCategory.OTHER.value
            )
            status = self._validate_enum_value(
                status, CalculationStatus, CalculationStatus.ERROR.value
            )

            # 1. Increment calculation counter
            self.calculations_total.labels(
                method=method, category=category, status=status
            ).inc()

            # 2. Observe duration if provided
            if duration is not None and duration > 0:
                self.calculation_duration_seconds.labels(
                    method=method, category=category
                ).observe(duration)

            # 3. Record emissions if provided
            if emissions_kg is not None and emissions_kg > 0:
                self.emissions_kg_total.labels(
                    treatment_method=TreatmentMethod.OTHER.value
                ).inc(emissions_kg)

                with self._stats_lock:
                    self._in_memory_stats["emissions_kg"] += emissions_kg

            # 4. Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats["calculations"] += 1
                if status == CalculationStatus.ERROR.value:
                    self._in_memory_stats["errors"] += 1

            logger.debug(
                "Recorded calculation: method=%s, category=%s, status=%s, "
                "duration=%.3fs, emissions=%.2f kg",
                method, category, status,
                duration if duration else 0.0,
                emissions_kg if emissions_kg else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record calculation metrics: %s", e, exc_info=True)

    def record_products_processed(
        self,
        category: str,
        count: int = 1,
    ) -> None:
        """
        Record products processed for end-of-life calculations.

        Args:
            category: Product category
            count: Number of products processed

        Example:
            >>> collector.record_products_processed("electronics", count=50)
        """
        try:
            category = self._validate_enum_value(
                category, ProductCategory, ProductCategory.OTHER.value
            )

            if count > 0:
                self.products_processed_total.labels(category=category).inc(count)

                with self._stats_lock:
                    self._in_memory_stats["products_processed"] += count

        except Exception as e:
            logger.error("Failed to record products processed: %s", e, exc_info=True)

    def record_landfill_emissions(
        self,
        material_type: str,
        emissions_kg: float,
    ) -> None:
        """
        Record landfill emissions from end-of-life treatment.

        Args:
            material_type: Material type being landfilled
            emissions_kg: Emissions in kg CO2e

        Example:
            >>> collector.record_landfill_emissions("paper_cardboard", 5.2)
        """
        try:
            material_type = self._validate_enum_value(
                material_type, MaterialType, MaterialType.OTHER.value
            )

            if emissions_kg is not None and emissions_kg > 0:
                self.landfill_emissions_total.labels(
                    material_type=material_type
                ).inc(emissions_kg)

                self.emissions_kg_total.labels(
                    treatment_method=TreatmentMethod.LANDFILL.value
                ).inc(emissions_kg)

                with self._stats_lock:
                    self._in_memory_stats["landfill_emissions_kg"] += emissions_kg
                    self._in_memory_stats["emissions_kg"] += emissions_kg

        except Exception as e:
            logger.error("Failed to record landfill emissions: %s", e, exc_info=True)

    def record_incineration_emissions(
        self,
        material_type: str,
        emissions_kg: float,
    ) -> None:
        """
        Record incineration emissions from end-of-life treatment.

        Args:
            material_type: Material type being incinerated
            emissions_kg: Emissions in kg CO2e

        Example:
            >>> collector.record_incineration_emissions("plastic_pet", 8.7)
        """
        try:
            material_type = self._validate_enum_value(
                material_type, MaterialType, MaterialType.OTHER.value
            )

            if emissions_kg is not None and emissions_kg > 0:
                self.incineration_emissions_total.labels(
                    material_type=material_type
                ).inc(emissions_kg)

                self.emissions_kg_total.labels(
                    treatment_method=TreatmentMethod.INCINERATION.value
                ).inc(emissions_kg)

                with self._stats_lock:
                    self._in_memory_stats["incineration_emissions_kg"] += emissions_kg
                    self._in_memory_stats["emissions_kg"] += emissions_kg

        except Exception as e:
            logger.error("Failed to record incineration emissions: %s", e, exc_info=True)

    def record_recycling_emissions(
        self,
        material_type: str,
        emissions_kg: float,
    ) -> None:
        """
        Record recycling process emissions from end-of-life treatment.

        Args:
            material_type: Material type being recycled
            emissions_kg: Process emissions in kg CO2e

        Example:
            >>> collector.record_recycling_emissions("aluminum", 3.1)
        """
        try:
            material_type = self._validate_enum_value(
                material_type, MaterialType, MaterialType.OTHER.value
            )

            if emissions_kg is not None and emissions_kg > 0:
                self.recycling_emissions_total.labels(
                    material_type=material_type
                ).inc(emissions_kg)

                self.emissions_kg_total.labels(
                    treatment_method=TreatmentMethod.RECYCLING.value
                ).inc(emissions_kg)

                with self._stats_lock:
                    self._in_memory_stats["recycling_emissions_kg"] += emissions_kg
                    self._in_memory_stats["emissions_kg"] += emissions_kg

        except Exception as e:
            logger.error("Failed to record recycling emissions: %s", e, exc_info=True)

    def record_avoided_emissions(
        self,
        avoided_type: str,
        emissions_kg: float,
    ) -> None:
        """
        Record avoided emissions from recycling or energy recovery.

        Args:
            avoided_type: Type of avoided emissions (recycling/energy_recovery)
            emissions_kg: Avoided emissions in kg CO2e (positive value)

        Example:
            >>> collector.record_avoided_emissions("recycling", 15.0)
        """
        try:
            avoided_type = self._validate_enum_value(
                avoided_type, AvoidedEmissionType, AvoidedEmissionType.RECYCLING.value
            )

            if emissions_kg is not None and emissions_kg > 0:
                self.avoided_emissions_total.labels(type=avoided_type).inc(emissions_kg)

                with self._stats_lock:
                    self._in_memory_stats["avoided_emissions_kg"] += emissions_kg

        except Exception as e:
            logger.error("Failed to record avoided emissions: %s", e, exc_info=True)

    def record_compliance_check(
        self,
        framework: str,
        status: str,
    ) -> None:
        """
        Record a compliance check result.

        Args:
            framework: Compliance framework (ghg_protocol/iso_14064/csrd_esrs/etc)
            status: Check result (compliant/non_compliant/warning/not_applicable)

        Example:
            >>> collector.record_compliance_check("ghg_protocol", "compliant")
        """
        try:
            framework = self._validate_enum_value(
                framework, ComplianceFramework, ComplianceFramework.GHG_PROTOCOL.value
            )
            status = self._validate_enum_value(
                status, ComplianceStatus, ComplianceStatus.WARNING.value
            )

            self.compliance_checks_total.labels(
                framework=framework, status=status
            ).inc()

            with self._stats_lock:
                self._in_memory_stats["compliance_checks"] += 1

        except Exception as e:
            logger.error("Failed to record compliance check: %s", e, exc_info=True)

    def record_dc_rule_trigger(self, rule_id: str) -> None:
        """
        Record a double-counting rule trigger.

        Args:
            rule_id: Rule identifier (e.g., "DC-CAT1-EOL", "DC-CAT5-EOL")

        Example:
            >>> collector.record_dc_rule_trigger("DC-CAT5-EOL")
        """
        try:
            if rule_id is None:
                rule_id = "unknown"
            elif len(rule_id) > 64:
                rule_id = rule_id[:64]

            self.dc_rule_triggers_total.labels(rule_id=rule_id).inc()

            with self._stats_lock:
                self._in_memory_stats["dc_triggers"] += 1

        except Exception as e:
            logger.error("Failed to record DC rule trigger: %s", e, exc_info=True)

    def record_pipeline_stage(
        self,
        stage: str,
        duration: float,
    ) -> None:
        """
        Record pipeline stage duration.

        Args:
            stage: Pipeline stage name
            duration: Stage duration in seconds

        Example:
            >>> collector.record_pipeline_stage("validate", 0.012)
        """
        try:
            stage = self._validate_enum_value(
                stage, PipelineStage, PipelineStage.VALIDATE.value
            )

            if duration is not None and duration > 0:
                self.pipeline_stage_duration_seconds.labels(stage=stage).observe(duration)

        except Exception as e:
            logger.error("Failed to record pipeline stage: %s", e, exc_info=True)

    def set_circularity_score(self, score: float) -> None:
        """
        Set the Material Circularity Index gauge.

        Args:
            score: Circularity score (0.0 to 1.0)

        Example:
            >>> collector.set_circularity_score(0.72)
        """
        try:
            score = max(0.0, min(1.0, score))
            self.circularity_score.set(score)
        except Exception as e:
            logger.error("Failed to set circularity score: %s", e, exc_info=True)

    def set_diversion_rate(self, rate: float) -> None:
        """
        Set the waste diversion rate gauge.

        Args:
            rate: Diversion rate (0.0 to 1.0)

        Example:
            >>> collector.set_diversion_rate(0.85)
        """
        try:
            rate = max(0.0, min(1.0, rate))
            self.diversion_rate.set(rate)
        except Exception as e:
            logger.error("Failed to set diversion rate: %s", e, exc_info=True)

    # ======================================================================
    # Active calculation tracking
    # ======================================================================

    def inc_active(self, amount: float = 1) -> None:
        """Increment active calculations gauge."""
        try:
            self.active_calculations.inc(amount)
        except Exception as e:
            logger.error("Failed to increment active calculations: %s", e, exc_info=True)

    def dec_active(self, amount: float = 1) -> None:
        """Decrement active calculations gauge."""
        try:
            self.active_calculations.dec(amount)
        except Exception as e:
            logger.error("Failed to decrement active calculations: %s", e, exc_info=True)

    # ======================================================================
    # Summary and reset
    # ======================================================================

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get in-memory metrics summary.

        Returns:
            Dictionary with current metrics counters

        Example:
            >>> summary = collector.get_metrics_summary()
            >>> print(summary['calculations'])
        """
        with self._stats_lock:
            return {
                "calculations": self._in_memory_stats["calculations"],
                "errors": self._in_memory_stats["errors"],
                "emissions_kg": self._in_memory_stats["emissions_kg"],
                "products_processed": self._in_memory_stats["products_processed"],
                "landfill_emissions_kg": self._in_memory_stats["landfill_emissions_kg"],
                "incineration_emissions_kg": self._in_memory_stats["incineration_emissions_kg"],
                "recycling_emissions_kg": self._in_memory_stats["recycling_emissions_kg"],
                "avoided_emissions_kg": self._in_memory_stats["avoided_emissions_kg"],
                "compliance_checks": self._in_memory_stats["compliance_checks"],
                "dc_triggers": self._in_memory_stats["dc_triggers"],
                "uptime_seconds": (
                    datetime.now(timezone.utc) - self._start_time
                ).total_seconds(),
                "prometheus_available": PROMETHEUS_AVAILABLE,
            }

    def reset(self) -> None:
        """
        Reset all in-memory counters (for testing).

        Does not reset Prometheus collectors.
        """
        with self._stats_lock:
            for key in self._in_memory_stats:
                if isinstance(self._in_memory_stats[key], float):
                    self._in_memory_stats[key] = 0.0
                else:
                    self._in_memory_stats[key] = 0

            self._start_time = datetime.now(timezone.utc)

        logger.info("MetricsCollector in-memory stats reset")


# ===========================================================================
# THREAD-SAFE SINGLETON ACCESS
# ===========================================================================


_metrics_instance: Optional[MetricsCollector] = None
_metrics_lock: threading.Lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """
    Get singleton MetricsCollector instance.

    Thread-safe lazy initialization using double-checked locking.

    Returns:
        MetricsCollector singleton instance

    Example:
        >>> collector = get_metrics_collector()
        >>> collector.record_calculation(...)
    """
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = MetricsCollector()

    return _metrics_instance


def reset_metrics_collector() -> None:
    """
    Reset singleton MetricsCollector instance.

    Clears the singleton, forcing next get_metrics_collector() call
    to create a fresh instance. Primarily for testing.

    Example:
        >>> reset_metrics_collector()
        >>> collector = get_metrics_collector()  # Fresh instance
    """
    global _metrics_instance

    with _metrics_lock:
        _metrics_instance = None

    logger.info("MetricsCollector singleton reset")


# ===========================================================================
# CONTEXT MANAGERS
# ===========================================================================


@contextmanager
def track_calculation(
    method: str = "waste_type_specific",
    category: str = "other",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for tracking calculation duration and status.

    Automatically records calculation metrics on exit, including duration
    and success/error status.

    Args:
        method: Calculation method
        category: Product category

    Yields:
        Mutable context dictionary for adding emissions and metadata

    Example:
        >>> with track_calculation("waste_type_specific", "electronics") as ctx:
        ...     result = calculate_eol_emissions(product)
        ...     ctx["emissions_kg"] = result.total_co2e_kg
    """
    collector = get_metrics_collector()
    collector.inc_active()

    ctx: Dict[str, Any] = {
        "method": method,
        "category": category,
        "emissions_kg": None,
        "status": "success",
    }

    start = time.monotonic()
    try:
        yield ctx
    except Exception:
        ctx["status"] = "error"
        raise
    finally:
        duration = time.monotonic() - start
        collector.dec_active()

        collector.record_calculation(
            method=ctx["method"],
            category=ctx["category"],
            status=ctx["status"],
            duration=duration,
            emissions_kg=ctx.get("emissions_kg"),
        )


@contextmanager
def track_pipeline_stage(
    stage: str,
) -> Generator[None, None, None]:
    """
    Context manager for tracking pipeline stage duration.

    Automatically records pipeline stage duration on exit.

    Args:
        stage: Pipeline stage name

    Yields:
        None

    Example:
        >>> with track_pipeline_stage("validate"):
        ...     validated = validate_input(data)
    """
    collector = get_metrics_collector()
    start = time.monotonic()
    try:
        yield
    finally:
        duration = time.monotonic() - start
        collector.record_pipeline_stage(stage=stage, duration=duration)


# ===========================================================================
# MODULE EXPORTS
# ===========================================================================

__all__ = [
    # Enumerations
    "CalculationMethod",
    "TreatmentMethod",
    "ProductCategory",
    "MaterialType",
    "AvoidedEmissionType",
    "CalculationStatus",
    "ComplianceFramework",
    "ComplianceStatus",
    "PipelineStage",
    # Collector class
    "MetricsCollector",
    # Singleton access
    "get_metrics_collector",
    "reset_metrics_collector",
    # Context managers
    "track_calculation",
    "track_pipeline_stage",
    # Constants
    "PROMETHEUS_AVAILABLE",
]
