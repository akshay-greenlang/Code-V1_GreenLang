# -*- coding: utf-8 -*-
"""
Use of Sold Products Prometheus Metrics - AGENT-MRV-024

14 Prometheus metrics with gl_usp_ prefix for monitoring the
GL-MRV-S3-011 Use of Sold Products Agent.

This module provides Prometheus metrics tracking for use-of-sold-products
emissions calculations (Scope 3, Category 11) including direct emissions
(fuel-consuming, refrigerant-containing, chemical products), indirect
emissions (electricity-consuming, heating, steam products), and fuels
and feedstocks sold for combustion.

Thread-safe singleton pattern with graceful fallback if Prometheus
unavailable.

Metrics prefix: gl_usp_

14 Prometheus Metrics:
    1.  gl_usp_calculations_total              - Counter: total calculations performed
    2.  gl_usp_calculation_duration_seconds     - Histogram: calculation duration
    3.  gl_usp_emissions_kg_total              - Counter: total emissions in kgCO2e
    4.  gl_usp_products_processed_total        - Counter: total products processed
    5.  gl_usp_direct_emissions_total          - Counter: direct emissions by type
    6.  gl_usp_indirect_emissions_total        - Counter: indirect emissions by type
    7.  gl_usp_fuel_sales_total                - Counter: fuel sales processed
    8.  gl_usp_compliance_checks_total         - Counter: compliance checks
    9.  gl_usp_dc_rule_triggers_total          - Counter: double-counting rule triggers
    10. gl_usp_pipeline_stage_duration_seconds  - Histogram: pipeline stage duration
    11. gl_usp_lifetime_years_avg              - Gauge: average product lifetime
    12. gl_usp_dqi_score                       - Gauge: current DQI score
    13. gl_usp_uncertainty_width               - Gauge: uncertainty interval width
    14. gl_usp_active_calculations             - Gauge: active calculations

GHG Protocol Scope 3 Category 11 covers use of sold products:
    A. Direct use-phase emissions from products that directly emit GHGs
       during use (vehicles, appliances with refrigerants, chemical products).
    B. Indirect use-phase emissions from products that consume energy
       during use (electricity, heating, steam).
    C. Fuels and feedstocks sold by the reporting company that produce
       emissions when combusted or processed by end users.

Calculation methods:
    - Direct: product emissions = EF x activity x lifetime x units sold
    - Indirect: product emissions = energy consumption x grid EF x lifetime x units sold
    - Fuels: fuel emissions = volume sold x fuel EF

Example:
    >>> metrics = get_metrics_collector()
    >>> metrics.record_calculation(
    ...     method="direct_fuel",
    ...     category="fuel_consuming",
    ...     status="success",
    ...     duration=0.045,
    ...     co2e_kg=12500.0
    ... )

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-011
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
# Enumerations -- Use of Sold Products domain-specific label value sets
# ===========================================================================


class CalculationMethodLabel(str, Enum):
    """
    Calculation methods for use-of-sold-products emissions.

    GHG Protocol Scope 3 Category 11 distinguishes three emission types
    with different calculation approaches:
        - direct_fuel: Fuel-consuming products (vehicles, generators)
        - direct_refrigerant: Refrigerant-containing products (HVAC, fridges)
        - direct_chemical: Chemical products (solvents, aerosols)
        - indirect_electricity: Electricity-consuming products
        - indirect_heating: Heating-consuming products
        - indirect_steam: Steam-consuming products
        - fuel_sales: Fuels sold for combustion
        - feedstock: Feedstocks sold for processing
    """
    DIRECT_FUEL = "direct_fuel"
    DIRECT_REFRIGERANT = "direct_refrigerant"
    DIRECT_CHEMICAL = "direct_chemical"
    INDIRECT_ELECTRICITY = "indirect_electricity"
    INDIRECT_HEATING = "indirect_heating"
    INDIRECT_STEAM = "indirect_steam"
    FUEL_SALES = "fuel_sales"
    FEEDSTOCK = "feedstock"


class ProductCategoryLabel(str, Enum):
    """
    Product categories tracked for use-of-sold-products emissions.

    Covers the primary product types defined in the GHG Protocol
    Technical Guidance for Calculating Scope 3 Emissions, Category 11.
    """
    FUEL_CONSUMING = "fuel_consuming"
    REFRIGERANT_CONTAINING = "refrigerant_containing"
    CHEMICAL = "chemical"
    ELECTRICITY_CONSUMING = "electricity_consuming"
    HEATING_CONSUMING = "heating_consuming"
    STEAM_CONSUMING = "steam_consuming"
    FUEL_PRODUCT = "fuel_product"
    FEEDSTOCK_PRODUCT = "feedstock_product"
    MIXED = "mixed"


class CalculationStatusLabel(str, Enum):
    """Calculation operation status for use-of-sold-products calculations."""
    SUCCESS = "success"
    ERROR = "error"
    VALIDATION_ERROR = "validation_error"


class EmissionTypeLabel(str, Enum):
    """Emission type labels for high-level tracking."""
    DIRECT = "direct"
    INDIRECT = "indirect"
    FUELS = "fuels"


class DirectEmissionTypeLabel(str, Enum):
    """Direct emission sub-type labels."""
    FUEL = "fuel"
    REFRIGERANT = "refrigerant"
    CHEMICAL = "chemical"


class IndirectEmissionTypeLabel(str, Enum):
    """Indirect emission sub-type labels."""
    ELECTRICITY = "electricity"
    HEATING = "heating"
    STEAM = "steam"


class FuelTypeLabel(str, Enum):
    """Fuel type labels for fuel sales tracking."""
    GASOLINE = "gasoline"
    DIESEL = "diesel"
    NATURAL_GAS = "natural_gas"
    COAL = "coal"
    LPG = "lpg"
    KEROSENE = "kerosene"
    FUEL_OIL = "fuel_oil"
    BIOFUEL = "biofuel"
    HYDROGEN = "hydrogen"
    OTHER = "other"


class FrameworkLabel(str, Enum):
    """Compliance frameworks for use-of-sold-products reporting."""
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD = "csrd"
    CDP = "cdp"
    SBTI = "sbti"
    GRI = "gri"
    SEC_CLIMATE = "sec_climate"


class ComplianceStatusLabel(str, Enum):
    """Compliance check result status."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class PipelineStageLabel(str, Enum):
    """Pipeline stage labels for duration tracking."""
    VALIDATE = "validate"
    CLASSIFY = "classify"
    NORMALIZE = "normalize"
    RESOLVE_EFS = "resolve_efs"
    CALCULATE_DIRECT = "calculate_direct"
    CALCULATE_INDIRECT = "calculate_indirect"
    CALCULATE_FUELS = "calculate_fuels"
    APPLY_LIFETIME = "apply_lifetime"
    APPLY_DEGRADATION = "apply_degradation"
    COMPLIANCE = "compliance"
    AGGREGATE = "aggregate"
    SEAL = "seal"


class DCRuleLabel(str, Enum):
    """Double-counting prevention rule labels."""
    DC_SCOPE1_OVERLAP = "dc_scope1_overlap"
    DC_CAT1_OVERLAP = "dc_cat1_overlap"
    DC_CAT12_OVERLAP = "dc_cat12_overlap"
    DC_FUEL_FEEDSTOCK = "dc_fuel_feedstock"
    DC_BOUNDARY = "dc_boundary"


# ===========================================================================
# MetricsCollector -- Thread-safe Singleton
# ===========================================================================


class MetricsCollector:
    """
    Thread-safe singleton metrics collector for Use of Sold Products Agent (MRV-024).

    Provides 14 Prometheus metrics for tracking Scope 3 Category 11
    use-of-sold-products emissions calculations, covering direct emissions,
    indirect emissions, and fuels/feedstocks sold.

    All metrics use the ``gl_usp_`` prefix to ensure namespace isolation
    within the GreenLang Prometheus ecosystem.

    Scope 3 Category 11 Sub-Categories Tracked:
        A. Direct use-phase emissions (fuel, refrigerant, chemical)
        B. Indirect use-phase emissions (electricity, heating, steam)
        C. Fuels and feedstocks sold

    Attributes:
        calculations_total: Counter for total calculation operations
        calculation_duration_seconds: Histogram for operation durations
        emissions_kg_total: Counter for total emissions in kgCO2e
        products_processed_total: Counter for products processed
        direct_emissions_total: Counter for direct emissions by type
        indirect_emissions_total: Counter for indirect emissions by type
        fuel_sales_total: Counter for fuel sales processed
        compliance_checks_total: Counter for compliance checks
        dc_rule_triggers_total: Counter for double-counting rule triggers
        pipeline_stage_duration_seconds: Histogram for pipeline stage durations
        lifetime_years_avg: Gauge for average product lifetime
        dqi_score: Gauge for current DQI score
        uncertainty_width: Gauge for uncertainty interval width
        active_calculations: Gauge for active calculations

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_calculation(
        ...     method="direct_fuel",
        ...     category="fuel_consuming",
        ...     status="success",
        ...     duration=0.045,
        ...     co2e_kg=12500.0
        ... )
        >>> summary = collector.get_stats()
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
        self._start_time: datetime = datetime.utcnow()
        self._stats_lock: threading.Lock = threading.Lock()
        self._in_memory_stats: Dict[str, Any] = {
            "calculations": 0,
            "emissions_kg_co2e": 0.0,
            "products_processed": 0,
            "direct_emissions": 0,
            "indirect_emissions": 0,
            "fuel_sales": 0,
            "compliance_checks": 0,
            "dc_rule_triggers": 0,
            "errors": 0,
        }

        self._init_metrics()

        logger.info(
            "MetricsCollector initialized (Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable",
        )

    def _init_metrics(self) -> None:
        """
        Initialize all 14 Prometheus metrics with gl_usp_ prefix.

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
        # 1. gl_usp_calculations_total (Counter)
        #    Total use-of-sold-products emission calculations performed.
        #    Labels:
        #      - method: direct_fuel, direct_refrigerant, direct_chemical,
        #                indirect_electricity, indirect_heating, indirect_steam,
        #                fuel_sales, feedstock
        #      - category: fuel_consuming, refrigerant_containing, chemical,
        #                  electricity_consuming, etc.
        #      - status: success, error, validation_error
        # ------------------------------------------------------------------
        self.calculations_total = _safe_create(
            Counter,
            "gl_usp_calculations_total",
            "Total use-of-sold-products emission calculations performed",
            ["method", "category", "status"],
        )

        # ------------------------------------------------------------------
        # 2. gl_usp_calculation_duration_seconds (Histogram)
        #    Duration of calculation operations in seconds.
        #    Labels:
        #      - method: Calculation method used
        #      - category: Product category
        #    Buckets tuned for typical use-of-sold-products calculation
        #    latencies, from simple fuel EF lookups to complex lifetime
        #    modeling with degradation curves.
        # ------------------------------------------------------------------
        self.calculation_duration_seconds = _safe_create(
            Histogram,
            "gl_usp_calculation_duration_seconds",
            "Duration of use-of-sold-products calculation operations",
            ["method", "category"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        # ------------------------------------------------------------------
        # 3. gl_usp_emissions_kg_total (Counter)
        #    Total emissions calculated in kilograms CO2-equivalent.
        #    Labels:
        #      - emission_type: direct, indirect, fuels
        #    Tracks cumulative emissions output by emission type for rate
        #    calculation and type breakdown.
        # ------------------------------------------------------------------
        self.emissions_kg_total = _safe_create(
            Counter,
            "gl_usp_emissions_kg_total",
            "Total use-of-sold-products emissions calculated in kgCO2e",
            ["emission_type"],
        )

        # ------------------------------------------------------------------
        # 4. gl_usp_products_processed_total (Counter)
        #    Total products processed.
        #    Labels:
        #      - category: Product category
        #    Tracks product volume by category for throughput analysis.
        # ------------------------------------------------------------------
        self.products_processed_total = _safe_create(
            Counter,
            "gl_usp_products_processed_total",
            "Total products processed for use-of-sold-products calculations",
            ["category"],
        )

        # ------------------------------------------------------------------
        # 5. gl_usp_direct_emissions_total (Counter)
        #    Total direct use-phase emissions by sub-type.
        #    Labels:
        #      - type: fuel, refrigerant, chemical
        #    Tracks direct emission volume by sub-type for product-level
        #    hot-spot analysis.
        # ------------------------------------------------------------------
        self.direct_emissions_total = _safe_create(
            Counter,
            "gl_usp_direct_emissions_total",
            "Total direct use-phase emissions by sub-type in kgCO2e",
            ["type"],
        )

        # ------------------------------------------------------------------
        # 6. gl_usp_indirect_emissions_total (Counter)
        #    Total indirect use-phase emissions by sub-type.
        #    Labels:
        #      - type: electricity, heating, steam
        #    Tracks indirect emission volume by energy type for grid-factor
        #    sensitivity analysis and product energy labeling.
        # ------------------------------------------------------------------
        self.indirect_emissions_total = _safe_create(
            Counter,
            "gl_usp_indirect_emissions_total",
            "Total indirect use-phase emissions by sub-type in kgCO2e",
            ["type"],
        )

        # ------------------------------------------------------------------
        # 7. gl_usp_fuel_sales_total (Counter)
        #    Total fuel sales processed by fuel type.
        #    Labels:
        #      - fuel_type: gasoline, diesel, natural_gas, coal, lpg,
        #                   kerosene, fuel_oil, biofuel, hydrogen, other
        #    Tracks fuel sales volume for portfolio emission intensity
        #    monitoring (critical for O&G companies).
        # ------------------------------------------------------------------
        self.fuel_sales_total = _safe_create(
            Counter,
            "gl_usp_fuel_sales_total",
            "Total fuel sales processed for use-of-sold-products",
            ["fuel_type"],
        )

        # ------------------------------------------------------------------
        # 8. gl_usp_compliance_checks_total (Counter)
        #    Total compliance checks performed.
        #    Labels:
        #      - framework: ghg_protocol, iso_14064, csrd, cdp, sbti,
        #                   gri, sec_climate
        #      - status: compliant, partially_compliant, non_compliant,
        #                warning, not_applicable
        # ------------------------------------------------------------------
        self.compliance_checks_total = _safe_create(
            Counter,
            "gl_usp_compliance_checks_total",
            "Total use-of-sold-products compliance checks performed",
            ["framework", "status"],
        )

        # ------------------------------------------------------------------
        # 9. gl_usp_dc_rule_triggers_total (Counter)
        #    Total double-counting prevention rule triggers.
        #    Labels:
        #      - rule_id: dc_scope1_overlap, dc_cat1_overlap,
        #                 dc_cat12_overlap, dc_fuel_feedstock, dc_boundary
        #    Tracks how often double-counting prevention rules fire.
        # ------------------------------------------------------------------
        self.dc_rule_triggers_total = _safe_create(
            Counter,
            "gl_usp_dc_rule_triggers_total",
            "Total double-counting prevention rule triggers",
            ["rule_id"],
        )

        # ------------------------------------------------------------------
        # 10. gl_usp_pipeline_stage_duration_seconds (Histogram)
        #     Duration of individual pipeline stages in seconds.
        #     Labels:
        #       - stage: validate, classify, normalize, resolve_efs,
        #                calculate_direct, calculate_indirect,
        #                calculate_fuels, apply_lifetime, apply_degradation,
        #                compliance, aggregate, seal
        # ------------------------------------------------------------------
        self.pipeline_stage_duration_seconds = _safe_create(
            Histogram,
            "gl_usp_pipeline_stage_duration_seconds",
            "Duration of use-of-sold-products pipeline stages",
            ["stage"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0),
        )

        # ------------------------------------------------------------------
        # 11. gl_usp_lifetime_years_avg (Gauge)
        #     Average product lifetime in years across recent calculations.
        #     No labels -- single gauge for overall lifetime monitoring.
        # ------------------------------------------------------------------
        self.lifetime_years_avg = _safe_create(
            Gauge,
            "gl_usp_lifetime_years_avg",
            "Average product lifetime in years",
        )

        # ------------------------------------------------------------------
        # 12. gl_usp_dqi_score (Gauge)
        #     Current data quality indicator score (1=best, 5=worst).
        #     No labels -- single gauge for overall data quality monitoring.
        # ------------------------------------------------------------------
        self.dqi_score = _safe_create(
            Gauge,
            "gl_usp_dqi_score",
            "Current data quality indicator score for use-of-sold-products",
        )

        # ------------------------------------------------------------------
        # 13. gl_usp_uncertainty_width (Gauge)
        #     Current uncertainty interval width as a percentage.
        #     No labels -- single gauge for uncertainty monitoring.
        # ------------------------------------------------------------------
        self.uncertainty_width = _safe_create(
            Gauge,
            "gl_usp_uncertainty_width",
            "Uncertainty interval width percentage for use-of-sold-products",
        )

        # ------------------------------------------------------------------
        # 14. gl_usp_active_calculations (Gauge)
        #     Number of currently active (in-flight) calculations.
        #     No labels -- single gauge for concurrency monitoring.
        # ------------------------------------------------------------------
        self.active_calculations = _safe_create(
            Gauge,
            "gl_usp_active_calculations",
            "Number of currently active use-of-sold-products calculations",
        )

        # ------------------------------------------------------------------
        # Agent info metric (non-numeric metadata)
        # ------------------------------------------------------------------
        self.agent_info = _safe_create(
            Info,
            "gl_usp_agent",
            "Use of Sold Products Agent metadata",
        )
        if PROMETHEUS_AVAILABLE:
            try:
                self.agent_info.info({
                    "agent_id": "GL-MRV-S3-011",
                    "version": "1.0.0",
                    "scope": "scope_3_category_11",
                    "description": "Use of Sold Products emissions calculator",
                })
            except Exception:
                pass

    # ======================================================================
    # Primary recording methods
    # ======================================================================

    def record_calculation(
        self,
        method: str,
        category: str,
        status: str,
        duration: float,
        co2e_kg: float,
        emission_type: str = "direct",
    ) -> None:
        """
        Record a use-of-sold-products emission calculation operation.

        This is the primary recording method that increments the calculations
        counter, observes the duration histogram, and tracks emissions output.

        Args:
            method: Calculation method (direct_fuel/direct_refrigerant/
                     direct_chemical/indirect_electricity/indirect_heating/
                     indirect_steam/fuel_sales/feedstock)
            category: Product category (fuel_consuming/refrigerant_containing/
                       chemical/electricity_consuming/etc.)
            status: Calculation status (success/error/validation_error)
            duration: Operation duration in seconds
            co2e_kg: Emissions calculated in kgCO2e
            emission_type: Emission type (direct/indirect/fuels)

        Example:
            >>> collector.record_calculation(
            ...     method="direct_fuel",
            ...     category="fuel_consuming",
            ...     status="success",
            ...     duration=0.045,
            ...     co2e_kg=12500.0,
            ...     emission_type="direct"
            ... )
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethodLabel, CalculationMethodLabel.DIRECT_FUEL.value
            )
            category = self._validate_enum_value(
                category, ProductCategoryLabel, ProductCategoryLabel.FUEL_CONSUMING.value
            )
            status = self._validate_enum_value(
                status, CalculationStatusLabel, CalculationStatusLabel.ERROR.value
            )
            emission_type = self._validate_enum_value(
                emission_type, EmissionTypeLabel, EmissionTypeLabel.DIRECT.value
            )

            self.calculations_total.labels(
                method=method, category=category, status=status
            ).inc()

            if duration is not None and duration > 0:
                self.calculation_duration_seconds.labels(
                    method=method, category=category
                ).observe(duration)

            if co2e_kg is not None and co2e_kg > 0:
                self.emissions_kg_total.labels(
                    emission_type=emission_type
                ).inc(co2e_kg)

                with self._stats_lock:
                    self._in_memory_stats["emissions_kg_co2e"] += co2e_kg

            with self._stats_lock:
                self._in_memory_stats["calculations"] += 1
                if status == CalculationStatusLabel.ERROR.value:
                    self._in_memory_stats["errors"] += 1

            logger.debug(
                "Recorded calculation: method=%s, category=%s, status=%s, "
                "duration=%.3fs, co2e=%.2f kgCO2e, type=%s",
                method, category, status,
                duration if duration else 0.0,
                co2e_kg if co2e_kg else 0.0,
                emission_type,
            )

        except Exception as e:
            logger.error("Failed to record calculation metrics: %s", e, exc_info=True)

    def record_product(self, category: str) -> None:
        """
        Record a product processed.

        Args:
            category: Product category

        Example:
            >>> collector.record_product(category="fuel_consuming")
        """
        try:
            category = self._validate_enum_value(
                category, ProductCategoryLabel, ProductCategoryLabel.FUEL_CONSUMING.value
            )

            self.products_processed_total.labels(category=category).inc()

            with self._stats_lock:
                self._in_memory_stats["products_processed"] += 1

            logger.debug("Recorded product: category=%s", category)

        except Exception as e:
            logger.error("Failed to record product metrics: %s", e, exc_info=True)

    def record_direct_emission(
        self,
        emission_subtype: str,
        co2e_kg: float,
    ) -> None:
        """
        Record a direct use-phase emission calculation.

        Args:
            emission_subtype: Direct emission sub-type (fuel/refrigerant/chemical)
            co2e_kg: Emissions in kgCO2e

        Example:
            >>> collector.record_direct_emission(
            ...     emission_subtype="fuel",
            ...     co2e_kg=8500.0
            ... )
        """
        try:
            emission_subtype = self._validate_enum_value(
                emission_subtype, DirectEmissionTypeLabel, DirectEmissionTypeLabel.FUEL.value
            )

            if co2e_kg is not None and co2e_kg > 0:
                self.direct_emissions_total.labels(type=emission_subtype).inc(co2e_kg)

            with self._stats_lock:
                self._in_memory_stats["direct_emissions"] += 1

            logger.debug(
                "Recorded direct emission: type=%s, co2e=%.2f kgCO2e",
                emission_subtype,
                co2e_kg if co2e_kg else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record direct emission metrics: %s", e, exc_info=True)

    def record_indirect_emission(
        self,
        emission_subtype: str,
        co2e_kg: float,
    ) -> None:
        """
        Record an indirect use-phase emission calculation.

        Args:
            emission_subtype: Indirect emission sub-type (electricity/heating/steam)
            co2e_kg: Emissions in kgCO2e

        Example:
            >>> collector.record_indirect_emission(
            ...     emission_subtype="electricity",
            ...     co2e_kg=3200.0
            ... )
        """
        try:
            emission_subtype = self._validate_enum_value(
                emission_subtype, IndirectEmissionTypeLabel, IndirectEmissionTypeLabel.ELECTRICITY.value
            )

            if co2e_kg is not None and co2e_kg > 0:
                self.indirect_emissions_total.labels(type=emission_subtype).inc(co2e_kg)

            with self._stats_lock:
                self._in_memory_stats["indirect_emissions"] += 1

            logger.debug(
                "Recorded indirect emission: type=%s, co2e=%.2f kgCO2e",
                emission_subtype,
                co2e_kg if co2e_kg else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record indirect emission metrics: %s", e, exc_info=True)

    def record_fuel_sale(self, fuel_type: str) -> None:
        """
        Record a fuel sale processed.

        Args:
            fuel_type: Fuel type (gasoline/diesel/natural_gas/coal/etc.)

        Example:
            >>> collector.record_fuel_sale(fuel_type="gasoline")
        """
        try:
            fuel_type = self._validate_enum_value(
                fuel_type, FuelTypeLabel, FuelTypeLabel.OTHER.value
            )

            self.fuel_sales_total.labels(fuel_type=fuel_type).inc()

            with self._stats_lock:
                self._in_memory_stats["fuel_sales"] += 1

            logger.debug("Recorded fuel sale: fuel_type=%s", fuel_type)

        except Exception as e:
            logger.error("Failed to record fuel sale metrics: %s", e, exc_info=True)

    def record_compliance_check(
        self,
        framework: str,
        status: str,
    ) -> None:
        """
        Record a compliance check result.

        Args:
            framework: Compliance framework (ghg_protocol/iso_14064/csrd/
                        cdp/sbti/gri/sec_climate)
            status: Check result (compliant/partially_compliant/non_compliant/
                     warning/not_applicable)

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
        Record a double-counting prevention rule trigger.

        Args:
            rule_id: Double-counting rule identifier

        Example:
            >>> collector.record_dc_rule_trigger(rule_id="dc_scope1_overlap")
        """
        try:
            rule_id = self._validate_enum_value(
                rule_id, DCRuleLabel, DCRuleLabel.DC_BOUNDARY.value
            )

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
            stage: Pipeline stage name
            duration: Stage duration in seconds

        Example:
            >>> collector.record_pipeline_stage(stage="validate", duration=0.005)
        """
        try:
            stage = self._validate_enum_value(
                stage, PipelineStageLabel, PipelineStageLabel.VALIDATE.value
            )

            if duration is not None and duration > 0:
                self.pipeline_stage_duration_seconds.labels(stage=stage).observe(duration)

            logger.debug(
                "Recorded pipeline stage: stage=%s, duration=%.3fs",
                stage, duration if duration else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record pipeline stage metrics: %s", e, exc_info=True)

    def update_lifetime_avg(self, lifetime_years: float) -> None:
        """
        Update the average product lifetime gauge.

        Args:
            lifetime_years: Average lifetime in years

        Example:
            >>> collector.update_lifetime_avg(lifetime_years=12.5)
        """
        try:
            if lifetime_years is not None and lifetime_years >= 0:
                self.lifetime_years_avg.set(lifetime_years)
                logger.debug("Updated lifetime_years_avg: %.1f", lifetime_years)
        except Exception as e:
            logger.error("Failed to update lifetime avg: %s", e, exc_info=True)

    def update_dqi_score(self, score: float) -> None:
        """
        Update the DQI score gauge.

        Args:
            score: DQI score (1=best, 5=worst)

        Example:
            >>> collector.update_dqi_score(score=2.3)
        """
        try:
            if score is not None and 0 <= score <= 5:
                self.dqi_score.set(score)
                logger.debug("Updated dqi_score: %.2f", score)
        except Exception as e:
            logger.error("Failed to update DQI score: %s", e, exc_info=True)

    def update_uncertainty_width(self, width_pct: float) -> None:
        """
        Update the uncertainty interval width gauge.

        Args:
            width_pct: Uncertainty width as percentage

        Example:
            >>> collector.update_uncertainty_width(width_pct=15.2)
        """
        try:
            if width_pct is not None and width_pct >= 0:
                self.uncertainty_width.set(width_pct)
                logger.debug("Updated uncertainty_width: %.1f%%", width_pct)
        except Exception as e:
            logger.error("Failed to update uncertainty width: %s", e, exc_info=True)

    # ======================================================================
    # Summary and lifecycle
    # ======================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get a summary of metrics collected since initialization.

        Returns:
            Dictionary with metrics summary including counts, uptime, rates,
            and emission type breakdown.

        Example:
            >>> stats = collector.get_stats()
            >>> print(stats['calculations'])
            42
        """
        try:
            uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()
            uptime_hours = uptime_seconds / 3600 if uptime_seconds > 0 else 1.0

            with self._stats_lock:
                stats_snapshot = dict(self._in_memory_stats)

            summary: Dict[str, Any] = {
                "prometheus_available": PROMETHEUS_AVAILABLE,
                "agent": "use_of_sold_products",
                "agent_id": "GL-MRV-S3-011",
                "prefix": "gl_usp_",
                "scope": "Scope 3 Category 11",
                "description": "Use of Sold Products",
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
                    "products_per_hour": (
                        stats_snapshot["products_processed"] / uptime_hours
                    ),
                    "direct_emissions_per_hour": (
                        stats_snapshot["direct_emissions"] / uptime_hours
                    ),
                    "indirect_emissions_per_hour": (
                        stats_snapshot["indirect_emissions"] / uptime_hours
                    ),
                    "fuel_sales_per_hour": (
                        stats_snapshot["fuel_sales"] / uptime_hours
                    ),
                    "compliance_checks_per_hour": (
                        stats_snapshot["compliance_checks"] / uptime_hours
                    ),
                    "errors_per_hour": (
                        stats_snapshot["errors"] / uptime_hours
                    ),
                },
                "emission_breakdown": {
                    "direct_emissions": stats_snapshot["direct_emissions"],
                    "indirect_emissions": stats_snapshot["indirect_emissions"],
                    "fuel_sales": stats_snapshot["fuel_sales"],
                },
                "operational": {
                    "compliance_checks": stats_snapshot["compliance_checks"],
                    "dc_rule_triggers": stats_snapshot["dc_rule_triggers"],
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
                "agent": "use_of_sold_products",
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
            >>> MetricsCollector.reset()
            >>> collector = MetricsCollector()  # Fresh instance
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

        Note: This only resets the in-memory counters used for get_stats().
        Prometheus metrics are cumulative and cannot be reset without
        restarting the process.

        Example:
            >>> collector.reset_stats()
            >>> stats = collector.get_stats()
            >>> assert stats['calculations'] == 0
        """
        try:
            with self._stats_lock:
                self._in_memory_stats = {
                    "calculations": 0,
                    "emissions_kg_co2e": 0.0,
                    "products_processed": 0,
                    "direct_emissions": 0,
                    "indirect_emissions": 0,
                    "fuel_sales": 0,
                    "compliance_checks": 0,
                    "dc_rule_triggers": 0,
                    "errors": 0,
                }
            self._start_time = datetime.utcnow()

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
            >>> MetricsCollector._validate_enum_value(
            ...     "direct_fuel", CalculationMethodLabel, "direct_fuel"
            ... )
            'direct_fuel'
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

    Thread-safe accessor for the global metrics instance. Prefer this
    function over direct instantiation for consistency across the
    use-of-sold-products agent codebase.

    Returns:
        MetricsCollector singleton instance

    Example:
        >>> from greenlang.use_of_sold_products.metrics import get_metrics_collector
        >>> collector = get_metrics_collector()
        >>> collector.record_calculation(
        ...     method="direct_fuel",
        ...     category="fuel_consuming",
        ...     status="success",
        ...     duration=0.045,
        ...     co2e_kg=12500.0
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

    Example:
        >>> from greenlang.use_of_sold_products.metrics import reset_metrics_collector
        >>> reset_metrics_collector()
    """
    MetricsCollector.reset()


# ===========================================================================
# Context manager helpers
# ===========================================================================


@contextmanager
def track_calculation(
    method: str = "direct_fuel",
    category: str = "fuel_consuming",
    emission_type: str = "direct",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a calculation's lifecycle.

    Automatically increments/decrements the active calculation gauge,
    measures wall-clock duration, and records the calculation when the
    context exits. The caller can set ``context['co2e_kg']`` inside the
    block to include emissions in the recorded metric.

    Args:
        method: Calculation method (default: "direct_fuel")
        category: Product category (default: "fuel_consuming")
        emission_type: Emission type (default: "direct")

    Yields:
        Mutable context dict. Set ``context['co2e_kg']`` to record emissions.

    Example:
        >>> with track_calculation(method="indirect_electricity",
        ...                        category="electricity_consuming",
        ...                        emission_type="indirect") as ctx:
        ...     result = calculate_indirect_emissions(product)
        ...     ctx['co2e_kg'] = result.co2e_kg
    """
    collector = get_metrics_collector()
    context: Dict[str, Any] = {
        "co2e_kg": 0.0,
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
            category=category,
            status=context["status"],
            duration=duration,
            co2e_kg=context.get("co2e_kg", 0.0),
            emission_type=emission_type,
        )


@contextmanager
def track_pipeline_stage(stage: str) -> Generator[None, None, None]:
    """
    Context manager that tracks a pipeline stage's duration.

    Automatically measures wall-clock duration and records the stage
    when the context exits.

    Args:
        stage: Pipeline stage name

    Yields:
        None

    Example:
        >>> with track_pipeline_stage("validate"):
        ...     validated_data = validate_input(raw_data)
    """
    collector = get_metrics_collector()
    start = time.monotonic()

    try:
        yield
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
    "ProductCategoryLabel",
    "CalculationStatusLabel",
    "EmissionTypeLabel",
    "DirectEmissionTypeLabel",
    "IndirectEmissionTypeLabel",
    "FuelTypeLabel",
    "FrameworkLabel",
    "ComplianceStatusLabel",
    "PipelineStageLabel",
    "DCRuleLabel",
    # Singleton class
    "MetricsCollector",
    # Module-level accessors
    "get_metrics_collector",
    "reset_metrics_collector",
    # Context managers
    "track_calculation",
    "track_pipeline_stage",
]
