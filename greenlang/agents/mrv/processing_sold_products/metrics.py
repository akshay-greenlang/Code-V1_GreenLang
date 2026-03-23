# -*- coding: utf-8 -*-
"""
Processing of Sold Products Prometheus Metrics - AGENT-MRV-023

14 Prometheus metrics with gl_psp_ prefix for monitoring the
GL-MRV-S3-010 Processing of Sold Products Agent.

This module provides Prometheus metrics tracking for processing of sold
products emissions calculations (Scope 3, Category 10) including
site-specific (direct/energy/fuel), average-data, and spend-based
calculation methods across 12 intermediate product categories, 18
processing types, and 8 multi-step processing chains.

Thread-safe singleton pattern with graceful fallback if Prometheus
unavailable.

Metrics prefix: gl_psp_

14 Prometheus Metrics:
    1.  gl_psp_calculations_total                - Counter: total calculations by method/category/status
    2.  gl_psp_calculation_duration_seconds      - Histogram: calculation duration in seconds
    3.  gl_psp_emissions_kg_total                - Counter: total emissions calculated in kgCO2e
    4.  gl_psp_products_processed_total          - Counter: products processed by category
    5.  gl_psp_ef_lookups_total                  - Counter: emission factor lookups by source/hit
    6.  gl_psp_grid_ef_lookups_total             - Counter: grid EF lookups by region
    7.  gl_psp_chain_calculations_total          - Counter: multi-step processing chain calculations
    8.  gl_psp_compliance_checks_total           - Counter: compliance checks by framework/status
    9.  gl_psp_dc_rule_triggers_total            - Counter: double-counting rule triggers by rule_id
    10. gl_psp_pipeline_stage_duration_seconds   - Histogram: pipeline stage duration by stage name
    11. gl_psp_batch_jobs_total                  - Counter: batch jobs by status
    12. gl_psp_dqi_score                         - Gauge: latest DQI composite score
    13. gl_psp_uncertainty_width                 - Gauge: latest uncertainty CI width
    14. gl_psp_active_calculations               - Gauge: in-flight calculations

GHG Protocol Scope 3 Category 10 covers processing of sold products:
    A. Emissions from third-party processing of intermediate products sold
       by the reporting company, before end use by the consumer.
    B. Applies when the reporting company sells intermediate products
       requiring further transformation (machining, molding, refining, etc.)
    C. Site-specific methods preferred: direct measurement, energy-based,
       or fuel-based emissions from downstream processors.
    D. Average-data method: product category x processing type EFs.
    E. Spend-based method: downstream revenue x EEIO sector factors.
    F. Excludes end-of-life (Cat 12) and use-phase (Cat 11) emissions.

Calculation methods defined by GHG Protocol:
    - Site-specific direct: customer-reported total processing emissions
    - Site-specific energy: energy consumption x grid/fuel EF
    - Site-specific fuel: fuel consumption x combustion EF
    - Average-data: product category x processing type EF (kgCO2e/tonne)
    - Spend-based: downstream revenue x EEIO sector factor

Example:
    >>> metrics = get_metrics_collector()
    >>> metrics.record_calculation(
    ...     method="average_data",
    ...     category="metals_ferrous",
    ...     status="success",
    ...     duration=0.042,
    ...     emissions_kg=1250.5,
    ... )

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-010
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
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not available, metrics will be no-ops")
    _PROMETHEUS_AVAILABLE = False

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
# Enumerations -- Processing of Sold Products domain label value sets
# ===========================================================================


class CalculationMethodLabel(str, Enum):
    """
    Calculation methods for processing of sold products emissions.

    GHG Protocol Scope 3 Category 10 supports five approaches in
    order of data quality preference:
        - Site-specific direct: customer-reported total processing emissions
        - Site-specific energy: energy consumption x grid/fuel EF
        - Site-specific fuel: fuel consumption x combustion EF
        - Average-data: product category x processing type EF
        - Spend-based: downstream revenue x EEIO sector factor
    """
    SITE_SPECIFIC_DIRECT = "site_specific_direct"
    SITE_SPECIFIC_ENERGY = "site_specific_energy"
    SITE_SPECIFIC_FUEL = "site_specific_fuel"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"


class IntermediateProductLabel(str, Enum):
    """
    Intermediate product categories sold to downstream processors.

    Covers 12 primary categories of intermediate goods that undergo
    further processing after sale, per GHG Protocol Scope 3 Cat 10.
    """
    METALS_FERROUS = "metals_ferrous"
    METALS_NON_FERROUS = "metals_non_ferrous"
    PLASTICS_THERMOPLASTIC = "plastics_thermoplastic"
    PLASTICS_THERMOSET = "plastics_thermoset"
    CHEMICALS = "chemicals"
    FOOD_INGREDIENTS = "food_ingredients"
    TEXTILES = "textiles"
    ELECTRONICS = "electronics"
    GLASS_CERAMICS = "glass_ceramics"
    WOOD_PAPER = "wood_paper"
    MINERALS = "minerals"
    AGRICULTURAL = "agricultural"


class CalculationStatusLabel(str, Enum):
    """Calculation operation status."""
    SUCCESS = "success"
    ERROR = "error"
    VALIDATION_ERROR = "validation_error"


class EFSourceLabel(str, Enum):
    """Emission factor data sources for processing EF lookups."""
    DEFRA = "defra"
    EPA = "epa"
    ECOINVENT = "ecoinvent"
    IEA = "iea"
    BEIS = "beis"
    CUSTOMER = "customer"


class EFHitLabel(str, Enum):
    """Emission factor lookup cache result."""
    HIT = "hit"
    MISS = "miss"


class GridRegionLabel(str, Enum):
    """Grid regions for electricity emission factor lookups."""
    US = "US"
    GB = "GB"
    DE = "DE"
    FR = "FR"
    CN = "CN"
    IN = "IN"
    JP = "JP"
    KR = "KR"
    BR = "BR"
    CA = "CA"
    AU = "AU"
    MX = "MX"
    IT = "IT"
    ES = "ES"
    PL = "PL"
    GLOBAL = "GLOBAL"


class ProcessingChainLabel(str, Enum):
    """Multi-step processing chain types."""
    METALS_AUTOMOTIVE = "metals_automotive"
    ALUMINUM_PACKAGING = "aluminum_packaging"
    PLASTIC_PACKAGING = "plastic_packaging"
    SEMICONDUCTOR = "semiconductor"
    FOOD_PRODUCTS = "food_products"
    TEXTILE_GARMENTS = "textile_garments"
    GLASS_BOTTLES = "glass_bottles"
    PAPER_PRODUCTS = "paper_products"


class FrameworkLabel(str, Enum):
    """Compliance frameworks for Category 10 reporting."""
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD = "csrd"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"
    GRI = "gri"


class ComplianceStatusLabel(str, Enum):
    """Compliance check result status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class BatchJobStatusLabel(str, Enum):
    """Batch job status labels."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStageLabel(str, Enum):
    """Pipeline stage names for stage-level duration tracking."""
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


class DCRuleLabel(str, Enum):
    """Double-counting prevention rule identifiers."""
    DC_PSP_001 = "DC-PSP-001"
    DC_PSP_002 = "DC-PSP-002"
    DC_PSP_003 = "DC-PSP-003"
    DC_PSP_004 = "DC-PSP-004"
    DC_PSP_005 = "DC-PSP-005"
    DC_PSP_006 = "DC-PSP-006"
    DC_PSP_007 = "DC-PSP-007"
    DC_PSP_008 = "DC-PSP-008"


# ===========================================================================
# MetricsCollector -- Thread-safe Singleton
# ===========================================================================


class MetricsCollector:
    """
    Thread-safe singleton metrics collector for Processing of Sold Products
    Agent (MRV-023).

    Provides 14 Prometheus metrics for tracking Scope 3 Category 10
    processing of sold products emissions calculations.

    All metrics use the ``gl_psp_`` prefix for namespace isolation within
    the GreenLang Prometheus ecosystem.

    Attributes:
        calculations_total: Counter for calculation operations
        calculation_duration_seconds: Histogram for calculation latency
        emissions_kg_total: Counter for total emissions in kgCO2e
        products_processed_total: Counter for products processed by category
        ef_lookups_total: Counter for EF lookups by source and hit/miss
        grid_ef_lookups_total: Counter for grid EF lookups by region
        chain_calculations_total: Counter for processing chain calculations
        compliance_checks_total: Counter for compliance checks
        dc_rule_triggers_total: Counter for double-counting rule triggers
        pipeline_stage_duration_seconds: Histogram for stage durations
        batch_jobs_total: Counter for batch job status
        dqi_score: Gauge for latest DQI composite score
        uncertainty_width: Gauge for latest uncertainty CI width
        active_calculations: Gauge for in-flight calculations

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_calculation(
        ...     method="average_data",
        ...     category="metals_ferrous",
        ...     status="success",
        ...     duration=0.042,
        ...     emissions_kg=1250.5,
        ... )
    """

    _instance: Optional["MetricsCollector"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "MetricsCollector":
        """Thread-safe singleton instantiation using double-checked locking."""
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
        self._stats_lock: threading.RLock = threading.RLock()
        self._in_memory_stats: Dict[str, Any] = {
            "calculations": 0,
            "emissions_kg_co2e": 0.0,
            "products_processed": 0,
            "ef_lookups": 0,
            "ef_hits": 0,
            "ef_misses": 0,
            "grid_ef_lookups": 0,
            "chain_calculations": 0,
            "compliance_checks": 0,
            "dc_rule_triggers": 0,
            "batch_jobs": 0,
            "batch_completed": 0,
            "batch_failed": 0,
            "latest_dqi_score": 0.0,
            "latest_uncertainty_width": 0.0,
            "errors": 0,
        }

        self._init_metrics()

        logger.info(
            "MetricsCollector initialized for gl_psp_ (Prometheus: %s)",
            "available" if _PROMETHEUS_AVAILABLE else "unavailable",
        )

    def _init_metrics(self) -> None:
        """
        Initialize all 14 Prometheus metrics with gl_psp_ prefix.

        When Prometheus is available, metric registration may fail if the
        metrics were previously registered (e.g., after reset in tests).
        We unregister and re-register to obtain fresh collector objects.
        """
        if _PROMETHEUS_AVAILABLE:
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
        # 1. gl_psp_calculations_total (Counter)
        #    Total processing of sold products emission calculations.
        #    Labels: method, category, status
        # ------------------------------------------------------------------
        self.calculations_total = _safe_create(
            Counter,
            "gl_psp_calculations_total",
            "Total processing of sold products emission calculations",
            ["method", "category", "status"],
        )

        # ------------------------------------------------------------------
        # 2. gl_psp_calculation_duration_seconds (Histogram)
        #    Duration of calculation operations in seconds.
        #    Labels: method, category
        #    Buckets tuned for processing emissions calculation latencies.
        # ------------------------------------------------------------------
        self.calculation_duration_seconds = _safe_create(
            Histogram,
            "gl_psp_calculation_duration_seconds",
            "Duration of processing of sold products calculation operations",
            ["method", "category"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
        )

        # ------------------------------------------------------------------
        # 3. gl_psp_emissions_kg_total (Counter)
        #    Total emissions calculated in kgCO2e.
        #    Labels: method, category
        # ------------------------------------------------------------------
        self.emissions_kg_total = _safe_create(
            Counter,
            "gl_psp_emissions_kg_total",
            "Total processing of sold products emissions in kgCO2e",
            ["method", "category"],
        )

        # ------------------------------------------------------------------
        # 4. gl_psp_products_processed_total (Counter)
        #    Total intermediate products processed.
        #    Labels: category
        # ------------------------------------------------------------------
        self.products_processed_total = _safe_create(
            Counter,
            "gl_psp_products_processed_total",
            "Total intermediate products processed by category",
            ["category"],
        )

        # ------------------------------------------------------------------
        # 5. gl_psp_ef_lookups_total (Counter)
        #    Total emission factor lookups.
        #    Labels: source, hit
        # ------------------------------------------------------------------
        self.ef_lookups_total = _safe_create(
            Counter,
            "gl_psp_ef_lookups_total",
            "Total emission factor lookups by source and cache status",
            ["source", "hit"],
        )

        # ------------------------------------------------------------------
        # 6. gl_psp_grid_ef_lookups_total (Counter)
        #    Total grid electricity emission factor lookups.
        #    Labels: region
        # ------------------------------------------------------------------
        self.grid_ef_lookups_total = _safe_create(
            Counter,
            "gl_psp_grid_ef_lookups_total",
            "Total grid electricity EF lookups by region",
            ["region"],
        )

        # ------------------------------------------------------------------
        # 7. gl_psp_chain_calculations_total (Counter)
        #    Total multi-step processing chain calculations.
        #    Labels: chain_type
        # ------------------------------------------------------------------
        self.chain_calculations_total = _safe_create(
            Counter,
            "gl_psp_chain_calculations_total",
            "Total multi-step processing chain calculations",
            ["chain_type"],
        )

        # ------------------------------------------------------------------
        # 8. gl_psp_compliance_checks_total (Counter)
        #    Total compliance checks performed.
        #    Labels: framework, status
        # ------------------------------------------------------------------
        self.compliance_checks_total = _safe_create(
            Counter,
            "gl_psp_compliance_checks_total",
            "Total compliance checks by framework and status",
            ["framework", "status"],
        )

        # ------------------------------------------------------------------
        # 9. gl_psp_dc_rule_triggers_total (Counter)
        #    Total double-counting rule triggers.
        #    Labels: rule_id
        # ------------------------------------------------------------------
        self.dc_rule_triggers_total = _safe_create(
            Counter,
            "gl_psp_dc_rule_triggers_total",
            "Total double-counting prevention rule triggers",
            ["rule_id"],
        )

        # ------------------------------------------------------------------
        # 10. gl_psp_pipeline_stage_duration_seconds (Histogram)
        #     Duration of individual pipeline stages in seconds.
        #     Labels: stage
        #     Buckets tuned for per-stage latencies.
        # ------------------------------------------------------------------
        self.pipeline_stage_duration_seconds = _safe_create(
            Histogram,
            "gl_psp_pipeline_stage_duration_seconds",
            "Duration of individual pipeline stages",
            ["stage"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        # ------------------------------------------------------------------
        # 11. gl_psp_batch_jobs_total (Counter)
        #     Total batch jobs by status.
        #     Labels: status
        # ------------------------------------------------------------------
        self.batch_jobs_total = _safe_create(
            Counter,
            "gl_psp_batch_jobs_total",
            "Total batch jobs by status",
            ["status"],
        )

        # ------------------------------------------------------------------
        # 12. gl_psp_dqi_score (Gauge)
        #     Latest data quality indicator composite score.
        #     No labels -- single current value.
        # ------------------------------------------------------------------
        self.dqi_score = _safe_create(
            Gauge,
            "gl_psp_dqi_score",
            "Latest DQI composite score (1.0-5.0)",
        )

        # ------------------------------------------------------------------
        # 13. gl_psp_uncertainty_width (Gauge)
        #     Latest uncertainty confidence interval width.
        #     No labels -- single current value.
        # ------------------------------------------------------------------
        self.uncertainty_width = _safe_create(
            Gauge,
            "gl_psp_uncertainty_width",
            "Latest uncertainty confidence interval width (kgCO2e)",
        )

        # ------------------------------------------------------------------
        # 14. gl_psp_active_calculations (Gauge)
        #     Number of currently in-flight calculations.
        #     No labels -- simple gauge.
        # ------------------------------------------------------------------
        self.active_calculations = _safe_create(
            Gauge,
            "gl_psp_active_calculations",
            "Number of currently active processing calculations",
        )

        # ------------------------------------------------------------------
        # Agent info metric (supplementary)
        # ------------------------------------------------------------------
        self.agent_info = _safe_create(
            Info,
            "gl_psp_agent",
            "Processing of Sold Products Agent metadata",
        )
        if _PROMETHEUS_AVAILABLE:
            try:
                self.agent_info.info({
                    "agent_id": "GL-MRV-S3-010",
                    "version": "1.0.0",
                    "scope": "scope_3_category_10",
                    "description": "Processing of Sold Products",
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
        emissions_kg: float,
    ) -> None:
        """
        Record a processing of sold products emission calculation.

        Increments the calculations counter, observes the duration histogram,
        and accumulates emissions. Also updates in-memory statistics for
        the summary endpoint.

        Args:
            method: Calculation method (site_specific_direct/site_specific_energy/
                    site_specific_fuel/average_data/spend_based)
            category: Intermediate product category (metals_ferrous/plastics_thermoplastic/etc.)
            status: Calculation status (success/error/validation_error)
            duration: Operation duration in seconds
            emissions_kg: Emissions calculated in kgCO2e

        Example:
            >>> collector.record_calculation(
            ...     method="average_data",
            ...     category="metals_ferrous",
            ...     status="success",
            ...     duration=0.042,
            ...     emissions_kg=1250.5,
            ... )
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethodLabel, CalculationMethodLabel.AVERAGE_DATA.value
            )
            category = self._validate_enum_value(
                category, IntermediateProductLabel, IntermediateProductLabel.METALS_FERROUS.value
            )
            status = self._validate_enum_value(
                status, CalculationStatusLabel, CalculationStatusLabel.ERROR.value
            )

            self.calculations_total.labels(
                method=method, category=category, status=status
            ).inc()

            if duration is not None and duration > 0:
                self.calculation_duration_seconds.labels(
                    method=method, category=category
                ).observe(duration)

            if emissions_kg is not None and emissions_kg > 0:
                self.emissions_kg_total.labels(
                    method=method, category=category
                ).inc(emissions_kg)
                with self._stats_lock:
                    self._in_memory_stats["emissions_kg_co2e"] += emissions_kg

            with self._stats_lock:
                self._in_memory_stats["calculations"] += 1
                if status == CalculationStatusLabel.ERROR.value:
                    self._in_memory_stats["errors"] += 1

            logger.debug(
                "Recorded calculation: method=%s, category=%s, status=%s, "
                "duration=%.3fs, emissions=%.2f kgCO2e",
                method, category, status,
                duration if duration else 0.0,
                emissions_kg if emissions_kg else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record calculation metrics: %s", e, exc_info=True)

    def record_product(self, category: str) -> None:
        """
        Record an intermediate product processed.

        Args:
            category: Intermediate product category (metals_ferrous/etc.)

        Example:
            >>> collector.record_product("plastics_thermoplastic")
        """
        try:
            category = self._validate_enum_value(
                category, IntermediateProductLabel, IntermediateProductLabel.METALS_FERROUS.value
            )

            self.products_processed_total.labels(category=category).inc()

            with self._stats_lock:
                self._in_memory_stats["products_processed"] += 1

            logger.debug("Recorded product: category=%s", category)

        except Exception as e:
            logger.error("Failed to record product metrics: %s", e, exc_info=True)

    def record_ef_lookup(self, source: str, hit: bool) -> None:
        """
        Record an emission factor lookup operation.

        Args:
            source: EF data source (defra/epa/ecoinvent/iea/beis/customer)
            hit: True if cache hit, False if cache miss

        Example:
            >>> collector.record_ef_lookup("ecoinvent", hit=True)
        """
        try:
            source = self._validate_enum_value(
                source, EFSourceLabel, EFSourceLabel.EPA.value
            )
            hit_label = EFHitLabel.HIT.value if hit else EFHitLabel.MISS.value

            self.ef_lookups_total.labels(source=source, hit=hit_label).inc()

            with self._stats_lock:
                self._in_memory_stats["ef_lookups"] += 1
                if hit:
                    self._in_memory_stats["ef_hits"] += 1
                else:
                    self._in_memory_stats["ef_misses"] += 1

            logger.debug(
                "Recorded EF lookup: source=%s, hit=%s", source, hit_label,
            )

        except Exception as e:
            logger.error("Failed to record EF lookup metrics: %s", e, exc_info=True)

    def record_grid_ef_lookup(self, region: str) -> None:
        """
        Record a grid electricity emission factor lookup.

        Args:
            region: Grid region code (US/GB/DE/FR/CN/IN/JP/KR/BR/CA/AU/MX/IT/ES/PL/GLOBAL)

        Example:
            >>> collector.record_grid_ef_lookup("DE")
        """
        try:
            region = self._validate_enum_value(
                region, GridRegionLabel, GridRegionLabel.GLOBAL.value
            )

            self.grid_ef_lookups_total.labels(region=region).inc()

            with self._stats_lock:
                self._in_memory_stats["grid_ef_lookups"] += 1

            logger.debug("Recorded grid EF lookup: region=%s", region)

        except Exception as e:
            logger.error("Failed to record grid EF lookup metrics: %s", e, exc_info=True)

    def record_chain_calculation(self, chain_type: str) -> None:
        """
        Record a multi-step processing chain calculation.

        Args:
            chain_type: Processing chain type (metals_automotive/aluminum_packaging/
                        plastic_packaging/semiconductor/food_products/textile_garments/
                        glass_bottles/paper_products)

        Example:
            >>> collector.record_chain_calculation("metals_automotive")
        """
        try:
            chain_type = self._validate_enum_value(
                chain_type, ProcessingChainLabel, ProcessingChainLabel.METALS_AUTOMOTIVE.value
            )

            self.chain_calculations_total.labels(chain_type=chain_type).inc()

            with self._stats_lock:
                self._in_memory_stats["chain_calculations"] += 1

            logger.debug("Recorded chain calculation: chain_type=%s", chain_type)

        except Exception as e:
            logger.error("Failed to record chain calculation metrics: %s", e, exc_info=True)

    def record_compliance_check(self, framework: str, status: str) -> None:
        """
        Record a compliance check result.

        Args:
            framework: Compliance framework (ghg_protocol/iso_14064/csrd/cdp/sbti/sb_253/gri)
            status: Check result (compliant/non_compliant/partially_compliant/warning/not_applicable)

        Example:
            >>> collector.record_compliance_check("ghg_protocol", "compliant")
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
            logger.error("Failed to record compliance metrics: %s", e, exc_info=True)

    def record_dc_rule(self, rule_id: str, triggered: bool = True) -> None:
        """
        Record a double-counting prevention rule evaluation.

        Only increments the counter when the rule was actually triggered
        (i.e., a potential double-counting overlap was detected and blocked).

        Args:
            rule_id: Double-counting rule identifier (DC-PSP-001 through DC-PSP-008)
            triggered: Whether the rule was triggered (default True)

        Example:
            >>> collector.record_dc_rule("DC-PSP-001", triggered=True)
        """
        try:
            if not triggered:
                return

            rule_id = self._validate_enum_value(
                rule_id, DCRuleLabel, DCRuleLabel.DC_PSP_001.value
            )

            self.dc_rule_triggers_total.labels(rule_id=rule_id).inc()

            with self._stats_lock:
                self._in_memory_stats["dc_rule_triggers"] += 1

            logger.debug("Recorded DC rule trigger: rule_id=%s", rule_id)

        except Exception as e:
            logger.error("Failed to record DC rule metrics: %s", e, exc_info=True)

    def record_pipeline_stage(self, stage: str, duration: float) -> None:
        """
        Record a pipeline stage duration.

        Args:
            stage: Pipeline stage name (validate/classify/normalize/resolve_efs/
                   calculate/allocate/aggregate/compliance/provenance/seal)
            duration: Stage duration in seconds

        Example:
            >>> collector.record_pipeline_stage("resolve_efs", 0.125)
        """
        try:
            stage = self._validate_enum_value(
                stage, PipelineStageLabel, PipelineStageLabel.CALCULATE.value
            )

            if duration is not None and duration >= 0:
                self.pipeline_stage_duration_seconds.labels(
                    stage=stage
                ).observe(duration)

            logger.debug(
                "Recorded pipeline stage: stage=%s, duration=%.3fs",
                stage, duration if duration else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record pipeline stage metrics: %s", e, exc_info=True)

    def record_batch_job(self, status: str) -> None:
        """
        Record a batch job status transition.

        Args:
            status: Batch job status (pending/running/completed/failed)

        Example:
            >>> collector.record_batch_job("completed")
        """
        try:
            status = self._validate_enum_value(
                status, BatchJobStatusLabel, BatchJobStatusLabel.PENDING.value
            )

            self.batch_jobs_total.labels(status=status).inc()

            with self._stats_lock:
                self._in_memory_stats["batch_jobs"] += 1
                if status == BatchJobStatusLabel.COMPLETED.value:
                    self._in_memory_stats["batch_completed"] += 1
                elif status == BatchJobStatusLabel.FAILED.value:
                    self._in_memory_stats["batch_failed"] += 1

            logger.debug("Recorded batch job: status=%s", status)

        except Exception as e:
            logger.error("Failed to record batch job metrics: %s", e, exc_info=True)

    def update_dqi_score(self, score: float) -> None:
        """
        Update the latest DQI composite score gauge.

        Args:
            score: DQI composite score (1.0 = highest quality, 5.0 = lowest quality)

        Example:
            >>> collector.update_dqi_score(2.3)
        """
        try:
            if score is not None:
                clamped = max(0.0, min(5.0, float(score)))
                self.dqi_score.set(clamped)

                with self._stats_lock:
                    self._in_memory_stats["latest_dqi_score"] = clamped

                logger.debug("Updated DQI score: %.2f", clamped)

        except Exception as e:
            logger.error("Failed to update DQI score: %s", e, exc_info=True)

    def update_uncertainty_width(self, width: float) -> None:
        """
        Update the latest uncertainty confidence interval width gauge.

        Args:
            width: Uncertainty CI width in kgCO2e (upper - lower bound)

        Example:
            >>> collector.update_uncertainty_width(125.4)
        """
        try:
            if width is not None:
                clamped = max(0.0, float(width))
                self.uncertainty_width.set(clamped)

                with self._stats_lock:
                    self._in_memory_stats["latest_uncertainty_width"] = clamped

                logger.debug("Updated uncertainty width: %.2f kgCO2e", clamped)

        except Exception as e:
            logger.error("Failed to update uncertainty width: %s", e, exc_info=True)

    def increment_active(self) -> None:
        """
        Increment the active calculations gauge.

        Call at the start of a calculation operation.

        Example:
            >>> collector.increment_active()
        """
        try:
            self.active_calculations.inc()
        except Exception as e:
            logger.error("Failed to increment active calculations: %s", e, exc_info=True)

    def decrement_active(self) -> None:
        """
        Decrement the active calculations gauge.

        Call at the end of a calculation operation (success or failure).

        Example:
            >>> collector.decrement_active()
        """
        try:
            self.active_calculations.dec()
        except Exception as e:
            logger.error("Failed to decrement active calculations: %s", e, exc_info=True)

    # ======================================================================
    # Summary and lifecycle
    # ======================================================================

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of metrics collected since initialization.

        Returns:
            Dictionary with metrics summary including counts, rates, and
            current gauge values. Useful for health check endpoints and
            operational dashboards.

        Example:
            >>> summary = collector.get_metrics_summary()
            >>> summary["calculations"]
            42
            >>> summary["rates"]["calculations_per_hour"]
            168.0
        """
        try:
            now = datetime.now(timezone.utc)
            uptime_seconds = (now - self._start_time).total_seconds()
            uptime_hours = uptime_seconds / 3600 if uptime_seconds > 0 else 1.0

            with self._stats_lock:
                stats_snapshot = dict(self._in_memory_stats)

            ef_total = stats_snapshot["ef_lookups"]
            ef_hit_rate = (
                stats_snapshot["ef_hits"] / ef_total
                if ef_total > 0
                else 0.0
            )

            summary: Dict[str, Any] = {
                "prometheus_available": _PROMETHEUS_AVAILABLE,
                "agent": "processing_sold_products",
                "agent_id": "GL-MRV-S3-010",
                "prefix": "gl_psp_",
                "scope": "Scope 3 Category 10",
                "description": "Processing of Sold Products",
                "metrics_count": 14,
                "uptime_seconds": uptime_seconds,
                "uptime_hours": uptime_seconds / 3600,
                "start_time": self._start_time.isoformat(),
                "current_time": now.isoformat(),
                **stats_snapshot,
                "ef_hit_rate": ef_hit_rate,
                "rates": {
                    "calculations_per_hour": stats_snapshot["calculations"] / uptime_hours,
                    "emissions_kg_per_hour": stats_snapshot["emissions_kg_co2e"] / uptime_hours,
                    "products_per_hour": stats_snapshot["products_processed"] / uptime_hours,
                    "ef_lookups_per_hour": stats_snapshot["ef_lookups"] / uptime_hours,
                    "compliance_checks_per_hour": stats_snapshot["compliance_checks"] / uptime_hours,
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
                "prometheus_available": _PROMETHEUS_AVAILABLE,
                "agent": "processing_sold_products",
            }

    def reset_all(self) -> None:
        """
        Reset in-memory statistics for testing purposes.

        Resets the in-memory counters and the start timestamp. Prometheus
        metrics themselves are cumulative and cannot be reset; this only
        affects the in-memory summary statistics.

        WARNING: Only call in test teardown. Not safe for concurrent use.

        Example:
            >>> collector.reset_all()
        """
        try:
            with self._stats_lock:
                self._in_memory_stats = {
                    "calculations": 0,
                    "emissions_kg_co2e": 0.0,
                    "products_processed": 0,
                    "ef_lookups": 0,
                    "ef_hits": 0,
                    "ef_misses": 0,
                    "grid_ef_lookups": 0,
                    "chain_calculations": 0,
                    "compliance_checks": 0,
                    "dc_rule_triggers": 0,
                    "batch_jobs": 0,
                    "batch_completed": 0,
                    "batch_failed": 0,
                    "latest_dqi_score": 0.0,
                    "latest_uncertainty_width": 0.0,
                    "errors": 0,
                }
            self._start_time = datetime.now(timezone.utc)
            logger.info("Reset in-memory statistics for MetricsCollector")
        except Exception as e:
            logger.error("Failed to reset statistics: %s", e, exc_info=True)

    @classmethod
    def reset_singleton(cls) -> None:
        """
        Reset the singleton instance for testing purposes.

        Destroys the current singleton so the next call to the constructor
        or ``get_metrics_collector()`` creates a fresh instance.

        WARNING: Not safe for concurrent use. Only call in test teardown.

        Example:
            >>> MetricsCollector.reset_singleton()
        """
        with cls._lock:
            if cls._instance is not None:
                if hasattr(cls._instance, "_initialized"):
                    del cls._instance._initialized
                cls._instance = None

                global _collector_instance
                _collector_instance = None

                logger.info("MetricsCollector singleton reset")

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

        If the value is None or not a valid member of the enum, the default
        value is returned and a warning is logged.

        Args:
            value: The string value to validate
            enum_class: The Enum class to validate against
            default: The default value if validation fails

        Returns:
            Validated value string or default
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

_collector_instance: Optional[MetricsCollector] = None
_collector_lock: threading.RLock = threading.RLock()


def get_metrics_collector() -> MetricsCollector:
    """
    Get the singleton MetricsCollector instance.

    Thread-safe accessor. Creates the singleton on first call.

    Returns:
        MetricsCollector singleton instance

    Example:
        >>> collector = get_metrics_collector()
        >>> collector.record_calculation(
        ...     method="average_data",
        ...     category="metals_ferrous",
        ...     status="success",
        ...     duration=0.042,
        ...     emissions_kg=1250.5,
        ... )
    """
    global _collector_instance

    if _collector_instance is None:
        with _collector_lock:
            if _collector_instance is None:
                _collector_instance = MetricsCollector()

    return _collector_instance


def reset_metrics_collector() -> None:
    """
    Reset the singleton MetricsCollector instance for testing purposes.

    Destroys the current singleton so the next call to
    ``get_metrics_collector()`` creates a fresh instance.

    Example:
        >>> reset_metrics_collector()
        >>> collector = get_metrics_collector()  # brand new instance
    """
    MetricsCollector.reset_singleton()


# ===========================================================================
# Context manager helpers
# ===========================================================================


@contextmanager
def track_calculation(
    method: str = "average_data",
    category: str = "metals_ferrous",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a calculation's lifecycle.

    Automatically increments/decrements the active calculation gauge,
    measures wall-clock duration, and records the calculation on exit.
    Set ``context['emissions_kg']`` inside the block to include emissions.

    Args:
        method: Calculation method (default: "average_data")
        category: Intermediate product category (default: "metals_ferrous")

    Yields:
        Mutable context dict. Set ``context['emissions_kg']`` to record emissions.

    Example:
        >>> with track_calculation(method="site_specific_energy", category="electronics") as ctx:
        ...     result = calculate_processing_emissions(product)
        ...     ctx['emissions_kg'] = result.total_co2e_kg
    """
    collector = get_metrics_collector()
    context: Dict[str, Any] = {
        "emissions_kg": 0.0,
        "status": "success",
    }

    collector.increment_active()
    start = time.monotonic()

    try:
        yield context
    except Exception:
        context["status"] = "error"
        raise
    finally:
        duration = time.monotonic() - start
        collector.decrement_active()
        collector.record_calculation(
            method=method,
            category=category,
            status=context["status"],
            duration=duration,
            emissions_kg=context.get("emissions_kg", 0.0),
        )


@contextmanager
def track_pipeline_stage(
    stage: str,
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a single pipeline stage's duration.

    Args:
        stage: Pipeline stage name (validate/classify/normalize/resolve_efs/
               calculate/allocate/aggregate/compliance/provenance/seal)

    Yields:
        Mutable context dict for the caller to annotate.

    Example:
        >>> with track_pipeline_stage("resolve_efs") as ctx:
        ...     efs = resolve_emission_factors(product_inputs)
        ...     ctx['ef_count'] = len(efs)
    """
    collector = get_metrics_collector()
    context: Dict[str, Any] = {}
    start = time.monotonic()

    try:
        yield context
    finally:
        duration = time.monotonic() - start
        collector.record_pipeline_stage(stage=stage, duration=duration)


@contextmanager
def track_batch_job() -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for tracking batch job lifecycle.

    Automatically records pending->running->completed/failed transitions.

    Yields:
        Mutable context dict. Set ``context['size']`` to record batch size.

    Example:
        >>> with track_batch_job() as ctx:
        ...     results = process_batch(products)
        ...     ctx['size'] = len(products)
    """
    collector = get_metrics_collector()
    context: Dict[str, Any] = {
        "size": 0,
        "status": "completed",
    }

    collector.record_batch_job("pending")
    collector.record_batch_job("running")

    try:
        yield context
    except Exception:
        context["status"] = "failed"
        collector.record_batch_job("failed")
        raise
    else:
        collector.record_batch_job("completed")


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    # Availability flag
    "_PROMETHEUS_AVAILABLE",
    # Enums
    "CalculationMethodLabel",
    "IntermediateProductLabel",
    "CalculationStatusLabel",
    "EFSourceLabel",
    "EFHitLabel",
    "GridRegionLabel",
    "ProcessingChainLabel",
    "FrameworkLabel",
    "ComplianceStatusLabel",
    "BatchJobStatusLabel",
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
    "track_batch_job",
]
