# -*- coding: utf-8 -*-
"""
Investments Prometheus Metrics - AGENT-MRV-028

14 Prometheus metrics with gl_inv_ prefix for monitoring the
GL-MRV-S3-015 Investments Agent.

This module provides Prometheus metrics tracking for financed emissions
calculations (Scope 3, Category 15) including equity, debt, project
finance, commercial real estate, mortgages, motor vehicle loans, and
sovereign bonds following the PCAF Global GHG Accounting and Reporting
Standard.

Thread-safe singleton pattern with graceful fallback if Prometheus
unavailable.

Metrics prefix: gl_inv_

14 Prometheus Metrics:
    1.  gl_inv_calculations_total              - Counter: total calculations
    2.  gl_inv_calculation_duration_seconds     - Histogram: calculation duration
    3.  gl_inv_financed_emissions_kgco2e_total  - Counter: total financed emissions
    4.  gl_inv_positions_processed_total        - Counter: positions processed
    5.  gl_inv_portfolio_aggregations_total     - Counter: portfolio aggregations
    6.  gl_inv_compliance_checks_total          - Counter: compliance checks
    7.  gl_inv_db_query_duration_seconds        - Histogram: DB query duration
    8.  gl_inv_cache_operations_total           - Counter: cache operations
    9.  gl_inv_pcaf_data_quality_score          - Gauge: PCAF data quality
    10. gl_inv_pipeline_stage_duration_seconds  - Histogram: pipeline stage duration
    11. gl_inv_batch_size                       - Histogram: batch sizes
    12. gl_inv_errors_total                     - Counter: errors
    13. gl_inv_waci                             - Gauge: WACI by portfolio
    14. gl_inv_active_calculations              - Gauge: active calculations

GHG Protocol Scope 3 Category 15 covers investments:
    A. Equity investments (listed and unlisted)
    B. Debt investments (corporate bonds and loans)
    C. Project finance
    D. Commercial real estate
    E. Mortgages (residential and commercial)
    F. Motor vehicle loans
    G. Sovereign bonds

PCAF asset classes map to calculation methods:
    - Equity: EVIC-based attribution (outstanding / EVIC)
    - Debt: Balance sheet attribution (outstanding / (equity + debt))
    - Project finance: Proportional project share
    - CRE: Property-level or benchmark EUI
    - Mortgages: LTV-based property attribution
    - Motor vehicles: Distance-based or spend-based
    - Sovereign: GDP-based national attribution

Example:
    >>> metrics = InvestmentsMetrics()
    >>> metrics.record_calculation(
    ...     method="evic_based",
    ...     asset_class="listed_equity",
    ...     status="success",
    ...     duration=0.042,
    ...     financed_emissions_kgco2e=1250.6,
    ... )

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-015
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
# Enumerations -- Investments domain-specific label value sets
# ===========================================================================


class CalculationMethodLabel(str, Enum):
    """
    Calculation methods for financed emissions.

    PCAF defines asset-class-specific approaches:
        - EVIC-based: For equity (outstanding / EVIC)
        - Balance sheet: For debt (outstanding / (equity + debt))
        - Project share: For project finance
        - Property level: For CRE with actual energy data
        - Benchmark EUI: For CRE/mortgage with estimated data
        - Distance based: For motor vehicles with known mileage
        - GDP based: For sovereign bonds
        - Spend based: Fallback using EEIO factors
    """
    EVIC_BASED = "evic_based"
    BALANCE_SHEET = "balance_sheet"
    PROJECT_SHARE = "project_share"
    PROPERTY_LEVEL = "property_level"
    BENCHMARK_EUI = "benchmark_eui"
    DISTANCE_BASED = "distance_based"
    GDP_BASED = "gdp_based"
    SPEND_BASED = "spend_based"
    REVENUE_BASED = "revenue_based"


class AssetClassLabel(str, Enum):
    """
    PCAF asset classes for Category 15 investments.

    The seven asset classes defined by PCAF, each with specific
    attribution and data quality scoring methodologies.
    """
    LISTED_EQUITY = "listed_equity"
    UNLISTED_EQUITY = "unlisted_equity"
    CORPORATE_BOND = "corporate_bond"
    CORPORATE_LOAN = "corporate_loan"
    PROJECT_FINANCE = "project_finance"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGE = "mortgage"
    MOTOR_VEHICLE = "motor_vehicle"
    SOVEREIGN_BOND = "sovereign_bond"
    SECURITIZED_DEBT = "securitized_debt"


class CalculationStatusLabel(str, Enum):
    """Calculation operation status."""
    SUCCESS = "success"
    ERROR = "error"
    VALIDATION_ERROR = "validation_error"
    SKIPPED = "skipped"


class PCAFQualityLabel(str, Enum):
    """
    PCAF data quality scores.

    Score 1 = highest quality (audited/verified emissions)
    Score 5 = lowest quality (estimated using EEIO or sector averages)
    """
    SCORE_1 = "score_1"
    SCORE_2 = "score_2"
    SCORE_3 = "score_3"
    SCORE_4 = "score_4"
    SCORE_5 = "score_5"


class FrameworkLabel(str, Enum):
    """Compliance frameworks for investment emissions reporting."""
    GHG_PROTOCOL = "ghg_protocol"
    PCAF = "pcaf"
    ISO_14064 = "iso_14064"
    CSRD = "csrd"
    CDP = "cdp"
    SBTI = "sbti"
    GRI = "gri"
    SEC_CLIMATE = "sec_climate"
    EU_TAXONOMY = "eu_taxonomy"


class ComplianceStatusLabel(str, Enum):
    """Compliance check result status."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class SectorLabel(str, Enum):
    """GICS sectors for WACI breakdown."""
    ENERGY = "energy"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    HEALTH_CARE = "health_care"
    FINANCIALS = "financials"
    INFORMATION_TECHNOLOGY = "information_technology"
    COMMUNICATION_SERVICES = "communication_services"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    SOVEREIGN = "sovereign"
    OTHER = "other"


class CacheOperationLabel(str, Enum):
    """Cache operation types."""
    GET = "get"
    SET = "set"
    DELETE = "delete"
    INVALIDATE = "invalidate"


class CacheResultLabel(str, Enum):
    """Cache operation results."""
    HIT = "hit"
    MISS = "miss"


class DBOperationLabel(str, Enum):
    """Database operation types."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    UPSERT = "upsert"


class PipelineStageLabel(str, Enum):
    """Pipeline stage names for duration tracking."""
    VALIDATE = "validate"
    CLASSIFY = "classify"
    RESOLVE_FINANCIAL = "resolve_financial"
    RESOLVE_EMISSIONS = "resolve_emissions"
    CALCULATE_ATTRIBUTION = "calculate_attribution"
    CALCULATE_FINANCED = "calculate_financed"
    SCORE_DATA_QUALITY = "score_data_quality"
    COMPLIANCE = "compliance"
    AGGREGATE = "aggregate"
    SEAL = "seal"


class ErrorTypeLabel(str, Enum):
    """Error type categories."""
    VALIDATION = "validation"
    CALCULATION = "calculation"
    DATABASE = "database"
    CACHE = "cache"
    INTEGRATION = "integration"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class EngineLabel(str, Enum):
    """Engine names for error attribution."""
    INVESTMENT_DATABASE = "investment_database"
    EQUITY_CALCULATOR = "equity_calculator"
    DEBT_CALCULATOR = "debt_calculator"
    REAL_ASSET_CALCULATOR = "real_asset_calculator"
    SOVEREIGN_CALCULATOR = "sovereign_calculator"
    COMPLIANCE_CHECKER = "compliance_checker"
    PIPELINE = "pipeline"


# ===========================================================================
# InvestmentsMetrics -- Thread-safe Singleton
# ===========================================================================


class InvestmentsMetrics:
    """
    Thread-safe singleton metrics collector for Investments Agent (MRV-028).

    Provides 14 Prometheus metrics for tracking Scope 3 Category 15
    financed emissions calculations across all PCAF asset classes.

    All metrics use the ``gl_inv_`` prefix to ensure namespace isolation
    within the GreenLang Prometheus ecosystem.

    Attributes:
        calculations_total: Counter for total calculation operations
        calculation_duration_seconds: Histogram for operation durations
        financed_emissions_kgco2e_total: Counter for total financed emissions
        positions_processed_total: Counter for positions processed
        portfolio_aggregations_total: Counter for portfolio aggregations
        compliance_checks_total: Counter for compliance checks
        db_query_duration_seconds: Histogram for DB query durations
        cache_operations_total: Counter for cache operations
        pcaf_data_quality_score: Gauge for PCAF quality scores
        pipeline_stage_duration_seconds: Histogram for pipeline stages
        batch_size: Histogram for batch sizes
        errors_total: Counter for errors
        waci: Gauge for WACI by portfolio
        active_calculations: Gauge for in-flight calculations

    Example:
        >>> metrics = InvestmentsMetrics()
        >>> metrics.record_calculation(
        ...     method="evic_based",
        ...     asset_class="listed_equity",
        ...     status="success",
        ...     duration=0.042,
        ...     financed_emissions_kgco2e=1250.6,
        ... )
    """

    _instance: Optional["InvestmentsMetrics"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "InvestmentsMetrics":
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
            "financed_emissions_kgco2e": 0.0,
            "positions_processed": 0,
            "portfolio_aggregations": 0,
            "compliance_checks": 0,
            "errors": 0,
        }

        self._init_metrics()

        logger.info(
            "InvestmentsMetrics initialized (Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable",
        )

    def _init_metrics(self) -> None:
        """
        Initialize all 14 Prometheus metrics with gl_inv_ prefix.

        When Prometheus is available, metric registration may fail if the
        metrics were previously registered. In that case we unregister
        and re-register to obtain fresh collector objects.
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
        # 1. gl_inv_calculations_total (Counter)
        #    Total financed emission calculations performed.
        #    Labels:
        #      - method: evic_based, balance_sheet, project_share,
        #                property_level, benchmark_eui, distance_based,
        #                gdp_based, spend_based, revenue_based
        #      - asset_class: PCAF asset class
        #      - status: success, error, validation_error, skipped
        # ------------------------------------------------------------------
        self.calculations_total = _safe_create(
            Counter,
            "gl_inv_calculations_total",
            "Total financed emission calculations performed",
            ["method", "asset_class", "status"],
        )

        # ------------------------------------------------------------------
        # 2. gl_inv_calculation_duration_seconds (Histogram)
        #    Duration of calculation operations in seconds.
        #    Labels:
        #      - method: Calculation method used
        #      - asset_class: PCAF asset class
        #    Buckets tuned for investment calculation latencies:
        #      5-50ms for single-position EVIC lookups,
        #      100ms-1s for portfolio aggregations,
        #      1-10s for full portfolio Monte Carlo.
        # ------------------------------------------------------------------
        self.calculation_duration_seconds = _safe_create(
            Histogram,
            "gl_inv_calculation_duration_seconds",
            "Duration of financed emission calculation operations",
            ["method", "asset_class"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        # ------------------------------------------------------------------
        # 3. gl_inv_financed_emissions_kgco2e_total (Counter)
        #    Total financed emissions calculated in kgCO2e.
        #    Labels:
        #      - asset_class: PCAF asset class
        #      - sector: GICS sector
        #    Tracks cumulative financed emissions for rate calculation.
        # ------------------------------------------------------------------
        self.financed_emissions_kgco2e_total = _safe_create(
            Counter,
            "gl_inv_financed_emissions_kgco2e_total",
            "Total financed emissions calculated in kgCO2e",
            ["asset_class", "sector"],
        )

        # ------------------------------------------------------------------
        # 4. gl_inv_positions_processed_total (Counter)
        #    Total investment positions processed.
        #    Labels:
        #      - asset_class: PCAF asset class
        #      - pcaf_quality: PCAF data quality score (score_1..score_5)
        #    Tracks position throughput by asset class and quality tier.
        # ------------------------------------------------------------------
        self.positions_processed_total = _safe_create(
            Counter,
            "gl_inv_positions_processed_total",
            "Total investment positions processed",
            ["asset_class", "pcaf_quality"],
        )

        # ------------------------------------------------------------------
        # 5. gl_inv_portfolio_aggregations_total (Counter)
        #    Total portfolio-level aggregation operations.
        #    Labels:
        #      - portfolio_id: Portfolio identifier
        #    Tracks aggregation frequency per portfolio.
        # ------------------------------------------------------------------
        self.portfolio_aggregations_total = _safe_create(
            Counter,
            "gl_inv_portfolio_aggregations_total",
            "Total portfolio-level aggregation operations",
            ["portfolio_id"],
        )

        # ------------------------------------------------------------------
        # 6. gl_inv_compliance_checks_total (Counter)
        #    Total compliance checks performed.
        #    Labels:
        #      - framework: ghg_protocol, pcaf, iso_14064, csrd, cdp,
        #                   sbti, gri, sec_climate, eu_taxonomy
        #      - status: compliant, partially_compliant, non_compliant,
        #                warning, not_applicable
        # ------------------------------------------------------------------
        self.compliance_checks_total = _safe_create(
            Counter,
            "gl_inv_compliance_checks_total",
            "Total investments compliance checks performed",
            ["framework", "status"],
        )

        # ------------------------------------------------------------------
        # 7. gl_inv_db_query_duration_seconds (Histogram)
        #    Duration of database query operations.
        #    Labels:
        #      - operation: select, insert, update, delete, upsert
        #      - table: Target table name
        # ------------------------------------------------------------------
        self.db_query_duration_seconds = _safe_create(
            Histogram,
            "gl_inv_db_query_duration_seconds",
            "Duration of investments database query operations",
            ["operation", "table"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
        )

        # ------------------------------------------------------------------
        # 8. gl_inv_cache_operations_total (Counter)
        #    Total cache operations performed.
        #    Labels:
        #      - operation: get, set, delete, invalidate
        #      - hit_miss: hit, miss
        # ------------------------------------------------------------------
        self.cache_operations_total = _safe_create(
            Counter,
            "gl_inv_cache_operations_total",
            "Total investments cache operations",
            ["operation", "hit_miss"],
        )

        # ------------------------------------------------------------------
        # 9. gl_inv_pcaf_data_quality_score (Gauge)
        #    Current PCAF data quality score by asset class.
        #    Labels:
        #      - asset_class: PCAF asset class
        #    Tracks the weighted average PCAF DQ score (1-5).
        # ------------------------------------------------------------------
        self.pcaf_data_quality_score = _safe_create(
            Gauge,
            "gl_inv_pcaf_data_quality_score",
            "PCAF data quality score by asset class (1=best, 5=worst)",
            ["asset_class"],
        )

        # ------------------------------------------------------------------
        # 10. gl_inv_pipeline_stage_duration_seconds (Histogram)
        #     Duration of individual pipeline stages.
        #     Labels:
        #       - stage: Pipeline stage name
        # ------------------------------------------------------------------
        self.pipeline_stage_duration_seconds = _safe_create(
            Histogram,
            "gl_inv_pipeline_stage_duration_seconds",
            "Duration of investments pipeline stages",
            ["stage"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )

        # ------------------------------------------------------------------
        # 11. gl_inv_batch_size (Histogram)
        #     Size of batch calculation operations (position count).
        #     Labels:
        #       - asset_class: PCAF asset class
        # ------------------------------------------------------------------
        self.batch_size = _safe_create(
            Histogram,
            "gl_inv_batch_size",
            "Batch calculation size for investments operations",
            ["asset_class"],
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000, 50000),
        )

        # ------------------------------------------------------------------
        # 12. gl_inv_errors_total (Counter)
        #     Total errors encountered by engine and error type.
        #     Labels:
        #       - engine: investment_database, equity_calculator, etc.
        #       - error_type: validation, calculation, database, etc.
        # ------------------------------------------------------------------
        self.errors_total = _safe_create(
            Counter,
            "gl_inv_errors_total",
            "Total errors in investments calculations",
            ["engine", "error_type"],
        )

        # ------------------------------------------------------------------
        # 13. gl_inv_waci (Gauge)
        #     Weighted Average Carbon Intensity by portfolio and sector.
        #     Labels:
        #       - portfolio_id: Portfolio identifier
        #       - sector: GICS sector
        #     WACI = SUM(portfolio_weight_i * carbon_intensity_i)
        #     Unit: tCO2e / $M revenue
        # ------------------------------------------------------------------
        self.waci = _safe_create(
            Gauge,
            "gl_inv_waci",
            "Weighted Average Carbon Intensity (tCO2e/$M revenue)",
            ["portfolio_id", "sector"],
        )

        # ------------------------------------------------------------------
        # 14. gl_inv_active_calculations (Gauge)
        #     Number of currently active (in-flight) calculations.
        #     Labels:
        #       - asset_class: PCAF asset class
        # ------------------------------------------------------------------
        self.active_calculations = _safe_create(
            Gauge,
            "gl_inv_active_calculations",
            "Number of currently active investments calculations",
            ["asset_class"],
        )

        # ------------------------------------------------------------------
        # Agent info metric (non-numeric metadata)
        # ------------------------------------------------------------------
        self.agent_info = _safe_create(
            Info,
            "gl_inv_agent",
            "Investments Agent metadata",
        )
        if PROMETHEUS_AVAILABLE:
            try:
                self.agent_info.info({
                    "agent_id": "GL-MRV-S3-015",
                    "version": "1.0.0",
                    "scope": "scope_3_category_15",
                    "description": "Investments financed emissions calculator",
                    "standard": "PCAF",
                })
            except Exception:
                pass

    # ======================================================================
    # Primary recording methods
    # ======================================================================

    def record_calculation(
        self,
        method: str,
        asset_class: str,
        status: str,
        duration: float,
        financed_emissions_kgco2e: float = 0.0,
        sector: str = "other",
    ) -> None:
        """
        Record a financed emission calculation operation.

        Args:
            method: Calculation method (evic_based, balance_sheet, etc.)
            asset_class: PCAF asset class
            status: Calculation status (success/error/validation_error)
            duration: Operation duration in seconds
            financed_emissions_kgco2e: Financed emissions in kgCO2e
            sector: GICS sector for emissions breakdown
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethodLabel, CalculationMethodLabel.EVIC_BASED.value
            )
            asset_class = self._validate_enum_value(
                asset_class, AssetClassLabel, AssetClassLabel.LISTED_EQUITY.value
            )
            status = self._validate_enum_value(
                status, CalculationStatusLabel, CalculationStatusLabel.ERROR.value
            )
            sector = self._validate_enum_value(
                sector, SectorLabel, SectorLabel.OTHER.value
            )

            self.calculations_total.labels(
                method=method, asset_class=asset_class, status=status
            ).inc()

            if duration is not None and duration > 0:
                self.calculation_duration_seconds.labels(
                    method=method, asset_class=asset_class
                ).observe(duration)

            if financed_emissions_kgco2e is not None and financed_emissions_kgco2e > 0:
                self.financed_emissions_kgco2e_total.labels(
                    asset_class=asset_class, sector=sector
                ).inc(financed_emissions_kgco2e)

                with self._stats_lock:
                    self._in_memory_stats["financed_emissions_kgco2e"] += financed_emissions_kgco2e

            with self._stats_lock:
                self._in_memory_stats["calculations"] += 1
                if status == CalculationStatusLabel.ERROR.value:
                    self._in_memory_stats["errors"] += 1

            logger.debug(
                "Recorded calculation: method=%s, asset_class=%s, status=%s, "
                "duration=%.3fs, emissions=%.2f kgCO2e",
                method, asset_class, status,
                duration if duration else 0.0,
                financed_emissions_kgco2e if financed_emissions_kgco2e else 0.0,
            )

        except Exception as e:
            logger.error("Failed to record calculation metrics: %s", e, exc_info=True)

    def record_position(
        self,
        asset_class: str,
        pcaf_quality: int,
    ) -> None:
        """
        Record a processed investment position.

        Args:
            asset_class: PCAF asset class
            pcaf_quality: PCAF data quality score (1-5)
        """
        try:
            asset_class = self._validate_enum_value(
                asset_class, AssetClassLabel, AssetClassLabel.LISTED_EQUITY.value
            )
            quality_label = f"score_{max(1, min(5, pcaf_quality))}"

            self.positions_processed_total.labels(
                asset_class=asset_class, pcaf_quality=quality_label
            ).inc()

            with self._stats_lock:
                self._in_memory_stats["positions_processed"] += 1

        except Exception as e:
            logger.error("Failed to record position metrics: %s", e, exc_info=True)

    def record_portfolio_aggregation(self, portfolio_id: str) -> None:
        """
        Record a portfolio aggregation operation.

        Args:
            portfolio_id: Portfolio identifier
        """
        try:
            self.portfolio_aggregations_total.labels(
                portfolio_id=portfolio_id
            ).inc()

            with self._stats_lock:
                self._in_memory_stats["portfolio_aggregations"] += 1

        except Exception as e:
            logger.error("Failed to record portfolio aggregation: %s", e, exc_info=True)

    def record_compliance_check(
        self, framework: str, status: str
    ) -> None:
        """
        Record a compliance check.

        Args:
            framework: Regulatory framework name
            status: Compliance status
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

        except Exception as e:
            logger.error("Failed to record compliance check: %s", e, exc_info=True)

    def record_db_query(
        self, operation: str, table: str, duration: float
    ) -> None:
        """
        Record a database query operation.

        Args:
            operation: DB operation type (select/insert/update/delete)
            table: Target table name
            duration: Query duration in seconds
        """
        try:
            operation = self._validate_enum_value(
                operation, DBOperationLabel, DBOperationLabel.SELECT.value
            )
            self.db_query_duration_seconds.labels(
                operation=operation, table=table
            ).observe(duration)

        except Exception as e:
            logger.error("Failed to record DB query metrics: %s", e, exc_info=True)

    def record_cache_operation(
        self, operation: str, hit_miss: str
    ) -> None:
        """
        Record a cache operation.

        Args:
            operation: Cache operation type (get/set/delete/invalidate)
            hit_miss: Cache result (hit/miss)
        """
        try:
            operation = self._validate_enum_value(
                operation, CacheOperationLabel, CacheOperationLabel.GET.value
            )
            hit_miss = self._validate_enum_value(
                hit_miss, CacheResultLabel, CacheResultLabel.MISS.value
            )
            self.cache_operations_total.labels(
                operation=operation, hit_miss=hit_miss
            ).inc()

        except Exception as e:
            logger.error("Failed to record cache operation: %s", e, exc_info=True)

    def set_pcaf_data_quality(
        self, asset_class: str, score: float
    ) -> None:
        """
        Set the current PCAF data quality score for an asset class.

        Args:
            asset_class: PCAF asset class
            score: PCAF data quality score (1.0-5.0)
        """
        try:
            asset_class = self._validate_enum_value(
                asset_class, AssetClassLabel, AssetClassLabel.LISTED_EQUITY.value
            )
            self.pcaf_data_quality_score.labels(
                asset_class=asset_class
            ).set(score)

        except Exception as e:
            logger.error("Failed to set PCAF quality score: %s", e, exc_info=True)

    def record_pipeline_stage(
        self, stage: str, duration: float
    ) -> None:
        """
        Record pipeline stage duration.

        Args:
            stage: Pipeline stage name
            duration: Stage duration in seconds
        """
        try:
            stage = self._validate_enum_value(
                stage, PipelineStageLabel, PipelineStageLabel.VALIDATE.value
            )
            self.pipeline_stage_duration_seconds.labels(
                stage=stage
            ).observe(duration)

        except Exception as e:
            logger.error("Failed to record pipeline stage: %s", e, exc_info=True)

    def record_batch(
        self, asset_class: str, size: int
    ) -> None:
        """
        Record batch processing size.

        Args:
            asset_class: PCAF asset class
            size: Number of positions in batch
        """
        try:
            asset_class = self._validate_enum_value(
                asset_class, AssetClassLabel, AssetClassLabel.LISTED_EQUITY.value
            )
            self.batch_size.labels(
                asset_class=asset_class
            ).observe(size)

        except Exception as e:
            logger.error("Failed to record batch size: %s", e, exc_info=True)

    def record_error(
        self, engine: str, error_type: str
    ) -> None:
        """
        Record an error.

        Args:
            engine: Engine where error occurred
            error_type: Error category
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

        except Exception as e:
            logger.error("Failed to record error metric: %s", e, exc_info=True)

    def set_waci(
        self, portfolio_id: str, sector: str, value: float
    ) -> None:
        """
        Set WACI gauge for a portfolio and sector.

        Args:
            portfolio_id: Portfolio identifier
            sector: GICS sector
            value: WACI value (tCO2e/$M revenue)
        """
        try:
            sector = self._validate_enum_value(
                sector, SectorLabel, SectorLabel.OTHER.value
            )
            self.waci.labels(
                portfolio_id=portfolio_id, sector=sector
            ).set(value)

        except Exception as e:
            logger.error("Failed to set WACI gauge: %s", e, exc_info=True)

    # ======================================================================
    # Utility methods
    # ======================================================================

    @staticmethod
    def _validate_enum_value(value: str, enum_cls: type, default: str) -> str:
        """
        Validate and normalize a label value against an enum.

        Args:
            value: Raw label value
            enum_cls: Enum class to validate against
            default: Default value if validation fails

        Returns:
            Validated label value string
        """
        if value is None:
            return default
        value_lower = str(value).lower()
        valid_values = {e.value for e in enum_cls}
        if value_lower in valid_values:
            return value_lower
        return default

    def get_stats(self) -> Dict[str, Any]:
        """
        Get in-memory statistics summary.

        Returns:
            Dictionary of current statistics
        """
        with self._stats_lock:
            stats = self._in_memory_stats.copy()

        stats["uptime_seconds"] = (
            datetime.utcnow() - self._start_time
        ).total_seconds()
        stats["prometheus_available"] = PROMETHEUS_AVAILABLE
        return stats

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance.

        Primarily for test teardown. Clears the singleton so a fresh
        instance is created on next access.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance = None
                logger.info("InvestmentsMetrics singleton reset")


# ===========================================================================
# Module-level singleton accessor
# ===========================================================================

_metrics_instance: Optional[InvestmentsMetrics] = None
_metrics_lock: threading.Lock = threading.Lock()


def get_metrics() -> InvestmentsMetrics:
    """
    Get the singleton InvestmentsMetrics instance.

    Thread-safe accessor for the global metrics instance.

    Returns:
        InvestmentsMetrics singleton instance

    Example:
        >>> from greenlang.agents.mrv.investments.metrics import get_metrics
        >>> metrics = get_metrics()
        >>> metrics.record_calculation(
        ...     method="evic_based",
        ...     asset_class="listed_equity",
        ...     status="success",
        ...     duration=0.042,
        ...     financed_emissions_kgco2e=1250.6,
        ... )
    """
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = InvestmentsMetrics()

    return _metrics_instance


def reset_metrics() -> None:
    """
    Reset the singleton metrics instance for testing purposes.

    Example:
        >>> from greenlang.agents.mrv.investments.metrics import reset_metrics
        >>> reset_metrics()
    """
    InvestmentsMetrics.reset()


# ===========================================================================
# Context manager helpers
# ===========================================================================


@contextmanager
def track_calculation(
    method: str = "evic_based",
    asset_class: str = "listed_equity",
    sector: str = "other",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a calculation's lifecycle.

    Automatically increments/decrements the active calculation gauge,
    measures wall-clock duration, and records the calculation when the
    context exits. The caller can set ``context['financed_emissions_kgco2e']``
    inside the block to include emissions in the recorded metric.

    Args:
        method: Calculation method (default: "evic_based")
        asset_class: PCAF asset class (default: "listed_equity")
        sector: GICS sector (default: "other")

    Yields:
        Mutable context dict.

    Example:
        >>> with track_calculation(method="evic_based", asset_class="listed_equity") as ctx:
        ...     result = calculate_financed_emissions(position)
        ...     ctx['financed_emissions_kgco2e'] = result.financed_emissions_kgco2e
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        "financed_emissions_kgco2e": 0.0,
        "status": "success",
    }

    metrics.active_calculations.labels(asset_class=asset_class).inc()
    start = time.monotonic()

    try:
        yield context
    except Exception:
        context["status"] = "error"
        raise
    finally:
        duration = time.monotonic() - start
        metrics.active_calculations.labels(asset_class=asset_class).dec()
        metrics.record_calculation(
            method=method,
            asset_class=asset_class,
            status=context["status"],
            duration=duration,
            financed_emissions_kgco2e=context.get("financed_emissions_kgco2e", 0.0),
            sector=sector,
        )


@contextmanager
def track_batch(asset_class: str = "listed_equity") -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a batch job's lifecycle.

    Yields:
        Mutable context dict. Set ``context['size']`` to record batch size.

    Example:
        >>> with track_batch(asset_class="corporate_bond") as ctx:
        ...     results = process_batch(positions)
        ...     ctx['size'] = len(positions)
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        "size": 0,
        "status": "success",
    }

    try:
        yield context
    except Exception:
        context["status"] = "error"
        raise
    finally:
        metrics.record_batch(
            asset_class=asset_class,
            size=context.get("size", 0),
        )


@contextmanager
def track_pipeline_stage(stage: str) -> Generator[None, None, None]:
    """
    Context manager that tracks a pipeline stage's duration.

    Args:
        stage: Pipeline stage name

    Example:
        >>> with track_pipeline_stage("resolve_emissions"):
        ...     emissions = resolve_emission_factors(position)
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
    # Enums
    "CalculationMethodLabel",
    "AssetClassLabel",
    "CalculationStatusLabel",
    "PCAFQualityLabel",
    "FrameworkLabel",
    "ComplianceStatusLabel",
    "SectorLabel",
    "CacheOperationLabel",
    "CacheResultLabel",
    "DBOperationLabel",
    "PipelineStageLabel",
    "ErrorTypeLabel",
    "EngineLabel",
    # Singleton class
    "InvestmentsMetrics",
    # Module-level accessors
    "get_metrics",
    "reset_metrics",
    # Context managers
    "track_calculation",
    "track_batch",
    "track_pipeline_stage",
]
