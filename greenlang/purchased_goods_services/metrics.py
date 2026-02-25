"""
Metrics Collection for Purchased Goods & Services Agent (AGENT-MRV-014).

This module provides Prometheus metrics tracking for purchased goods and services
emissions calculations including spend-based, supplier-specific, hybrid methods,
data quality scoring, materiality analysis, and compliance checks.

Thread-safe singleton pattern with graceful fallback if Prometheus unavailable.

Metrics prefix: gl_pgs_

Example:
    >>> metrics = PurchasedGoodsServicesMetrics()
    >>> metrics.record_calculation(
    ...     tenant_id="tenant-123",
    ...     method="supplier_specific",
    ...     status="success",
    ...     duration_s=2.5,
    ...     emissions_kgco2e=15000.0,
    ...     spend_usd=50000.0,
    ...     material_category="steel"
    ... )
"""

import logging
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Graceful Prometheus import
try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not available, metrics will be no-ops")
    PROMETHEUS_AVAILABLE = False
    # Mock classes for graceful degradation
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def inc(self, amount=1):
            pass

    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def observe(self, amount):
            pass

    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def set(self, value):
            pass
        def inc(self, amount=1):
            pass
        def dec(self, amount=1):
            pass

    class Info:
        def __init__(self, *args, **kwargs):
            pass
        def info(self, data):
            pass


class CalculationMethod(str, Enum):
    """Calculation methods for purchased goods/services emissions."""
    SPEND_BASED = "spend_based"
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"
    AVERAGE_DATA = "average_data"
    EEIO = "eeio"  # Environmentally-Extended Input-Output
    LCA = "lca"  # Life Cycle Assessment


class CalculationStatus(str, Enum):
    """Calculation operation status."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    INSUFFICIENT_DATA = "insufficient_data"


class MaterialCategory(str, Enum):
    """Material/service categories for emissions tracking."""
    RAW_MATERIALS = "raw_materials"
    MANUFACTURED_GOODS = "manufactured_goods"
    CAPITAL_GOODS = "capital_goods"
    SERVICES = "services"
    PACKAGING = "packaging"
    ELECTRONICS = "electronics"
    CHEMICALS = "chemicals"
    METALS = "metals"
    PLASTICS = "plastics"
    TEXTILES = "textiles"
    FOOD_BEVERAGES = "food_beverages"
    CONSTRUCTION_MATERIALS = "construction_materials"
    ENERGY_PRODUCTS = "energy_products"
    TRANSPORTATION_EQUIPMENT = "transportation_equipment"
    OTHER = "other"


class SpendClassification(str, Enum):
    """Spend data classification types."""
    GOODS = "goods"
    SERVICES = "services"
    CAPITAL_EXPENDITURE = "capital_expenditure"
    OPERATIONAL_EXPENDITURE = "operational_expenditure"
    DIRECT_MATERIALS = "direct_materials"
    INDIRECT_MATERIALS = "indirect_materials"


class ProcurementType(str, Enum):
    """Procurement transaction types."""
    PURCHASE_ORDER = "purchase_order"
    CONTRACT = "contract"
    INVOICE = "invoice"
    REQUISITION = "requisition"
    BLANKET_ORDER = "blanket_order"
    SPOT_BUY = "spot_buy"


class DataSource(str, Enum):
    """Data sources for supplier assessments."""
    CDP_QUESTIONNAIRE = "cdp_questionnaire"
    SUPPLIER_SURVEY = "supplier_survey"
    ERP_SYSTEM = "erp_system"
    THIRD_PARTY_DATABASE = "third_party_database"
    LCA_DATABASE = "lca_database"
    INDUSTRY_AVERAGE = "industry_average"
    ECOVADIS = "ecovadis"
    DJSI = "djsi"


class QualityDimension(str, Enum):
    """Data quality dimensions (GHG Protocol)."""
    TECHNOLOGY_REPRESENTATIVENESS = "technology_representativeness"
    TEMPORAL_REPRESENTATIVENESS = "temporal_representativeness"
    GEOGRAPHICAL_REPRESENTATIVENESS = "geographical_representativeness"
    COMPLETENESS = "completeness"
    RELIABILITY = "reliability"
    OVERALL = "overall"


class MaterialityLevel(str, Enum):
    """Materiality levels for hotspot analysis."""
    CRITICAL = "critical"  # Top 10% emissions contribution
    HIGH = "high"  # 10-30%
    MEDIUM = "medium"  # 30-60%
    LOW = "low"  # 60-90%
    NEGLIGIBLE = "negligible"  # Bottom 10%


class Framework(str, Enum):
    """Compliance frameworks."""
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    ISO_14067 = "iso_14067"
    CSRD = "csrd"
    TCFD = "tcfd"
    CDP = "cdp"
    SBTI = "sbti"
    ESRS = "esrs"


class ComplianceStatus(str, Enum):
    """Compliance check status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class BatchStatus(str, Enum):
    """Batch job status."""
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


class ErrorType(str, Enum):
    """Error types for tracking."""
    VALIDATION_ERROR = "validation_error"
    CALCULATION_ERROR = "calculation_error"
    DATABASE_ERROR = "database_error"
    API_ERROR = "api_error"
    TIMEOUT_ERROR = "timeout_error"
    CONFIGURATION_ERROR = "configuration_error"
    DATA_NOT_FOUND = "data_not_found"
    SUPPLIER_DATA_UNAVAILABLE = "supplier_data_unavailable"
    EF_LOOKUP_FAILED = "ef_lookup_failed"


class PurchasedGoodsServicesMetrics:
    """
    Thread-safe singleton metrics collector for Purchased Goods & Services Agent.

    Provides 12 Prometheus metrics for tracking emissions calculations, spend
    processing, supplier assessments, data quality scoring, hotspot analysis,
    and compliance checks.

    Attributes:
        calculations_total: Counter for calculation operations
        calculation_duration_seconds: Histogram for operation duration
        emissions_kgco2e_total: Counter for total emissions calculated
        spend_processed_usd_total: Counter for spend processed
        items_processed_total: Counter for items/transactions processed
        suppliers_assessed_total: Counter for suppliers assessed
        dqi_scores: Histogram for Data Quality Indicator scores
        coverage_percentage: Gauge for data coverage by method
        compliance_checks_total: Counter for compliance checks
        batch_jobs_total: Counter for batch processing jobs
        hotspot_items_total: Counter for hotspot items identified
        errors_total: Counter for errors

    Example:
        >>> metrics = PurchasedGoodsServicesMetrics()
        >>> metrics.record_calculation(
        ...     tenant_id="tenant-123",
        ...     method="supplier_specific",
        ...     status="success",
        ...     duration_s=2.5,
        ...     emissions_kgco2e=15000.0,
        ...     spend_usd=50000.0,
        ...     material_category="steel"
        ... )
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize metrics (only once due to singleton)."""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._start_time = datetime.utcnow()
        self._in_memory_stats = {
            'calculations': 0,
            'emissions_kgco2e': 0.0,
            'spend_usd': 0.0,
            'items_processed': 0,
            'suppliers_assessed': 0,
            'compliance_checks': 0,
            'batch_jobs': 0,
            'hotspot_items': 0,
            'errors': 0,
        }

        # Initialize Prometheus metrics
        self._init_metrics()

        logger.info(
            "PurchasedGoodsServicesMetrics initialized (Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable"
        )

    def _init_metrics(self):
        """Initialize all Prometheus metrics."""

        # 1. Calculations total counter
        self.calculations_total = Counter(
            'gl_pgs_calculations_total',
            'Total number of purchased goods/services emissions calculations',
            ['method', 'status', 'tenant_id']
        )

        # 2. Calculation duration histogram
        self.calculation_duration_seconds = Histogram(
            'gl_pgs_calculation_duration_seconds',
            'Duration of calculation operations in seconds',
            ['method'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
        )

        # 3. Emissions total counter (kgCO2e)
        self.emissions_kgco2e_total = Counter(
            'gl_pgs_emissions_kgco2e_total',
            'Total emissions calculated in kgCO2e',
            ['method', 'material_category']
        )

        # 4. Spend processed counter (USD)
        self.spend_processed_usd_total = Counter(
            'gl_pgs_spend_processed_usd_total',
            'Total spend amount processed in USD',
            ['method', 'classification']
        )

        # 5. Items/transactions processed counter
        self.items_processed_total = Counter(
            'gl_pgs_items_processed_total',
            'Total number of items/transactions processed',
            ['method', 'procurement_type']
        )

        # 6. Suppliers assessed counter
        self.suppliers_assessed_total = Counter(
            'gl_pgs_suppliers_assessed_total',
            'Total number of suppliers assessed',
            ['data_source']
        )

        # 7. Data Quality Indicator (DQI) scores histogram
        self.dqi_scores = Histogram(
            'gl_pgs_dqi_scores',
            'Data Quality Indicator score distribution (1.0-5.0, GHG Protocol)',
            ['dimension'],
            buckets=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        )

        # 8. Coverage percentage gauge
        self.coverage_percentage = Gauge(
            'gl_pgs_coverage_percentage',
            'Percentage of spend/items covered by each method',
            ['method', 'tenant_id']
        )

        # 9. Compliance checks counter
        self.compliance_checks_total = Counter(
            'gl_pgs_compliance_checks_total',
            'Total number of compliance checks performed',
            ['framework', 'status']
        )

        # 10. Batch jobs counter
        self.batch_jobs_total = Counter(
            'gl_pgs_batch_jobs_total',
            'Total number of batch processing jobs',
            ['status']
        )

        # 11. Hotspot items counter
        self.hotspot_items_total = Counter(
            'gl_pgs_hotspot_items_total',
            'Total number of hotspot items identified',
            ['materiality_level']
        )

        # 12. Errors counter
        self.errors_total = Counter(
            'gl_pgs_errors_total',
            'Total number of errors by type and operation',
            ['error_type', 'operation']
        )

    def record_calculation(
        self,
        tenant_id: str,
        method: str,
        status: str,
        duration_s: float,
        emissions_kgco2e: Optional[float] = None,
        spend_usd: Optional[float] = None,
        material_category: Optional[str] = None
    ) -> None:
        """
        Record a calculation operation.

        Args:
            tenant_id: Tenant identifier
            method: Calculation method (spend_based/supplier_specific/hybrid/etc)
            status: Calculation status (success/failed/partial/insufficient_data)
            duration_s: Operation duration in seconds
            emissions_kgco2e: Emissions calculated in kgCO2e (optional)
            spend_usd: Spend amount in USD (optional)
            material_category: Material/service category (optional)

        Example:
            >>> metrics.record_calculation(
            ...     tenant_id="tenant-123",
            ...     method="supplier_specific",
            ...     status="success",
            ...     duration_s=2.5,
            ...     emissions_kgco2e=15000.0,
            ...     spend_usd=50000.0,
            ...     material_category="metals"
            ... )
        """
        try:
            # Validate method
            if method not in [m.value for m in CalculationMethod]:
                logger.warning(f"Invalid calculation method: {method}")
                method = CalculationMethod.SPEND_BASED.value

            # Validate status
            if status not in [s.value for s in CalculationStatus]:
                logger.warning(f"Invalid calculation status: {status}")
                status = CalculationStatus.FAILED.value

            # Record calculation count
            self.calculations_total.labels(
                method=method,
                status=status,
                tenant_id=tenant_id
            ).inc()

            # Record duration
            self.calculation_duration_seconds.labels(
                method=method
            ).observe(duration_s)

            # Record emissions if provided
            if emissions_kgco2e is not None and emissions_kgco2e > 0:
                # Validate material category
                if material_category is None:
                    material_category = MaterialCategory.OTHER.value
                elif material_category not in [c.value for c in MaterialCategory]:
                    logger.warning(f"Invalid material category: {material_category}")
                    material_category = MaterialCategory.OTHER.value

                self.emissions_kgco2e_total.labels(
                    method=method,
                    material_category=material_category
                ).inc(emissions_kgco2e)

                # Update in-memory stats
                self._in_memory_stats['emissions_kgco2e'] += emissions_kgco2e

            # Record spend if provided
            if spend_usd is not None and spend_usd > 0:
                self._in_memory_stats['spend_usd'] += spend_usd

            # Update in-memory stats
            self._in_memory_stats['calculations'] += 1

            logger.debug(
                f"Recorded calculation: tenant={tenant_id}, method={method}, "
                f"status={status}, duration={duration_s:.3f}s, "
                f"emissions={emissions_kgco2e or 0:.2f} kgCO2e"
            )

        except Exception as e:
            logger.error(f"Failed to record calculation metrics: {e}", exc_info=True)

    def record_spend_processed(
        self,
        method: str,
        classification: str,
        spend_usd: float
    ) -> None:
        """
        Record spend processing.

        Args:
            method: Calculation method used
            classification: Spend classification (goods/services/capex/opex/etc)
            spend_usd: Spend amount in USD

        Example:
            >>> metrics.record_spend_processed(
            ...     method="spend_based",
            ...     classification="goods",
            ...     spend_usd=100000.0
            ... )
        """
        try:
            # Validate method
            if method not in [m.value for m in CalculationMethod]:
                logger.warning(f"Invalid calculation method: {method}")
                method = CalculationMethod.SPEND_BASED.value

            # Validate classification
            if classification not in [c.value for c in SpendClassification]:
                logger.warning(f"Invalid spend classification: {classification}")
                classification = SpendClassification.GOODS.value

            # Record spend
            self.spend_processed_usd_total.labels(
                method=method,
                classification=classification
            ).inc(spend_usd)

            logger.debug(
                f"Recorded spend: method={method}, classification={classification}, "
                f"amount=${spend_usd:,.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to record spend metrics: {e}", exc_info=True)

    def record_items_processed(
        self,
        method: str,
        procurement_type: str,
        count: int = 1
    ) -> None:
        """
        Record items/transactions processed.

        Args:
            method: Calculation method used
            procurement_type: Type of procurement transaction
            count: Number of items processed (default: 1)

        Example:
            >>> metrics.record_items_processed(
            ...     method="hybrid",
            ...     procurement_type="purchase_order",
            ...     count=50
            ... )
        """
        try:
            # Validate method
            if method not in [m.value for m in CalculationMethod]:
                logger.warning(f"Invalid calculation method: {method}")
                method = CalculationMethod.SPEND_BASED.value

            # Validate procurement type
            if procurement_type not in [p.value for p in ProcurementType]:
                logger.warning(f"Invalid procurement type: {procurement_type}")
                procurement_type = ProcurementType.PURCHASE_ORDER.value

            # Record items
            self.items_processed_total.labels(
                method=method,
                procurement_type=procurement_type
            ).inc(count)

            # Update in-memory stats
            self._in_memory_stats['items_processed'] += count

            logger.debug(
                f"Recorded items: method={method}, type={procurement_type}, count={count}"
            )

        except Exception as e:
            logger.error(f"Failed to record items metrics: {e}", exc_info=True)

    def record_supplier_assessed(
        self,
        data_source: str,
        count: int = 1
    ) -> None:
        """
        Record supplier assessment.

        Args:
            data_source: Source of supplier data
            count: Number of suppliers assessed (default: 1)

        Example:
            >>> metrics.record_supplier_assessed(
            ...     data_source="cdp_questionnaire",
            ...     count=5
            ... )
        """
        try:
            # Validate data source
            if data_source not in [s.value for s in DataSource]:
                logger.warning(f"Invalid data source: {data_source}")
                data_source = DataSource.SUPPLIER_SURVEY.value

            # Record supplier assessment
            self.suppliers_assessed_total.labels(
                data_source=data_source
            ).inc(count)

            # Update in-memory stats
            self._in_memory_stats['suppliers_assessed'] += count

            logger.debug(
                f"Recorded supplier assessment: source={data_source}, count={count}"
            )

        except Exception as e:
            logger.error(f"Failed to record supplier assessment metrics: {e}", exc_info=True)

    def record_dqi_score(
        self,
        dimension: str,
        score: float
    ) -> None:
        """
        Record a Data Quality Indicator (DQI) score.

        Args:
            dimension: Quality dimension (GHG Protocol: 1.0-5.0 scale)
            score: DQI score (1.0 = highest quality, 5.0 = lowest quality)

        Example:
            >>> metrics.record_dqi_score(
            ...     dimension="technology_representativeness",
            ...     score=2.5
            ... )
        """
        try:
            # Validate dimension
            if dimension not in [d.value for d in QualityDimension]:
                logger.warning(f"Invalid quality dimension: {dimension}")
                dimension = QualityDimension.OVERALL.value

            # Clamp score to valid range (1.0-5.0)
            score = max(1.0, min(5.0, score))

            # Record DQI score
            self.dqi_scores.labels(
                dimension=dimension
            ).observe(score)

            logger.debug(
                f"Recorded DQI score: dimension={dimension}, score={score:.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to record DQI score metrics: {e}", exc_info=True)

    def update_coverage(
        self,
        method: str,
        tenant_id: str,
        percentage: float
    ) -> None:
        """
        Update coverage percentage for a calculation method.

        Args:
            method: Calculation method
            tenant_id: Tenant identifier
            percentage: Coverage percentage (0.0-100.0)

        Example:
            >>> metrics.update_coverage(
            ...     method="supplier_specific",
            ...     tenant_id="tenant-123",
            ...     percentage=67.5
            ... )
        """
        try:
            # Validate method
            if method not in [m.value for m in CalculationMethod]:
                logger.warning(f"Invalid calculation method: {method}")
                method = CalculationMethod.SPEND_BASED.value

            # Clamp percentage to valid range
            percentage = max(0.0, min(100.0, percentage))

            # Update coverage gauge
            self.coverage_percentage.labels(
                method=method,
                tenant_id=tenant_id
            ).set(percentage)

            logger.debug(
                f"Updated coverage: method={method}, tenant={tenant_id}, "
                f"percentage={percentage:.1f}%"
            )

        except Exception as e:
            logger.error(f"Failed to update coverage metrics: {e}", exc_info=True)

    def record_compliance_check(
        self,
        framework: str,
        status: str
    ) -> None:
        """
        Record a compliance check operation.

        Args:
            framework: Compliance framework
            status: Compliance status (compliant/non_compliant/warning/not_applicable)

        Example:
            >>> metrics.record_compliance_check(
            ...     framework="ghg_protocol",
            ...     status="compliant"
            ... )
        """
        try:
            # Validate framework
            if framework not in [f.value for f in Framework]:
                logger.warning(f"Invalid framework: {framework}")
                framework = Framework.GHG_PROTOCOL.value

            # Validate status
            if status not in [s.value for s in ComplianceStatus]:
                logger.warning(f"Invalid compliance status: {status}")
                status = ComplianceStatus.NOT_APPLICABLE.value

            # Record compliance check
            self.compliance_checks_total.labels(
                framework=framework,
                status=status
            ).inc()

            # Update in-memory stats
            self._in_memory_stats['compliance_checks'] += 1

            logger.debug(
                f"Recorded compliance check: framework={framework}, status={status}"
            )

        except Exception as e:
            logger.error(f"Failed to record compliance check metrics: {e}", exc_info=True)

    def record_batch_job(
        self,
        status: str,
        items_processed: Optional[int] = None
    ) -> None:
        """
        Record a batch processing job.

        Args:
            status: Batch job status (completed/failed/partial/timeout)
            items_processed: Number of items processed in batch (optional)

        Example:
            >>> metrics.record_batch_job(
            ...     status="completed",
            ...     items_processed=1000
            ... )
        """
        try:
            # Validate status
            if status not in [s.value for s in BatchStatus]:
                logger.warning(f"Invalid batch status: {status}")
                status = BatchStatus.FAILED.value

            # Record batch job
            self.batch_jobs_total.labels(
                status=status
            ).inc()

            # Update in-memory stats
            self._in_memory_stats['batch_jobs'] += 1

            logger.debug(
                f"Recorded batch job: status={status}, "
                f"items={items_processed or 'N/A'}"
            )

        except Exception as e:
            logger.error(f"Failed to record batch job metrics: {e}", exc_info=True)

    def record_hotspot_item(
        self,
        materiality_level: str,
        count: int = 1
    ) -> None:
        """
        Record hotspot items identified in materiality analysis.

        Args:
            materiality_level: Materiality level (critical/high/medium/low/negligible)
            count: Number of hotspot items (default: 1)

        Example:
            >>> metrics.record_hotspot_item(
            ...     materiality_level="critical",
            ...     count=3
            ... )
        """
        try:
            # Validate materiality level
            if materiality_level not in [m.value for m in MaterialityLevel]:
                logger.warning(f"Invalid materiality level: {materiality_level}")
                materiality_level = MaterialityLevel.MEDIUM.value

            # Record hotspot items
            self.hotspot_items_total.labels(
                materiality_level=materiality_level
            ).inc(count)

            # Update in-memory stats
            self._in_memory_stats['hotspot_items'] += count

            logger.debug(
                f"Recorded hotspot items: level={materiality_level}, count={count}"
            )

        except Exception as e:
            logger.error(f"Failed to record hotspot item metrics: {e}", exc_info=True)

    def record_error(
        self,
        error_type: str,
        operation: str
    ) -> None:
        """
        Record an error occurrence.

        Args:
            error_type: Type of error
            operation: Operation where error occurred

        Example:
            >>> metrics.record_error(
            ...     error_type="supplier_data_unavailable",
            ...     operation="calculate_supplier_specific"
            ... )
        """
        try:
            # Validate error type
            if error_type not in [e.value for e in ErrorType]:
                logger.warning(f"Invalid error type: {error_type}")
                error_type = ErrorType.VALIDATION_ERROR.value

            # Record error
            self.errors_total.labels(
                error_type=error_type,
                operation=operation
            ).inc()

            # Update in-memory stats
            self._in_memory_stats['errors'] += 1

            logger.debug(f"Recorded error: type={error_type}, operation={operation}")

        except Exception as e:
            logger.error(f"Failed to record error metrics: {e}", exc_info=True)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of metrics collected since initialization.

        Returns:
            Dictionary with metrics summary including counts and uptime

        Example:
            >>> summary = metrics.get_metrics_summary()
            >>> print(summary['calculations'])
            5432
        """
        try:
            uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()

            summary = {
                'prometheus_available': PROMETHEUS_AVAILABLE,
                'uptime_seconds': uptime_seconds,
                'uptime_hours': uptime_seconds / 3600,
                'start_time': self._start_time.isoformat(),
                'current_time': datetime.utcnow().isoformat(),
                **self._in_memory_stats,
                'rates': {
                    'calculations_per_hour': (
                        self._in_memory_stats['calculations'] / (uptime_seconds / 3600)
                        if uptime_seconds > 0 else 0
                    ),
                    'emissions_kgco2e_per_hour': (
                        self._in_memory_stats['emissions_kgco2e'] / (uptime_seconds / 3600)
                        if uptime_seconds > 0 else 0
                    ),
                    'spend_usd_per_hour': (
                        self._in_memory_stats['spend_usd'] / (uptime_seconds / 3600)
                        if uptime_seconds > 0 else 0
                    ),
                    'items_per_hour': (
                        self._in_memory_stats['items_processed'] / (uptime_seconds / 3600)
                        if uptime_seconds > 0 else 0
                    ),
                    'errors_per_hour': (
                        self._in_memory_stats['errors'] / (uptime_seconds / 3600)
                        if uptime_seconds > 0 else 0
                    ),
                }
            }

            logger.debug(f"Generated metrics summary: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate metrics summary: {e}", exc_info=True)
            return {
                'error': str(e),
                'prometheus_available': PROMETHEUS_AVAILABLE,
            }

    def record_batch_calculation(
        self,
        tenant_id: str,
        method: str,
        batch_size: int,
        successful: int,
        failed: int,
        duration_s: float,
        total_emissions_kgco2e: float,
        total_spend_usd: float
    ) -> None:
        """
        Record a batch calculation operation with aggregated metrics.

        Args:
            tenant_id: Tenant identifier
            method: Calculation method used
            batch_size: Total number of items processed
            successful: Number of successful calculations
            failed: Number of failed calculations
            duration_s: Total batch processing duration
            total_emissions_kgco2e: Total emissions calculated
            total_spend_usd: Total spend processed

        Example:
            >>> metrics.record_batch_calculation(
            ...     tenant_id="tenant-123",
            ...     method="hybrid",
            ...     batch_size=1000,
            ...     successful=985,
            ...     failed=15,
            ...     duration_s=45.5,
            ...     total_emissions_kgco2e=2500000.0,
            ...     total_spend_usd=10000000.0
            ... )
        """
        try:
            # Record batch job
            status = BatchStatus.COMPLETED.value if failed == 0 else BatchStatus.PARTIAL.value
            self.record_batch_job(status, batch_size)

            # Record successful calculations
            if successful > 0:
                self.calculations_total.labels(
                    method=method,
                    status=CalculationStatus.SUCCESS.value,
                    tenant_id=tenant_id
                ).inc(successful)

            # Record failed calculations
            if failed > 0:
                self.calculations_total.labels(
                    method=method,
                    status=CalculationStatus.FAILED.value,
                    tenant_id=tenant_id
                ).inc(failed)

            # Record average duration
            avg_duration = duration_s / batch_size if batch_size > 0 else 0
            self.calculation_duration_seconds.labels(
                method=method
            ).observe(avg_duration)

            # Record total emissions
            if total_emissions_kgco2e > 0:
                self.emissions_kgco2e_total.labels(
                    method=method,
                    material_category=MaterialCategory.OTHER.value
                ).inc(total_emissions_kgco2e)

            # Update in-memory stats
            self._in_memory_stats['calculations'] += batch_size
            self._in_memory_stats['emissions_kgco2e'] += total_emissions_kgco2e
            self._in_memory_stats['spend_usd'] += total_spend_usd

            logger.info(
                f"Recorded batch calculation: tenant={tenant_id}, method={method}, "
                f"batch_size={batch_size}, successful={successful}, failed={failed}, "
                f"duration={duration_s:.2f}s, emissions={total_emissions_kgco2e:,.2f} kgCO2e, "
                f"spend=${total_spend_usd:,.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to record batch calculation metrics: {e}", exc_info=True)

    def reset_stats(self) -> None:
        """
        Reset in-memory statistics (not Prometheus metrics).

        Note: This only resets the in-memory counters used for get_metrics_summary().
        Prometheus metrics are cumulative and cannot be reset without restarting.

        Example:
            >>> metrics.reset_stats()
        """
        try:
            self._in_memory_stats = {
                'calculations': 0,
                'emissions_kgco2e': 0.0,
                'spend_usd': 0.0,
                'items_processed': 0,
                'suppliers_assessed': 0,
                'compliance_checks': 0,
                'batch_jobs': 0,
                'hotspot_items': 0,
                'errors': 0,
            }
            self._start_time = datetime.utcnow()

            logger.info("Reset in-memory statistics")

        except Exception as e:
            logger.error(f"Failed to reset statistics: {e}", exc_info=True)


# Singleton instance for module-level access
_metrics_instance = None
_metrics_lock = threading.Lock()


def get_metrics() -> PurchasedGoodsServicesMetrics:
    """
    Get the singleton PurchasedGoodsServicesMetrics instance.

    Thread-safe accessor for the global metrics instance.

    Returns:
        PurchasedGoodsServicesMetrics singleton instance

    Example:
        >>> from greenlang.purchased_goods_services.metrics import get_metrics
        >>> metrics = get_metrics()
        >>> metrics.record_calculation(...)
    """
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = PurchasedGoodsServicesMetrics()

    return _metrics_instance


__all__ = [
    'PurchasedGoodsServicesMetrics',
    'get_metrics',
    'CalculationMethod',
    'CalculationStatus',
    'MaterialCategory',
    'SpendClassification',
    'ProcurementType',
    'DataSource',
    'QualityDimension',
    'MaterialityLevel',
    'Framework',
    'ComplianceStatus',
    'BatchStatus',
    'ErrorType',
    'PROMETHEUS_AVAILABLE',
]
