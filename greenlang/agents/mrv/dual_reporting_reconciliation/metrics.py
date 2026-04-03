"""
Metrics Collection for Dual Reporting Reconciliation Agent (AGENT-MRV-013).

This module provides Prometheus metrics tracking for dual reporting reconciliation
operations including discrepancy detection, quality scoring, compliance checks,
and performance monitoring.

Thread-safe singleton pattern with graceful fallback if Prometheus unavailable.

Metrics prefix: gl_drr_

Example:
    >>> metrics = DualReportingReconciliationMetrics()
    >>> metrics.record_reconciliation(
    ...     tenant_id="tenant-123",
    ...     status="success",
    ...     energy_type="natural_gas",
    ...     duration_s=1.2,
    ...     discrepancy_pct=5.3,
    ...     pif=85.0
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


class ReconciliationStatus(str, Enum):
    """Reconciliation operation status."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class DiscrepancyType(str, Enum):
    """Types of discrepancies detected."""
    ENERGY_MISMATCH = "energy_mismatch"
    EMISSION_MISMATCH = "emission_mismatch"
    MISSING_DATA = "missing_data"
    TEMPORAL_GAP = "temporal_gap"
    UNIT_INCONSISTENCY = "unit_inconsistency"
    ALLOCATION_ERROR = "allocation_error"
    CONVERSION_ERROR = "conversion_error"
    ROUNDING_ERROR = "rounding_error"


class Materiality(str, Enum):
    """Materiality levels for discrepancies."""
    CRITICAL = "critical"  # >25%
    HIGH = "high"  # 10-25%
    MEDIUM = "medium"  # 5-10%
    LOW = "low"  # 1-5%
    NEGLIGIBLE = "negligible"  # <1%


class Direction(str, Enum):
    """Direction of discrepancy."""
    OVER = "over"  # Energy > Emissions
    UNDER = "under"  # Energy < Emissions
    NEUTRAL = "neutral"  # Balanced


class QualityDimension(str, Enum):
    """Data quality dimensions."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    OVERALL = "overall"


class QualityGrade(str, Enum):
    """Quality grade classifications."""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"  # 75-89%
    FAIR = "fair"  # 60-74%
    POOR = "poor"  # <60%


class Framework(str, Enum):
    """Reporting frameworks."""
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD = "csrd"
    TCFD = "tcfd"
    CDP = "cdp"
    SECR = "secr"
    ESOS = "esos"


class ComplianceStatus(str, Enum):
    """Compliance check status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class ErrorType(str, Enum):
    """Error types for tracking."""
    VALIDATION_ERROR = "validation_error"
    CALCULATION_ERROR = "calculation_error"
    DATABASE_ERROR = "database_error"
    API_ERROR = "api_error"
    TIMEOUT_ERROR = "timeout_error"
    CONFIGURATION_ERROR = "configuration_error"
    DATA_NOT_FOUND = "data_not_found"


class DualReportingReconciliationMetrics:
    """
    Thread-safe singleton metrics collector for Dual Reporting Reconciliation Agent.

    Provides 12 Prometheus metrics for tracking reconciliation operations,
    discrepancy detection, quality scoring, compliance checks, and errors.

    Attributes:
        reconciliations_total: Counter for reconciliation operations
        reconciliation_duration: Histogram for operation duration
        discrepancies_total: Counter for detected discrepancies
        discrepancy_pct: Histogram for discrepancy percentages
        quality_score: Histogram for quality scores
        quality_grade_total: Counter for quality grades
        reports_generated_total: Counter for generated reports
        trend_analyses_total: Counter for trend analyses
        compliance_checks_total: Counter for compliance checks
        batch_size: Histogram for batch sizes
        errors_total: Counter for errors
        pif_value: Histogram for PIF (Percentage in Form) values

    Example:
        >>> metrics = DualReportingReconciliationMetrics()
        >>> metrics.record_reconciliation(
        ...     tenant_id="tenant-123",
        ...     status="success",
        ...     energy_type="natural_gas",
        ...     duration_s=1.2,
        ...     discrepancy_pct=5.3,
        ...     pif=85.0
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
            'reconciliations': 0,
            'discrepancies': 0,
            'reports_generated': 0,
            'trend_analyses': 0,
            'compliance_checks': 0,
            'errors': 0,
        }

        # Initialize Prometheus metrics
        self._init_metrics()

        logger.info(
            "DualReportingReconciliationMetrics initialized (Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable"
        )

    def _init_metrics(self):
        """Initialize all Prometheus metrics."""

        # 1. Reconciliations total counter
        self.reconciliations_total = Counter(
            'gl_drr_reconciliations_total',
            'Total number of dual reporting reconciliations performed',
            ['tenant_id', 'status', 'energy_type']
        )

        # 2. Reconciliation duration histogram
        self.reconciliation_duration = Histogram(
            'gl_drr_reconciliation_duration_seconds',
            'Duration of reconciliation operations in seconds',
            ['energy_type'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        )

        # 3. Discrepancies total counter
        self.discrepancies_total = Counter(
            'gl_drr_discrepancies_total',
            'Total number of discrepancies detected',
            ['discrepancy_type', 'materiality', 'direction']
        )

        # 4. Discrepancy percentage histogram
        self.discrepancy_pct = Histogram(
            'gl_drr_discrepancy_pct',
            'Discrepancy percentage distribution',
            ['energy_type'],
            buckets=[1, 5, 10, 15, 25, 50, 75, 100, 200]
        )

        # 5. Quality score histogram
        self.quality_score = Histogram(
            'gl_drr_quality_score',
            'Data quality score distribution (0.0-1.0)',
            ['dimension'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        # 6. Quality grade counter
        self.quality_grade_total = Counter(
            'gl_drr_quality_grade_total',
            'Total count by quality grade classification',
            ['grade']
        )

        # 7. Reports generated counter
        self.reports_generated_total = Counter(
            'gl_drr_reports_generated_total',
            'Total number of reconciliation reports generated',
            ['framework']
        )

        # 8. Trend analyses counter
        self.trend_analyses_total = Counter(
            'gl_drr_trend_analyses_total',
            'Total number of trend analyses performed',
            ['tenant_id']
        )

        # 9. Compliance checks counter
        self.compliance_checks_total = Counter(
            'gl_drr_compliance_checks_total',
            'Total number of compliance checks performed',
            ['framework', 'status']
        )

        # 10. Batch size histogram
        self.batch_size = Histogram(
            'gl_drr_batch_size',
            'Distribution of reconciliation batch sizes (number of periods)',
            ['tenant_id'],
            buckets=[1, 3, 6, 12, 24, 60]
        )

        # 11. Errors counter
        self.errors_total = Counter(
            'gl_drr_errors_total',
            'Total number of errors by type and operation',
            ['error_type', 'operation']
        )

        # 12. PIF value histogram
        self.pif_value = Histogram(
            'gl_drr_pif_value',
            'Percentage in Form (PIF) distribution',
            ['tenant_id'],
            buckets=[-50, -25, 0, 10, 25, 50, 75, 90, 100]
        )

    def record_reconciliation(
        self,
        tenant_id: str,
        status: str,
        energy_type: str,
        duration_s: float,
        discrepancy_pct: Optional[float] = None,
        pif: Optional[float] = None
    ) -> None:
        """
        Record a reconciliation operation.

        Args:
            tenant_id: Tenant identifier
            status: Reconciliation status (success/failed/partial/skipped)
            energy_type: Type of energy being reconciled
            duration_s: Operation duration in seconds
            discrepancy_pct: Discrepancy percentage (optional)
            pif: Percentage in Form value (optional)

        Example:
            >>> metrics.record_reconciliation(
            ...     tenant_id="tenant-123",
            ...     status="success",
            ...     energy_type="natural_gas",
            ...     duration_s=1.2,
            ...     discrepancy_pct=5.3,
            ...     pif=85.0
            ... )
        """
        try:
            # Validate status
            if status not in [s.value for s in ReconciliationStatus]:
                logger.warning("Invalid reconciliation status: %s", status)
                status = ReconciliationStatus.FAILED.value

            # Record reconciliation count
            self.reconciliations_total.labels(
                tenant_id=tenant_id,
                status=status,
                energy_type=energy_type
            ).inc()

            # Record duration
            self.reconciliation_duration.labels(
                energy_type=energy_type
            ).observe(duration_s)

            # Record discrepancy percentage if provided
            if discrepancy_pct is not None:
                self.discrepancy_pct.labels(
                    energy_type=energy_type
                ).observe(abs(discrepancy_pct))

            # Record PIF if provided
            if pif is not None:
                self.pif_value.labels(
                    tenant_id=tenant_id
                ).observe(pif)

            # Update in-memory stats
            self._in_memory_stats['reconciliations'] += 1

            logger.debug(
                f"Recorded reconciliation: tenant={tenant_id}, status={status}, "
                f"energy_type={energy_type}, duration={duration_s:.3f}s"
            )

        except Exception as e:
            logger.error("Failed to record reconciliation metrics: %s", e, exc_info=True)

    def record_discrepancy(
        self,
        discrepancy_type: str,
        materiality: str,
        direction: str
    ) -> None:
        """
        Record a detected discrepancy.

        Args:
            discrepancy_type: Type of discrepancy
            materiality: Materiality level (critical/high/medium/low/negligible)
            direction: Direction of discrepancy (over/under/neutral)

        Example:
            >>> metrics.record_discrepancy(
            ...     discrepancy_type="energy_mismatch",
            ...     materiality="high",
            ...     direction="over"
            ... )
        """
        try:
            # Validate inputs
            if discrepancy_type not in [d.value for d in DiscrepancyType]:
                logger.warning("Invalid discrepancy type: %s", discrepancy_type)
                discrepancy_type = DiscrepancyType.ENERGY_MISMATCH.value

            if materiality not in [m.value for m in Materiality]:
                logger.warning("Invalid materiality: %s", materiality)
                materiality = Materiality.MEDIUM.value

            if direction not in [d.value for d in Direction]:
                logger.warning("Invalid direction: %s", direction)
                direction = Direction.NEUTRAL.value

            # Record discrepancy
            self.discrepancies_total.labels(
                discrepancy_type=discrepancy_type,
                materiality=materiality,
                direction=direction
            ).inc()

            # Update in-memory stats
            self._in_memory_stats['discrepancies'] += 1

            logger.debug(
                f"Recorded discrepancy: type={discrepancy_type}, "
                f"materiality={materiality}, direction={direction}"
            )

        except Exception as e:
            logger.error("Failed to record discrepancy metrics: %s", e, exc_info=True)

    def record_quality_score(
        self,
        dimension: str,
        score: float,
        grade: Optional[str] = None
    ) -> None:
        """
        Record a data quality score.

        Args:
            dimension: Quality dimension being measured
            score: Quality score (0.0-1.0)
            grade: Quality grade (optional, auto-calculated if not provided)

        Example:
            >>> metrics.record_quality_score(
            ...     dimension="completeness",
            ...     score=0.85,
            ...     grade="good"
            ... )
        """
        try:
            # Validate dimension
            if dimension not in [d.value for d in QualityDimension]:
                logger.warning("Invalid quality dimension: %s", dimension)
                dimension = QualityDimension.OVERALL.value

            # Clamp score to valid range
            score = max(0.0, min(1.0, score))

            # Record quality score
            self.quality_score.labels(
                dimension=dimension
            ).observe(score)

            # Auto-calculate grade if not provided
            if grade is None:
                score_pct = score * 100
                if score_pct >= 90:
                    grade = QualityGrade.EXCELLENT.value
                elif score_pct >= 75:
                    grade = QualityGrade.GOOD.value
                elif score_pct >= 60:
                    grade = QualityGrade.FAIR.value
                else:
                    grade = QualityGrade.POOR.value

            # Validate grade
            if grade not in [g.value for g in QualityGrade]:
                logger.warning("Invalid quality grade: %s", grade)
                grade = QualityGrade.FAIR.value

            # Record quality grade
            self.quality_grade_total.labels(
                grade=grade
            ).inc()

            logger.debug(
                f"Recorded quality score: dimension={dimension}, "
                f"score={score:.3f}, grade={grade}"
            )

        except Exception as e:
            logger.error("Failed to record quality score metrics: %s", e, exc_info=True)

    def record_report_generated(self, framework: str) -> None:
        """
        Record a generated reconciliation report.

        Args:
            framework: Reporting framework (ghg_protocol/iso_14064/csrd/etc)

        Example:
            >>> metrics.record_report_generated("ghg_protocol")
        """
        try:
            # Validate framework
            if framework not in [f.value for f in Framework]:
                logger.warning("Invalid framework: %s", framework)
                framework = Framework.GHG_PROTOCOL.value

            # Record report generation
            self.reports_generated_total.labels(
                framework=framework
            ).inc()

            # Update in-memory stats
            self._in_memory_stats['reports_generated'] += 1

            logger.debug("Recorded report generation: framework=%s", framework)

        except Exception as e:
            logger.error("Failed to record report generation metrics: %s", e, exc_info=True)

    def record_trend_analysis(self, tenant_id: str) -> None:
        """
        Record a trend analysis operation.

        Args:
            tenant_id: Tenant identifier

        Example:
            >>> metrics.record_trend_analysis("tenant-123")
        """
        try:
            # Record trend analysis
            self.trend_analyses_total.labels(
                tenant_id=tenant_id
            ).inc()

            # Update in-memory stats
            self._in_memory_stats['trend_analyses'] += 1

            logger.debug("Recorded trend analysis: tenant=%s", tenant_id)

        except Exception as e:
            logger.error("Failed to record trend analysis metrics: %s", e, exc_info=True)

    def record_compliance_check(self, framework: str, status: str) -> None:
        """
        Record a compliance check operation.

        Args:
            framework: Reporting framework
            status: Compliance status (compliant/non_compliant/warning/not_applicable)

        Example:
            >>> metrics.record_compliance_check("ghg_protocol", "compliant")
        """
        try:
            # Validate framework
            if framework not in [f.value for f in Framework]:
                logger.warning("Invalid framework: %s", framework)
                framework = Framework.GHG_PROTOCOL.value

            # Validate status
            if status not in [s.value for s in ComplianceStatus]:
                logger.warning("Invalid compliance status: %s", status)
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
            logger.error("Failed to record compliance check metrics: %s", e, exc_info=True)

    def record_batch(self, tenant_id: str, batch_size: int) -> None:
        """
        Record a reconciliation batch operation.

        Args:
            tenant_id: Tenant identifier
            batch_size: Number of periods in the batch

        Example:
            >>> metrics.record_batch("tenant-123", 12)
        """
        try:
            # Record batch size
            self.batch_size.labels(
                tenant_id=tenant_id
            ).observe(batch_size)

            logger.debug("Recorded batch: tenant=%s, size=%s", tenant_id, batch_size)

        except Exception as e:
            logger.error("Failed to record batch metrics: %s", e, exc_info=True)

    def record_error(self, error_type: str, operation: str) -> None:
        """
        Record an error occurrence.

        Args:
            error_type: Type of error
            operation: Operation where error occurred

        Example:
            >>> metrics.record_error("validation_error", "reconcile")
        """
        try:
            # Validate error type
            if error_type not in [e.value for e in ErrorType]:
                logger.warning("Invalid error type: %s", error_type)
                error_type = ErrorType.VALIDATION_ERROR.value

            # Record error
            self.errors_total.labels(
                error_type=error_type,
                operation=operation
            ).inc()

            # Update in-memory stats
            self._in_memory_stats['errors'] += 1

            logger.debug("Recorded error: type=%s, operation=%s", error_type, operation)

        except Exception as e:
            logger.error("Failed to record error metrics: %s", e, exc_info=True)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of metrics collected since initialization.

        Returns:
            Dictionary with metrics summary including counts and uptime

        Example:
            >>> summary = metrics.get_metrics_summary()
            >>> print(summary['reconciliations'])
            1234
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
                    'reconciliations_per_hour': (
                        self._in_memory_stats['reconciliations'] / (uptime_seconds / 3600)
                        if uptime_seconds > 0 else 0
                    ),
                    'discrepancies_per_hour': (
                        self._in_memory_stats['discrepancies'] / (uptime_seconds / 3600)
                        if uptime_seconds > 0 else 0
                    ),
                    'reports_per_hour': (
                        self._in_memory_stats['reports_generated'] / (uptime_seconds / 3600)
                        if uptime_seconds > 0 else 0
                    ),
                    'errors_per_hour': (
                        self._in_memory_stats['errors'] / (uptime_seconds / 3600)
                        if uptime_seconds > 0 else 0
                    ),
                }
            }

            logger.debug("Generated metrics summary: %s", summary)
            return summary

        except Exception as e:
            logger.error("Failed to generate metrics summary: %s", e, exc_info=True)
            return {
                'error': str(e),
                'prometheus_available': PROMETHEUS_AVAILABLE,
            }

    def record_batch_reconciliation(
        self,
        tenant_id: str,
        batch_size: int,
        successful: int,
        failed: int,
        duration_s: float,
        avg_discrepancy_pct: float,
        avg_pif: float
    ) -> None:
        """
        Record a batch reconciliation operation with aggregated metrics.

        Args:
            tenant_id: Tenant identifier
            batch_size: Total number of periods processed
            successful: Number of successful reconciliations
            failed: Number of failed reconciliations
            duration_s: Total batch processing duration
            avg_discrepancy_pct: Average discrepancy percentage
            avg_pif: Average PIF value

        Example:
            >>> metrics.record_batch_reconciliation(
            ...     tenant_id="tenant-123",
            ...     batch_size=12,
            ...     successful=11,
            ...     failed=1,
            ...     duration_s=15.5,
            ...     avg_discrepancy_pct=3.2,
            ...     avg_pif=87.5
            ... )
        """
        try:
            # Record batch size
            self.record_batch(tenant_id, batch_size)

            # Record successful reconciliations
            for _ in range(successful):
                self.reconciliations_total.labels(
                    tenant_id=tenant_id,
                    status=ReconciliationStatus.SUCCESS.value,
                    energy_type="batch"
                ).inc()

            # Record failed reconciliations
            for _ in range(failed):
                self.reconciliations_total.labels(
                    tenant_id=tenant_id,
                    status=ReconciliationStatus.FAILED.value,
                    energy_type="batch"
                ).inc()

            # Record average metrics
            avg_duration = duration_s / batch_size if batch_size > 0 else 0
            self.reconciliation_duration.labels(
                energy_type="batch"
            ).observe(avg_duration)

            self.discrepancy_pct.labels(
                energy_type="batch"
            ).observe(abs(avg_discrepancy_pct))

            self.pif_value.labels(
                tenant_id=tenant_id
            ).observe(avg_pif)

            # Update in-memory stats
            self._in_memory_stats['reconciliations'] += batch_size

            logger.info(
                f"Recorded batch reconciliation: tenant={tenant_id}, "
                f"batch_size={batch_size}, successful={successful}, "
                f"failed={failed}, duration={duration_s:.2f}s"
            )

        except Exception as e:
            logger.error("Failed to record batch reconciliation metrics: %s", e, exc_info=True)

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
                'reconciliations': 0,
                'discrepancies': 0,
                'reports_generated': 0,
                'trend_analyses': 0,
                'compliance_checks': 0,
                'errors': 0,
            }
            self._start_time = datetime.utcnow()

            logger.info("Reset in-memory statistics")

        except Exception as e:
            logger.error("Failed to reset statistics: %s", e, exc_info=True)


# Singleton instance for module-level access
_metrics_instance = None
_metrics_lock = threading.Lock()


def get_metrics() -> DualReportingReconciliationMetrics:
    """
    Get the singleton DualReportingReconciliationMetrics instance.

    Thread-safe accessor for the global metrics instance.

    Returns:
        DualReportingReconciliationMetrics singleton instance

    Example:
        >>> from greenlang.agents.mrv.dual_reporting_reconciliation.metrics import get_metrics
        >>> metrics = get_metrics()
        >>> metrics.record_reconciliation(...)
    """
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = DualReportingReconciliationMetrics()

    return _metrics_instance


__all__ = [
    'DualReportingReconciliationMetrics',
    'get_metrics',
    'ReconciliationStatus',
    'DiscrepancyType',
    'Materiality',
    'Direction',
    'QualityDimension',
    'QualityGrade',
    'Framework',
    'ComplianceStatus',
    'ErrorType',
    'PROMETHEUS_AVAILABLE',
]
