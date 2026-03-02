# -*- coding: utf-8 -*-
"""
Scope 3 Category Mapper Prometheus Metrics - AGENT-MRV-029

13 Prometheus metrics with gl_scm_ prefix for monitoring the
GL-MRV-X-040 Scope 3 Category Mapper Agent.

This module provides Prometheus metrics tracking for Scope 3 category
classification and routing operations. Covers classification throughput,
routing distribution, double-counting detection, completeness scoring,
compliance tracking, and pipeline stage latencies.

Thread-safe singleton pattern with graceful fallback if Prometheus
unavailable.

Metrics prefix: gl_scm_

13 Prometheus Metrics:
    1.  gl_scm_classifications_total              - Counter: total classifications
    2.  gl_scm_classification_duration_seconds     - Histogram: classification latency
    3.  gl_scm_batch_size                          - Histogram: batch sizes
    4.  gl_scm_routing_total                       - Counter: records routed
    5.  gl_scm_routing_duration_seconds            - Histogram: routing latency
    6.  gl_scm_double_counting_detected_total      - Counter: DC detections
    7.  gl_scm_unmapped_records_total              - Counter: unmapped records
    8.  gl_scm_confidence_score                    - Histogram: confidence distribution
    9.  gl_scm_completeness_score                  - Gauge: completeness score
    10. gl_scm_categories_active                   - Gauge: active categories
    11. gl_scm_compliance_score                    - Gauge: compliance score
    12. gl_scm_errors_total                        - Counter: processing errors
    13. gl_scm_pipeline_stage_duration_seconds     - Histogram: pipeline stages

Example:
    >>> metrics = Scope3CategoryMapperMetrics()
    >>> metrics.record_classification(
    ...     category="cat_1",
    ...     method="naics",
    ...     confidence_level="high",
    ...     duration=0.003,
    ... )

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-X-040
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
# Enumerations -- Category Mapper domain-specific label value sets
# ===========================================================================


class CategoryLabel(str, Enum):
    """Scope 3 category labels (1-15 plus unmapped)."""

    CAT_1 = "cat_1"
    CAT_2 = "cat_2"
    CAT_3 = "cat_3"
    CAT_4 = "cat_4"
    CAT_5 = "cat_5"
    CAT_6 = "cat_6"
    CAT_7 = "cat_7"
    CAT_8 = "cat_8"
    CAT_9 = "cat_9"
    CAT_10 = "cat_10"
    CAT_11 = "cat_11"
    CAT_12 = "cat_12"
    CAT_13 = "cat_13"
    CAT_14 = "cat_14"
    CAT_15 = "cat_15"
    UNMAPPED = "unmapped"


class ClassificationMethodLabel(str, Enum):
    """Classification method labels."""

    NAICS = "naics"
    ISIC = "isic"
    UNSPSC = "unspsc"
    HS_CODE = "hs_code"
    GL_ACCOUNT = "gl_account"
    KEYWORD = "keyword"
    MANUAL = "manual"


class ConfidenceLevelLabel(str, Enum):
    """Classification confidence levels."""

    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class SourceTypeLabel(str, Enum):
    """Data source type labels."""

    SPEND = "spend"
    PURCHASE_ORDER = "purchase_order"
    BOM = "bom"
    TRAVEL = "travel"
    FLEET = "fleet"
    WASTE = "waste"
    LEASE = "lease"
    LOGISTICS = "logistics"
    PRODUCT_SALES = "product_sales"
    INVESTMENT = "investment"
    FRANCHISE = "franchise"
    ENERGY = "energy"
    SUPPLIER = "supplier"


class RoutingActionLabel(str, Enum):
    """Routing action labels."""

    ROUTE = "route"
    SPLIT_ROUTE = "split_route"
    QUEUE_REVIEW = "queue_review"
    EXCLUDE = "exclude"


class DoubleCountingRuleLabel(str, Enum):
    """Double-counting rule labels."""

    DC_SCM_001 = "dc_scm_001"
    DC_SCM_002 = "dc_scm_002"
    DC_SCM_003 = "dc_scm_003"
    DC_SCM_004 = "dc_scm_004"
    DC_SCM_005 = "dc_scm_005"
    DC_SCM_006 = "dc_scm_006"
    DC_SCM_007 = "dc_scm_007"
    DC_SCM_008 = "dc_scm_008"
    DC_SCM_009 = "dc_scm_009"
    DC_SCM_010 = "dc_scm_010"


class CompanyTypeLabel(str, Enum):
    """Company type labels for completeness scoring."""

    MANUFACTURER = "manufacturer"
    SERVICES = "services"
    FINANCIAL = "financial"
    RETAILER = "retailer"
    ENERGY = "energy"
    MINING = "mining"
    AGRICULTURE = "agriculture"
    TRANSPORT = "transport"


class ComplianceFrameworkLabel(str, Enum):
    """Compliance framework labels."""

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD = "csrd"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"
    SEC_CLIMATE = "sec_climate"
    ISSB_S2 = "issb_s2"


class PipelineStageLabel(str, Enum):
    """Pipeline stage names for duration tracking."""

    VALIDATE = "validate"
    SOURCE_CLASSIFY = "source_classify"
    CODE_LOOKUP = "code_lookup"
    SPEND_CLASSIFY = "spend_classify"
    BOUNDARY = "boundary"
    DOUBLE_COUNTING = "double_counting"
    SPLIT = "split"
    RECOMMEND = "recommend"
    COMPLETENESS = "completeness"
    SEAL = "seal"


class ErrorTypeLabel(str, Enum):
    """Error type categories."""

    VALIDATION = "validation"
    CLASSIFICATION = "classification"
    ROUTING = "routing"
    BOUNDARY = "boundary"
    DATABASE = "database"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


# ===========================================================================
# Scope3CategoryMapperMetrics -- Thread-safe Singleton
# ===========================================================================


class Scope3CategoryMapperMetrics:
    """
    Thread-safe singleton metrics collector for Category Mapper (MRV-029).

    Provides 13 Prometheus metrics for tracking Scope 3 category
    classification and routing across all 15 categories.

    All metrics use the ``gl_scm_`` prefix to ensure namespace isolation
    within the GreenLang Prometheus ecosystem.

    Example:
        >>> metrics = Scope3CategoryMapperMetrics()
        >>> metrics.record_classification(
        ...     category="cat_1", method="naics",
        ...     confidence_level="high", duration=0.003,
        ... )
    """

    _instance: Optional["Scope3CategoryMapperMetrics"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "Scope3CategoryMapperMetrics":
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
            "classifications": 0,
            "routings": 0,
            "double_counting_detections": 0,
            "unmapped": 0,
            "errors": 0,
        }

        self._init_metrics()

        logger.info(
            "Scope3CategoryMapperMetrics initialized (Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable",
        )

    def _init_metrics(self) -> None:
        """
        Initialize all 13 Prometheus metrics with gl_scm_ prefix.

        When Prometheus is available, metric registration may fail if the
        metrics were previously registered. In that case we unregister
        and re-register to obtain fresh collector objects.
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
        # 1. gl_scm_classifications_total (Counter)
        #    Total records classified by category, method, and confidence.
        # ------------------------------------------------------------------
        self.classifications_total = _safe_create(
            Counter,
            "gl_scm_classifications_total",
            "Total records classified into Scope 3 categories",
            ["category", "method", "confidence_level"],
        )

        # ------------------------------------------------------------------
        # 2. gl_scm_classification_duration_seconds (Histogram)
        #    Classification latency per source type.
        # ------------------------------------------------------------------
        self.classification_duration_seconds = _safe_create(
            Histogram,
            "gl_scm_classification_duration_seconds",
            "Duration of classification operations in seconds",
            ["source_type"],
            buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
        )

        # ------------------------------------------------------------------
        # 3. gl_scm_batch_size (Histogram)
        #    Batch sizes processed per source type.
        # ------------------------------------------------------------------
        self.batch_size = _safe_create(
            Histogram,
            "gl_scm_batch_size",
            "Batch classification sizes",
            ["source_type"],
            buckets=(1, 10, 50, 100, 500, 1000, 5000, 10000, 25000, 50000),
        )

        # ------------------------------------------------------------------
        # 4. gl_scm_routing_total (Counter)
        #    Records routed to category agents by category and action.
        # ------------------------------------------------------------------
        self.routing_total = _safe_create(
            Counter,
            "gl_scm_routing_total",
            "Total records routed to category agents",
            ["category", "action"],
        )

        # ------------------------------------------------------------------
        # 5. gl_scm_routing_duration_seconds (Histogram)
        #    Routing operation latency.
        # ------------------------------------------------------------------
        self.routing_duration_seconds = _safe_create(
            Histogram,
            "gl_scm_routing_duration_seconds",
            "Duration of routing operations in seconds",
            [],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )

        # ------------------------------------------------------------------
        # 6. gl_scm_double_counting_detected_total (Counter)
        #    Double-counting detections by DC-SCM rule.
        # ------------------------------------------------------------------
        self.double_counting_detected_total = _safe_create(
            Counter,
            "gl_scm_double_counting_detected_total",
            "Double-counting overlaps detected by rule",
            ["rule"],
        )

        # ------------------------------------------------------------------
        # 7. gl_scm_unmapped_records_total (Counter)
        #    Records that could not be classified by source type.
        # ------------------------------------------------------------------
        self.unmapped_records_total = _safe_create(
            Counter,
            "gl_scm_unmapped_records_total",
            "Records that could not be classified into a Scope 3 category",
            ["source_type"],
        )

        # ------------------------------------------------------------------
        # 8. gl_scm_confidence_score (Histogram)
        #    Distribution of classification confidence scores by category.
        # ------------------------------------------------------------------
        self.confidence_score = _safe_create(
            Histogram,
            "gl_scm_confidence_score",
            "Distribution of classification confidence scores",
            ["category"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
        )

        # ------------------------------------------------------------------
        # 9. gl_scm_completeness_score (Gauge)
        #    Current completeness score (0-100) by company type.
        # ------------------------------------------------------------------
        self.completeness_score = _safe_create(
            Gauge,
            "gl_scm_completeness_score",
            "Category completeness score (0-100) by company type",
            ["company_type"],
        )

        # ------------------------------------------------------------------
        # 10. gl_scm_categories_active (Gauge)
        #     Number of active categories with data by organization.
        # ------------------------------------------------------------------
        self.categories_active = _safe_create(
            Gauge,
            "gl_scm_categories_active",
            "Number of active Scope 3 categories with data",
            ["organization_id"],
        )

        # ------------------------------------------------------------------
        # 11. gl_scm_compliance_score (Gauge)
        #     Framework compliance score (0-100) by framework.
        # ------------------------------------------------------------------
        self.compliance_score = _safe_create(
            Gauge,
            "gl_scm_compliance_score",
            "Compliance score (0-100) by framework",
            ["framework"],
        )

        # ------------------------------------------------------------------
        # 12. gl_scm_errors_total (Counter)
        #     Processing errors by error type.
        # ------------------------------------------------------------------
        self.errors_total = _safe_create(
            Counter,
            "gl_scm_errors_total",
            "Total processing errors in category mapper",
            ["error_type"],
        )

        # ------------------------------------------------------------------
        # 13. gl_scm_pipeline_stage_duration_seconds (Histogram)
        #     Duration of individual pipeline stages.
        # ------------------------------------------------------------------
        self.pipeline_stage_duration_seconds = _safe_create(
            Histogram,
            "gl_scm_pipeline_stage_duration_seconds",
            "Duration of category mapper pipeline stages",
            ["stage"],
            buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )

        # ------------------------------------------------------------------
        # Agent info metric (non-numeric metadata)
        # ------------------------------------------------------------------
        self.agent_info = _safe_create(
            Info,
            "gl_scm_agent",
            "Scope 3 Category Mapper Agent metadata",
        )
        if PROMETHEUS_AVAILABLE:
            try:
                self.agent_info.info({
                    "agent_id": "GL-MRV-X-040",
                    "version": "1.0.0",
                    "scope": "scope_3_cross_cutting",
                    "description": "Scope 3 category mapper and router",
                    "categories": "1-15",
                })
            except Exception:
                pass

    # ======================================================================
    # Primary recording methods
    # ======================================================================

    def record_classification(
        self,
        category: str,
        method: str,
        confidence_level: str,
        duration: float,
        source_type: str = "spend",
        confidence_value: Optional[float] = None,
    ) -> None:
        """
        Record a classification operation.

        Args:
            category: Scope 3 category label (cat_1..cat_15 or unmapped).
            method: Classification method (naics, isic, gl_account, etc.).
            confidence_level: Confidence level (very_high..very_low).
            duration: Operation duration in seconds.
            source_type: Data source type.
            confidence_value: Raw confidence score 0.0-1.0 (optional).
        """
        try:
            category = self._validate_enum_value(
                category, CategoryLabel, CategoryLabel.UNMAPPED.value
            )
            method = self._validate_enum_value(
                method, ClassificationMethodLabel,
                ClassificationMethodLabel.KEYWORD.value,
            )
            confidence_level = self._validate_enum_value(
                confidence_level, ConfidenceLevelLabel,
                ConfidenceLevelLabel.MEDIUM.value,
            )
            source_type = self._validate_enum_value(
                source_type, SourceTypeLabel, SourceTypeLabel.SPEND.value
            )

            self.classifications_total.labels(
                category=category, method=method,
                confidence_level=confidence_level,
            ).inc()

            if duration > 0:
                self.classification_duration_seconds.labels(
                    source_type=source_type
                ).observe(duration)

            if confidence_value is not None:
                self.confidence_score.labels(category=category).observe(
                    confidence_value
                )

            if category == CategoryLabel.UNMAPPED.value:
                self.unmapped_records_total.labels(
                    source_type=source_type
                ).inc()
                with self._stats_lock:
                    self._in_memory_stats["unmapped"] += 1

            with self._stats_lock:
                self._in_memory_stats["classifications"] += 1

        except Exception as e:
            logger.error(
                "Failed to record classification metrics: %s", e, exc_info=True
            )

    def record_routing(
        self,
        category: str,
        action: str,
        duration: float,
    ) -> None:
        """
        Record a routing operation.

        Args:
            category: Target Scope 3 category label.
            action: Routing action (route, split_route, queue_review, exclude).
            duration: Routing duration in seconds.
        """
        try:
            category = self._validate_enum_value(
                category, CategoryLabel, CategoryLabel.UNMAPPED.value
            )
            action = self._validate_enum_value(
                action, RoutingActionLabel,
                RoutingActionLabel.QUEUE_REVIEW.value,
            )

            self.routing_total.labels(category=category, action=action).inc()

            if duration > 0:
                self.routing_duration_seconds.observe(duration)

            with self._stats_lock:
                self._in_memory_stats["routings"] += 1

        except Exception as e:
            logger.error(
                "Failed to record routing metrics: %s", e, exc_info=True
            )

    def record_double_counting(self, rule: str) -> None:
        """
        Record a double-counting detection.

        Args:
            rule: DC-SCM rule that was triggered (e.g., dc_scm_001).
        """
        try:
            self.double_counting_detected_total.labels(rule=rule).inc()

            with self._stats_lock:
                self._in_memory_stats["double_counting_detections"] += 1

        except Exception as e:
            logger.error("Failed to record DC detection: %s", e, exc_info=True)

    def record_error(self, error_type: str) -> None:
        """
        Record a processing error.

        Args:
            error_type: Error category (validation, classification, etc.).
        """
        try:
            error_type = self._validate_enum_value(
                error_type, ErrorTypeLabel, ErrorTypeLabel.UNKNOWN.value
            )
            self.errors_total.labels(error_type=error_type).inc()

            with self._stats_lock:
                self._in_memory_stats["errors"] += 1

        except Exception as e:
            logger.error("Failed to record error metric: %s", e, exc_info=True)

    def record_pipeline_stage(self, stage: str, duration: float) -> None:
        """
        Record pipeline stage duration.

        Args:
            stage: Pipeline stage name.
            duration: Stage duration in seconds.
        """
        try:
            stage = self._validate_enum_value(
                stage, PipelineStageLabel, PipelineStageLabel.VALIDATE.value
            )
            self.pipeline_stage_duration_seconds.labels(
                stage=stage
            ).observe(duration)

        except Exception as e:
            logger.error(
                "Failed to record pipeline stage: %s", e, exc_info=True
            )

    def record_batch(self, source_type: str, size: int) -> None:
        """
        Record batch processing size.

        Args:
            source_type: Data source type.
            size: Number of records in batch.
        """
        try:
            source_type = self._validate_enum_value(
                source_type, SourceTypeLabel, SourceTypeLabel.SPEND.value
            )
            self.batch_size.labels(source_type=source_type).observe(size)

        except Exception as e:
            logger.error("Failed to record batch size: %s", e, exc_info=True)

    def update_completeness(self, company_type: str, score: float) -> None:
        """
        Update completeness score gauge.

        Args:
            company_type: Company type (manufacturer, services, etc.).
            score: Completeness score 0-100.
        """
        try:
            company_type = self._validate_enum_value(
                company_type, CompanyTypeLabel,
                CompanyTypeLabel.SERVICES.value,
            )
            self.completeness_score.labels(
                company_type=company_type
            ).set(score)

        except Exception as e:
            logger.error(
                "Failed to update completeness score: %s", e, exc_info=True
            )

    def update_compliance(self, framework: str, score: float) -> None:
        """
        Update compliance score gauge.

        Args:
            framework: Compliance framework identifier.
            score: Compliance score 0-100.
        """
        try:
            framework = self._validate_enum_value(
                framework, ComplianceFrameworkLabel,
                ComplianceFrameworkLabel.GHG_PROTOCOL.value,
            )
            self.compliance_score.labels(framework=framework).set(score)

        except Exception as e:
            logger.error(
                "Failed to update compliance score: %s", e, exc_info=True
            )

    def update_categories_active(
        self, organization_id: str, count: int
    ) -> None:
        """
        Update active categories gauge.

        Args:
            organization_id: Organization identifier.
            count: Number of active categories.
        """
        try:
            self.categories_active.labels(
                organization_id=organization_id
            ).set(count)

        except Exception as e:
            logger.error(
                "Failed to update active categories: %s", e, exc_info=True
            )

    # ======================================================================
    # Utility methods
    # ======================================================================

    @staticmethod
    def _validate_enum_value(
        value: str, enum_cls: type, default: str
    ) -> str:
        """
        Validate and normalize a label value against an enum.

        Args:
            value: Raw label value.
            enum_cls: Enum class to validate against.
            default: Default value if validation fails.

        Returns:
            Validated label value string.
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
            Dictionary of current statistics.
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
                logger.info("Scope3CategoryMapperMetrics singleton reset")


# ===========================================================================
# Module-level singleton accessor
# ===========================================================================

_metrics_instance: Optional[Scope3CategoryMapperMetrics] = None
_metrics_lock: threading.Lock = threading.Lock()


def get_metrics() -> Scope3CategoryMapperMetrics:
    """
    Get the singleton Scope3CategoryMapperMetrics instance.

    Thread-safe accessor for the global metrics instance.

    Returns:
        Scope3CategoryMapperMetrics singleton instance.

    Example:
        >>> from greenlang.scope3_category_mapper.metrics import get_metrics
        >>> metrics = get_metrics()
        >>> metrics.record_classification(
        ...     category="cat_1", method="naics",
        ...     confidence_level="high", duration=0.003,
        ... )
    """
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = Scope3CategoryMapperMetrics()

    return _metrics_instance


def reset_metrics() -> None:
    """
    Reset the singleton metrics instance for testing purposes.

    Example:
        >>> from greenlang.scope3_category_mapper.metrics import reset_metrics
        >>> reset_metrics()
    """
    Scope3CategoryMapperMetrics.reset()


# ===========================================================================
# Convenience helper functions (module-level)
# ===========================================================================


def record_classification(
    category: str,
    method: str,
    confidence_level: str,
    duration: float,
    source_type: str = "spend",
    confidence_value: Optional[float] = None,
) -> None:
    """
    Record a classification via the singleton metrics instance.

    Args:
        category: Scope 3 category label.
        method: Classification method.
        confidence_level: Confidence level.
        duration: Duration in seconds.
        source_type: Data source type.
        confidence_value: Raw confidence 0.0-1.0 (optional).
    """
    get_metrics().record_classification(
        category=category,
        method=method,
        confidence_level=confidence_level,
        duration=duration,
        source_type=source_type,
        confidence_value=confidence_value,
    )


def record_routing(category: str, action: str, duration: float) -> None:
    """
    Record a routing operation via the singleton metrics instance.

    Args:
        category: Target category label.
        action: Routing action.
        duration: Duration in seconds.
    """
    get_metrics().record_routing(
        category=category, action=action, duration=duration,
    )


def record_error(error_type: str) -> None:
    """
    Record an error via the singleton metrics instance.

    Args:
        error_type: Error category.
    """
    get_metrics().record_error(error_type=error_type)


def record_pipeline_stage(stage: str, duration: float) -> None:
    """
    Record a pipeline stage duration via the singleton metrics instance.

    Args:
        stage: Pipeline stage name.
        duration: Duration in seconds.
    """
    get_metrics().record_pipeline_stage(stage=stage, duration=duration)


def update_completeness(company_type: str, score: float) -> None:
    """
    Update completeness score via the singleton metrics instance.

    Args:
        company_type: Company type.
        score: Completeness score 0-100.
    """
    get_metrics().update_completeness(
        company_type=company_type, score=score,
    )


def update_compliance(framework: str, score: float) -> None:
    """
    Update compliance score via the singleton metrics instance.

    Args:
        framework: Compliance framework.
        score: Compliance score 0-100.
    """
    get_metrics().update_compliance(framework=framework, score=score)


# ===========================================================================
# Context manager helpers
# ===========================================================================


@contextmanager
def track_classification(
    source_type: str = "spend",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a classification operation's lifecycle.

    Automatically measures wall-clock duration and records the operation
    when the context exits.

    Args:
        source_type: Data source type.

    Yields:
        Mutable context dict. Set ``context['category']``,
        ``context['method']``, ``context['confidence_level']``, and
        ``context['confidence_value']`` inside the block.

    Example:
        >>> with track_classification(source_type="spend") as ctx:
        ...     result = classifier.classify(record)
        ...     ctx["category"] = f"cat_{result.category}"
        ...     ctx["method"] = result.method
        ...     ctx["confidence_level"] = result.confidence_level
    """
    context: Dict[str, Any] = {
        "category": "unmapped",
        "method": "keyword",
        "confidence_level": "medium",
        "confidence_value": None,
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
        metrics = get_metrics()
        if context["status"] == "success":
            metrics.record_classification(
                category=context["category"],
                method=context["method"],
                confidence_level=context["confidence_level"],
                duration=duration,
                source_type=source_type,
                confidence_value=context.get("confidence_value"),
            )
        else:
            metrics.record_error("classification")


@contextmanager
def track_pipeline_stage(stage: str) -> Generator[None, None, None]:
    """
    Context manager that tracks a pipeline stage's duration.

    Args:
        stage: Pipeline stage name.

    Example:
        >>> with track_pipeline_stage("code_lookup"):
        ...     result = database.lookup_naics(code)
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
    "CategoryLabel",
    "ClassificationMethodLabel",
    "ConfidenceLevelLabel",
    "SourceTypeLabel",
    "RoutingActionLabel",
    "DoubleCountingRuleLabel",
    "CompanyTypeLabel",
    "ComplianceFrameworkLabel",
    "PipelineStageLabel",
    "ErrorTypeLabel",
    # Singleton class
    "Scope3CategoryMapperMetrics",
    # Module-level accessors
    "get_metrics",
    "reset_metrics",
    # Convenience functions
    "record_classification",
    "record_routing",
    "record_error",
    "record_pipeline_stage",
    "update_completeness",
    "update_compliance",
    # Context managers
    "track_classification",
    "track_pipeline_stage",
]
