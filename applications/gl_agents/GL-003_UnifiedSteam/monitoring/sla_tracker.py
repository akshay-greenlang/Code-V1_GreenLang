"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - SLA Tracking

This module provides Service Level Agreement (SLA) monitoring and compliance
tracking for the steam system optimization agent.

SLA Types:
    - Computation Latency SLA: Response time requirements
    - Availability SLA: System uptime requirements
    - Data Freshness SLA: Data currency requirements
    - Optimization Frequency SLA: Optimization cycle requirements

Example:
    >>> tracker = SLATracker()
    >>> tracker.define_sla(
    ...     name="computation_latency",
    ...     metric=SLAMetricType.LATENCY_P95,
    ...     threshold=500.0,  # milliseconds
    ...     time_window=timedelta(hours=1)
    ... )
    >>> result = tracker.check_sla_compliance("computation_latency")
    >>> print(f"SLA compliant: {result.is_compliant}")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import logging
import hashlib

logger = logging.getLogger(__name__)


class SLAMetricType(Enum):
    """Types of SLA metrics."""
    # Latency metrics
    LATENCY_AVG = "latency_avg"
    LATENCY_P50 = "latency_p50"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    LATENCY_MAX = "latency_max"

    # Availability metrics
    AVAILABILITY = "availability"
    UPTIME = "uptime"

    # Throughput metrics
    THROUGHPUT = "throughput"
    REQUEST_RATE = "request_rate"

    # Error metrics
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"

    # Data metrics
    DATA_FRESHNESS = "data_freshness"
    DATA_QUALITY = "data_quality"

    # Custom metrics
    CUSTOM = "custom"


class SLAComparisonOperator(Enum):
    """Comparison operators for SLA thresholds."""
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    EQUAL = "eq"


@dataclass
class SLADefinition:
    """Definition of a Service Level Agreement."""
    name: str
    metric: SLAMetricType
    threshold: float
    time_window: timedelta
    comparison: SLAComparisonOperator = SLAComparisonOperator.LESS_THAN_OR_EQUAL
    description: str = ""
    owner: str = ""
    priority: int = 1  # 1 = highest priority
    enabled: bool = True
    alert_on_breach: bool = True
    breach_tolerance_percent: float = 0.0  # Allow X% breach before alerting
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "metric": self.metric.value,
            "threshold": self.threshold,
            "time_window_seconds": self.time_window.total_seconds(),
            "comparison": self.comparison.value,
            "description": self.description,
            "owner": self.owner,
            "priority": self.priority,
            "enabled": self.enabled,
            "alert_on_breach": self.alert_on_breach,
            "breach_tolerance_percent": self.breach_tolerance_percent,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
        }


@dataclass
class SLAViolation:
    """Record of an SLA violation."""
    violation_id: str
    sla_name: str
    metric: SLAMetricType
    threshold: float
    actual_value: float
    comparison: SLAComparisonOperator
    occurred_at: datetime
    duration_seconds: float = 0.0
    resolved_at: Optional[datetime] = None
    is_resolved: bool = False
    impact_description: str = ""
    root_cause: str = ""
    remediation_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "violation_id": self.violation_id,
            "sla_name": self.sla_name,
            "metric": self.metric.value,
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "comparison": self.comparison.value,
            "occurred_at": self.occurred_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "is_resolved": self.is_resolved,
            "impact_description": self.impact_description,
            "root_cause": self.root_cause,
            "remediation_actions": self.remediation_actions,
        }


@dataclass
class SLAComplianceResult:
    """Result of an SLA compliance check."""
    sla_name: str
    is_compliant: bool
    metric: SLAMetricType
    threshold: float
    actual_value: float
    comparison: SLAComparisonOperator
    compliance_percent: float
    time_window: timedelta
    checked_at: datetime
    samples_count: int = 0
    breaches_count: int = 0
    current_violation: Optional[SLAViolation] = None
    trend: str = "stable"  # improving, stable, degrading
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "sla_name": self.sla_name,
            "is_compliant": self.is_compliant,
            "metric": self.metric.value,
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "comparison": self.comparison.value,
            "compliance_percent": self.compliance_percent,
            "time_window_seconds": self.time_window.total_seconds(),
            "checked_at": self.checked_at.isoformat(),
            "samples_count": self.samples_count,
            "breaches_count": self.breaches_count,
            "current_violation": (
                self.current_violation.to_dict() if self.current_violation else None
            ),
            "trend": self.trend,
            "message": self.message,
        }


@dataclass
class SLAReport:
    """SLA compliance report for a time period."""
    report_id: str
    report_period_start: datetime
    report_period_end: datetime
    generated_at: datetime
    total_slas: int = 0
    compliant_slas: int = 0
    non_compliant_slas: int = 0
    overall_compliance_percent: float = 0.0
    sla_results: Dict[str, SLAComplianceResult] = field(default_factory=dict)
    violations: List[SLAViolation] = field(default_factory=list)
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "report_id": self.report_id,
            "report_period": {
                "start": self.report_period_start.isoformat(),
                "end": self.report_period_end.isoformat(),
            },
            "generated_at": self.generated_at.isoformat(),
            "summary": {
                "total_slas": self.total_slas,
                "compliant_slas": self.compliant_slas,
                "non_compliant_slas": self.non_compliant_slas,
                "overall_compliance_percent": self.overall_compliance_percent,
            },
            "sla_results": {k: v.to_dict() for k, v in self.sla_results.items()},
            "violations": [v.to_dict() for v in self.violations],
            "text_summary": self.summary,
            "recommendations": self.recommendations,
        }


class SLATracker:
    """
    SLA tracking and compliance monitoring system.

    This class provides comprehensive SLA definition, monitoring, and reporting
    for the steam system optimization agent. It tracks computation latency,
    system availability, data freshness, and custom metrics.

    Attributes:
        namespace: SLA namespace for identification

    Example:
        >>> tracker = SLATracker(namespace="unifiedsteam")
        >>> tracker.define_sla(
        ...     name="computation_latency",
        ...     metric=SLAMetricType.LATENCY_P95,
        ...     threshold=500.0,
        ...     time_window=timedelta(hours=1)
        ... )
        >>> result = tracker.check_sla_compliance("computation_latency")
    """

    def __init__(
        self,
        namespace: str = "unifiedsteam",
        metrics_collector: Optional[Any] = None,
        on_violation_callback: Optional[Callable[[SLAViolation], None]] = None,
    ) -> None:
        """
        Initialize SLATracker.

        Args:
            namespace: SLA namespace for identification
            metrics_collector: Optional MetricsCollector for metric data
            on_violation_callback: Optional callback for SLA violations
        """
        self.namespace = namespace
        self._metrics = metrics_collector
        self._on_violation = on_violation_callback

        # SLA storage
        self._slas: Dict[str, SLADefinition] = {}

        # Violation tracking
        self._active_violations: Dict[str, SLAViolation] = {}  # sla_name -> violation
        self._violation_history: List[SLAViolation] = []
        self._max_history_size = 1000

        # Metric data storage (when no external collector)
        self._metric_data: Dict[str, List[tuple]] = {}  # metric_name -> [(timestamp, value)]
        self._max_data_points = 10000

        # Statistics
        self._stats = {
            "checks_performed": 0,
            "violations_detected": 0,
            "violations_resolved": 0,
        }

        # Register default SLAs
        self._register_default_slas()

        logger.info("SLATracker initialized: namespace=%s", namespace)

    def _register_default_slas(self) -> None:
        """Register default SLA definitions for steam system."""
        default_slas = [
            SLADefinition(
                name="computation_latency_p95",
                metric=SLAMetricType.LATENCY_P95,
                threshold=500.0,  # 500ms
                time_window=timedelta(hours=1),
                comparison=SLAComparisonOperator.LESS_THAN_OR_EQUAL,
                description="95th percentile computation latency must be <= 500ms",
                owner="engineering",
                priority=1,
            ),
            SLADefinition(
                name="computation_latency_p99",
                metric=SLAMetricType.LATENCY_P99,
                threshold=1000.0,  # 1000ms
                time_window=timedelta(hours=1),
                comparison=SLAComparisonOperator.LESS_THAN_OR_EQUAL,
                description="99th percentile computation latency must be <= 1000ms",
                owner="engineering",
                priority=2,
            ),
            SLADefinition(
                name="system_availability",
                metric=SLAMetricType.AVAILABILITY,
                threshold=99.5,  # 99.5%
                time_window=timedelta(days=30),
                comparison=SLAComparisonOperator.GREATER_THAN_OR_EQUAL,
                description="System availability must be >= 99.5% over 30 days",
                owner="operations",
                priority=1,
            ),
            SLADefinition(
                name="data_freshness",
                metric=SLAMetricType.DATA_FRESHNESS,
                threshold=60.0,  # 60 seconds
                time_window=timedelta(hours=1),
                comparison=SLAComparisonOperator.LESS_THAN_OR_EQUAL,
                description="Process data must be no older than 60 seconds",
                owner="data_engineering",
                priority=1,
            ),
            SLADefinition(
                name="error_rate",
                metric=SLAMetricType.ERROR_RATE,
                threshold=1.0,  # 1%
                time_window=timedelta(hours=1),
                comparison=SLAComparisonOperator.LESS_THAN_OR_EQUAL,
                description="Error rate must be <= 1% per hour",
                owner="engineering",
                priority=1,
            ),
            SLADefinition(
                name="optimization_success_rate",
                metric=SLAMetricType.SUCCESS_RATE,
                threshold=95.0,  # 95%
                time_window=timedelta(hours=24),
                comparison=SLAComparisonOperator.GREATER_THAN_OR_EQUAL,
                description="Optimization success rate must be >= 95% per day",
                owner="engineering",
                priority=2,
            ),
        ]

        for sla in default_slas:
            self._slas[sla.name] = sla

    def define_sla(
        self,
        name: str,
        metric: SLAMetricType,
        threshold: float,
        time_window: timedelta,
        comparison: SLAComparisonOperator = SLAComparisonOperator.LESS_THAN_OR_EQUAL,
        description: str = "",
        owner: str = "",
        priority: int = 1,
        alert_on_breach: bool = True,
        breach_tolerance_percent: float = 0.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> SLADefinition:
        """
        Define a new SLA.

        Args:
            name: Unique SLA name
            metric: Type of metric to track
            threshold: Threshold value
            time_window: Time window for evaluation
            comparison: Comparison operator
            description: Human-readable description
            owner: Team/person responsible
            priority: Priority level (1 = highest)
            alert_on_breach: Whether to alert on breach
            breach_tolerance_percent: Allowed breach percentage
            tags: Optional tags for categorization

        Returns:
            Created SLADefinition
        """
        sla = SLADefinition(
            name=name,
            metric=metric,
            threshold=threshold,
            time_window=time_window,
            comparison=comparison,
            description=description,
            owner=owner,
            priority=priority,
            alert_on_breach=alert_on_breach,
            breach_tolerance_percent=breach_tolerance_percent,
            tags=tags or {},
        )

        self._slas[name] = sla
        logger.info("SLA defined: %s (metric=%s, threshold=%s)", name, metric.value, threshold)

        return sla

    def update_sla(
        self,
        name: str,
        **kwargs: Any,
    ) -> Optional[SLADefinition]:
        """
        Update an existing SLA definition.

        Args:
            name: SLA name to update
            **kwargs: Fields to update

        Returns:
            Updated SLADefinition or None if not found
        """
        if name not in self._slas:
            logger.warning("SLA not found for update: %s", name)
            return None

        sla = self._slas[name]

        for key, value in kwargs.items():
            if hasattr(sla, key):
                setattr(sla, key, value)

        sla.updated_at = datetime.now(timezone.utc)
        logger.info("SLA updated: %s", name)

        return sla

    def delete_sla(self, name: str) -> bool:
        """
        Delete an SLA definition.

        Args:
            name: SLA name to delete

        Returns:
            True if deleted, False if not found
        """
        if name not in self._slas:
            return False

        del self._slas[name]
        logger.info("SLA deleted: %s", name)
        return True

    def get_sla(self, name: str) -> Optional[SLADefinition]:
        """Get SLA definition by name."""
        return self._slas.get(name)

    def list_slas(self) -> List[SLADefinition]:
        """List all SLA definitions."""
        return sorted(self._slas.values(), key=lambda s: s.priority)

    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record a metric value for SLA tracking.

        Args:
            metric_name: Metric name
            value: Metric value
            timestamp: Optional timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        if metric_name not in self._metric_data:
            self._metric_data[metric_name] = []

        self._metric_data[metric_name].append((timestamp, value))

        # Trim old data
        if len(self._metric_data[metric_name]) > self._max_data_points:
            self._metric_data[metric_name] = self._metric_data[metric_name][
                -self._max_data_points:
            ]

    def _get_metric_values(
        self,
        metric: SLAMetricType,
        time_window: timedelta,
    ) -> List[float]:
        """Get metric values within time window."""
        now = datetime.now(timezone.utc)
        cutoff = now - time_window

        # Try to get from metrics collector first
        if self._metrics:
            summary = self._metrics.get_metrics_summary(time_window)
            return self._extract_metric_from_summary(metric, summary)

        # Fall back to internal storage
        metric_name = metric.value
        if metric_name not in self._metric_data:
            return []

        return [
            value for timestamp, value in self._metric_data[metric_name]
            if timestamp >= cutoff
        ]

    def _extract_metric_from_summary(
        self,
        metric: SLAMetricType,
        summary: Any,
    ) -> List[float]:
        """Extract metric values from MetricsSummary."""
        metric_mapping = {
            SLAMetricType.LATENCY_AVG: "avg_computation_time_ms",
            SLAMetricType.LATENCY_P50: "p50_computation_time_ms",
            SLAMetricType.LATENCY_P95: "p95_computation_time_ms",
            SLAMetricType.LATENCY_P99: "p99_computation_time_ms",
        }

        attr_name = metric_mapping.get(metric)
        if attr_name and hasattr(summary, attr_name):
            value = getattr(summary, attr_name)
            return [value] if value > 0 else []

        return []

    def _compute_metric_value(
        self,
        metric: SLAMetricType,
        values: List[float],
    ) -> float:
        """Compute aggregate metric value from samples."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        if metric == SLAMetricType.LATENCY_AVG:
            return sum(values) / n
        elif metric == SLAMetricType.LATENCY_P50:
            return sorted_values[int(n * 0.50)]
        elif metric == SLAMetricType.LATENCY_P95:
            return sorted_values[int(n * 0.95)] if n > 1 else sorted_values[0]
        elif metric == SLAMetricType.LATENCY_P99:
            return sorted_values[int(n * 0.99)] if n > 1 else sorted_values[0]
        elif metric == SLAMetricType.LATENCY_MAX:
            return max(values)
        elif metric in [SLAMetricType.AVAILABILITY, SLAMetricType.UPTIME]:
            return sum(values) / n  # Average availability
        elif metric == SLAMetricType.ERROR_RATE:
            return sum(values) / n  # Average error rate
        elif metric == SLAMetricType.SUCCESS_RATE:
            return sum(values) / n  # Average success rate
        elif metric == SLAMetricType.DATA_FRESHNESS:
            return max(values)  # Max data age
        elif metric == SLAMetricType.THROUGHPUT:
            return sum(values)  # Total throughput
        else:
            return sum(values) / n  # Default to average

    def _check_threshold(
        self,
        actual: float,
        threshold: float,
        comparison: SLAComparisonOperator,
    ) -> bool:
        """Check if actual value meets threshold."""
        if comparison == SLAComparisonOperator.LESS_THAN:
            return actual < threshold
        elif comparison == SLAComparisonOperator.LESS_THAN_OR_EQUAL:
            return actual <= threshold
        elif comparison == SLAComparisonOperator.GREATER_THAN:
            return actual > threshold
        elif comparison == SLAComparisonOperator.GREATER_THAN_OR_EQUAL:
            return actual >= threshold
        elif comparison == SLAComparisonOperator.EQUAL:
            return actual == threshold
        return False

    def check_sla_compliance(
        self,
        sla_name: str,
    ) -> SLAComplianceResult:
        """
        Check compliance for a specific SLA.

        Args:
            sla_name: Name of SLA to check

        Returns:
            SLAComplianceResult with compliance status
        """
        now = datetime.now(timezone.utc)
        self._stats["checks_performed"] += 1

        if sla_name not in self._slas:
            return SLAComplianceResult(
                sla_name=sla_name,
                is_compliant=False,
                metric=SLAMetricType.CUSTOM,
                threshold=0.0,
                actual_value=0.0,
                comparison=SLAComparisonOperator.EQUAL,
                compliance_percent=0.0,
                time_window=timedelta(hours=1),
                checked_at=now,
                message=f"SLA not found: {sla_name}",
            )

        sla = self._slas[sla_name]

        if not sla.enabled:
            return SLAComplianceResult(
                sla_name=sla_name,
                is_compliant=True,
                metric=sla.metric,
                threshold=sla.threshold,
                actual_value=0.0,
                comparison=sla.comparison,
                compliance_percent=100.0,
                time_window=sla.time_window,
                checked_at=now,
                message="SLA is disabled",
            )

        # Get metric values
        values = self._get_metric_values(sla.metric, sla.time_window)
        samples_count = len(values)

        if samples_count == 0:
            return SLAComplianceResult(
                sla_name=sla_name,
                is_compliant=True,  # Assume compliant if no data
                metric=sla.metric,
                threshold=sla.threshold,
                actual_value=0.0,
                comparison=sla.comparison,
                compliance_percent=100.0,
                time_window=sla.time_window,
                checked_at=now,
                samples_count=0,
                message="No metric data available",
            )

        # Compute actual value
        actual_value = self._compute_metric_value(sla.metric, values)

        # Check compliance
        is_compliant = self._check_threshold(actual_value, sla.threshold, sla.comparison)

        # Count breaches (for per-sample metrics)
        breaches_count = sum(
            1 for v in values
            if not self._check_threshold(v, sla.threshold, sla.comparison)
        )

        # Calculate compliance percentage
        compliance_percent = ((samples_count - breaches_count) / samples_count * 100) if samples_count > 0 else 100.0

        # Check for active violation
        current_violation = None
        if not is_compliant:
            current_violation = self._handle_violation(sla, actual_value)
        else:
            self._resolve_violation(sla_name)

        # Determine trend
        trend = self._calculate_trend(sla_name, actual_value)

        # Build message
        message = self._build_compliance_message(sla, is_compliant, actual_value, compliance_percent)

        return SLAComplianceResult(
            sla_name=sla_name,
            is_compliant=is_compliant,
            metric=sla.metric,
            threshold=sla.threshold,
            actual_value=actual_value,
            comparison=sla.comparison,
            compliance_percent=compliance_percent,
            time_window=sla.time_window,
            checked_at=now,
            samples_count=samples_count,
            breaches_count=breaches_count,
            current_violation=current_violation,
            trend=trend,
            message=message,
        )

    def _handle_violation(
        self,
        sla: SLADefinition,
        actual_value: float,
    ) -> SLAViolation:
        """Handle SLA violation."""
        now = datetime.now(timezone.utc)

        # Check for existing violation
        if sla.name in self._active_violations:
            violation = self._active_violations[sla.name]
            violation.duration_seconds = (now - violation.occurred_at).total_seconds()
            violation.actual_value = actual_value  # Update with latest value
            return violation

        # Create new violation
        violation_id = hashlib.sha256(
            f"{sla.name}:{now.isoformat()}".encode()
        ).hexdigest()[:16]

        violation = SLAViolation(
            violation_id=violation_id,
            sla_name=sla.name,
            metric=sla.metric,
            threshold=sla.threshold,
            actual_value=actual_value,
            comparison=sla.comparison,
            occurred_at=now,
        )

        self._active_violations[sla.name] = violation
        self._stats["violations_detected"] += 1

        logger.warning(
            "SLA violation detected: %s (actual=%s, threshold=%s)",
            sla.name,
            actual_value,
            sla.threshold,
        )

        # Invoke callback
        if self._on_violation and sla.alert_on_breach:
            try:
                self._on_violation(violation)
            except Exception as e:
                logger.error("Violation callback failed: %s", e)

        return violation

    def _resolve_violation(self, sla_name: str) -> Optional[SLAViolation]:
        """Resolve an active violation."""
        if sla_name not in self._active_violations:
            return None

        violation = self._active_violations[sla_name]
        violation.is_resolved = True
        violation.resolved_at = datetime.now(timezone.utc)
        violation.duration_seconds = (
            violation.resolved_at - violation.occurred_at
        ).total_seconds()

        # Move to history
        self._add_to_violation_history(violation)
        del self._active_violations[sla_name]

        self._stats["violations_resolved"] += 1

        logger.info(
            "SLA violation resolved: %s (duration=%.1fs)",
            sla_name,
            violation.duration_seconds,
        )

        return violation

    def _add_to_violation_history(self, violation: SLAViolation) -> None:
        """Add violation to history."""
        self._violation_history.append(violation)
        if len(self._violation_history) > self._max_history_size:
            self._violation_history = self._violation_history[-self._max_history_size:]

    def _calculate_trend(self, sla_name: str, current_value: float) -> str:
        """Calculate trend based on recent values."""
        metric_name = f"sla_trend:{sla_name}"
        if metric_name not in self._metric_data:
            self._metric_data[metric_name] = []

        self._metric_data[metric_name].append(
            (datetime.now(timezone.utc), current_value)
        )

        # Keep only recent values
        recent = self._metric_data[metric_name][-10:]
        self._metric_data[metric_name] = recent

        if len(recent) < 3:
            return "stable"

        # Compare recent average to older average
        values = [v for _, v in recent]
        recent_avg = sum(values[-3:]) / 3
        older_avg = sum(values[:-3]) / max(1, len(values) - 3)

        if recent_avg < older_avg * 0.95:
            return "improving"
        elif recent_avg > older_avg * 1.05:
            return "degrading"
        return "stable"

    def _build_compliance_message(
        self,
        sla: SLADefinition,
        is_compliant: bool,
        actual_value: float,
        compliance_percent: float,
    ) -> str:
        """Build human-readable compliance message."""
        comparison_text = {
            SLAComparisonOperator.LESS_THAN: "<",
            SLAComparisonOperator.LESS_THAN_OR_EQUAL: "<=",
            SLAComparisonOperator.GREATER_THAN: ">",
            SLAComparisonOperator.GREATER_THAN_OR_EQUAL: ">=",
            SLAComparisonOperator.EQUAL: "==",
        }.get(sla.comparison, "?")

        status = "COMPLIANT" if is_compliant else "BREACH"

        return (
            f"[{status}] {sla.name}: {actual_value:.2f} {comparison_text} {sla.threshold} "
            f"({compliance_percent:.1f}% compliance)"
        )

    def check_all_slas(self) -> Dict[str, SLAComplianceResult]:
        """
        Check compliance for all defined SLAs.

        Returns:
            Dictionary of SLA name to compliance result
        """
        results = {}
        for sla_name in self._slas:
            results[sla_name] = self.check_sla_compliance(sla_name)
        return results

    def get_sla_report(
        self,
        time_window: Optional[timedelta] = None,
    ) -> SLAReport:
        """
        Generate SLA compliance report.

        Args:
            time_window: Time window for report (default: 24 hours)

        Returns:
            SLAReport with comprehensive compliance data
        """
        if time_window is None:
            time_window = timedelta(hours=24)

        now = datetime.now(timezone.utc)
        start = now - time_window

        # Generate report ID
        report_id = hashlib.sha256(
            f"{self.namespace}:{now.isoformat()}".encode()
        ).hexdigest()[:16]

        # Check all SLAs
        sla_results = self.check_all_slas()

        # Calculate summary
        total_slas = len(sla_results)
        compliant_slas = sum(1 for r in sla_results.values() if r.is_compliant)
        non_compliant_slas = total_slas - compliant_slas
        overall_compliance = (compliant_slas / total_slas * 100) if total_slas > 0 else 100.0

        # Get violations in time window
        violations = [
            v for v in self._violation_history
            if v.occurred_at >= start
        ]
        violations.extend(self._active_violations.values())

        # Generate recommendations
        recommendations = self._generate_recommendations(sla_results)

        # Generate summary
        summary = self._generate_summary(
            total_slas, compliant_slas, non_compliant_slas, overall_compliance
        )

        return SLAReport(
            report_id=report_id,
            report_period_start=start,
            report_period_end=now,
            generated_at=now,
            total_slas=total_slas,
            compliant_slas=compliant_slas,
            non_compliant_slas=non_compliant_slas,
            overall_compliance_percent=overall_compliance,
            sla_results=sla_results,
            violations=violations,
            summary=summary,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        results: Dict[str, SLAComplianceResult],
    ) -> List[str]:
        """Generate recommendations based on SLA results."""
        recommendations = []

        for name, result in results.items():
            if not result.is_compliant:
                sla = self._slas.get(name)
                if sla:
                    if sla.metric in [SLAMetricType.LATENCY_P95, SLAMetricType.LATENCY_P99]:
                        recommendations.append(
                            f"Review computation performance for {name}. "
                            f"Consider optimizing algorithms or scaling resources."
                        )
                    elif sla.metric == SLAMetricType.AVAILABILITY:
                        recommendations.append(
                            f"Investigate availability issues for {name}. "
                            f"Review recent incidents and implement redundancy."
                        )
                    elif sla.metric == SLAMetricType.ERROR_RATE:
                        recommendations.append(
                            f"Address error rate increase for {name}. "
                            f"Review logs and implement additional error handling."
                        )
                    elif sla.metric == SLAMetricType.DATA_FRESHNESS:
                        recommendations.append(
                            f"Improve data freshness for {name}. "
                            f"Check data pipeline latency and source connectivity."
                        )

            elif result.trend == "degrading":
                recommendations.append(
                    f"Monitor {name} closely - performance trend is degrading."
                )

        return recommendations

    def _generate_summary(
        self,
        total: int,
        compliant: int,
        non_compliant: int,
        overall_percent: float,
    ) -> str:
        """Generate text summary for report."""
        if overall_percent >= 99.0:
            status = "Excellent"
        elif overall_percent >= 95.0:
            status = "Good"
        elif overall_percent >= 90.0:
            status = "Needs Attention"
        else:
            status = "Critical"

        return (
            f"SLA Compliance Report: {status}\n"
            f"Overall compliance: {overall_percent:.1f}%\n"
            f"SLAs meeting target: {compliant}/{total}\n"
            f"SLAs breaching: {non_compliant}/{total}"
        )

    def get_active_violations(self) -> List[SLAViolation]:
        """Get all active SLA violations."""
        return list(self._active_violations.values())

    def get_violation_history(
        self,
        time_window: Optional[timedelta] = None,
        sla_name: Optional[str] = None,
    ) -> List[SLAViolation]:
        """
        Get violation history.

        Args:
            time_window: Optional time window filter
            sla_name: Optional SLA name filter

        Returns:
            List of historical violations
        """
        violations = self._violation_history

        if time_window:
            cutoff = datetime.now(timezone.utc) - time_window
            violations = [v for v in violations if v.occurred_at >= cutoff]

        if sla_name:
            violations = [v for v in violations if v.sla_name == sla_name]

        return sorted(violations, key=lambda v: v.occurred_at, reverse=True)

    def get_statistics(self) -> Dict[str, Any]:
        """Get SLA tracker statistics."""
        return {
            **self._stats,
            "total_slas": len(self._slas),
            "active_violations": len(self._active_violations),
            "history_size": len(self._violation_history),
        }
