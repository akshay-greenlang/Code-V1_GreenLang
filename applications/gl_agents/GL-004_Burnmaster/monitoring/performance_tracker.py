"""
GL-004 BURNMASTER Performance Tracker Module

This module provides optimizer performance tracking for combustion optimization
operations, including recommendation accuracy tracking, optimizer contribution
analysis, value delivery computation, and baseline comparison.

Example:
    >>> tracker = OptimizerPerformanceTracker()
    >>> tracker.track_recommendation_accuracy(predicted, actual)
    >>> report = tracker.generate_performance_report(period)
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import statistics
import uuid

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class PerformanceLevel(str, Enum):
    """Performance level assessment."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    CRITICAL = "CRITICAL"


class ValueCategory(str, Enum):
    """Category of value delivered."""
    FUEL_SAVINGS = "FUEL_SAVINGS"
    EMISSIONS_REDUCTION = "EMISSIONS_REDUCTION"
    EFFICIENCY_GAIN = "EFFICIENCY_GAIN"
    MAINTENANCE_AVOIDANCE = "MAINTENANCE_AVOIDANCE"
    SAFETY_IMPROVEMENT = "SAFETY_IMPROVEMENT"
    THROUGHPUT_INCREASE = "THROUGHPUT_INCREASE"


# =============================================================================
# DATA MODELS
# =============================================================================

class DateRange(BaseModel):
    """Date range for queries."""

    start: datetime = Field(..., description="Start of date range")
    end: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="End of date range"
    )


class RecommendationRecord(BaseModel):
    """Record of a single recommendation."""

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Recommendation identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Recommendation timestamp"
    )
    unit_id: str = Field(..., description="Unit identifier")

    # Predicted values
    predicted_values: Dict[str, float] = Field(
        ..., description="Predicted parameter values"
    )

    # Actual values (filled in after observation)
    actual_values: Optional[Dict[str, float]] = Field(
        None, description="Actual parameter values"
    )

    # Accuracy metrics (computed after actual values available)
    accuracy_metrics: Optional[Dict[str, float]] = Field(
        None, description="Per-parameter accuracy"
    )
    overall_accuracy: Optional[float] = Field(
        None, description="Overall accuracy score"
    )

    # Acceptance
    accepted: bool = Field(default=False, description="Recommendation accepted")
    implemented_at: Optional[datetime] = Field(
        None, description="Implementation timestamp"
    )

    # Metadata
    recommendation_type: str = Field(
        default="optimization", description="Recommendation type"
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Model confidence"
    )


class AccuracyMetrics(BaseModel):
    """Accuracy metrics for predictions."""

    # Overall
    overall_accuracy: float = Field(
        ..., ge=0.0, le=100.0, description="Overall accuracy percentage"
    )
    mean_absolute_error: float = Field(..., description="Mean absolute error")
    mean_squared_error: float = Field(..., description="Mean squared error")
    root_mean_squared_error: float = Field(..., description="Root MSE")

    # Per-parameter
    parameter_accuracy: Dict[str, float] = Field(
        default_factory=dict, description="Accuracy per parameter"
    )
    parameter_mae: Dict[str, float] = Field(
        default_factory=dict, description="MAE per parameter"
    )

    # Counts
    total_predictions: int = Field(default=0, ge=0, description="Total predictions")
    accurate_predictions: int = Field(
        default=0, ge=0, description="Predictions within threshold"
    )

    # Trend
    accuracy_trend: str = Field(
        default="STABLE", description="IMPROVING, STABLE, or DEGRADING"
    )


class ContributionMetrics(BaseModel):
    """Metrics for optimizer contribution."""

    # Efficiency
    efficiency_baseline: float = Field(..., description="Baseline efficiency %")
    efficiency_optimized: float = Field(..., description="Optimized efficiency %")
    efficiency_contribution: float = Field(..., description="Efficiency improvement %")

    # Emissions
    nox_baseline: float = Field(..., description="Baseline NOx ppm")
    nox_optimized: float = Field(..., description="Optimized NOx ppm")
    nox_reduction_percent: float = Field(..., description="NOx reduction %")

    co_baseline: float = Field(..., description="Baseline CO ppm")
    co_optimized: float = Field(..., description="Optimized CO ppm")
    co_reduction_percent: float = Field(..., description="CO reduction %")

    # Fuel
    fuel_baseline: float = Field(..., description="Baseline fuel consumption")
    fuel_optimized: float = Field(..., description="Optimized fuel consumption")
    fuel_savings_percent: float = Field(..., description="Fuel savings %")

    # Stability
    stability_baseline: float = Field(..., description="Baseline stability score")
    stability_optimized: float = Field(..., description="Optimized stability score")
    stability_improvement: float = Field(..., description="Stability improvement")


class ValueMetrics(BaseModel):
    """Metrics for value delivered by optimizer."""

    period: DateRange = Field(..., description="Value calculation period")

    # Financial value
    total_value_usd: float = Field(default=0.0, description="Total value in USD")
    fuel_savings_usd: float = Field(default=0.0, description="Fuel cost savings USD")
    emissions_value_usd: float = Field(
        default=0.0, description="Emissions reduction value USD"
    )
    maintenance_savings_usd: float = Field(
        default=0.0, description="Maintenance savings USD"
    )

    # Physical savings
    fuel_saved_mmbtu: float = Field(default=0.0, description="Fuel saved in MMBtu")
    nox_reduced_tons: float = Field(default=0.0, description="NOx reduced in tons")
    co_reduced_tons: float = Field(default=0.0, description="CO reduced in tons")
    co2_avoided_tons: float = Field(default=0.0, description="CO2 avoided in tons")

    # Operational
    efficiency_points_gained: float = Field(
        default=0.0, description="Efficiency percentage points gained"
    )
    uptime_hours_gained: float = Field(
        default=0.0, description="Additional uptime hours"
    )

    # Value breakdown by category
    value_by_category: Dict[str, float] = Field(
        default_factory=dict, description="Value breakdown by category"
    )

    # Confidence
    value_confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence in value calculation"
    )


class ComparisonResult(BaseModel):
    """Result of comparing current to baseline performance."""

    comparison_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Comparison identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Comparison timestamp"
    )

    # Overall assessment
    overall_improvement: float = Field(
        ..., description="Overall improvement percentage"
    )
    performance_level: PerformanceLevel = Field(
        ..., description="Performance level vs baseline"
    )

    # Per-metric comparison
    metric_comparisons: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Comparison per metric (baseline, current, delta, delta_pct)"
    )

    # Summary
    improved_metrics: List[str] = Field(
        default_factory=list, description="Metrics that improved"
    )
    degraded_metrics: List[str] = Field(
        default_factory=list, description="Metrics that degraded"
    )
    unchanged_metrics: List[str] = Field(
        default_factory=list, description="Metrics that stayed the same"
    )

    # Context
    baseline_period: Optional[DateRange] = Field(
        None, description="Baseline measurement period"
    )
    current_period: Optional[DateRange] = Field(
        None, description="Current measurement period"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list, description="Performance recommendations"
    )


class PerformanceReport(BaseModel):
    """Comprehensive performance report."""

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Report identifier"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report generation time"
    )
    period: DateRange = Field(..., description="Report period")

    # Overall performance
    overall_performance: PerformanceLevel = Field(
        ..., description="Overall performance level"
    )
    performance_score: float = Field(
        ..., ge=0.0, le=100.0, description="Performance score 0-100"
    )

    # Accuracy
    accuracy_metrics: AccuracyMetrics = Field(
        ..., description="Prediction accuracy metrics"
    )

    # Contribution
    contribution_metrics: Optional[ContributionMetrics] = Field(
        None, description="Optimizer contribution metrics"
    )

    # Value
    value_metrics: ValueMetrics = Field(..., description="Value delivered metrics")

    # Recommendations
    total_recommendations: int = Field(
        default=0, ge=0, description="Total recommendations generated"
    )
    accepted_recommendations: int = Field(
        default=0, ge=0, description="Recommendations accepted"
    )
    acceptance_rate: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Acceptance rate percentage"
    )

    # Comparison to baseline
    baseline_comparison: Optional[ComparisonResult] = Field(
        None, description="Comparison to baseline"
    )

    # Trends
    accuracy_trend: str = Field(
        default="STABLE", description="Accuracy trend direction"
    )
    value_trend: str = Field(
        default="STABLE", description="Value trend direction"
    )

    # Issues and recommendations
    issues: List[str] = Field(default_factory=list, description="Performance issues")
    improvement_opportunities: List[str] = Field(
        default_factory=list, description="Improvement opportunities"
    )

    # Audit
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")


# =============================================================================
# OPTIMIZER PERFORMANCE TRACKER
# =============================================================================

class OptimizerPerformanceTracker:
    """
    Comprehensive optimizer performance tracking for combustion optimization.

    Tracks recommendation accuracy, optimizer contribution vs baseline,
    value delivered, and generates performance reports.

    Attributes:
        accuracy_threshold: Threshold for considering a prediction accurate

    Example:
        >>> tracker = OptimizerPerformanceTracker()
        >>> tracker.track_recommendation_accuracy(predicted, actual)
        >>> value = tracker.compute_value_delivered(period)
        >>> report = tracker.generate_performance_report(period)
    """

    # Default pricing for value calculations
    DEFAULT_FUEL_PRICE_PER_MMBTU = 5.0  # USD per MMBtu
    DEFAULT_CARBON_PRICE_PER_TON = 50.0  # USD per ton CO2
    DEFAULT_NOX_CREDIT_PER_TON = 5000.0  # USD per ton NOx

    def __init__(
        self,
        accuracy_threshold: float = 5.0,
        fuel_price: Optional[float] = None,
        carbon_price: Optional[float] = None
    ):
        """
        Initialize the OptimizerPerformanceTracker.

        Args:
            accuracy_threshold: Percentage threshold for accurate predictions
            fuel_price: Fuel price per MMBtu
            carbon_price: Carbon price per ton CO2
        """
        self.accuracy_threshold = accuracy_threshold
        self.fuel_price = fuel_price or self.DEFAULT_FUEL_PRICE_PER_MMBTU
        self.carbon_price = carbon_price or self.DEFAULT_CARBON_PRICE_PER_TON

        self._recommendations: List[RecommendationRecord] = []
        self._baseline_data: Dict[str, Dict[str, float]] = {}
        self._value_history: List[ValueMetrics] = []

        logger.info(
            f"OptimizerPerformanceTracker initialized with "
            f"accuracy_threshold={accuracy_threshold}%"
        )

    def track_recommendation_accuracy(
        self,
        predicted: Dict[str, float],
        actual: Dict[str, float],
        unit_id: str = "default",
        recommendation_id: Optional[str] = None
    ) -> None:
        """
        Track accuracy of a recommendation's predictions.

        Args:
            predicted: Predicted parameter values
            actual: Actual observed values
            unit_id: Unit identifier
            recommendation_id: Optional ID to update existing recommendation
        """
        # Calculate per-parameter accuracy
        accuracy_metrics = {}
        for param, pred_value in predicted.items():
            if param in actual:
                actual_value = actual[param]
                if actual_value != 0:
                    error_pct = abs((pred_value - actual_value) / actual_value) * 100
                    accuracy = max(0, 100 - error_pct)
                else:
                    accuracy = 100 if pred_value == 0 else 0
                accuracy_metrics[param] = accuracy

        # Calculate overall accuracy
        overall_accuracy = (
            statistics.mean(accuracy_metrics.values())
            if accuracy_metrics else 0.0
        )

        # Create or update record
        if recommendation_id:
            # Find and update existing recommendation
            for rec in self._recommendations:
                if rec.recommendation_id == recommendation_id:
                    rec.actual_values = actual
                    rec.accuracy_metrics = accuracy_metrics
                    rec.overall_accuracy = overall_accuracy
                    break
        else:
            # Create new record
            record = RecommendationRecord(
                unit_id=unit_id,
                predicted_values=predicted,
                actual_values=actual,
                accuracy_metrics=accuracy_metrics,
                overall_accuracy=overall_accuracy,
            )
            self._recommendations.append(record)

        logger.info(
            f"Tracked recommendation accuracy for {unit_id}: "
            f"overall={overall_accuracy:.1f}%"
        )

    def track_optimizer_contribution(
        self,
        baseline: Dict[str, float],
        optimized: Dict[str, float],
        unit_id: str = "default"
    ) -> ContributionMetrics:
        """
        Track optimizer contribution compared to baseline.

        Args:
            baseline: Baseline (unoptimized) performance metrics
            optimized: Optimized performance metrics
            unit_id: Unit identifier

        Returns:
            ContributionMetrics with comparison details
        """
        # Store baseline for future comparisons
        self._baseline_data[unit_id] = baseline

        # Calculate contributions
        def calc_improvement(base: float, opt: float, higher_is_better: bool) -> float:
            if base == 0:
                return 0.0
            if higher_is_better:
                return ((opt - base) / base) * 100
            else:
                return ((base - opt) / base) * 100

        metrics = ContributionMetrics(
            # Efficiency (higher is better)
            efficiency_baseline=baseline.get('efficiency', 0),
            efficiency_optimized=optimized.get('efficiency', 0),
            efficiency_contribution=calc_improvement(
                baseline.get('efficiency', 0),
                optimized.get('efficiency', 0),
                higher_is_better=True
            ),

            # NOx (lower is better)
            nox_baseline=baseline.get('nox_ppm', 0),
            nox_optimized=optimized.get('nox_ppm', 0),
            nox_reduction_percent=calc_improvement(
                baseline.get('nox_ppm', 0),
                optimized.get('nox_ppm', 0),
                higher_is_better=False
            ),

            # CO (lower is better)
            co_baseline=baseline.get('co_ppm', 0),
            co_optimized=optimized.get('co_ppm', 0),
            co_reduction_percent=calc_improvement(
                baseline.get('co_ppm', 0),
                optimized.get('co_ppm', 0),
                higher_is_better=False
            ),

            # Fuel (lower is better)
            fuel_baseline=baseline.get('fuel_consumption', 0),
            fuel_optimized=optimized.get('fuel_consumption', 0),
            fuel_savings_percent=calc_improvement(
                baseline.get('fuel_consumption', 0),
                optimized.get('fuel_consumption', 0),
                higher_is_better=False
            ),

            # Stability (higher is better)
            stability_baseline=baseline.get('stability', 0),
            stability_optimized=optimized.get('stability', 0),
            stability_improvement=optimized.get('stability', 0) - baseline.get('stability', 0),
        )

        logger.info(
            f"Tracked optimizer contribution for {unit_id}: "
            f"efficiency +{metrics.efficiency_contribution:.1f}%, "
            f"NOx -{metrics.nox_reduction_percent:.1f}%"
        )

        return metrics

    def compute_value_delivered(self, period: DateRange) -> ValueMetrics:
        """
        Compute total value delivered by optimizer over a period.

        Args:
            period: Date range for value calculation

        Returns:
            ValueMetrics with value breakdown
        """
        # Filter recommendations in period
        period_recs = [
            r for r in self._recommendations
            if period.start <= r.timestamp <= period.end
        ]

        # Calculate physical savings from baseline comparisons
        fuel_saved = 0.0
        nox_reduced_tons = 0.0
        co_reduced_tons = 0.0
        efficiency_gained = 0.0

        for unit_id, baseline in self._baseline_data.items():
            # Get average optimized values for the unit
            unit_recs = [r for r in period_recs if r.unit_id == unit_id and r.actual_values]
            if not unit_recs:
                continue

            # Average actual values
            avg_optimized = {}
            for rec in unit_recs:
                for param, value in (rec.actual_values or {}).items():
                    if param not in avg_optimized:
                        avg_optimized[param] = []
                    avg_optimized[param].append(value)

            avg_optimized = {k: statistics.mean(v) for k, v in avg_optimized.items()}

            # Calculate savings
            if 'fuel_consumption' in baseline and 'fuel_consumption' in avg_optimized:
                fuel_saved += max(0, baseline['fuel_consumption'] - avg_optimized['fuel_consumption'])

            if 'nox_ppm' in baseline and 'nox_ppm' in avg_optimized:
                # Convert ppm reduction to tons (simplified calculation)
                nox_reduction_ppm = max(0, baseline['nox_ppm'] - avg_optimized['nox_ppm'])
                nox_reduced_tons += nox_reduction_ppm * 0.001  # Simplified conversion

            if 'efficiency' in baseline and 'efficiency' in avg_optimized:
                efficiency_gained += max(0, avg_optimized['efficiency'] - baseline['efficiency'])

        # Calculate CO2 avoided (from fuel savings)
        # Approximate: 0.053 tons CO2 per MMBtu natural gas
        co2_avoided = fuel_saved * 0.053

        # Calculate financial values
        fuel_savings_usd = fuel_saved * self.fuel_price
        emissions_value_usd = (
            co2_avoided * self.carbon_price +
            nox_reduced_tons * self.DEFAULT_NOX_CREDIT_PER_TON
        )
        total_value = fuel_savings_usd + emissions_value_usd

        # Create value metrics
        metrics = ValueMetrics(
            period=period,
            total_value_usd=total_value,
            fuel_savings_usd=fuel_savings_usd,
            emissions_value_usd=emissions_value_usd,
            fuel_saved_mmbtu=fuel_saved,
            nox_reduced_tons=nox_reduced_tons,
            co2_avoided_tons=co2_avoided,
            efficiency_points_gained=efficiency_gained,
            value_by_category={
                ValueCategory.FUEL_SAVINGS.value: fuel_savings_usd,
                ValueCategory.EMISSIONS_REDUCTION.value: emissions_value_usd,
            }
        )

        # Store in history
        self._value_history.append(metrics)

        logger.info(
            f"Computed value for period: ${total_value:.2f} total, "
            f"${fuel_savings_usd:.2f} fuel savings"
        )

        return metrics

    def compare_to_baseline(
        self,
        current: Dict[str, float],
        baseline: Dict[str, float]
    ) -> ComparisonResult:
        """
        Compare current performance to baseline.

        Args:
            current: Current performance metrics
            baseline: Baseline performance metrics

        Returns:
            ComparisonResult with detailed comparison
        """
        metric_comparisons = {}
        improved = []
        degraded = []
        unchanged = []

        # Metrics where higher is better
        higher_is_better = {'efficiency', 'stability', 'availability', 'turndown'}

        for metric in set(current.keys()) | set(baseline.keys()):
            curr_val = current.get(metric, 0)
            base_val = baseline.get(metric, 0)

            if base_val != 0:
                delta_pct = ((curr_val - base_val) / abs(base_val)) * 100
            else:
                delta_pct = 0 if curr_val == 0 else 100

            metric_comparisons[metric] = {
                'baseline': base_val,
                'current': curr_val,
                'delta': curr_val - base_val,
                'delta_percent': delta_pct,
            }

            # Categorize improvement/degradation
            if metric in higher_is_better:
                if delta_pct > 2:
                    improved.append(metric)
                elif delta_pct < -2:
                    degraded.append(metric)
                else:
                    unchanged.append(metric)
            else:  # Lower is better
                if delta_pct < -2:
                    improved.append(metric)
                elif delta_pct > 2:
                    degraded.append(metric)
                else:
                    unchanged.append(metric)

        # Calculate overall improvement
        improvements = []
        for metric, comp in metric_comparisons.items():
            if metric in higher_is_better:
                improvements.append(comp['delta_percent'])
            else:
                improvements.append(-comp['delta_percent'])  # Invert for "lower is better"

        overall_improvement = statistics.mean(improvements) if improvements else 0.0

        # Determine performance level
        if overall_improvement >= 10:
            level = PerformanceLevel.EXCELLENT
        elif overall_improvement >= 5:
            level = PerformanceLevel.GOOD
        elif overall_improvement >= 0:
            level = PerformanceLevel.ACCEPTABLE
        elif overall_improvement >= -5:
            level = PerformanceLevel.POOR
        else:
            level = PerformanceLevel.CRITICAL

        # Generate recommendations
        recommendations = []
        if degraded:
            recommendations.append(
                f"Investigate degradation in: {', '.join(degraded)}"
            )
        if len(improved) > len(degraded):
            recommendations.append(
                "Overall improvement trend is positive. Continue current strategy."
            )

        result = ComparisonResult(
            overall_improvement=overall_improvement,
            performance_level=level,
            metric_comparisons=metric_comparisons,
            improved_metrics=improved,
            degraded_metrics=degraded,
            unchanged_metrics=unchanged,
            recommendations=recommendations,
        )

        logger.info(
            f"Baseline comparison: {overall_improvement:.1f}% overall, "
            f"level={level.value}"
        )

        return result

    def generate_performance_report(self, period: DateRange) -> PerformanceReport:
        """
        Generate comprehensive performance report.

        Args:
            period: Date range for the report

        Returns:
            PerformanceReport with all performance metrics
        """
        # Filter recommendations in period
        period_recs = [
            r for r in self._recommendations
            if period.start <= r.timestamp <= period.end
        ]

        # Calculate accuracy metrics
        accuracy_data = [
            r for r in period_recs
            if r.overall_accuracy is not None
        ]

        if accuracy_data:
            accuracies = [r.overall_accuracy for r in accuracy_data]
            overall_accuracy = statistics.mean(accuracies)

            # Calculate MAE, MSE
            errors = []
            for rec in accuracy_data:
                if rec.predicted_values and rec.actual_values:
                    for param in rec.predicted_values:
                        if param in rec.actual_values:
                            errors.append(
                                abs(rec.predicted_values[param] - rec.actual_values[param])
                            )

            mae = statistics.mean(errors) if errors else 0
            mse = statistics.mean([e**2 for e in errors]) if errors else 0
            rmse = mse ** 0.5

            accurate_count = sum(
                1 for r in accuracy_data
                if r.overall_accuracy >= (100 - self.accuracy_threshold)
            )

            accuracy_metrics = AccuracyMetrics(
                overall_accuracy=overall_accuracy,
                mean_absolute_error=mae,
                mean_squared_error=mse,
                root_mean_squared_error=rmse,
                total_predictions=len(accuracy_data),
                accurate_predictions=accurate_count,
            )
        else:
            accuracy_metrics = AccuracyMetrics(
                overall_accuracy=0,
                mean_absolute_error=0,
                mean_squared_error=0,
                root_mean_squared_error=0,
            )

        # Calculate value metrics
        value_metrics = self.compute_value_delivered(period)

        # Calculate acceptance rate
        total_recs = len(period_recs)
        accepted_recs = sum(1 for r in period_recs if r.accepted)
        acceptance_rate = (accepted_recs / total_recs * 100) if total_recs > 0 else 0

        # Determine overall performance
        score = (
            accuracy_metrics.overall_accuracy * 0.4 +
            min(100, value_metrics.total_value_usd / 1000) * 0.3 +  # Normalize value
            acceptance_rate * 0.3
        )

        if score >= 85:
            overall_performance = PerformanceLevel.EXCELLENT
        elif score >= 70:
            overall_performance = PerformanceLevel.GOOD
        elif score >= 55:
            overall_performance = PerformanceLevel.ACCEPTABLE
        elif score >= 40:
            overall_performance = PerformanceLevel.POOR
        else:
            overall_performance = PerformanceLevel.CRITICAL

        # Identify issues and opportunities
        issues = []
        opportunities = []

        if accuracy_metrics.overall_accuracy < 80:
            issues.append(
                f"Prediction accuracy below target: {accuracy_metrics.overall_accuracy:.1f}%"
            )
            opportunities.append("Consider model retraining or parameter tuning")

        if acceptance_rate < 50:
            issues.append(f"Low recommendation acceptance rate: {acceptance_rate:.1f}%")
            opportunities.append("Review recommendation quality and operator feedback")

        report = PerformanceReport(
            period=period,
            overall_performance=overall_performance,
            performance_score=score,
            accuracy_metrics=accuracy_metrics,
            value_metrics=value_metrics,
            total_recommendations=total_recs,
            accepted_recommendations=accepted_recs,
            acceptance_rate=acceptance_rate,
            issues=issues,
            improvement_opportunities=opportunities,
        )

        # Compute provenance hash
        report.provenance_hash = self._compute_provenance(report)

        logger.info(
            f"Generated performance report: score={score:.1f}, "
            f"level={overall_performance.value}"
        )

        return report

    def record_recommendation_acceptance(
        self,
        recommendation_id: str,
        accepted: bool
    ) -> bool:
        """
        Record whether a recommendation was accepted.

        Args:
            recommendation_id: Recommendation identifier
            accepted: Whether recommendation was accepted

        Returns:
            True if recommendation found and updated
        """
        for rec in self._recommendations:
            if rec.recommendation_id == recommendation_id:
                rec.accepted = accepted
                if accepted:
                    rec.implemented_at = datetime.now(timezone.utc)
                logger.info(
                    f"Recommendation {recommendation_id} "
                    f"{'accepted' if accepted else 'rejected'}"
                )
                return True
        return False

    def _compute_provenance(self, report: PerformanceReport) -> str:
        """Compute SHA-256 provenance hash for audit."""
        content = report.json(exclude={'provenance_hash'})
        return hashlib.sha256(content.encode()).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            'total_recommendations': len(self._recommendations),
            'recommendations_with_accuracy': sum(
                1 for r in self._recommendations if r.overall_accuracy is not None
            ),
            'baseline_units_tracked': len(self._baseline_data),
            'value_history_entries': len(self._value_history),
        }
