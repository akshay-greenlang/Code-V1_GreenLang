# -*- coding: utf-8 -*-
"""
Trend Analysis for Combustion Quality - GL-005 CombustionSense
==============================================================

Provides real-time trend analysis for combustion parameters including:
    - Short-term and long-term trend detection
    - Efficiency degradation monitoring
    - Predictive maintenance indicators
    - Seasonal pattern detection

Analysis Methods:
    - Moving average analysis
    - Linear regression trending
    - Change point detection
    - Statistical process control (SPC)

Author: GL-DataEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import statistics
import math
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class TrendDirection(Enum):
    """Direction of trend."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    OSCILLATING = "oscillating"


class TrendSignificance(Enum):
    """Statistical significance of trend."""
    SIGNIFICANT = "significant"
    MARGINAL = "marginal"
    NOT_SIGNIFICANT = "not_significant"


class CombustionQuality(Enum):
    """Overall combustion quality assessment."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DataSample:
    """Single data sample for trend analysis."""
    parameter: str
    value: float
    timestamp: datetime


@dataclass
class TrendResult:
    """Result of trend analysis."""
    parameter: str
    direction: TrendDirection
    significance: TrendSignificance
    slope: float              # Rate of change
    r_squared: float          # Coefficient of determination
    window_mean: float
    window_std: float
    samples_analyzed: int
    analysis_period_hours: float


@dataclass
class CombustionQualityReport:
    """Comprehensive combustion quality report."""
    timestamp: datetime
    quality_score: float       # 0-100
    quality_level: CombustionQuality
    component_scores: Dict[str, float]
    trends: Dict[str, TrendResult]
    recommendations: List[str]
    alerts: List[str]


# =============================================================================
# TREND ANALYZER
# =============================================================================

class TrendAnalyzer:
    """
    Analyzes trends in combustion parameters.

    Features:
        - Multi-timescale analysis
        - Statistical significance testing
        - Quality scoring
        - Predictive insights
    """

    def __init__(
        self,
        short_window_minutes: int = 60,
        long_window_hours: int = 24
    ):
        self.short_window = short_window_minutes
        self.long_window = long_window_hours

        self.data_buffers: Dict[str, deque] = {}
        self.baseline_values: Dict[str, float] = {}
        self.quality_weights: Dict[str, float] = {
            "O2": 0.25,
            "CO": 0.25,
            "efficiency": 0.30,
            "stability": 0.20,
        }

    def add_sample(self, sample: DataSample) -> None:
        """
        Add a data sample for analysis.

        Args:
            sample: Data sample to add
        """
        if sample.parameter not in self.data_buffers:
            # Keep enough samples for long window (assuming 1 sample/minute)
            max_samples = self.long_window * 60
            self.data_buffers[sample.parameter] = deque(maxlen=max_samples)

        self.data_buffers[sample.parameter].append(sample)

    def analyze_trend(
        self,
        parameter: str,
        window_minutes: Optional[int] = None
    ) -> Optional[TrendResult]:
        """
        Analyze trend for a parameter.

        Args:
            parameter: Parameter name
            window_minutes: Analysis window (default: short window)

        Returns:
            TrendResult or None if insufficient data
        """
        if parameter not in self.data_buffers:
            return None

        window = window_minutes or self.short_window
        cutoff_time = datetime.now() - timedelta(minutes=window)

        samples = [
            s for s in self.data_buffers[parameter]
            if s.timestamp >= cutoff_time
        ]

        if len(samples) < 10:
            return None

        values = [s.value for s in samples]
        timestamps = [(s.timestamp - samples[0].timestamp).total_seconds() / 60
                     for s in samples]

        # Calculate linear regression
        slope, intercept, r_squared = self._linear_regression(timestamps, values)

        # Determine trend direction
        direction = self._determine_direction(slope, values)

        # Determine significance
        significance = self._determine_significance(r_squared, len(values))

        # Calculate statistics
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0

        return TrendResult(
            parameter=parameter,
            direction=direction,
            significance=significance,
            slope=slope,
            r_squared=r_squared,
            window_mean=mean_val,
            window_std=std_val,
            samples_analyzed=len(samples),
            analysis_period_hours=window / 60,
        )

    def _linear_regression(
        self,
        x: List[float],
        y: List[float]
    ) -> Tuple[float, float, float]:
        """
        Calculate simple linear regression.

        Returns:
            Tuple of (slope, intercept, r_squared)
        """
        n = len(x)
        if n < 2:
            return 0.0, 0.0, 0.0

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)
        sum_yy = sum(yi * yi for yi in y)

        denom = n * sum_xx - sum_x * sum_x
        if denom == 0:
            return 0.0, 0.0, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        y_pred = [slope * xi + intercept for xi in x]
        ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return slope, intercept, max(0, min(1, r_squared))

    def _determine_direction(
        self,
        slope: float,
        values: List[float]
    ) -> TrendDirection:
        """Determine trend direction."""
        if len(values) < 2:
            return TrendDirection.STABLE

        mean_val = statistics.mean(values)
        threshold = abs(mean_val) * 0.01 if mean_val != 0 else 0.01

        if slope > threshold:
            return TrendDirection.INCREASING
        elif slope < -threshold:
            return TrendDirection.DECREASING
        else:
            # Check for oscillation
            std_val = statistics.stdev(values)
            cv = std_val / abs(mean_val) if mean_val != 0 else 0

            if cv > 0.1:
                return TrendDirection.OSCILLATING
            else:
                return TrendDirection.STABLE

    def _determine_significance(
        self,
        r_squared: float,
        n_samples: int
    ) -> TrendSignificance:
        """Determine statistical significance of trend."""
        # Simple threshold-based significance
        if n_samples < 20:
            if r_squared > 0.8:
                return TrendSignificance.SIGNIFICANT
            elif r_squared > 0.5:
                return TrendSignificance.MARGINAL
            else:
                return TrendSignificance.NOT_SIGNIFICANT
        else:
            if r_squared > 0.6:
                return TrendSignificance.SIGNIFICANT
            elif r_squared > 0.3:
                return TrendSignificance.MARGINAL
            else:
                return TrendSignificance.NOT_SIGNIFICANT

    def assess_combustion_quality(self) -> CombustionQualityReport:
        """
        Assess overall combustion quality.

        Returns:
            CombustionQualityReport with comprehensive assessment
        """
        component_scores = {}
        trends = {}
        recommendations = []
        alerts = []

        # Analyze each parameter
        for param in ["O2", "CO", "efficiency", "stability"]:
            trend = self.analyze_trend(param)
            if trend:
                trends[param] = trend

                # Calculate component score (0-100)
                score = self._calculate_component_score(param, trend)
                component_scores[param] = score

                # Generate recommendations
                recs, alts = self._generate_insights(param, trend)
                recommendations.extend(recs)
                alerts.extend(alts)

        # Calculate overall quality score
        quality_score = self._calculate_overall_score(component_scores)

        # Determine quality level
        quality_level = self._determine_quality_level(quality_score)

        return CombustionQualityReport(
            timestamp=datetime.now(),
            quality_score=quality_score,
            quality_level=quality_level,
            component_scores=component_scores,
            trends=trends,
            recommendations=recommendations,
            alerts=alerts,
        )

    def _calculate_component_score(
        self,
        parameter: str,
        trend: TrendResult
    ) -> float:
        """Calculate quality score for a component (0-100)."""
        base_score = 100.0

        # Penalize based on deviation from optimal
        if parameter == "O2":
            # Optimal O2 is 3-5%
            optimal_low, optimal_high = 3.0, 5.0
            if trend.window_mean < optimal_low:
                base_score -= (optimal_low - trend.window_mean) * 20
            elif trend.window_mean > optimal_high:
                base_score -= (trend.window_mean - optimal_high) * 10

        elif parameter == "CO":
            # Optimal CO is < 100 ppm
            if trend.window_mean > 400:
                base_score -= 50
            elif trend.window_mean > 200:
                base_score -= 25
            elif trend.window_mean > 100:
                base_score -= 10

        # Penalize for adverse trends
        if trend.direction == TrendDirection.INCREASING and parameter == "CO":
            base_score -= 15
        elif trend.direction == TrendDirection.DECREASING and parameter == "O2":
            base_score -= 15
        elif trend.direction == TrendDirection.OSCILLATING:
            base_score -= 10

        # Penalize for high variability
        if trend.window_std > 0:
            cv = trend.window_std / abs(trend.window_mean) if trend.window_mean != 0 else 0
            if cv > 0.2:
                base_score -= 20
            elif cv > 0.1:
                base_score -= 10

        return max(0.0, min(100.0, base_score))

    def _calculate_overall_score(
        self,
        component_scores: Dict[str, float]
    ) -> float:
        """Calculate weighted overall quality score."""
        if not component_scores:
            return 50.0

        weighted_sum = 0.0
        weight_sum = 0.0

        for param, score in component_scores.items():
            weight = self.quality_weights.get(param, 0.1)
            weighted_sum += score * weight
            weight_sum += weight

        if weight_sum == 0:
            return 50.0

        return weighted_sum / weight_sum

    def _determine_quality_level(self, score: float) -> CombustionQuality:
        """Determine quality level from score."""
        if score >= 90:
            return CombustionQuality.EXCELLENT
        elif score >= 75:
            return CombustionQuality.GOOD
        elif score >= 60:
            return CombustionQuality.ACCEPTABLE
        elif score >= 40:
            return CombustionQuality.POOR
        else:
            return CombustionQuality.CRITICAL

    def _generate_insights(
        self,
        parameter: str,
        trend: TrendResult
    ) -> Tuple[List[str], List[str]]:
        """Generate recommendations and alerts from trend."""
        recommendations = []
        alerts = []

        if parameter == "O2":
            if trend.window_mean < 2.0:
                alerts.append("O2 critically low - risk of incomplete combustion")
                recommendations.append("Increase combustion air immediately")
            elif trend.direction == TrendDirection.DECREASING and trend.significance == TrendSignificance.SIGNIFICANT:
                recommendations.append("O2 trending down - consider adjusting air-fuel ratio")

        elif parameter == "CO":
            if trend.window_mean > 400:
                alerts.append("CO elevated - check for incomplete combustion")
            if trend.direction == TrendDirection.INCREASING:
                recommendations.append("CO trending up - check O2 levels and burner condition")

        elif parameter == "efficiency":
            if trend.direction == TrendDirection.DECREASING and trend.significance == TrendSignificance.SIGNIFICANT:
                recommendations.append("Efficiency declining - schedule tune-up")
                recommendations.append("Check for fouling, air leaks, or fuel quality issues")

        return recommendations, alerts

    def detect_change_point(
        self,
        parameter: str,
        window_minutes: int = 120
    ) -> Optional[datetime]:
        """
        Detect significant change point in data.

        Args:
            parameter: Parameter name
            window_minutes: Analysis window

        Returns:
            Timestamp of detected change point or None
        """
        if parameter not in self.data_buffers:
            return None

        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        samples = [s for s in self.data_buffers[parameter] if s.timestamp >= cutoff]

        if len(samples) < 20:
            return None

        values = [s.value for s in samples]
        n = len(values)

        # Simple change point detection using CUSUM
        mean = statistics.mean(values)
        cusum_pos = [0.0]
        cusum_neg = [0.0]

        for v in values:
            cusum_pos.append(max(0, cusum_pos[-1] + (v - mean)))
            cusum_neg.append(min(0, cusum_neg[-1] + (v - mean)))

        # Find maximum deviation
        max_pos_idx = cusum_pos.index(max(cusum_pos))
        min_neg_idx = cusum_neg.index(min(cusum_neg))

        # Use whichever has larger magnitude
        if abs(cusum_pos[max_pos_idx]) > abs(cusum_neg[min_neg_idx]):
            change_idx = max_pos_idx
        else:
            change_idx = min_neg_idx

        if 0 < change_idx < n:
            return samples[change_idx - 1].timestamp

        return None


if __name__ == "__main__":
    # Example usage
    analyzer = TrendAnalyzer()

    # Simulate data
    base_time = datetime.now() - timedelta(hours=1)

    for i in range(120):
        # O2 slowly decreasing
        o2_sample = DataSample(
            parameter="O2",
            value=4.0 - (i * 0.01) + (0.1 * math.sin(i / 10)),
            timestamp=base_time + timedelta(minutes=i),
        )
        analyzer.add_sample(o2_sample)

        # CO slowly increasing
        co_sample = DataSample(
            parameter="CO",
            value=50.0 + (i * 0.5),
            timestamp=base_time + timedelta(minutes=i),
        )
        analyzer.add_sample(co_sample)

    # Analyze trends
    o2_trend = analyzer.analyze_trend("O2")
    co_trend = analyzer.analyze_trend("CO")

    if o2_trend:
        print(f"O2 Trend: {o2_trend.direction.value}, Slope: {o2_trend.slope:.4f}, R²: {o2_trend.r_squared:.3f}")

    if co_trend:
        print(f"CO Trend: {co_trend.direction.value}, Slope: {co_trend.slope:.4f}, R²: {co_trend.r_squared:.3f}")

    # Get quality report
    report = analyzer.assess_combustion_quality()
    print(f"\nCombustion Quality: {report.quality_level.value} (Score: {report.quality_score:.1f})")
    print(f"Recommendations: {report.recommendations}")
