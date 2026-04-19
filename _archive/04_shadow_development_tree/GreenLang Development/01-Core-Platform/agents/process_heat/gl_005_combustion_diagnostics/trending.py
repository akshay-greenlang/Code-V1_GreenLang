# -*- coding: utf-8 -*-
"""
GL-005 Trending and Long-Term Analysis Module
=============================================

This module provides long-term trending, historical analysis, and performance
tracking capabilities for the GL-005 COMBUSENSE agent.

Key Capabilities:
    - Time-series data aggregation (hourly, daily, weekly, monthly)
    - Trend detection and significance testing
    - Seasonality detection
    - Baseline comparison
    - Performance degradation tracking
    - Compliance period summaries

ZERO-HALLUCINATION GUARANTEE:
    All trend analysis uses documented statistical methods.
    Seasonality detection uses deterministic algorithms.
    No AI/ML speculation in trend interpretation.

Author: GreenLang Process Heat Team
Version: 1.0.0
Status: Production Ready
"""

import hashlib
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.process_heat.gl_005_combustion_diagnostics.config import (
    TrendingConfig,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
    CQIResult,
    FlueGasReading,
    TrendDirection,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TimeSeriesPoint:
    """Single point in a time series."""

    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedPeriod:
    """Aggregated data for a time period."""

    period_start: datetime
    period_end: datetime
    count: int
    mean: float
    min_value: float
    max_value: float
    std_dev: float
    sum_value: float


@dataclass
class TrendAnalysisResult:
    """Result of trend analysis."""

    parameter: str
    period_start: datetime
    period_end: datetime
    data_points: int

    # Trend statistics
    slope: float  # Rate of change per day
    intercept: float
    r_squared: float  # Goodness of fit

    # Significance
    is_significant: bool
    p_value: float
    confidence_interval: Tuple[float, float]

    # Direction
    direction: TrendDirection

    # Summary
    total_change: float  # Change over period
    percent_change: float  # Percentage change


@dataclass
class SeasonalityResult:
    """Result of seasonality detection."""

    parameter: str
    seasonal_detected: bool
    seasonal_period_hours: Optional[int]  # e.g., 24 for daily
    seasonal_strength: float  # 0-1, strength of seasonal pattern
    seasonal_pattern: List[float]  # Average values by period


@dataclass
class BaselineComparison:
    """Comparison of current performance to baseline."""

    parameter: str
    baseline_period: Tuple[datetime, datetime]
    comparison_period: Tuple[datetime, datetime]

    baseline_mean: float
    current_mean: float
    absolute_change: float
    percent_change: float

    is_significant: bool
    direction: TrendDirection


# =============================================================================
# TIME SERIES STORAGE
# =============================================================================

class TimeSeriesStore:
    """
    In-memory time series data store.

    Stores raw data and aggregated summaries for efficient trending analysis.
    Implements automatic data aging based on retention policies.
    """

    def __init__(self, config: TrendingConfig) -> None:
        """
        Initialize time series store.

        Args:
            config: Trending configuration
        """
        self.config = config
        self._raw_data: Dict[str, List[TimeSeriesPoint]] = defaultdict(list)
        self._hourly_data: Dict[str, List[AggregatedPeriod]] = defaultdict(list)
        self._daily_data: Dict[str, List[AggregatedPeriod]] = defaultdict(list)
        self._weekly_data: Dict[str, List[AggregatedPeriod]] = defaultdict(list)
        self._monthly_data: Dict[str, List[AggregatedPeriod]] = defaultdict(list)

        logger.info(
            f"TimeSeriesStore initialized "
            f"(raw_retention={config.raw_data_retention_days}d, "
            f"aggregated_retention={config.aggregated_data_retention_days}d)"
        )

    def add_point(
        self,
        parameter: str,
        timestamp: datetime,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a data point to the store.

        Args:
            parameter: Parameter name (e.g., "cqi", "oxygen", "efficiency")
            timestamp: Data timestamp
            value: Measured value
            metadata: Optional metadata
        """
        point = TimeSeriesPoint(
            timestamp=timestamp,
            value=value,
            metadata=metadata or {},
        )
        self._raw_data[parameter].append(point)

        # Age out old data
        self._age_raw_data(parameter)

    def get_raw_data(
        self,
        parameter: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[TimeSeriesPoint]:
        """
        Get raw data points for a parameter.

        Args:
            parameter: Parameter name
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of TimeSeriesPoint
        """
        data = self._raw_data.get(parameter, [])

        if start_time:
            data = [p for p in data if p.timestamp >= start_time]
        if end_time:
            data = [p for p in data if p.timestamp <= end_time]

        return data

    def aggregate_hourly(self, parameter: str) -> List[AggregatedPeriod]:
        """Aggregate raw data to hourly summaries."""
        return self._aggregate_data(parameter, timedelta(hours=1))

    def aggregate_daily(self, parameter: str) -> List[AggregatedPeriod]:
        """Aggregate raw data to daily summaries."""
        return self._aggregate_data(parameter, timedelta(days=1))

    def aggregate_weekly(self, parameter: str) -> List[AggregatedPeriod]:
        """Aggregate raw data to weekly summaries."""
        return self._aggregate_data(parameter, timedelta(weeks=1))

    def aggregate_monthly(self, parameter: str) -> List[AggregatedPeriod]:
        """Aggregate raw data to monthly summaries (30-day periods)."""
        return self._aggregate_data(parameter, timedelta(days=30))

    def _aggregate_data(
        self,
        parameter: str,
        period: timedelta,
    ) -> List[AggregatedPeriod]:
        """Aggregate data into fixed periods."""
        raw_data = self._raw_data.get(parameter, [])
        if not raw_data:
            return []

        # Sort by timestamp
        sorted_data = sorted(raw_data, key=lambda p: p.timestamp)

        # Group by period
        periods: Dict[datetime, List[float]] = defaultdict(list)
        for point in sorted_data:
            # Round down to period start
            period_start = self._get_period_start(point.timestamp, period)
            periods[period_start].append(point.value)

        # Calculate aggregates
        aggregates = []
        for period_start, values in sorted(periods.items()):
            if not values:
                continue

            n = len(values)
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0
            std_dev = math.sqrt(variance)

            agg = AggregatedPeriod(
                period_start=period_start,
                period_end=period_start + period,
                count=n,
                mean=mean,
                min_value=min(values),
                max_value=max(values),
                std_dev=std_dev,
                sum_value=sum(values),
            )
            aggregates.append(agg)

        return aggregates

    def _get_period_start(self, timestamp: datetime, period: timedelta) -> datetime:
        """Get the start of the period containing timestamp."""
        if period == timedelta(hours=1):
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif period == timedelta(days=1):
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == timedelta(weeks=1):
            # Start of week (Monday)
            days_since_monday = timestamp.weekday()
            start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            return start - timedelta(days=days_since_monday)
        else:
            # Generic period - round down
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
            delta_seconds = (timestamp - epoch).total_seconds()
            period_seconds = period.total_seconds()
            period_number = int(delta_seconds // period_seconds)
            return epoch + timedelta(seconds=period_number * period_seconds)

    def _age_raw_data(self, parameter: str) -> None:
        """Remove raw data older than retention period."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self.config.raw_data_retention_days
        )
        self._raw_data[parameter] = [
            p for p in self._raw_data[parameter] if p.timestamp >= cutoff
        ]


# =============================================================================
# TREND ANALYZER
# =============================================================================

class TrendAnalyzer:
    """
    Time series trend analysis engine.

    Uses linear regression and statistical tests to detect significant trends
    in combustion parameters over time.

    DETERMINISTIC: Uses documented statistical methods.
    """

    def __init__(self, config: TrendingConfig) -> None:
        """
        Initialize trend analyzer.

        Args:
            config: Trending configuration
        """
        self.config = config
        logger.info(
            f"TrendAnalyzer initialized "
            f"(window={config.trend_detection_window_days}d, "
            f"significance={config.trend_significance_threshold})"
        )

    def analyze_trend(
        self,
        parameter: str,
        data_points: List[TimeSeriesPoint],
    ) -> TrendAnalysisResult:
        """
        Analyze trend in time series data.

        Uses ordinary least squares (OLS) linear regression to fit a trend
        line and assess significance.

        Args:
            parameter: Parameter name
            data_points: Time series data points

        Returns:
            TrendAnalysisResult with trend statistics
        """
        if len(data_points) < 2:
            return self._empty_trend_result(parameter)

        # Sort by timestamp
        sorted_data = sorted(data_points, key=lambda p: p.timestamp)

        # Convert to numeric arrays
        start_time = sorted_data[0].timestamp
        x_values = [
            (p.timestamp - start_time).total_seconds() / 86400  # Days
            for p in sorted_data
        ]
        y_values = [p.value for p in sorted_data]

        # Calculate linear regression
        n = len(x_values)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return self._empty_trend_result(parameter)

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Calculate R-squared
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate standard error and t-statistic
        if n > 2:
            mse = ss_res / (n - 2)
            se_slope = math.sqrt(mse / denominator) if denominator > 0 else 0
            t_stat = slope / se_slope if se_slope > 0 else 0

            # Approximate p-value (using normal approximation for large n)
            # For exact p-value, would need t-distribution
            p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))

            # 95% confidence interval
            ci_margin = 1.96 * se_slope
            confidence_interval = (slope - ci_margin, slope + ci_margin)
        else:
            p_value = 1.0
            confidence_interval = (slope, slope)

        # Determine significance
        is_significant = p_value < self.config.trend_significance_threshold

        # Determine direction
        if not is_significant:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.DEGRADING  # Assume increasing = bad for most params
        else:
            direction = TrendDirection.IMPROVING

        # Calculate total and percent change
        total_days = x_values[-1] - x_values[0] if x_values else 0
        total_change = slope * total_days
        first_value = y_values[0] if y_values else 1
        percent_change = (total_change / first_value * 100) if first_value != 0 else 0

        return TrendAnalysisResult(
            parameter=parameter,
            period_start=sorted_data[0].timestamp,
            period_end=sorted_data[-1].timestamp,
            data_points=n,
            slope=round(slope, 6),
            intercept=round(intercept, 4),
            r_squared=round(r_squared, 4),
            is_significant=is_significant,
            p_value=round(p_value, 4),
            confidence_interval=(round(confidence_interval[0], 6), round(confidence_interval[1], 6)),
            direction=direction,
            total_change=round(total_change, 4),
            percent_change=round(percent_change, 2),
        )

    def _empty_trend_result(self, parameter: str) -> TrendAnalysisResult:
        """Return empty trend result for insufficient data."""
        now = datetime.now(timezone.utc)
        return TrendAnalysisResult(
            parameter=parameter,
            period_start=now,
            period_end=now,
            data_points=0,
            slope=0.0,
            intercept=0.0,
            r_squared=0.0,
            is_significant=False,
            p_value=1.0,
            confidence_interval=(0.0, 0.0),
            direction=TrendDirection.UNKNOWN,
            total_change=0.0,
            percent_change=0.0,
        )

    def _normal_cdf(self, x: float) -> float:
        """Approximate cumulative distribution function for standard normal."""
        # Using error function approximation
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def detect_seasonality(
        self,
        parameter: str,
        data_points: List[TimeSeriesPoint],
        period_hours: int = 24,
    ) -> SeasonalityResult:
        """
        Detect seasonal patterns in time series.

        Uses autocorrelation to detect periodic patterns.

        Args:
            parameter: Parameter name
            data_points: Time series data points
            period_hours: Period to test (e.g., 24 for daily)

        Returns:
            SeasonalityResult with pattern analysis
        """
        if len(data_points) < period_hours * 2:
            return SeasonalityResult(
                parameter=parameter,
                seasonal_detected=False,
                seasonal_period_hours=period_hours,
                seasonal_strength=0.0,
                seasonal_pattern=[],
            )

        # Sort and extract values
        sorted_data = sorted(data_points, key=lambda p: p.timestamp)
        values = [p.value for p in sorted_data]

        # Calculate autocorrelation at the specified lag
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n

        if variance == 0:
            return SeasonalityResult(
                parameter=parameter,
                seasonal_detected=False,
                seasonal_period_hours=period_hours,
                seasonal_strength=0.0,
                seasonal_pattern=[],
            )

        # Approximate lag in data points (assume roughly hourly data)
        lag = period_hours

        if lag >= n:
            return SeasonalityResult(
                parameter=parameter,
                seasonal_detected=False,
                seasonal_period_hours=period_hours,
                seasonal_strength=0.0,
                seasonal_pattern=[],
            )

        # Calculate autocorrelation
        autocov = sum((values[i] - mean) * (values[i + lag] - mean) for i in range(n - lag)) / (n - lag)
        autocorr = autocov / variance if variance > 0 else 0

        # Seasonal strength (absolute autocorrelation)
        strength = abs(autocorr)
        seasonal_detected = strength > 0.3  # Threshold for significant seasonality

        # Calculate seasonal pattern (average by hour-of-day)
        pattern = []
        if seasonal_detected and period_hours == 24:
            hourly_values: Dict[int, List[float]] = defaultdict(list)
            for point in sorted_data:
                hour = point.timestamp.hour
                hourly_values[hour].append(point.value)

            for hour in range(24):
                if hourly_values[hour]:
                    pattern.append(sum(hourly_values[hour]) / len(hourly_values[hour]))
                else:
                    pattern.append(mean)

        return SeasonalityResult(
            parameter=parameter,
            seasonal_detected=seasonal_detected,
            seasonal_period_hours=period_hours,
            seasonal_strength=round(strength, 4),
            seasonal_pattern=[round(v, 4) for v in pattern],
        )

    def compare_to_baseline(
        self,
        parameter: str,
        baseline_data: List[TimeSeriesPoint],
        current_data: List[TimeSeriesPoint],
    ) -> BaselineComparison:
        """
        Compare current performance to baseline period.

        Args:
            parameter: Parameter name
            baseline_data: Data from baseline period
            current_data: Data from current period

        Returns:
            BaselineComparison with change analysis
        """
        if not baseline_data or not current_data:
            now = datetime.now(timezone.utc)
            return BaselineComparison(
                parameter=parameter,
                baseline_period=(now, now),
                comparison_period=(now, now),
                baseline_mean=0.0,
                current_mean=0.0,
                absolute_change=0.0,
                percent_change=0.0,
                is_significant=False,
                direction=TrendDirection.UNKNOWN,
            )

        # Calculate means
        baseline_values = [p.value for p in baseline_data]
        current_values = [p.value for p in current_data]

        baseline_mean = sum(baseline_values) / len(baseline_values)
        current_mean = sum(current_values) / len(current_values)

        # Calculate change
        absolute_change = current_mean - baseline_mean
        percent_change = (absolute_change / baseline_mean * 100) if baseline_mean != 0 else 0

        # Statistical significance test (two-sample t-test approximation)
        n1, n2 = len(baseline_values), len(current_values)
        var1 = sum((v - baseline_mean) ** 2 for v in baseline_values) / (n1 - 1) if n1 > 1 else 0
        var2 = sum((v - current_mean) ** 2 for v in current_values) / (n2 - 1) if n2 > 1 else 0

        pooled_se = math.sqrt(var1 / n1 + var2 / n2) if (var1 > 0 or var2 > 0) else 0
        t_stat = abs(absolute_change) / pooled_se if pooled_se > 0 else 0
        p_value = 2 * (1 - self._normal_cdf(t_stat))

        is_significant = p_value < self.config.trend_significance_threshold

        # Determine direction
        if not is_significant:
            direction = TrendDirection.STABLE
        elif absolute_change > 0:
            direction = TrendDirection.DEGRADING
        else:
            direction = TrendDirection.IMPROVING

        # Get period ranges
        baseline_sorted = sorted(baseline_data, key=lambda p: p.timestamp)
        current_sorted = sorted(current_data, key=lambda p: p.timestamp)

        return BaselineComparison(
            parameter=parameter,
            baseline_period=(baseline_sorted[0].timestamp, baseline_sorted[-1].timestamp),
            comparison_period=(current_sorted[0].timestamp, current_sorted[-1].timestamp),
            baseline_mean=round(baseline_mean, 4),
            current_mean=round(current_mean, 4),
            absolute_change=round(absolute_change, 4),
            percent_change=round(percent_change, 2),
            is_significant=is_significant,
            direction=direction,
        )


# =============================================================================
# TRENDING ENGINE
# =============================================================================

class TrendingEngine:
    """
    Integrated Trending Engine for GL-005.

    Combines time series storage, trend analysis, and reporting
    into a unified interface for long-term combustion performance tracking.

    Example:
        >>> config = TrendingConfig()
        >>> engine = TrendingEngine(config)
        >>> engine.add_cqi_result(cqi_result)
        >>> trend = engine.get_cqi_trend()
        >>> print(f"CQI trend: {trend.direction.value}")
    """

    def __init__(self, config: TrendingConfig) -> None:
        """
        Initialize trending engine.

        Args:
            config: Trending configuration
        """
        self.config = config
        self.store = TimeSeriesStore(config)
        self.analyzer = TrendAnalyzer(config)

        self._audit_trail: List[Dict[str, Any]] = []

        logger.info("TrendingEngine initialized")

    def add_cqi_result(self, cqi_result: CQIResult) -> None:
        """
        Add CQI result to trending store.

        Args:
            cqi_result: CQI calculation result
        """
        self.store.add_point(
            parameter="cqi",
            timestamp=cqi_result.calculation_timestamp,
            value=cqi_result.cqi_score,
            metadata={"rating": cqi_result.cqi_rating.value},
        )

        # Also store component values
        for component in cqi_result.components:
            self.store.add_point(
                parameter=f"cqi_{component.component}",
                timestamp=cqi_result.calculation_timestamp,
                value=component.raw_value,
            )

        # Store efficiency
        self.store.add_point(
            parameter="efficiency",
            timestamp=cqi_result.calculation_timestamp,
            value=cqi_result.combustion_efficiency_pct,
        )

    def add_flue_gas_reading(self, reading: FlueGasReading) -> None:
        """
        Add flue gas reading to trending store.

        Args:
            reading: Flue gas reading
        """
        self.store.add_point("oxygen", reading.timestamp, reading.oxygen_pct)
        self.store.add_point("co2", reading.timestamp, reading.co2_pct)
        self.store.add_point("co", reading.timestamp, reading.co_ppm)
        self.store.add_point("nox", reading.timestamp, reading.nox_ppm)
        self.store.add_point("stack_temp", reading.timestamp, reading.flue_gas_temp_c)

        if reading.combustibles_pct is not None:
            self.store.add_point("combustibles", reading.timestamp, reading.combustibles_pct)

    def get_cqi_trend(
        self,
        days: Optional[int] = None,
    ) -> TrendAnalysisResult:
        """
        Get CQI trend analysis.

        Args:
            days: Number of days to analyze (default from config)

        Returns:
            TrendAnalysisResult for CQI
        """
        days = days or self.config.trend_detection_window_days
        start_time = datetime.now(timezone.utc) - timedelta(days=days)
        data = self.store.get_raw_data("cqi", start_time=start_time)
        return self.analyzer.analyze_trend("cqi", data)

    def get_efficiency_trend(
        self,
        days: Optional[int] = None,
    ) -> TrendAnalysisResult:
        """
        Get combustion efficiency trend analysis.

        Args:
            days: Number of days to analyze

        Returns:
            TrendAnalysisResult for efficiency
        """
        days = days or self.config.trend_detection_window_days
        start_time = datetime.now(timezone.utc) - timedelta(days=days)
        data = self.store.get_raw_data("efficiency", start_time=start_time)
        return self.analyzer.analyze_trend("efficiency", data)

    def get_parameter_trend(
        self,
        parameter: str,
        days: Optional[int] = None,
    ) -> TrendAnalysisResult:
        """
        Get trend analysis for any parameter.

        Args:
            parameter: Parameter name
            days: Number of days to analyze

        Returns:
            TrendAnalysisResult
        """
        days = days or self.config.trend_detection_window_days
        start_time = datetime.now(timezone.utc) - timedelta(days=days)
        data = self.store.get_raw_data(parameter, start_time=start_time)
        return self.analyzer.analyze_trend(parameter, data)

    def get_daily_summary(self, parameter: str) -> List[AggregatedPeriod]:
        """Get daily aggregated data for a parameter."""
        return self.store.aggregate_daily(parameter)

    def get_weekly_summary(self, parameter: str) -> List[AggregatedPeriod]:
        """Get weekly aggregated data for a parameter."""
        return self.store.aggregate_weekly(parameter)

    def check_seasonality(
        self,
        parameter: str,
    ) -> SeasonalityResult:
        """
        Check for daily seasonal patterns in a parameter.

        Args:
            parameter: Parameter name

        Returns:
            SeasonalityResult
        """
        if not self.config.detect_seasonality:
            return SeasonalityResult(
                parameter=parameter,
                seasonal_detected=False,
                seasonal_period_hours=24,
                seasonal_strength=0.0,
                seasonal_pattern=[],
            )

        data = self.store.get_raw_data(parameter)
        return self.analyzer.detect_seasonality(parameter, data, period_hours=24)

    def compare_to_baseline(
        self,
        parameter: str,
        baseline_days: Optional[int] = None,
        comparison_days: int = 7,
    ) -> BaselineComparison:
        """
        Compare recent performance to baseline.

        Args:
            parameter: Parameter name
            baseline_days: Days for baseline period (default from config)
            comparison_days: Days for comparison period

        Returns:
            BaselineComparison
        """
        baseline_days = baseline_days or self.config.baseline_period_days
        now = datetime.now(timezone.utc)

        # Baseline period: from (baseline_days + comparison_days) ago to comparison_days ago
        baseline_end = now - timedelta(days=comparison_days)
        baseline_start = baseline_end - timedelta(days=baseline_days)

        # Comparison period: last comparison_days
        comparison_start = now - timedelta(days=comparison_days)

        baseline_data = self.store.get_raw_data(
            parameter,
            start_time=baseline_start,
            end_time=baseline_end,
        )
        current_data = self.store.get_raw_data(
            parameter,
            start_time=comparison_start,
        )

        return self.analyzer.compare_to_baseline(parameter, baseline_data, current_data)

    def get_trend_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive trend summary for all key parameters.

        Returns:
            Dictionary with trend summaries
        """
        parameters = ["cqi", "efficiency", "oxygen", "co", "nox", "stack_temp"]
        summary = {}

        for param in parameters:
            trend = self.get_parameter_trend(param)
            summary[param] = {
                "direction": trend.direction.value,
                "slope_per_day": trend.slope,
                "percent_change": trend.percent_change,
                "is_significant": trend.is_significant,
                "data_points": trend.data_points,
            }

        return summary

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get trending engine audit trail."""
        return self._audit_trail.copy()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_moving_average(
    values: List[float],
    window: int = 7,
) -> List[float]:
    """
    Calculate simple moving average.

    Args:
        values: Input values
        window: Window size

    Returns:
        List of moving averages
    """
    if len(values) < window:
        return values.copy()

    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_values = values[start:i + 1]
        result.append(sum(window_values) / len(window_values))

    return result


def calculate_exponential_moving_average(
    values: List[float],
    alpha: float = 0.2,
) -> List[float]:
    """
    Calculate exponential moving average.

    Args:
        values: Input values
        alpha: Smoothing factor (0-1)

    Returns:
        List of EMAs
    """
    if not values:
        return []

    result = [values[0]]
    for i in range(1, len(values)):
        ema = alpha * values[i] + (1 - alpha) * result[-1]
        result.append(ema)

    return result
