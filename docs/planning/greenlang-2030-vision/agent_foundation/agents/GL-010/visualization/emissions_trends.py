"""
GL-010 EMISSIONWATCH - Emissions Trends Visualization

Time-series emissions visualization module for the EmissionsComplianceAgent.
Provides hourly, daily, monthly, and annual trend analysis with forecasting.

Author: GreenLang Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from datetime import datetime, timedelta
import json
import math
from abc import ABC, abstractmethod


class TimeResolution(Enum):
    """Time resolution for trend analysis."""
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"

    @property
    def display_format(self) -> str:
        """Return display format for timestamps."""
        formats = {
            "minute": "%H:%M",
            "hourly": "%H:00",
            "daily": "%Y-%m-%d",
            "weekly": "Week %W, %Y",
            "monthly": "%b %Y",
            "quarterly": "Q%q %Y",
            "annual": "%Y"
        }
        return formats.get(self.value, "%Y-%m-%d %H:%M")


class TrendDirection(Enum):
    """Trend direction enumeration."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class EmissionDataPoint:
    """Single emission data point."""
    timestamp: str
    value: float
    unit: str
    data_quality: float
    is_valid: bool = True
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "unit": self.unit,
            "data_quality": self.data_quality,
            "is_valid": self.is_valid,
            "flags": self.flags
        }


@dataclass
class TrendStatistics:
    """Statistical summary of trend data."""
    min_value: float
    max_value: float
    mean_value: float
    median_value: float
    std_deviation: float
    percentile_90: float
    percentile_95: float
    percentile_99: float
    total_points: int
    valid_points: int
    exceedance_count: int
    exceedance_hours: float
    trend_direction: TrendDirection
    trend_slope: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": self.mean_value,
            "median_value": self.median_value,
            "std_deviation": self.std_deviation,
            "percentile_90": self.percentile_90,
            "percentile_95": self.percentile_95,
            "percentile_99": self.percentile_99,
            "total_points": self.total_points,
            "valid_points": self.valid_points,
            "exceedance_count": self.exceedance_count,
            "exceedance_hours": self.exceedance_hours,
            "trend_direction": self.trend_direction.value,
            "trend_slope": self.trend_slope
        }


@dataclass
class TrendConfig:
    """Configuration for trend visualization."""
    pollutant: str
    pollutant_name: str
    unit: str
    permit_limit: float
    warning_threshold: float  # Usually 90% of limit
    resolution: TimeResolution
    show_rolling_average: bool = True
    rolling_window: int = 24  # hours
    show_forecast: bool = False
    forecast_periods: int = 24
    show_confidence_bands: bool = True
    confidence_level: float = 0.95
    highlight_anomalies: bool = True
    anomaly_threshold: float = 2.0  # std deviations
    color_blind_safe: bool = False


class StatisticsCalculator:
    """Calculate statistics for emission data."""

    @staticmethod
    def calculate(
        data: List[EmissionDataPoint],
        permit_limit: float
    ) -> TrendStatistics:
        """
        Calculate comprehensive statistics for emission data.

        Args:
            data: List of emission data points
            permit_limit: Regulatory permit limit

        Returns:
            TrendStatistics object
        """
        if not data:
            return StatisticsCalculator._empty_stats()

        values = [d.value for d in data if d.is_valid]
        if not values:
            return StatisticsCalculator._empty_stats()

        sorted_values = sorted(values)
        n = len(sorted_values)

        # Basic statistics
        min_val = sorted_values[0]
        max_val = sorted_values[-1]
        mean_val = sum(sorted_values) / n
        median_val = sorted_values[n // 2] if n % 2 else \
            (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2

        # Standard deviation
        variance = sum((x - mean_val) ** 2 for x in sorted_values) / n
        std_dev = math.sqrt(variance)

        # Percentiles
        p90 = StatisticsCalculator._percentile(sorted_values, 90)
        p95 = StatisticsCalculator._percentile(sorted_values, 95)
        p99 = StatisticsCalculator._percentile(sorted_values, 99)

        # Exceedances
        exceedances = [v for v in values if v > permit_limit]
        exceedance_count = len(exceedances)

        # Trend analysis
        trend_direction, trend_slope = StatisticsCalculator._calculate_trend(values)

        return TrendStatistics(
            min_value=min_val,
            max_value=max_val,
            mean_value=mean_val,
            median_value=median_val,
            std_deviation=std_dev,
            percentile_90=p90,
            percentile_95=p95,
            percentile_99=p99,
            total_points=len(data),
            valid_points=n,
            exceedance_count=exceedance_count,
            exceedance_hours=exceedance_count,  # Assuming hourly data
            trend_direction=trend_direction,
            trend_slope=trend_slope
        )

    @staticmethod
    def _empty_stats() -> TrendStatistics:
        """Return empty statistics."""
        return TrendStatistics(
            min_value=0.0,
            max_value=0.0,
            mean_value=0.0,
            median_value=0.0,
            std_deviation=0.0,
            percentile_90=0.0,
            percentile_95=0.0,
            percentile_99=0.0,
            total_points=0,
            valid_points=0,
            exceedance_count=0,
            exceedance_hours=0.0,
            trend_direction=TrendDirection.STABLE,
            trend_slope=0.0
        )

    @staticmethod
    def _percentile(sorted_values: List[float], p: int) -> float:
        """Calculate percentile value."""
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_values[int(k)]
        return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)

    @staticmethod
    def _calculate_trend(values: List[float]) -> Tuple[TrendDirection, float]:
        """Calculate trend direction and slope using linear regression."""
        if len(values) < 2:
            return TrendDirection.STABLE, 0.0

        n = len(values)
        x = list(range(n))
        mean_x = sum(x) / n
        mean_y = sum(values) / n

        numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

        if denominator == 0:
            return TrendDirection.STABLE, 0.0

        slope = numerator / denominator

        # Determine trend direction based on slope significance
        std_y = math.sqrt(sum((v - mean_y) ** 2 for v in values) / n)
        threshold = 0.01 * std_y if std_y > 0 else 0.01

        if abs(slope) < threshold:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        return direction, slope


class RollingAverageCalculator:
    """Calculate rolling averages for emission data."""

    @staticmethod
    def calculate(
        data: List[EmissionDataPoint],
        window_size: int
    ) -> List[Dict[str, Any]]:
        """
        Calculate rolling average.

        Args:
            data: List of emission data points
            window_size: Number of periods for rolling window

        Returns:
            List of rolling average data points
        """
        if len(data) < window_size:
            return []

        result = []
        values = [d.value for d in data]

        for i in range(window_size - 1, len(values)):
            window = values[i - window_size + 1:i + 1]
            avg = sum(window) / len(window)
            result.append({
                "timestamp": data[i].timestamp,
                "value": avg,
                "window_size": window_size
            })

        return result


class AnomalyDetector:
    """Detect anomalies in emission data."""

    @staticmethod
    def detect(
        data: List[EmissionDataPoint],
        threshold_std: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies using z-score method.

        Args:
            data: List of emission data points
            threshold_std: Number of standard deviations for anomaly threshold

        Returns:
            List of anomaly data points
        """
        if len(data) < 3:
            return []

        values = [d.value for d in data if d.is_valid]
        if not values:
            return []

        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

        if std == 0:
            return []

        anomalies = []
        for dp in data:
            if not dp.is_valid:
                continue

            z_score = abs(dp.value - mean) / std
            if z_score > threshold_std:
                anomalies.append({
                    "timestamp": dp.timestamp,
                    "value": dp.value,
                    "z_score": z_score,
                    "deviation": dp.value - mean
                })

        return anomalies


class SimpleForecast:
    """Simple forecasting using linear regression."""

    @staticmethod
    def forecast(
        data: List[EmissionDataPoint],
        periods: int,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Generate forecast using linear regression.

        Args:
            data: Historical emission data
            periods: Number of periods to forecast
            confidence_level: Confidence level for prediction intervals

        Returns:
            Forecast data with confidence intervals
        """
        if len(data) < 10:
            return {"error": "Insufficient data for forecasting"}

        values = [d.value for d in data if d.is_valid]
        n = len(values)

        if n < 10:
            return {"error": "Insufficient valid data points"}

        # Linear regression
        x = list(range(n))
        mean_x = sum(x) / n
        mean_y = sum(values) / n

        numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

        if denominator == 0:
            return {"error": "Cannot calculate trend"}

        slope = numerator / denominator
        intercept = mean_y - slope * mean_x

        # Calculate standard error
        residuals = [values[i] - (slope * i + intercept) for i in range(n)]
        sse = sum(r ** 2 for r in residuals)
        mse = sse / (n - 2) if n > 2 else sse
        se = math.sqrt(mse)

        # Generate forecast
        forecast_values = []
        lower_bound = []
        upper_bound = []

        # Z-value for confidence interval (approximate)
        z = 1.96 if confidence_level >= 0.95 else 1.645

        for i in range(n, n + periods):
            predicted = slope * i + intercept
            forecast_values.append(predicted)

            # Prediction interval (simplified)
            margin = z * se * math.sqrt(1 + 1/n + ((i - mean_x) ** 2) / denominator)
            lower_bound.append(max(0, predicted - margin))
            upper_bound.append(predicted + margin)

        # Generate timestamps (simplified - assume hourly)
        last_ts = data[-1].timestamp
        try:
            base_dt = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))
        except ValueError:
            base_dt = datetime.now()

        forecast_timestamps = [
            (base_dt + timedelta(hours=i+1)).isoformat()
            for i in range(periods)
        ]

        return {
            "timestamps": forecast_timestamps,
            "values": forecast_values,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "confidence_level": confidence_level,
            "model": "linear_regression",
            "slope": slope,
            "intercept": intercept
        }


class EmissionsTrendChart:
    """Generate emissions trend visualization."""

    def __init__(self, config: TrendConfig):
        """
        Initialize trend chart generator.

        Args:
            config: Trend visualization configuration
        """
        self.config = config
        self._data: List[EmissionDataPoint] = []
        self._statistics: Optional[TrendStatistics] = None
        self._rolling_avg: List[Dict[str, Any]] = []
        self._anomalies: List[Dict[str, Any]] = []
        self._forecast: Optional[Dict[str, Any]] = None

    def set_data(self, data: List[EmissionDataPoint]) -> None:
        """
        Set emission data for visualization.

        Args:
            data: List of emission data points
        """
        self._data = data
        self._calculate_derived_data()

    def _calculate_derived_data(self) -> None:
        """Calculate statistics, rolling averages, anomalies, and forecast."""
        if not self._data:
            return

        # Statistics
        self._statistics = StatisticsCalculator.calculate(
            self._data,
            self.config.permit_limit
        )

        # Rolling average
        if self.config.show_rolling_average:
            self._rolling_avg = RollingAverageCalculator.calculate(
                self._data,
                self.config.rolling_window
            )

        # Anomalies
        if self.config.highlight_anomalies:
            self._anomalies = AnomalyDetector.detect(
                self._data,
                self.config.anomaly_threshold
            )

        # Forecast
        if self.config.show_forecast:
            self._forecast = SimpleForecast.forecast(
                self._data,
                self.config.forecast_periods,
                self.config.confidence_level
            )

    def get_colors(self) -> Dict[str, str]:
        """Get color scheme based on configuration."""
        if self.config.color_blind_safe:
            return {
                "primary": "#0072B2",
                "limit": "#D55E00",
                "warning": "#E69F00",
                "compliant": "#009E73",
                "rolling": "#CC79A7",
                "forecast": "#56B4E9",
                "anomaly": "#D55E00",
                "confidence": "rgba(86, 180, 233, 0.2)"
            }
        return {
            "primary": "#3498DB",
            "limit": "#E74C3C",
            "warning": "#F39C12",
            "compliant": "#2ECC71",
            "rolling": "#9B59B6",
            "forecast": "#1ABC9C",
            "anomaly": "#E74C3C",
            "confidence": "rgba(26, 188, 156, 0.2)"
        }

    def build_hourly_trend(self) -> Dict[str, Any]:
        """
        Build hourly emissions trend chart.

        Returns:
            Plotly chart dictionary
        """
        colors = self.get_colors()
        timestamps = [d.timestamp for d in self._data]
        values = [d.value for d in self._data]

        traces = []

        # Main emission trace
        traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "name": self.config.pollutant_name,
            "x": timestamps,
            "y": values,
            "line": {"color": colors["primary"], "width": 2},
            "marker": {"size": 4},
            "hovertemplate": (
                f"<b>{self.config.pollutant_name}</b><br>"
                f"Time: %{{x}}<br>"
                f"Value: %{{y:.2f}} {self.config.unit}<extra></extra>"
            )
        })

        # Permit limit line
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "name": "Permit Limit",
            "x": timestamps,
            "y": [self.config.permit_limit] * len(timestamps),
            "line": {"color": colors["limit"], "width": 2, "dash": "dash"},
            "hovertemplate": f"Permit Limit: {self.config.permit_limit} {self.config.unit}<extra></extra>"
        })

        # Warning threshold line
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "name": "Warning (90%)",
            "x": timestamps,
            "y": [self.config.warning_threshold] * len(timestamps),
            "line": {"color": colors["warning"], "width": 1, "dash": "dot"},
            "hovertemplate": f"Warning: {self.config.warning_threshold} {self.config.unit}<extra></extra>"
        })

        # Rolling average
        if self.config.show_rolling_average and self._rolling_avg:
            ra_timestamps = [r["timestamp"] for r in self._rolling_avg]
            ra_values = [r["value"] for r in self._rolling_avg]
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "name": f"{self.config.rolling_window}h Rolling Avg",
                "x": ra_timestamps,
                "y": ra_values,
                "line": {"color": colors["rolling"], "width": 2},
                "hovertemplate": (
                    f"<b>{self.config.rolling_window}h Rolling Average</b><br>"
                    f"Time: %{{x}}<br>"
                    f"Value: %{{y:.2f}} {self.config.unit}<extra></extra>"
                )
            })

        # Anomalies
        if self.config.highlight_anomalies and self._anomalies:
            anom_timestamps = [a["timestamp"] for a in self._anomalies]
            anom_values = [a["value"] for a in self._anomalies]
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "name": "Anomalies",
                "x": anom_timestamps,
                "y": anom_values,
                "marker": {
                    "color": colors["anomaly"],
                    "size": 12,
                    "symbol": "x"
                },
                "hovertemplate": (
                    "<b>ANOMALY</b><br>"
                    f"Time: %{{x}}<br>"
                    f"Value: %{{y:.2f}} {self.config.unit}<extra></extra>"
                )
            })

        # Forecast
        if self.config.show_forecast and self._forecast and "values" in self._forecast:
            fc = self._forecast
            # Confidence band
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "name": "Upper Bound",
                "x": fc["timestamps"],
                "y": fc["upper_bound"],
                "line": {"width": 0},
                "showlegend": False,
                "hoverinfo": "skip"
            })
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "name": f"Forecast ({int(fc['confidence_level']*100)}% CI)",
                "x": fc["timestamps"],
                "y": fc["lower_bound"],
                "line": {"width": 0},
                "fill": "tonexty",
                "fillcolor": colors["confidence"],
                "hoverinfo": "skip"
            })
            # Forecast line
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "name": "Forecast",
                "x": fc["timestamps"],
                "y": fc["values"],
                "line": {"color": colors["forecast"], "width": 2, "dash": "dash"},
                "hovertemplate": (
                    "<b>Forecast</b><br>"
                    f"Time: %{{x}}<br>"
                    f"Value: %{{y:.2f}} {self.config.unit}<extra></extra>"
                )
            })

        # Build exceedance shapes
        shapes = self._build_exceedance_shapes(timestamps, values)

        layout = {
            "title": {
                "text": f"{self.config.pollutant_name} - Hourly Emissions Trend",
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "xaxis": {
                "title": "Time",
                "type": "date",
                "rangeslider": {"visible": True},
                "rangeselector": {
                    "buttons": [
                        {"count": 6, "label": "6h", "step": "hour", "stepmode": "backward"},
                        {"count": 24, "label": "24h", "step": "hour", "stepmode": "backward"},
                        {"count": 7, "label": "7d", "step": "day", "stepmode": "backward"},
                        {"step": "all", "label": "All"}
                    ]
                }
            },
            "yaxis": {
                "title": f"{self.config.pollutant_name} ({self.config.unit})",
                "rangemode": "tozero"
            },
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1
            },
            "shapes": shapes,
            "hovermode": "x unified",
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA"
        }

        return {
            "data": traces,
            "layout": layout,
            "config": {
                "responsive": True,
                "displayModeBar": True,
                "displaylogo": False
            }
        }

    def build_daily_summary(self, daily_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build daily rolling average summary chart.

        Args:
            daily_data: List of daily summary data

        Returns:
            Plotly chart dictionary
        """
        colors = self.get_colors()

        dates = [d["date"] for d in daily_data]
        daily_avg = [d["avg"] for d in daily_data]
        daily_max = [d["max"] for d in daily_data]
        daily_min = [d["min"] for d in daily_data]

        traces = [
            # Daily average
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Daily Average",
                "x": dates,
                "y": daily_avg,
                "line": {"color": colors["primary"], "width": 2},
                "marker": {"size": 6}
            },
            # Daily max
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Daily Max",
                "x": dates,
                "y": daily_max,
                "line": {"color": colors["anomaly"], "width": 1, "dash": "dot"}
            },
            # Daily min
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Daily Min",
                "x": dates,
                "y": daily_min,
                "line": {"color": colors["compliant"], "width": 1, "dash": "dot"}
            },
            # Permit limit
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Permit Limit",
                "x": dates,
                "y": [self.config.permit_limit] * len(dates),
                "line": {"color": colors["limit"], "width": 2, "dash": "dash"}
            }
        ]

        layout = {
            "title": {
                "text": f"{self.config.pollutant_name} - Daily Summary",
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "xaxis": {"title": "Date", "type": "date"},
            "yaxis": {
                "title": f"{self.config.pollutant_name} ({self.config.unit})",
                "rangemode": "tozero"
            },
            "legend": {"orientation": "h", "y": 1.1},
            "hovermode": "x unified",
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA"
        }

        return {"data": traces, "layout": layout}

    def build_monthly_comparison(
        self,
        monthly_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Build monthly comparison chart (current vs previous year).

        Args:
            monthly_data: Dictionary with year keys and monthly data values

        Returns:
            Plotly chart dictionary
        """
        colors = self.get_colors()
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        traces = []
        color_list = [colors["primary"], "#95A5A6", "#BDC3C7"]

        for idx, (year, data) in enumerate(monthly_data.items()):
            values = [d.get("avg", 0) for d in data]
            traces.append({
                "type": "bar",
                "name": str(year),
                "x": months[:len(values)],
                "y": values,
                "marker": {"color": color_list[idx % len(color_list)]}
            })

        # Add limit line
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "name": "Permit Limit",
            "x": months,
            "y": [self.config.permit_limit] * 12,
            "line": {"color": colors["limit"], "width": 2, "dash": "dash"}
        })

        layout = {
            "title": {
                "text": f"{self.config.pollutant_name} - Monthly Comparison",
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "xaxis": {"title": "Month"},
            "yaxis": {
                "title": f"Average {self.config.pollutant_name} ({self.config.unit})",
                "rangemode": "tozero"
            },
            "barmode": "group",
            "legend": {"orientation": "h", "y": 1.1},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA"
        }

        return {"data": traces, "layout": layout}

    def build_annual_summary(
        self,
        annual_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build annual emissions summary chart.

        Args:
            annual_data: List of annual summary data

        Returns:
            Plotly chart dictionary
        """
        colors = self.get_colors()

        years = [str(d["year"]) for d in annual_data]
        totals = [d["total"] for d in annual_data]
        averages = [d["avg"] for d in annual_data]
        exceedance_hours = [d.get("exceedance_hours", 0) for d in annual_data]

        traces = [
            # Annual average (bar)
            {
                "type": "bar",
                "name": "Annual Average",
                "x": years,
                "y": averages,
                "marker": {"color": colors["primary"]},
                "yaxis": "y"
            },
            # Exceedance hours (line on secondary axis)
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Exceedance Hours",
                "x": years,
                "y": exceedance_hours,
                "line": {"color": colors["anomaly"], "width": 2},
                "marker": {"size": 8},
                "yaxis": "y2"
            },
            # Permit limit
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Permit Limit",
                "x": years,
                "y": [self.config.permit_limit] * len(years),
                "line": {"color": colors["limit"], "width": 2, "dash": "dash"},
                "yaxis": "y"
            }
        ]

        layout = {
            "title": {
                "text": f"{self.config.pollutant_name} - Annual Summary",
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "xaxis": {"title": "Year"},
            "yaxis": {
                "title": f"Average {self.config.pollutant_name} ({self.config.unit})",
                "rangemode": "tozero"
            },
            "yaxis2": {
                "title": "Exceedance Hours",
                "overlaying": "y",
                "side": "right",
                "rangemode": "tozero"
            },
            "legend": {"orientation": "h", "y": 1.1},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA"
        }

        return {"data": traces, "layout": layout}

    def build_statistics_panel(self) -> Dict[str, Any]:
        """
        Build statistics summary panel.

        Returns:
            Plotly indicator/table chart dictionary
        """
        if not self._statistics:
            return {"data": [], "layout": {"title": "No statistics available"}}

        stats = self._statistics

        # Create indicator traces
        traces = [
            {
                "type": "indicator",
                "mode": "number",
                "value": stats.mean_value,
                "title": {"text": "Mean"},
                "number": {"suffix": f" {self.config.unit}"},
                "domain": {"row": 0, "column": 0}
            },
            {
                "type": "indicator",
                "mode": "number",
                "value": stats.max_value,
                "title": {"text": "Maximum"},
                "number": {"suffix": f" {self.config.unit}"},
                "domain": {"row": 0, "column": 1}
            },
            {
                "type": "indicator",
                "mode": "number",
                "value": stats.percentile_95,
                "title": {"text": "95th Percentile"},
                "number": {"suffix": f" {self.config.unit}"},
                "domain": {"row": 0, "column": 2}
            },
            {
                "type": "indicator",
                "mode": "number+delta",
                "value": stats.exceedance_count,
                "title": {"text": "Exceedances"},
                "delta": {"reference": 0, "increasing": {"color": "#E74C3C"}},
                "domain": {"row": 0, "column": 3}
            }
        ]

        layout = {
            "title": {
                "text": f"{self.config.pollutant_name} Statistics",
                "font": {"size": 16}
            },
            "grid": {"rows": 1, "columns": 4, "pattern": "independent"},
            "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
            "paper_bgcolor": "white"
        }

        return {"data": traces, "layout": layout}

    def _build_exceedance_shapes(
        self,
        timestamps: List[str],
        values: List[float]
    ) -> List[Dict[str, Any]]:
        """Build shapes to highlight exceedance periods."""
        shapes = []
        in_exceedance = False
        start_idx = 0

        for i, val in enumerate(values):
            if val > self.config.permit_limit and not in_exceedance:
                in_exceedance = True
                start_idx = i
            elif val <= self.config.permit_limit and in_exceedance:
                in_exceedance = False
                shapes.append({
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": timestamps[start_idx],
                    "y0": 0,
                    "x1": timestamps[i],
                    "y1": 1,
                    "fillcolor": "rgba(231, 76, 60, 0.1)",
                    "line": {"width": 0}
                })

        if in_exceedance and timestamps:
            shapes.append({
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": timestamps[start_idx],
                "y0": 0,
                "x1": timestamps[-1],
                "y1": 1,
                "fillcolor": "rgba(231, 76, 60, 0.1)",
                "line": {"width": 0}
            })

        return shapes

    def to_plotly_json(self) -> str:
        """Export hourly trend to Plotly JSON."""
        return json.dumps(self.build_hourly_trend(), indent=2)

    def get_statistics(self) -> Optional[TrendStatistics]:
        """Get calculated statistics."""
        return self._statistics

    def get_anomalies(self) -> List[Dict[str, Any]]:
        """Get detected anomalies."""
        return self._anomalies

    def get_forecast(self) -> Optional[Dict[str, Any]]:
        """Get forecast data."""
        return self._forecast


class EmissionsTrendDashboard:
    """Comprehensive emissions trend dashboard generator."""

    def __init__(
        self,
        facility_name: str,
        pollutants: List[TrendConfig],
        color_blind_safe: bool = False
    ):
        """
        Initialize trend dashboard.

        Args:
            facility_name: Name of the facility
            pollutants: List of pollutant configurations
            color_blind_safe: Use color-blind safe palette
        """
        self.facility_name = facility_name
        self.pollutants = pollutants
        self.color_blind_safe = color_blind_safe
        self._charts: Dict[str, EmissionsTrendChart] = {}

        for config in pollutants:
            config.color_blind_safe = color_blind_safe
            self._charts[config.pollutant] = EmissionsTrendChart(config)

    def set_pollutant_data(
        self,
        pollutant: str,
        data: List[EmissionDataPoint]
    ) -> None:
        """
        Set data for a specific pollutant.

        Args:
            pollutant: Pollutant identifier
            data: List of emission data points
        """
        if pollutant in self._charts:
            self._charts[pollutant].set_data(data)

    def generate_all_trends(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate trend charts for all pollutants.

        Returns:
            Dictionary of pollutant to chart data
        """
        return {
            pollutant: chart.build_hourly_trend()
            for pollutant, chart in self._charts.items()
        }

    def generate_combined_trend(self) -> Dict[str, Any]:
        """
        Generate combined multi-pollutant trend chart.

        Returns:
            Plotly chart dictionary
        """
        traces = []
        pollutant_colors = {
            "NOx": "#E74C3C",
            "SO2": "#9B59B6",
            "PM": "#3498DB",
            "CO": "#2ECC71",
            "VOC": "#F39C12"
        }

        for pollutant, chart in self._charts.items():
            if not chart._data:
                continue

            config = chart.config
            timestamps = [d.timestamp for d in chart._data]
            values = [d.value / config.permit_limit * 100 for d in chart._data]  # Normalize to %

            traces.append({
                "type": "scatter",
                "mode": "lines",
                "name": config.pollutant_name,
                "x": timestamps,
                "y": values,
                "line": {
                    "color": pollutant_colors.get(pollutant, "#3498DB"),
                    "width": 2
                },
                "hovertemplate": (
                    f"<b>{config.pollutant_name}</b><br>"
                    f"Time: %{{x}}<br>"
                    f"% of Limit: %{{y:.1f}}%<extra></extra>"
                )
            })

        # 100% limit line
        if traces:
            all_timestamps = []
            for chart in self._charts.values():
                all_timestamps.extend([d.timestamp for d in chart._data])

            if all_timestamps:
                traces.append({
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Permit Limit (100%)",
                    "x": [min(all_timestamps), max(all_timestamps)],
                    "y": [100, 100],
                    "line": {"color": "#E74C3C", "width": 2, "dash": "dash"}
                })

        layout = {
            "title": {
                "text": f"{self.facility_name} - Combined Emissions Trend",
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "xaxis": {
                "title": "Time",
                "type": "date",
                "rangeslider": {"visible": True}
            },
            "yaxis": {
                "title": "Percent of Permit Limit (%)",
                "range": [0, 150]
            },
            "legend": {"orientation": "h", "y": 1.1},
            "hovermode": "x unified",
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA"
        }

        return {"data": traces, "layout": layout}

    def to_html(self) -> str:
        """
        Generate standalone HTML trend dashboard.

        Returns:
            HTML string
        """
        all_charts = self.generate_all_trends()
        combined_chart = self.generate_combined_trend()

        charts_json = json.dumps({
            "combined": combined_chart,
            "individual": all_charts
        })

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emissions Trends - {self.facility_name}</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #f5f6fa;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 20px;
        }}
        .chart-container {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Emissions Trends - {self.facility_name}</h1>

        <div class="chart-container" id="combined-chart"></div>

        <h2>Individual Pollutant Trends</h2>
        <div class="charts-grid" id="individual-charts"></div>
    </div>

    <script>
        const chartData = {charts_json};

        // Render combined chart
        Plotly.newPlot('combined-chart',
            chartData.combined.data,
            chartData.combined.layout,
            {{responsive: true}}
        );

        // Render individual charts
        const container = document.getElementById('individual-charts');
        Object.entries(chartData.individual).forEach(([pollutant, chart]) => {{
            const div = document.createElement('div');
            div.className = 'chart-container';
            div.id = 'chart-' + pollutant;
            container.appendChild(div);

            Plotly.newPlot(div.id, chart.data, chart.layout, {{responsive: true}});
        }});
    </script>
</body>
</html>"""

        return html


def create_sample_trend_data(hours: int = 168) -> List[EmissionDataPoint]:
    """
    Create sample emission data for testing.

    Args:
        hours: Number of hours of data to generate

    Returns:
        List of sample emission data points
    """
    import random
    random.seed(42)

    base_value = 150.0
    data = []
    base_time = datetime(2024, 1, 15, 0, 0, 0)

    for i in range(hours):
        # Add some variation and occasional spikes
        variation = random.gauss(0, 15)
        spike = 50 if random.random() < 0.02 else 0  # 2% chance of spike
        value = max(0, base_value + variation + spike + math.sin(i / 24 * 2 * math.pi) * 20)

        data.append(EmissionDataPoint(
            timestamp=(base_time + timedelta(hours=i)).isoformat() + "Z",
            value=round(value, 2),
            unit="lb/hr",
            data_quality=random.uniform(95, 100),
            is_valid=random.random() > 0.01,  # 1% invalid data
            flags=[]
        ))

    return data


if __name__ == "__main__":
    # Demo usage
    config = TrendConfig(
        pollutant="NOx",
        pollutant_name="Nitrogen Oxides",
        unit="lb/hr",
        permit_limit=200.0,
        warning_threshold=180.0,
        resolution=TimeResolution.HOURLY,
        show_rolling_average=True,
        rolling_window=24,
        show_forecast=True,
        forecast_periods=24,
        highlight_anomalies=True
    )

    chart = EmissionsTrendChart(config)
    sample_data = create_sample_trend_data(168)
    chart.set_data(sample_data)

    print("Generated trend chart JSON (first 500 chars):")
    print(chart.to_plotly_json()[:500])

    stats = chart.get_statistics()
    if stats:
        print(f"\nStatistics:")
        print(f"  Mean: {stats.mean_value:.2f}")
        print(f"  Max: {stats.max_value:.2f}")
        print(f"  95th Percentile: {stats.percentile_95:.2f}")
        print(f"  Exceedances: {stats.exceedance_count}")
        print(f"  Trend: {stats.trend_direction.value}")
