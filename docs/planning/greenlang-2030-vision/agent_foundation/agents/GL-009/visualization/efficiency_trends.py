"""Efficiency Trends for GL-009 THERMALIQ Time-Series Analysis.

Generates time-series charts for efficiency trends, benchmarking,
and performance monitoring over time.

Features:
- Multi-metric trend visualization
- Moving averages and smoothing
- Benchmark comparisons
- Anomaly highlighting
- Plotly-compatible output
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import statistics


class TrendType(Enum):
    """Trend chart types."""
    EFFICIENCY = "efficiency"
    LOSSES = "losses"
    OUTPUT = "output"
    INPUT = "input"
    TEMPERATURE = "temperature"


@dataclass
class TrendPoint:
    """Single data point in trend."""
    timestamp: datetime
    value: float
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "label": self.label,
            "metadata": self.metadata
        }


@dataclass
class TrendData:
    """Complete trend data with statistics."""
    points: List[TrendPoint]
    title: str
    trend_type: TrendType
    y_axis_label: str = "Value"
    unit: str = ""
    moving_average_window: Optional[int] = None
    benchmark_value: Optional[float] = None
    benchmark_label: Optional[str] = None
    avg_value: float = 0
    min_value: float = 0
    max_value: float = 0
    std_dev: float = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_statistics(self):
        """Calculate trend statistics."""
        if not self.points:
            return

        values = [p.value for p in self.points]
        self.avg_value = statistics.mean(values)
        self.min_value = min(values)
        self.max_value = max(values)
        self.std_dev = statistics.stdev(values) if len(values) > 1 else 0

    def to_plotly_json(self) -> Dict:
        """Export to Plotly line chart format."""
        # Sort points by timestamp
        sorted_points = sorted(self.points, key=lambda p: p.timestamp)

        # Extract data
        timestamps = [p.timestamp for p in sorted_points]
        values = [p.value for p in sorted_points]
        hover_texts = [
            f"{p.timestamp.strftime('%Y-%m-%d %H:%M')}<br>{p.value:.2f} {self.unit}"
            for p in sorted_points
        ]

        # Main trace
        traces = [{
            "type": "scatter",
            "mode": "lines+markers",
            "x": timestamps,
            "y": values,
            "name": self.title,
            "line": {"color": "#3498DB", "width": 2},
            "marker": {"size": 6, "color": "#3498DB"},
            "hovertemplate": "%{text}<extra></extra>",
            "text": hover_texts
        }]

        # Add moving average if specified
        if self.moving_average_window and len(values) >= self.moving_average_window:
            ma_values = self._calculate_moving_average(
                values, self.moving_average_window
            )
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "x": timestamps,
                "y": ma_values,
                "name": f"{self.moving_average_window}-point MA",
                "line": {"color": "#E74C3C", "width": 2, "dash": "dash"},
                "hovertemplate": "MA: %{y:.2f} " + self.unit + "<extra></extra>"
            })

        # Add benchmark line if specified
        if self.benchmark_value is not None:
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "x": [timestamps[0], timestamps[-1]],
                "y": [self.benchmark_value, self.benchmark_value],
                "name": self.benchmark_label or "Benchmark",
                "line": {"color": "#2ECC71", "width": 2, "dash": "dot"},
                "hovertemplate": "Benchmark: %{y:.2f} " + self.unit + "<extra></extra>"
            })

        # Add average line
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "x": [timestamps[0], timestamps[-1]],
            "y": [self.avg_value, self.avg_value],
            "name": "Average",
            "line": {"color": "#95A5A6", "width": 1, "dash": "dash"},
            "hovertemplate": "Avg: %{y:.2f} " + self.unit + "<extra></extra>"
        })

        # Build layout
        layout = {
            "title": {
                "text": self.title,
                "font": {"size": 16, "color": "#333"}
            },
            "xaxis": {
                "title": "Time",
                "gridcolor": "#E5E5E5",
                "tickformat": "%Y-%m-%d"
            },
            "yaxis": {
                "title": f"{self.y_axis_label} ({self.unit})",
                "gridcolor": "#E5E5E5"
            },
            "margin": {"l": 60, "r": 20, "t": 80, "b": 60},
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "hovermode": "x unified",
            "legend": {
                "x": 0.01,
                "y": 0.99,
                "bgcolor": "rgba(255,255,255,0.8)",
                "bordercolor": "#DDD",
                "borderwidth": 1
            },
            "annotations": [
                {
                    "text": f"Avg: {self.avg_value:.2f} {self.unit} | "
                            f"Min: {self.min_value:.2f} | "
                            f"Max: {self.max_value:.2f} | "
                            f"Std Dev: {self.std_dev:.2f}",
                    "x": 0.5,
                    "y": -0.15,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 10, "color": "#555"}
                }
            ]
        }

        return {
            "data": traces,
            "layout": layout
        }

    def _calculate_moving_average(
        self, values: List[float], window: int
    ) -> List[Optional[float]]:
        """Calculate moving average."""
        ma = []
        for i in range(len(values)):
            if i < window - 1:
                ma.append(None)
            else:
                window_values = values[i - window + 1:i + 1]
                ma.append(statistics.mean(window_values))
        return ma

    def to_dict(self) -> Dict:
        """Export to dictionary."""
        return {
            "points": [p.to_dict() for p in self.points],
            "title": self.title,
            "trend_type": self.trend_type.value,
            "y_axis_label": self.y_axis_label,
            "unit": self.unit,
            "moving_average_window": self.moving_average_window,
            "benchmark_value": self.benchmark_value,
            "benchmark_label": self.benchmark_label,
            "avg_value": self.avg_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "std_dev": self.std_dev,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class EfficiencyTrends:
    """Generate efficiency trend charts for performance monitoring."""

    def __init__(self):
        """Initialize efficiency trends generator."""
        pass

    def generate_efficiency_trend(
        self,
        efficiency_data: List[Tuple[datetime, float]],
        title: str = "Thermal Efficiency Trend",
        benchmark_efficiency: Optional[float] = None,
        moving_average_days: int = 7
    ) -> TrendData:
        """Generate efficiency trend over time.

        Args:
            efficiency_data: List of (timestamp, efficiency_percent) tuples
            title: Chart title
            benchmark_efficiency: Benchmark efficiency for comparison
            moving_average_days: Days for moving average calculation

        Returns:
            Trend data with statistics
        """
        points = [
            TrendPoint(
                timestamp=ts,
                value=eff,
                metadata={"efficiency_percent": eff}
            )
            for ts, eff in efficiency_data
        ]

        trend = TrendData(
            points=points,
            title=title,
            trend_type=TrendType.EFFICIENCY,
            y_axis_label="Efficiency",
            unit="%",
            moving_average_window=moving_average_days,
            benchmark_value=benchmark_efficiency,
            benchmark_label="Target Efficiency"
        )

        trend.calculate_statistics()
        return trend

    def generate_loss_trend(
        self,
        loss_data: List[Tuple[datetime, Dict[str, float]]],
        title: str = "Heat Loss Trends",
        loss_categories: Optional[List[str]] = None
    ) -> Dict[str, TrendData]:
        """Generate trends for multiple loss categories.

        Args:
            loss_data: List of (timestamp, losses_dict) tuples
            title: Base chart title
            loss_categories: Specific loss categories to plot

        Returns:
            Dictionary of loss category to trend data
        """
        # Extract all loss categories if not specified
        if loss_categories is None:
            all_categories = set()
            for _, losses in loss_data:
                all_categories.update(losses.keys())
            loss_categories = sorted(all_categories)

        # Create trend for each category
        trends = {}
        for category in loss_categories:
            points = []
            for ts, losses in loss_data:
                value = losses.get(category, 0)
                points.append(TrendPoint(
                    timestamp=ts,
                    value=value,
                    metadata={"loss_category": category}
                ))

            trend = TrendData(
                points=points,
                title=f"{title}: {category.replace('_', ' ').title()}",
                trend_type=TrendType.LOSSES,
                y_axis_label="Heat Loss",
                unit="kW",
                moving_average_window=7
            )
            trend.calculate_statistics()
            trends[category] = trend

        return trends

    def generate_multi_metric_trend(
        self,
        metrics: Dict[str, List[Tuple[datetime, float]]],
        title: str = "Performance Metrics",
        y_axis_label: str = "Value",
        unit: str = ""
    ) -> Dict:
        """Generate multi-line trend chart for multiple metrics.

        Args:
            metrics: Dictionary of metric_name to [(timestamp, value)] data
            title: Chart title
            y_axis_label: Y-axis label
            unit: Unit of measurement

        Returns:
            Plotly figure dictionary
        """
        traces = []
        colors = [
            "#3498DB", "#E74C3C", "#2ECC71", "#F39C12",
            "#9B59B6", "#1ABC9C", "#E67E22", "#95A5A6"
        ]

        for idx, (metric_name, data) in enumerate(metrics.items()):
            sorted_data = sorted(data, key=lambda x: x[0])
            timestamps = [d[0] for d in sorted_data]
            values = [d[1] for d in sorted_data]

            color = colors[idx % len(colors)]

            traces.append({
                "type": "scatter",
                "mode": "lines+markers",
                "x": timestamps,
                "y": values,
                "name": metric_name,
                "line": {"color": color, "width": 2},
                "marker": {"size": 5, "color": color},
                "hovertemplate": f"{metric_name}: %{{y:.2f}} {unit}<extra></extra>"
            })

        layout = {
            "title": {
                "text": title,
                "font": {"size": 16, "color": "#333"}
            },
            "xaxis": {
                "title": "Time",
                "gridcolor": "#E5E5E5",
                "tickformat": "%Y-%m-%d"
            },
            "yaxis": {
                "title": f"{y_axis_label} ({unit})",
                "gridcolor": "#E5E5E5"
            },
            "margin": {"l": 60, "r": 20, "t": 80, "b": 60},
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "hovermode": "x unified",
            "legend": {
                "x": 0.01,
                "y": 0.99,
                "bgcolor": "rgba(255,255,255,0.8)",
                "bordercolor": "#DDD",
                "borderwidth": 1
            }
        }

        return {
            "data": traces,
            "layout": layout
        }

    def generate_comparison_chart(
        self,
        baseline_data: List[Tuple[datetime, float]],
        current_data: List[Tuple[datetime, float]],
        title: str = "Baseline vs Current Performance",
        y_axis_label: str = "Efficiency",
        unit: str = "%"
    ) -> Dict:
        """Generate comparison chart for baseline vs current performance.

        Args:
            baseline_data: Baseline performance data
            current_data: Current performance data
            title: Chart title
            y_axis_label: Y-axis label
            unit: Unit of measurement

        Returns:
            Plotly figure dictionary
        """
        # Sort data
        baseline_sorted = sorted(baseline_data, key=lambda x: x[0])
        current_sorted = sorted(current_data, key=lambda x: x[0])

        # Extract timestamps and values
        baseline_ts = [d[0] for d in baseline_sorted]
        baseline_vals = [d[1] for d in baseline_sorted]

        current_ts = [d[0] for d in current_sorted]
        current_vals = [d[1] for d in current_sorted]

        traces = [
            {
                "type": "scatter",
                "mode": "lines",
                "x": baseline_ts,
                "y": baseline_vals,
                "name": "Baseline",
                "line": {"color": "#95A5A6", "width": 2, "dash": "dash"},
                "fill": "tonexty",
                "fillcolor": "rgba(149,165,166,0.1)",
                "hovertemplate": f"Baseline: %{{y:.2f}} {unit}<extra></extra>"
            },
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": current_ts,
                "y": current_vals,
                "name": "Current",
                "line": {"color": "#3498DB", "width": 2},
                "marker": {"size": 5, "color": "#3498DB"},
                "hovertemplate": f"Current: %{{y:.2f}} {unit}<extra></extra>"
            }
        ]

        # Calculate improvement
        if baseline_vals and current_vals:
            baseline_avg = statistics.mean(baseline_vals)
            current_avg = statistics.mean(current_vals)
            improvement = current_avg - baseline_avg
            improvement_pct = (improvement / baseline_avg * 100) if baseline_avg > 0 else 0

            annotation_text = (
                f"Average Baseline: {baseline_avg:.2f} {unit} | "
                f"Average Current: {current_avg:.2f} {unit} | "
                f"Improvement: {improvement:+.2f} {unit} ({improvement_pct:+.1f}%)"
            )
        else:
            annotation_text = ""

        layout = {
            "title": {
                "text": title,
                "font": {"size": 16, "color": "#333"}
            },
            "xaxis": {
                "title": "Time",
                "gridcolor": "#E5E5E5",
                "tickformat": "%Y-%m-%d"
            },
            "yaxis": {
                "title": f"{y_axis_label} ({unit})",
                "gridcolor": "#E5E5E5"
            },
            "margin": {"l": 60, "r": 20, "t": 80, "b": 80},
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "hovermode": "x unified",
            "legend": {
                "x": 0.01,
                "y": 0.99,
                "bgcolor": "rgba(255,255,255,0.8)",
                "bordercolor": "#DDD",
                "borderwidth": 1
            },
            "annotations": [
                {
                    "text": annotation_text,
                    "x": 0.5,
                    "y": -0.18,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 10, "color": "#555"}
                }
            ]
        }

        return {
            "data": traces,
            "layout": layout
        }
