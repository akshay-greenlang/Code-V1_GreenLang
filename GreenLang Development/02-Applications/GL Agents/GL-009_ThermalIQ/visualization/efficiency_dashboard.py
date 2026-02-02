"""
ThermalIQ Efficiency Dashboard

Generates interactive efficiency dashboards with gauges, trend charts,
waterfall diagrams, and comparison visualizations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from datetime import datetime
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import numpy as np


class GaugeStyle(Enum):
    """Styles for efficiency gauges."""
    STANDARD = "standard"
    BULLET = "bullet"
    ANGULAR = "angular"
    SEMICIRCLE = "semicircle"


class TrendPeriod(Enum):
    """Time periods for trend analysis."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass
class GaugeConfig:
    """Configuration for efficiency gauge."""
    min_value: float = 0.0
    max_value: float = 100.0
    target_value: Optional[float] = None
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "poor": 70,
        "average": 80,
        "good": 90
    })
    colors: Dict[str, str] = field(default_factory=lambda: {
        "poor": "#e74c3c",
        "average": "#f39c12",
        "good": "#27ae60",
        "excellent": "#2ecc71"
    })
    unit: str = "%"
    show_target: bool = True
    title: str = "Efficiency"


@dataclass
class WaterfallConfig:
    """Configuration for waterfall chart."""
    show_connectors: bool = True
    positive_color: str = "#27ae60"
    negative_color: str = "#e74c3c"
    total_color: str = "#3498db"
    connector_color: str = "rgba(0,0,0,0.3)"


@dataclass
class TrendData:
    """Container for trend data."""
    timestamps: List[datetime]
    values: List[float]
    labels: Optional[List[str]] = None
    target: Optional[float] = None
    unit: str = "%"


@dataclass
class LossItem:
    """A loss item for waterfall chart."""
    name: str
    value: float
    is_intermediate: bool = False
    color: Optional[str] = None


class EfficiencyDashboard:
    """
    Generates interactive efficiency dashboards for thermal systems.

    Provides visualizations for:
    - Efficiency gauges with targets and thresholds
    - Trend charts showing historical performance
    - Waterfall charts for loss breakdown
    - Equipment comparison bar charts
    """

    # Default color schemes
    EFFICIENCY_COLORS = {
        "excellent": "#27ae60",  # Green
        "good": "#2ecc71",       # Light green
        "average": "#f39c12",    # Orange
        "poor": "#e74c3c",       # Red
        "critical": "#c0392b"    # Dark red
    }

    BRAND_COLORS = {
        "primary": "#2e7d32",    # GreenLang green
        "secondary": "#4caf50",
        "accent": "#81c784",
        "warning": "#ff9800",
        "danger": "#f44336"
    }

    def __init__(
        self,
        default_gauge_config: Optional[GaugeConfig] = None,
        default_waterfall_config: Optional[WaterfallConfig] = None,
        theme: str = "plotly_white"
    ):
        """
        Initialize the efficiency dashboard.

        Args:
            default_gauge_config: Default configuration for gauges
            default_waterfall_config: Default configuration for waterfall charts
            theme: Plotly theme to use
        """
        self.default_gauge_config = default_gauge_config or GaugeConfig()
        self.default_waterfall_config = default_waterfall_config or WaterfallConfig()
        self.theme = theme

    def create_efficiency_gauge(
        self,
        current: float,
        target: Optional[float] = None,
        config: Optional[GaugeConfig] = None,
        style: GaugeStyle = GaugeStyle.ANGULAR
    ) -> go.Figure:
        """
        Create an efficiency gauge visualization.

        Args:
            current: Current efficiency value
            target: Target efficiency value
            config: Gauge configuration
            style: Gauge style

        Returns:
            Plotly figure with efficiency gauge
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        config = config or self.default_gauge_config
        target = target if target is not None else config.target_value

        # Determine current status color
        if current >= config.thresholds.get("good", 90):
            bar_color = config.colors.get("excellent", "#2ecc71")
            status = "Excellent"
        elif current >= config.thresholds.get("average", 80):
            bar_color = config.colors.get("good", "#27ae60")
            status = "Good"
        elif current >= config.thresholds.get("poor", 70):
            bar_color = config.colors.get("average", "#f39c12")
            status = "Average"
        else:
            bar_color = config.colors.get("poor", "#e74c3c")
            status = "Poor"

        # Create gauge steps
        steps = [
            {"range": [config.min_value, config.thresholds.get("poor", 70)],
             "color": "rgba(231, 76, 60, 0.3)"},
            {"range": [config.thresholds.get("poor", 70), config.thresholds.get("average", 80)],
             "color": "rgba(243, 156, 18, 0.3)"},
            {"range": [config.thresholds.get("average", 80), config.thresholds.get("good", 90)],
             "color": "rgba(39, 174, 96, 0.3)"},
            {"range": [config.thresholds.get("good", 90), config.max_value],
             "color": "rgba(46, 204, 113, 0.3)"}
        ]

        # Create threshold line for target
        threshold = None
        if target is not None and config.show_target:
            threshold = {
                "line": {"color": "#2c3e50", "width": 4},
                "thickness": 0.75,
                "value": target
            }

        if style == GaugeStyle.ANGULAR:
            gauge_mode = "gauge+number+delta" if target else "gauge+number"
            delta_ref = target if target else None
        else:
            gauge_mode = "gauge+number"
            delta_ref = None

        fig = go.Figure(go.Indicator(
            mode=gauge_mode,
            value=current,
            number={
                "suffix": config.unit,
                "font": {"size": 40, "color": bar_color}
            },
            delta={
                "reference": delta_ref,
                "increasing": {"color": "#27ae60"},
                "decreasing": {"color": "#e74c3c"}
            } if delta_ref else None,
            title={
                "text": f"{config.title}<br><span style='font-size:0.8em;color:gray'>Status: {status}</span>",
                "font": {"size": 18}
            },
            gauge={
                "axis": {
                    "range": [config.min_value, config.max_value],
                    "tickwidth": 1,
                    "tickcolor": "#2c3e50"
                },
                "bar": {"color": bar_color, "thickness": 0.8},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#2c3e50",
                "steps": steps,
                "threshold": threshold
            }
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="white",
            font=dict(family="Arial")
        )

        return fig

    def create_trend_chart(
        self,
        history: Union[TrendData, List[Dict[str, Any]]],
        show_target: bool = True,
        show_average: bool = True,
        show_range: bool = False,
        title: str = "Efficiency Trend"
    ) -> go.Figure:
        """
        Create an efficiency trend chart.

        Args:
            history: Historical efficiency data
            show_target: Whether to show target line
            show_average: Whether to show moving average
            show_range: Whether to show min/max range
            title: Chart title

        Returns:
            Plotly figure with trend chart
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        # Convert to TrendData if needed
        if isinstance(history, list):
            timestamps = [h.get('timestamp', datetime.now()) for h in history]
            values = [h.get('value', h.get('efficiency', 0)) for h in history]
            target = history[0].get('target') if history else None
            history = TrendData(
                timestamps=timestamps,
                values=values,
                target=target
            )

        values = np.array(history.values)

        fig = go.Figure()

        # Main trend line
        fig.add_trace(go.Scatter(
            x=history.timestamps,
            y=values,
            mode='lines+markers',
            name='Efficiency',
            line=dict(color=self.BRAND_COLORS["primary"], width=2),
            marker=dict(size=6),
            hovertemplate=(
                "Date: %{x}<br>" +
                f"Efficiency: %{{y:.1f}}{history.unit}<br>" +
                "<extra></extra>"
            )
        ))

        # Moving average
        if show_average and len(values) >= 5:
            window = min(7, len(values) // 2)
            moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
            avg_timestamps = history.timestamps[window-1:]

            fig.add_trace(go.Scatter(
                x=avg_timestamps,
                y=moving_avg,
                mode='lines',
                name=f'{window}-Point Average',
                line=dict(color=self.BRAND_COLORS["secondary"], width=2, dash='dash'),
                hovertemplate=(
                    "Date: %{x}<br>" +
                    f"Average: %{{y:.1f}}{history.unit}<br>" +
                    "<extra></extra>"
                )
            ))

        # Target line
        if show_target and history.target is not None:
            fig.add_hline(
                y=history.target,
                line_dash="dot",
                line_color=self.BRAND_COLORS["accent"],
                annotation_text=f"Target: {history.target}{history.unit}",
                annotation_position="top right"
            )

        # Min/Max range
        if show_range and len(values) >= 5:
            window = min(7, len(values) // 2)

            # Calculate rolling min/max
            rolling_min = np.array([
                values[max(0, i-window+1):i+1].min()
                for i in range(len(values))
            ])
            rolling_max = np.array([
                values[max(0, i-window+1):i+1].max()
                for i in range(len(values))
            ])

            fig.add_trace(go.Scatter(
                x=history.timestamps,
                y=rolling_max,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            fig.add_trace(go.Scatter(
                x=history.timestamps,
                y=rolling_min,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(46, 125, 50, 0.1)',
                name='Range',
                hoverinfo='skip'
            ))

        # Add threshold zones
        fig.add_hrect(
            y0=0, y1=70,
            fillcolor="rgba(231, 76, 60, 0.1)",
            layer="below", line_width=0
        )
        fig.add_hrect(
            y0=70, y1=80,
            fillcolor="rgba(243, 156, 18, 0.1)",
            layer="below", line_width=0
        )
        fig.add_hrect(
            y0=80, y1=90,
            fillcolor="rgba(39, 174, 96, 0.1)",
            layer="below", line_width=0
        )
        fig.add_hrect(
            y0=90, y1=100,
            fillcolor="rgba(46, 204, 113, 0.1)",
            layer="below", line_width=0
        )

        # Calculate statistics
        avg_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)

        fig.update_layout(
            title=dict(
                text=f"{title}<br><sub>Avg: {avg_val:.1f}% | Min: {min_val:.1f}% | Max: {max_val:.1f}%</sub>",
                font=dict(size=16)
            ),
            xaxis_title="Date/Time",
            yaxis_title=f"Efficiency ({history.unit})",
            template=self.theme,
            height=400,
            width=800,
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99
            ),
            hovermode="x unified"
        )

        return fig

    def create_loss_waterfall(
        self,
        losses: Union[Dict[str, float], List[LossItem]],
        input_value: Optional[float] = None,
        title: str = "Heat Loss Breakdown",
        config: Optional[WaterfallConfig] = None
    ) -> go.Figure:
        """
        Create a waterfall chart showing loss breakdown.

        Args:
            losses: Dictionary of {loss_name: value} or list of LossItem
            input_value: Starting value (total heat input)
            title: Chart title
            config: Waterfall configuration

        Returns:
            Plotly figure with waterfall chart
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        config = config or self.default_waterfall_config

        # Convert dict to list if needed
        if isinstance(losses, dict):
            loss_items = [
                LossItem(name=name, value=value)
                for name, value in losses.items()
            ]
        else:
            loss_items = losses

        # Calculate values
        if input_value is None:
            input_value = sum(item.value for item in loss_items)

        # Build waterfall data
        names = ["Total Input"]
        values = [input_value]
        measures = ["absolute"]
        colors = [config.total_color]
        text_values = [f"{input_value:.1f} kW"]

        remaining = input_value

        for item in loss_items:
            names.append(item.name)
            values.append(-item.value)  # Negative for losses
            measures.append("relative")

            if item.color:
                colors.append(item.color)
            else:
                colors.append(config.negative_color)

            text_values.append(f"-{item.value:.1f} kW")
            remaining -= item.value

        # Add useful output (remaining)
        names.append("Useful Output")
        values.append(remaining)
        measures.append("total")
        colors.append(config.positive_color)
        text_values.append(f"{remaining:.1f} kW")

        fig = go.Figure(go.Waterfall(
            name="Energy Balance",
            orientation="v",
            measure=measures,
            x=names,
            y=values,
            textposition="outside",
            text=text_values,
            connector={
                "line": {"color": config.connector_color, "width": 2}
            } if config.show_connectors else {"line": {"width": 0}},
            decreasing={"marker": {"color": config.negative_color}},
            increasing={"marker": {"color": config.positive_color}},
            totals={"marker": {"color": config.total_color}},
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Value: %{text}<br>" +
                "<extra></extra>"
            )
        ))

        # Calculate efficiency
        efficiency = (remaining / input_value * 100) if input_value > 0 else 0

        fig.update_layout(
            title=dict(
                text=f"{title}<br><sub>Thermal Efficiency: {efficiency:.1f}%</sub>",
                font=dict(size=16)
            ),
            yaxis_title="Energy (kW)",
            template=self.theme,
            height=500,
            width=900,
            showlegend=False
        )

        # Rotate x-axis labels if many items
        if len(names) > 5:
            fig.update_xaxes(tickangle=45)

        return fig

    def create_comparison_bar(
        self,
        equipment_efficiencies: Dict[str, float],
        targets: Optional[Dict[str, float]] = None,
        title: str = "Equipment Efficiency Comparison",
        sort_by_value: bool = True
    ) -> go.Figure:
        """
        Create a bar chart comparing equipment efficiencies.

        Args:
            equipment_efficiencies: Dict of {equipment_name: efficiency}
            targets: Optional dict of target values
            title: Chart title
            sort_by_value: Whether to sort by efficiency value

        Returns:
            Plotly figure with comparison bar chart
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        # Sort equipment if requested
        if sort_by_value:
            equipment = sorted(
                equipment_efficiencies.items(),
                key=lambda x: x[1],
                reverse=True
            )
        else:
            equipment = list(equipment_efficiencies.items())

        names = [e[0] for e in equipment]
        values = [e[1] for e in equipment]

        # Assign colors based on efficiency
        colors = []
        for val in values:
            if val >= 90:
                colors.append(self.EFFICIENCY_COLORS["excellent"])
            elif val >= 80:
                colors.append(self.EFFICIENCY_COLORS["good"])
            elif val >= 70:
                colors.append(self.EFFICIENCY_COLORS["average"])
            else:
                colors.append(self.EFFICIENCY_COLORS["poor"])

        fig = go.Figure()

        # Main bars
        fig.add_trace(go.Bar(
            x=names,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition='outside',
            name='Actual',
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Efficiency: %{y:.1f}%<br>" +
                "<extra></extra>"
            )
        ))

        # Target markers
        if targets:
            target_values = [targets.get(name, None) for name in names]
            valid_targets = [(i, t) for i, t in enumerate(target_values) if t is not None]

            if valid_targets:
                fig.add_trace(go.Scatter(
                    x=[names[i] for i, _ in valid_targets],
                    y=[t for _, t in valid_targets],
                    mode='markers',
                    marker=dict(
                        symbol='line-ew',
                        size=20,
                        line=dict(width=3, color='#2c3e50')
                    ),
                    name='Target',
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        "Target: %{y:.1f}%<br>" +
                        "<extra></extra>"
                    )
                ))

        # Add average line
        avg_efficiency = np.mean(values)
        fig.add_hline(
            y=avg_efficiency,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Average: {avg_efficiency:.1f}%",
            annotation_position="top right"
        )

        # Add threshold lines
        fig.add_hline(y=90, line_dash="dot", line_color="rgba(39, 174, 96, 0.5)")
        fig.add_hline(y=80, line_dash="dot", line_color="rgba(243, 156, 18, 0.5)")

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title="Equipment",
            yaxis_title="Efficiency (%)",
            yaxis_range=[0, 105],
            template=self.theme,
            height=450,
            width=max(600, len(names) * 80),
            showlegend=bool(targets)
        )

        # Rotate labels if many items
        if len(names) > 6:
            fig.update_xaxes(tickangle=45)

        return fig

    def create_dashboard(
        self,
        current_efficiency: float,
        target_efficiency: float,
        history: Union[TrendData, List[Dict[str, Any]]],
        losses: Dict[str, float],
        equipment_comparison: Optional[Dict[str, float]] = None,
        title: str = "Thermal Efficiency Dashboard"
    ) -> go.Figure:
        """
        Create a comprehensive efficiency dashboard.

        Args:
            current_efficiency: Current efficiency value
            target_efficiency: Target efficiency value
            history: Historical efficiency data
            losses: Loss breakdown
            equipment_comparison: Optional equipment comparison data
            title: Dashboard title

        Returns:
            Plotly figure with full dashboard
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        # Determine layout based on whether we have equipment comparison
        if equipment_comparison:
            n_rows, n_cols = 2, 2
            specs = [
                [{"type": "indicator"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ]
            subplot_titles = (
                "Current Efficiency", "Efficiency Trend",
                "Loss Breakdown", "Equipment Comparison"
            )
        else:
            n_rows, n_cols = 2, 2
            specs = [
                [{"type": "indicator"}, {"type": "xy"}],
                [{"type": "xy", "colspan": 2}, None]
            ]
            subplot_titles = (
                "Current Efficiency", "Efficiency Trend",
                "Loss Breakdown", ""
            )

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            specs=specs,
            subplot_titles=subplot_titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # 1. Efficiency Gauge (row 1, col 1)
        gauge = self.create_efficiency_gauge(current_efficiency, target_efficiency)
        for trace in gauge.data:
            fig.add_trace(trace, row=1, col=1)

        # 2. Trend Chart (row 1, col 2)
        trend = self.create_trend_chart(history, title="")
        for trace in trend.data:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=2)

        # 3. Waterfall (row 2, col 1)
        waterfall = self.create_loss_waterfall(losses, title="")
        for trace in waterfall.data:
            fig.add_trace(trace, row=2, col=1)

        # 4. Equipment Comparison (row 2, col 2) if provided
        if equipment_comparison:
            comparison = self.create_comparison_bar(equipment_comparison, title="")
            for trace in comparison.data:
                trace.showlegend = False
                fig.add_trace(trace, row=2, col=2)

        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            height=800,
            width=1200,
            template=self.theme,
            showlegend=False,
            margin=dict(l=60, r=40, t=100, b=60)
        )

        return fig

    def create_kpi_cards(
        self,
        kpis: Dict[str, Dict[str, Any]],
        n_cols: int = 4
    ) -> go.Figure:
        """
        Create KPI card visualization.

        Args:
            kpis: Dict of {kpi_name: {value, unit, change, target}}
            n_cols: Number of columns for layout

        Returns:
            Plotly figure with KPI cards
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        n_kpis = len(kpis)
        n_rows = (n_kpis + n_cols - 1) // n_cols

        # Create specs for indicator subplots
        specs = [[{"type": "indicator"} for _ in range(n_cols)] for _ in range(n_rows)]

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            specs=specs,
            vertical_spacing=0.2,
            horizontal_spacing=0.1
        )

        for idx, (name, data) in enumerate(kpis.items()):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            value = data.get('value', 0)
            unit = data.get('unit', '')
            change = data.get('change')
            target = data.get('target')

            # Determine mode
            if change is not None:
                mode = "number+delta"
                delta = {
                    "reference": value - change,
                    "relative": True,
                    "position": "bottom"
                }
            else:
                mode = "number"
                delta = None

            fig.add_trace(
                go.Indicator(
                    mode=mode,
                    value=value,
                    number={"suffix": f" {unit}", "font": {"size": 32}},
                    delta=delta,
                    title={"text": name, "font": {"size": 14}},
                    domain={"row": row - 1, "column": col - 1}
                ),
                row=row,
                col=col
            )

        fig.update_layout(
            height=150 * n_rows,
            width=250 * n_cols,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white"
        )

        return fig

    def create_heatmap_comparison(
        self,
        data: Dict[str, Dict[str, float]],
        title: str = "Equipment Performance Heatmap",
        value_label: str = "Efficiency (%)"
    ) -> go.Figure:
        """
        Create a heatmap for multi-dimensional comparison.

        Args:
            data: Dict of {row_name: {col_name: value}}
            title: Chart title
            value_label: Label for values

        Returns:
            Plotly figure with heatmap
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        # Extract row and column names
        row_names = list(data.keys())
        col_names = list(set(
            col for row_data in data.values()
            for col in row_data.keys()
        ))

        # Build value matrix
        z_values = []
        annotations = []

        for i, row_name in enumerate(row_names):
            row_values = []
            for j, col_name in enumerate(col_names):
                val = data[row_name].get(col_name, 0)
                row_values.append(val)

                annotations.append(dict(
                    x=j,
                    y=i,
                    text=f"{val:.1f}",
                    showarrow=False,
                    font=dict(color="white" if val > 50 else "black")
                ))

            z_values.append(row_values)

        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=col_names,
            y=row_names,
            colorscale=[
                [0, self.EFFICIENCY_COLORS["poor"]],
                [0.5, self.EFFICIENCY_COLORS["average"]],
                [0.75, self.EFFICIENCY_COLORS["good"]],
                [1, self.EFFICIENCY_COLORS["excellent"]]
            ],
            colorbar=dict(title=value_label),
            hovertemplate=(
                "%{y} - %{x}<br>" +
                f"{value_label}: %{{z:.1f}}<br>" +
                "<extra></extra>"
            )
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            annotations=annotations,
            height=max(300, len(row_names) * 40),
            width=max(400, len(col_names) * 80),
            xaxis_title="Parameter",
            yaxis_title="Equipment"
        )

        return fig

    def export_dashboard(
        self,
        fig: go.Figure,
        path: Union[str, Path],
        format: str = "html",
        width: int = 1200,
        height: int = 800
    ) -> None:
        """
        Export dashboard to file.

        Args:
            fig: Plotly figure to export
            path: Output file path
            format: "html", "png", "svg", or "pdf"
            width: Image width (for image formats)
            height: Image height (for image formats)
        """
        path = Path(path)

        if format == "html":
            fig.write_html(
                str(path),
                include_plotlyjs=True,
                full_html=True
            )
        else:
            fig.write_image(
                str(path),
                format=format,
                width=width,
                height=height,
                scale=2.0
            )
