"""Loss Breakdown Charts for GL-009 THERMALIQ.

Generates pie charts, bar charts, and donut charts for visualizing
heat loss distribution and categories.

Features:
- Pie charts for loss distribution
- Bar charts for loss comparison
- Donut charts with center text
- Color-coded by loss type
- Percentage and absolute value labels
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime


class ChartType(Enum):
    """Chart types for loss breakdown."""
    PIE = "pie"
    DONUT = "donut"
    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"


@dataclass
class LossCategory:
    """Single loss category."""
    name: str
    value: float
    percentage: float
    color: str
    description: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "percentage": self.percentage,
            "color": self.color,
            "description": self.description
        }


@dataclass
class BreakdownChart:
    """Loss breakdown chart data."""
    categories: List[LossCategory]
    chart_type: ChartType
    title: str
    subtitle: Optional[str] = None
    total_value: float = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_plotly_json(self) -> Dict:
        """Export to Plotly format."""
        if self.chart_type == ChartType.PIE:
            return self._create_pie_chart()
        elif self.chart_type == ChartType.DONUT:
            return self._create_donut_chart()
        elif self.chart_type == ChartType.BAR:
            return self._create_bar_chart()
        elif self.chart_type == ChartType.HORIZONTAL_BAR:
            return self._create_horizontal_bar_chart()
        else:
            return self._create_pie_chart()

    def _create_pie_chart(self) -> Dict:
        """Create pie chart."""
        labels = [c.name for c in self.categories]
        values = [c.value for c in self.categories]
        colors = [c.color for c in self.categories]
        text_labels = [
            f"{c.name}<br>{c.value:.0f} kW ({c.percentage:.1f}%)"
            for c in self.categories
        ]

        return {
            "data": [{
                "type": "pie",
                "labels": labels,
                "values": values,
                "marker": {"colors": colors},
                "text": text_labels,
                "textinfo": "label+percent",
                "textposition": "outside",
                "hovertemplate": "%{text}<extra></extra>",
                "hole": 0
            }],
            "layout": {
                "title": {
                    "text": self.title,
                    "font": {"size": 16, "color": "#333"}
                },
                "margin": {"l": 20, "r": 20, "t": 80, "b": 60},
                "paper_bgcolor": "white",
                "showlegend": True,
                "legend": {
                    "x": 1.05,
                    "y": 0.5,
                    "bgcolor": "rgba(255,255,255,0.8)",
                    "bordercolor": "#DDD",
                    "borderwidth": 1
                },
                "annotations": [
                    {
                        "text": self.subtitle or f"Total Losses: {self.total_value:.0f} kW",
                        "x": 0.5,
                        "y": -0.15,
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 11, "color": "#555"}
                    }
                ]
            }
        }

    def _create_donut_chart(self) -> Dict:
        """Create donut chart with center text."""
        labels = [c.name for c in self.categories]
        values = [c.value for c in self.categories]
        colors = [c.color for c in self.categories]
        text_labels = [
            f"{c.name}<br>{c.value:.0f} kW ({c.percentage:.1f}%)"
            for c in self.categories
        ]

        return {
            "data": [{
                "type": "pie",
                "labels": labels,
                "values": values,
                "marker": {"colors": colors},
                "text": text_labels,
                "textinfo": "label+percent",
                "textposition": "outside",
                "hovertemplate": "%{text}<extra></extra>",
                "hole": 0.5
            }],
            "layout": {
                "title": {
                    "text": self.title,
                    "font": {"size": 16, "color": "#333"}
                },
                "margin": {"l": 20, "r": 20, "t": 80, "b": 60},
                "paper_bgcolor": "white",
                "showlegend": True,
                "legend": {
                    "x": 1.05,
                    "y": 0.5,
                    "bgcolor": "rgba(255,255,255,0.8)",
                    "bordercolor": "#DDD",
                    "borderwidth": 1
                },
                "annotations": [
                    {
                        "text": f"<b>Total</b><br>{self.total_value:.0f} kW",
                        "x": 0.5,
                        "y": 0.5,
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 14, "color": "#333"},
                        "align": "center"
                    },
                    {
                        "text": self.subtitle or "",
                        "x": 0.5,
                        "y": -0.15,
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 11, "color": "#555"}
                    }
                ]
            }
        }

    def _create_bar_chart(self) -> Dict:
        """Create vertical bar chart."""
        labels = [c.name for c in self.categories]
        values = [c.value for c in self.categories]
        colors = [c.color for c in self.categories]
        text_labels = [f"{c.value:.0f} kW ({c.percentage:.1f}%)" for c in self.categories]

        return {
            "data": [{
                "type": "bar",
                "x": labels,
                "y": values,
                "marker": {"color": colors},
                "text": text_labels,
                "textposition": "outside",
                "hovertemplate": "%{x}<br>%{text}<extra></extra>"
            }],
            "layout": {
                "title": {
                    "text": self.title,
                    "font": {"size": 16, "color": "#333"}
                },
                "xaxis": {
                    "title": "Loss Category",
                    "tickangle": -45
                },
                "yaxis": {
                    "title": "Heat Loss (kW)",
                    "gridcolor": "#E5E5E5"
                },
                "margin": {"l": 60, "r": 20, "t": 80, "b": 120},
                "paper_bgcolor": "white",
                "plot_bgcolor": "white",
                "annotations": [
                    {
                        "text": self.subtitle or f"Total Losses: {self.total_value:.0f} kW",
                        "x": 0.5,
                        "y": -0.25,
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 11, "color": "#555"}
                    }
                ]
            }
        }

    def _create_horizontal_bar_chart(self) -> Dict:
        """Create horizontal bar chart."""
        # Sort categories by value (descending)
        sorted_categories = sorted(self.categories, key=lambda c: c.value)

        labels = [c.name for c in sorted_categories]
        values = [c.value for c in sorted_categories]
        colors = [c.color for c in sorted_categories]
        text_labels = [
            f"{c.value:.0f} kW ({c.percentage:.1f}%)"
            for c in sorted_categories
        ]

        return {
            "data": [{
                "type": "bar",
                "orientation": "h",
                "y": labels,
                "x": values,
                "marker": {"color": colors},
                "text": text_labels,
                "textposition": "outside",
                "hovertemplate": "%{y}<br>%{text}<extra></extra>"
            }],
            "layout": {
                "title": {
                    "text": self.title,
                    "font": {"size": 16, "color": "#333"}
                },
                "xaxis": {
                    "title": "Heat Loss (kW)",
                    "gridcolor": "#E5E5E5"
                },
                "yaxis": {
                    "title": "Loss Category"
                },
                "margin": {"l": 150, "r": 120, "t": 80, "b": 60},
                "paper_bgcolor": "white",
                "plot_bgcolor": "white",
                "annotations": [
                    {
                        "text": self.subtitle or f"Total Losses: {self.total_value:.0f} kW",
                        "x": 0.5,
                        "y": -0.15,
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 11, "color": "#555"}
                    }
                ]
            }
        }

    def to_dict(self) -> Dict:
        """Export to dictionary."""
        return {
            "categories": [c.to_dict() for c in self.categories],
            "chart_type": self.chart_type.value,
            "title": self.title,
            "subtitle": self.subtitle,
            "total_value": self.total_value,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class LossBreakdown:
    """Generate loss breakdown visualizations."""

    # Color palette for loss types
    LOSS_COLORS = {
        "radiation": "#E74C3C",
        "convection": "#E67E22",
        "flue_gas": "#95A5A6",
        "stack": "#C0392B",
        "conduction": "#9B59B6",
        "unburned": "#34495E",
        "blowdown": "#1ABC9C",
        "moisture": "#16A085",
        "ambient": "#D35400",
        "leakage": "#F39C12",
        "other": "#7F8C8D"
    }

    def __init__(self):
        """Initialize loss breakdown generator."""
        pass

    def generate_pie_chart(
        self,
        losses: Dict[str, float],
        title: str = "Heat Loss Breakdown",
        total_input: Optional[float] = None
    ) -> BreakdownChart:
        """Generate pie chart of heat losses.

        Args:
            losses: Dictionary of loss name to value (kW)
            title: Chart title
            total_input: Total input energy for percentage calculation

        Returns:
            Pie chart breakdown
        """
        total_losses = sum(losses.values())

        categories = []
        for name, value in sorted(losses.items(), key=lambda x: -x[1]):
            if value > 0:
                percentage = (value / total_losses * 100) if total_losses > 0 else 0
                categories.append(LossCategory(
                    name=name.replace('_', ' ').title(),
                    value=value,
                    percentage=percentage,
                    color=self._get_color(name),
                    description=f"{name} heat loss"
                ))

        subtitle = None
        if total_input:
            loss_percentage = (total_losses / total_input * 100) if total_input > 0 else 0
            subtitle = (
                f"Total Losses: {total_losses:.0f} kW "
                f"({loss_percentage:.1f}% of input)"
            )

        return BreakdownChart(
            categories=categories,
            chart_type=ChartType.PIE,
            title=title,
            subtitle=subtitle,
            total_value=total_losses,
            metadata={"losses": losses, "total_input": total_input}
        )

    def generate_donut_chart(
        self,
        losses: Dict[str, float],
        title: str = "Heat Loss Distribution",
        total_input: Optional[float] = None
    ) -> BreakdownChart:
        """Generate donut chart of heat losses.

        Args:
            losses: Dictionary of loss name to value (kW)
            title: Chart title
            total_input: Total input energy for percentage calculation

        Returns:
            Donut chart breakdown
        """
        chart = self.generate_pie_chart(losses, title, total_input)
        chart.chart_type = ChartType.DONUT
        return chart

    def generate_bar_chart(
        self,
        losses: Dict[str, float],
        title: str = "Heat Loss Comparison",
        horizontal: bool = False
    ) -> BreakdownChart:
        """Generate bar chart of heat losses.

        Args:
            losses: Dictionary of loss name to value (kW)
            title: Chart title
            horizontal: Use horizontal orientation

        Returns:
            Bar chart breakdown
        """
        total_losses = sum(losses.values())

        categories = []
        for name, value in losses.items():
            if value > 0:
                percentage = (value / total_losses * 100) if total_losses > 0 else 0
                categories.append(LossCategory(
                    name=name.replace('_', ' ').title(),
                    value=value,
                    percentage=percentage,
                    color=self._get_color(name)
                ))

        chart_type = ChartType.HORIZONTAL_BAR if horizontal else ChartType.BAR

        return BreakdownChart(
            categories=categories,
            chart_type=chart_type,
            title=title,
            total_value=total_losses,
            metadata={"losses": losses}
        )

    def generate_comparison_chart(
        self,
        baseline_losses: Dict[str, float],
        current_losses: Dict[str, float],
        title: str = "Loss Comparison: Baseline vs Current"
    ) -> Dict:
        """Generate comparison chart for baseline vs current losses.

        Args:
            baseline_losses: Baseline loss values
            current_losses: Current loss values
            title: Chart title

        Returns:
            Plotly figure dictionary
        """
        # Combine all loss categories
        all_categories = set(baseline_losses.keys()) | set(current_losses.keys())

        categories = sorted(all_categories)
        baseline_values = [baseline_losses.get(cat, 0) for cat in categories]
        current_values = [current_losses.get(cat, 0) for cat in categories]

        # Calculate improvements
        improvements = [
            baseline - current
            for baseline, current in zip(baseline_values, current_values)
        ]

        # Format labels
        labels = [cat.replace('_', ' ').title() for cat in categories]

        traces = [
            {
                "type": "bar",
                "x": labels,
                "y": baseline_values,
                "name": "Baseline",
                "marker": {"color": "#95A5A6"},
                "hovertemplate": "Baseline: %{y:.0f} kW<extra></extra>"
            },
            {
                "type": "bar",
                "x": labels,
                "y": current_values,
                "name": "Current",
                "marker": {"color": "#3498DB"},
                "hovertemplate": "Current: %{y:.0f} kW<extra></extra>"
            }
        ]

        # Calculate totals
        baseline_total = sum(baseline_values)
        current_total = sum(current_values)
        total_improvement = baseline_total - current_total
        improvement_pct = (
            (total_improvement / baseline_total * 100)
            if baseline_total > 0 else 0
        )

        layout = {
            "title": {
                "text": title,
                "font": {"size": 16, "color": "#333"}
            },
            "xaxis": {
                "title": "Loss Category",
                "tickangle": -45
            },
            "yaxis": {
                "title": "Heat Loss (kW)",
                "gridcolor": "#E5E5E5"
            },
            "barmode": "group",
            "margin": {"l": 60, "r": 20, "t": 80, "b": 120},
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "legend": {
                "x": 0.01,
                "y": 0.99,
                "bgcolor": "rgba(255,255,255,0.8)",
                "bordercolor": "#DDD",
                "borderwidth": 1
            },
            "annotations": [
                {
                    "text": (
                        f"Baseline Total: {baseline_total:.0f} kW | "
                        f"Current Total: {current_total:.0f} kW | "
                        f"Reduction: {total_improvement:.0f} kW ({improvement_pct:.1f}%)"
                    ),
                    "x": 0.5,
                    "y": -0.25,
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

    def _get_color(self, loss_name: str) -> str:
        """Get color for loss type."""
        loss_lower = loss_name.lower()
        for key, color in self.LOSS_COLORS.items():
            if key in loss_lower:
                return color
        return self.LOSS_COLORS["other"]
