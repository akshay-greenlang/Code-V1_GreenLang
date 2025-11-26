"""Waterfall Chart for GL-009 THERMALIQ Heat Balance Breakdown.

Generates waterfall charts showing step-by-step energy transformation
from input through losses to final useful output.

Features:
- Sequential heat balance visualization
- Color-coded gains and losses
- Cumulative energy tracking
- Plotly-compatible output
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import hashlib
import json


class BarType(Enum):
    """Bar types in waterfall chart."""
    TOTAL = "total"      # Starting or ending total
    GAIN = "gain"        # Energy addition
    LOSS = "loss"        # Energy reduction
    SUBTOTAL = "subtotal"  # Intermediate total


@dataclass
class WaterfallBar:
    """Single bar in waterfall chart."""
    label: str
    value: float  # Absolute value (positive for gains, negative for losses)
    bar_type: BarType
    color: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "value": self.value,
            "bar_type": self.bar_type.value,
            "color": self.color,
            "description": self.description
        }


@dataclass
class WaterfallData:
    """Complete waterfall chart data."""
    bars: List[WaterfallBar]
    title: str
    subtitle: Optional[str] = None
    y_axis_label: str = "Energy (kW)"
    start_value: float = 0
    end_value: float = 0
    total_gains: float = 0
    total_losses: float = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_plotly_json(self) -> Dict:
        """Export to Plotly waterfall format."""
        # Build data arrays
        labels = []
        values = []
        measures = []
        colors = []
        text_labels = []

        cumulative = self.start_value

        for bar in self.bars:
            labels.append(bar.label)

            if bar.bar_type == BarType.TOTAL:
                measures.append("total")
                values.append(bar.value)
                cumulative = bar.value
                color = "#3498DB"
            elif bar.bar_type == BarType.SUBTOTAL:
                measures.append("total")
                values.append(cumulative)
                color = "#9B59B6"
            elif bar.bar_type == BarType.GAIN:
                measures.append("relative")
                values.append(bar.value)
                cumulative += bar.value
                color = bar.color or "#2ECC71"
            elif bar.bar_type == BarType.LOSS:
                measures.append("relative")
                values.append(-abs(bar.value))  # Ensure negative
                cumulative -= abs(bar.value)
                color = bar.color or "#E74C3C"
            else:
                measures.append("relative")
                values.append(bar.value)
                color = "#95A5A6"

            colors.append(bar.color or color)

            # Text labels
            if bar.bar_type in (BarType.TOTAL, BarType.SUBTOTAL):
                text_labels.append(f"{bar.value:.0f} kW")
            else:
                text_labels.append(f"{abs(bar.value):.0f} kW")

        # Create Plotly figure
        return {
            "data": [{
                "type": "waterfall",
                "orientation": "v",
                "x": labels,
                "y": values,
                "measure": measures,
                "text": text_labels,
                "textposition": "outside",
                "connector": {
                    "mode": "between",
                    "line": {"width": 1, "color": "#999", "dash": "dot"}
                },
                "increasing": {"marker": {"color": "#2ECC71"}},
                "decreasing": {"marker": {"color": "#E74C3C"}},
                "totals": {"marker": {"color": "#3498DB"}},
                "marker": {"color": colors},
                "hovertemplate": "%{x}<br>%{text}<br>Cumulative: %{y:.0f} kW<extra></extra>"
            }],
            "layout": {
                "title": {
                    "text": self.title,
                    "font": {"size": 16, "color": "#333"}
                },
                "xaxis": {
                    "title": "",
                    "tickangle": -45
                },
                "yaxis": {
                    "title": self.y_axis_label,
                    "gridcolor": "#E5E5E5"
                },
                "margin": {"l": 60, "r": 20, "t": 80, "b": 120},
                "paper_bgcolor": "white",
                "plot_bgcolor": "white",
                "showlegend": False,
                "annotations": [
                    {
                        "text": self.subtitle or f"Start: {self.start_value:.0f} kW | End: {self.end_value:.0f} kW | "
                                                f"Gains: +{self.total_gains:.0f} kW | Losses: -{self.total_losses:.0f} kW",
                        "x": 0.5,
                        "y": -0.25,
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 10, "color": "#555"}
                    }
                ]
            }
        }

    def to_dict(self) -> Dict:
        """Export to dictionary."""
        return {
            "bars": [b.to_dict() for b in self.bars],
            "title": self.title,
            "subtitle": self.subtitle,
            "y_axis_label": self.y_axis_label,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "total_gains": self.total_gains,
            "total_losses": self.total_losses,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class WaterfallChart:
    """Generate waterfall charts for heat balance breakdown."""

    def __init__(self):
        """Initialize waterfall chart generator."""
        pass

    def generate_from_heat_balance(
        self,
        input_energy: Dict[str, float],
        losses: Dict[str, float],
        useful_output: Dict[str, float],
        title: str = "Heat Balance Waterfall",
        include_subtotals: bool = True
    ) -> WaterfallData:
        """Generate waterfall chart from heat balance data.

        Args:
            input_energy: Dictionary of input energy streams
            losses: Dictionary of loss streams (positive values)
            useful_output: Dictionary of useful output streams
            title: Chart title
            include_subtotals: Include subtotal bars

        Returns:
            Waterfall chart data
        """
        bars = []

        # Starting total (total input)
        total_input = sum(input_energy.values())
        bars.append(WaterfallBar(
            label="Total Input",
            value=total_input,
            bar_type=BarType.TOTAL,
            color="#3498DB",
            description="Total energy input to the system"
        ))

        # Track cumulative
        cumulative = total_input
        total_losses_sum = 0
        total_gains_sum = 0

        # Add losses (as negative)
        for name, value in sorted(losses.items(), key=lambda x: -x[1]):
            if value > 0:
                bars.append(WaterfallBar(
                    label=name.replace('_', ' ').title(),
                    value=-value,  # Negative for losses
                    bar_type=BarType.LOSS,
                    color=self._get_loss_color(name),
                    description=f"{name.replace('_', ' ').title()} loss"
                ))
                cumulative -= value
                total_losses_sum += value

        # Subtotal after losses
        if include_subtotals and losses:
            bars.append(WaterfallBar(
                label="After Losses",
                value=cumulative,
                bar_type=BarType.SUBTOTAL,
                color="#9B59B6",
                description="Energy remaining after all losses"
            ))

        # Ending total (useful output)
        total_output = sum(useful_output.values())
        bars.append(WaterfallBar(
            label="Useful Output",
            value=total_output,
            bar_type=BarType.TOTAL,
            color="#2ECC71",
            description="Total useful energy output"
        ))

        return WaterfallData(
            bars=bars,
            title=title,
            subtitle=f"Efficiency: {(total_output / total_input * 100):.1f}%",
            start_value=total_input,
            end_value=total_output,
            total_gains=total_gains_sum,
            total_losses=total_losses_sum,
            metadata={
                "input_energy": input_energy,
                "losses": losses,
                "useful_output": useful_output
            }
        )

    def generate_detailed_breakdown(
        self,
        input_energy: Dict[str, float],
        process_losses: Dict[str, float],
        distribution_losses: Dict[str, float],
        useful_output: Dict[str, float],
        title: str = "Detailed Heat Balance"
    ) -> WaterfallData:
        """Generate detailed waterfall with process and distribution stages.

        Args:
            input_energy: Input energy streams
            process_losses: Losses during process stage
            distribution_losses: Losses during distribution stage
            useful_output: Useful output streams
            title: Chart title

        Returns:
            Detailed waterfall chart
        """
        bars = []

        # Starting total
        total_input = sum(input_energy.values())
        bars.append(WaterfallBar(
            label="Total Input",
            value=total_input,
            bar_type=BarType.TOTAL,
            color="#3498DB"
        ))

        # Process losses
        cumulative = total_input
        for name, value in sorted(process_losses.items(), key=lambda x: -x[1]):
            if value > 0:
                bars.append(WaterfallBar(
                    label=f"Process: {name.replace('_', ' ').title()}",
                    value=-value,
                    bar_type=BarType.LOSS,
                    color="#E74C3C"
                ))
                cumulative -= value

        # Subtotal after process
        bars.append(WaterfallBar(
            label="After Process",
            value=cumulative,
            bar_type=BarType.SUBTOTAL,
            color="#9B59B6"
        ))

        # Distribution losses
        for name, value in sorted(distribution_losses.items(), key=lambda x: -x[1]):
            if value > 0:
                bars.append(WaterfallBar(
                    label=f"Distribution: {name.replace('_', ' ').title()}",
                    value=-value,
                    bar_type=BarType.LOSS,
                    color="#E67E22"
                ))
                cumulative -= value

        # Final total
        total_output = sum(useful_output.values())
        bars.append(WaterfallBar(
            label="Useful Output",
            value=total_output,
            bar_type=BarType.TOTAL,
            color="#2ECC71"
        ))

        total_losses = sum(process_losses.values()) + sum(distribution_losses.values())

        return WaterfallData(
            bars=bars,
            title=title,
            start_value=total_input,
            end_value=total_output,
            total_gains=0,
            total_losses=total_losses,
            metadata={
                "input_energy": input_energy,
                "process_losses": process_losses,
                "distribution_losses": distribution_losses,
                "useful_output": useful_output
            }
        )

    def _get_loss_color(self, name: str) -> str:
        """Get color for loss type."""
        name_lower = name.lower()
        loss_colors = {
            "radiation": "#E74C3C",
            "convection": "#E67E22",
            "flue": "#95A5A6",
            "stack": "#C0392B",
            "conduction": "#9B59B6",
            "unburned": "#34495E",
            "blowdown": "#1ABC9C",
            "moisture": "#16A085"
        }

        for key, color in loss_colors.items():
            if key in name_lower:
                return color

        return "#E74C3C"  # Default red for losses
