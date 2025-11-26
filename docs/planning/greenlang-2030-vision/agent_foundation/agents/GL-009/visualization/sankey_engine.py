"""Sankey Diagram Engine for GL-009 THERMALIQ.

Generates interactive energy flow visualizations for thermal efficiency analysis.
Output: Plotly-compatible JSON for web rendering.

Features:
- Multi-stage energy flow visualization
- Automatic node positioning with manual override
- Color-coded by energy type, efficiency, or temperature
- Provenance hashing for data lineage tracking
- Support for complex industrial processes (boilers, furnaces, heat exchangers)
- Loss attribution and breakdown visualization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import hashlib
from datetime import datetime


class NodeType(Enum):
    """Node types in energy flow diagram."""
    INPUT = "input"           # Energy inputs (fuel, electricity)
    PROCESS = "process"       # Conversion processes
    OUTPUT = "output"         # Useful outputs (steam, heat)
    LOSS = "loss"             # Heat losses


class ColorScheme(Enum):
    """Color schemes for visualization."""
    EFFICIENCY = "efficiency"      # Green=high, Red=low
    ENERGY_TYPE = "energy_type"    # By fuel type
    TEMPERATURE = "temperature"    # Hot=red, Cold=blue
    PROCESS_STAGE = "process_stage"  # By process stage


@dataclass
class SankeyNode:
    """Node in Sankey diagram."""
    id: str
    label: str
    node_type: NodeType
    value_kw: float
    color: str = "#888888"
    x_position: Optional[float] = None  # 0-1 for manual positioning
    y_position: Optional[float] = None
    temperature_c: Optional[float] = None
    efficiency_percent: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "node_type": self.node_type.value,
            "value_kw": self.value_kw,
            "color": self.color,
            "x_position": self.x_position,
            "y_position": self.y_position,
            "temperature_c": self.temperature_c,
            "efficiency_percent": self.efficiency_percent,
            "metadata": self.metadata
        }


@dataclass
class SankeyLink:
    """Link between nodes in Sankey diagram."""
    source_id: str
    target_id: str
    value_kw: float
    color: str = "#cccccc"
    label: Optional[str] = None
    efficiency_percent: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "value_kw": self.value_kw,
            "color": self.color,
            "label": self.label,
            "efficiency_percent": self.efficiency_percent,
            "metadata": self.metadata
        }


@dataclass
class SankeyDiagram:
    """Complete Sankey diagram with metadata."""
    nodes: List[SankeyNode]
    links: List[SankeyLink]
    title: str
    subtitle: Optional[str] = None
    total_input_kw: float = 0
    total_output_kw: float = 0
    total_losses_kw: float = 0
    efficiency_percent: float = 0
    provenance_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_plotly_json(self) -> Dict:
        """Export to Plotly Sankey format."""
        node_map = {n.id: i for i, n in enumerate(self.nodes)}

        # Prepare node data
        node_labels = []
        node_colors = []
        node_x = []
        node_y = []
        node_customdata = []

        for n in self.nodes:
            node_labels.append(n.label)
            node_colors.append(n.color)

            if n.x_position is not None:
                node_x.append(n.x_position)
            if n.y_position is not None:
                node_y.append(n.y_position)

            # Custom data for hover
            customdata = {
                "type": n.node_type.value,
                "value": f"{n.value_kw:.2f} kW"
            }
            if n.temperature_c:
                customdata["temperature"] = f"{n.temperature_c:.1f}°C"
            if n.efficiency_percent:
                customdata["efficiency"] = f"{n.efficiency_percent:.1f}%"
            node_customdata.append(customdata)

        # Prepare link data
        link_sources = []
        link_targets = []
        link_values = []
        link_colors = []
        link_labels = []
        link_customdata = []

        for l in self.links:
            link_sources.append(node_map[l.source_id])
            link_targets.append(node_map[l.target_id])
            link_values.append(l.value_kw)
            link_colors.append(l.color)
            link_labels.append(l.label or "")

            customdata = {"value": f"{l.value_kw:.2f} kW"}
            if l.efficiency_percent:
                customdata["efficiency"] = f"{l.efficiency_percent:.1f}%"
            link_customdata.append(customdata)

        # Build Plotly figure
        node_data = {
            "pad": 15,
            "thickness": 20,
            "line": {"color": "black", "width": 0.5},
            "label": node_labels,
            "color": node_colors,
            "customdata": node_customdata,
            "hovertemplate": "%{label}<br>%{customdata}<extra></extra>"
        }

        if node_x:
            node_data["x"] = node_x
        if node_y:
            node_data["y"] = node_y

        link_data = {
            "source": link_sources,
            "target": link_targets,
            "value": link_values,
            "color": link_colors,
            "label": link_labels,
            "customdata": link_customdata,
            "hovertemplate": "%{customdata}<extra></extra>"
        }

        # Create annotations
        annotations = [
            {
                "text": f"Efficiency: {self.efficiency_percent:.1f}% | "
                        f"Input: {self.total_input_kw:.0f} kW | "
                        f"Output: {self.total_output_kw:.0f} kW | "
                        f"Losses: {self.total_losses_kw:.0f} kW",
                "x": 0.5,
                "y": -0.1,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 10, "color": "#555"}
            }
        ]

        if self.subtitle:
            annotations.append({
                "text": self.subtitle,
                "x": 0.5,
                "y": 1.05,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 11, "color": "#777"}
            })

        return {
            "data": [{
                "type": "sankey",
                "orientation": "h",
                "node": node_data,
                "link": link_data
            }],
            "layout": {
                "title": {
                    "text": self.title,
                    "font": {"size": 16, "color": "#333"}
                },
                "font": {"size": 12},
                "annotations": annotations,
                "margin": {"l": 20, "r": 20, "t": 80, "b": 60},
                "paper_bgcolor": "white",
                "plot_bgcolor": "white"
            }
        }

    def to_dict(self) -> Dict:
        """Export to dictionary."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "links": [l.to_dict() for l in self.links],
            "title": self.title,
            "subtitle": self.subtitle,
            "total_input_kw": self.total_input_kw,
            "total_output_kw": self.total_output_kw,
            "total_losses_kw": self.total_losses_kw,
            "efficiency_percent": self.efficiency_percent,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class SankeyEngine:
    """Generate Sankey diagrams for thermal efficiency visualization."""

    # Color palettes
    COLORS = {
        "input": {
            "fuel": "#FF6B6B",
            "natural_gas": "#FF8C42",
            "coal": "#34495E",
            "electricity": "#4ECDC4",
            "steam_in": "#45B7D1",
            "oil": "#8B4513",
            "biomass": "#228B22"
        },
        "output": {
            "steam": "#96CEB4",
            "hot_water": "#FFEAA7",
            "process_heat": "#DDA0DD",
            "power": "#87CEEB",
            "heating": "#FFA07A",
            "cooling": "#B0E0E6"
        },
        "loss": {
            "radiation": "#E74C3C",
            "convection": "#E67E22",
            "flue_gas": "#95A5A6",
            "conduction": "#9B59B6",
            "unburned": "#34495E",
            "blowdown": "#1ABC9C",
            "stack": "#C0392B",
            "ambient": "#D35400",
            "moisture": "#16A085"
        },
        "process": {
            "boiler": "#3498DB",
            "furnace": "#E74C3C",
            "heat_exchanger": "#2ECC71",
            "turbine": "#9B59B6",
            "economizer": "#1ABC9C",
            "preheater": "#F39C12"
        }
    }

    def __init__(self, color_scheme: ColorScheme = ColorScheme.EFFICIENCY):
        """Initialize Sankey engine.

        Args:
            color_scheme: Color scheme for visualization
        """
        self.color_scheme = color_scheme

    def generate_from_efficiency_result(
        self,
        energy_inputs: Dict[str, float],
        useful_outputs: Dict[str, float],
        losses: Dict[str, float],
        title: str = "Thermal Energy Flow",
        process_name: str = "Process",
        metadata: Optional[Dict[str, Any]] = None
    ) -> SankeyDiagram:
        """Generate Sankey diagram from efficiency calculation results.

        Args:
            energy_inputs: Dictionary of input energy streams (kW)
            useful_outputs: Dictionary of useful output streams (kW)
            losses: Dictionary of loss streams (kW)
            title: Diagram title
            process_name: Name of the process node
            metadata: Additional metadata

        Returns:
            Complete Sankey diagram
        """
        nodes = []
        links = []

        # Create input nodes (positioned on left)
        y_offset = 0.1
        y_step = 0.8 / max(len(energy_inputs), 1)
        for i, (name, value) in enumerate(sorted(energy_inputs.items())):
            nodes.append(SankeyNode(
                id=f"input_{name}",
                label=f"{name.replace('_', ' ').title()}\n{value:.0f} kW",
                node_type=NodeType.INPUT,
                value_kw=value,
                color=self._get_input_color(name),
                x_position=0.01,
                y_position=y_offset + i * y_step,
                metadata={"category": "input", "stream": name}
            ))

        # Create process node (centered)
        total_input = sum(energy_inputs.values())
        nodes.append(SankeyNode(
            id="process",
            label=f"{process_name}\n{total_input:.0f} kW",
            node_type=NodeType.PROCESS,
            value_kw=total_input,
            color=self._get_process_color(process_name.lower()),
            x_position=0.5,
            y_position=0.5,
            metadata={"category": "process", "name": process_name}
        ))

        # Create output nodes (positioned on right)
        y_offset = 0.1
        y_step = 0.4 / max(len(useful_outputs), 1)
        for i, (name, value) in enumerate(sorted(useful_outputs.items())):
            nodes.append(SankeyNode(
                id=f"output_{name}",
                label=f"{name.replace('_', ' ').title()}\n{value:.0f} kW",
                node_type=NodeType.OUTPUT,
                value_kw=value,
                color=self._get_output_color(name),
                x_position=0.99,
                y_position=y_offset + i * y_step,
                metadata={"category": "output", "stream": name}
            ))

        # Create loss nodes (positioned on right, below outputs)
        y_offset = 0.6
        y_step = 0.3 / max(len(losses), 1)
        for i, (name, value) in enumerate(sorted(losses.items())):
            if value > 0:
                nodes.append(SankeyNode(
                    id=f"loss_{name}",
                    label=f"{name.replace('_', ' ').title()}\n{value:.0f} kW",
                    node_type=NodeType.LOSS,
                    value_kw=value,
                    color=self._get_loss_color(name),
                    x_position=0.99,
                    y_position=y_offset + i * y_step,
                    metadata={"category": "loss", "stream": name}
                ))

        # Create links from inputs to process
        for name, value in energy_inputs.items():
            links.append(SankeyLink(
                source_id=f"input_{name}",
                target_id="process",
                value_kw=value,
                color=self._get_input_color(name) + "60",  # Add transparency
                label=f"{value:.0f} kW",
                metadata={"flow": f"{name} → process"}
            ))

        # Create links from process to outputs
        for name, value in useful_outputs.items():
            efficiency = (value / total_input * 100) if total_input > 0 else 0
            links.append(SankeyLink(
                source_id="process",
                target_id=f"output_{name}",
                value_kw=value,
                color=self._get_output_color(name) + "60",
                label=f"{value:.0f} kW ({efficiency:.1f}%)",
                efficiency_percent=efficiency,
                metadata={"flow": f"process → {name}"}
            ))

        # Create links from process to losses
        for name, value in losses.items():
            if value > 0:
                loss_percent = (value / total_input * 100) if total_input > 0 else 0
                links.append(SankeyLink(
                    source_id="process",
                    target_id=f"loss_{name}",
                    value_kw=value,
                    color=self._get_loss_color(name) + "60",
                    label=f"{value:.0f} kW ({loss_percent:.1f}%)",
                    efficiency_percent=loss_percent,
                    metadata={"flow": f"process → {name} loss"}
                ))

        # Calculate totals and efficiency
        total_output = sum(useful_outputs.values())
        total_losses = sum(losses.values())
        efficiency = (total_output / total_input * 100) if total_input > 0 else 0

        # Create provenance hash
        data = {
            "inputs": energy_inputs,
            "outputs": useful_outputs,
            "losses": losses,
            "timestamp": datetime.utcnow().isoformat()
        }
        provenance = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return SankeyDiagram(
            nodes=nodes,
            links=links,
            title=title,
            subtitle=f"Total Input: {total_input:.0f} kW | Useful Output: {total_output:.0f} kW | Losses: {total_losses:.0f} kW",
            total_input_kw=total_input,
            total_output_kw=total_output,
            total_losses_kw=total_losses,
            efficiency_percent=efficiency,
            provenance_hash=provenance,
            metadata=metadata or {}
        )

    def generate_multi_stage(
        self,
        stages: List[Dict[str, Any]],
        title: str = "Multi-Stage Energy Flow"
    ) -> SankeyDiagram:
        """Generate multi-stage Sankey diagram for complex processes.

        Args:
            stages: List of process stages, each with inputs, outputs, losses
            title: Diagram title

        Returns:
            Multi-stage Sankey diagram
        """
        nodes = []
        links = []
        x_positions = [0.01 + i * (0.98 / len(stages)) for i in range(len(stages) + 1)]

        for stage_idx, stage in enumerate(stages):
            stage_name = stage.get("name", f"Stage {stage_idx + 1}")
            inputs = stage.get("inputs", {})
            outputs = stage.get("outputs", {})
            losses = stage.get("losses", {})

            # Create stage node
            total_input = sum(inputs.values())
            nodes.append(SankeyNode(
                id=f"stage_{stage_idx}",
                label=f"{stage_name}\n{total_input:.0f} kW",
                node_type=NodeType.PROCESS,
                value_kw=total_input,
                color=self._get_process_color(stage_name.lower()),
                x_position=x_positions[stage_idx],
                y_position=0.5
            ))

            # Create input nodes for first stage
            if stage_idx == 0:
                for name, value in inputs.items():
                    nodes.append(SankeyNode(
                        id=f"input_{name}",
                        label=f"{name.replace('_', ' ').title()}\n{value:.0f} kW",
                        node_type=NodeType.INPUT,
                        value_kw=value,
                        color=self._get_input_color(name),
                        x_position=0.01
                    ))
                    links.append(SankeyLink(
                        source_id=f"input_{name}",
                        target_id=f"stage_{stage_idx}",
                        value_kw=value,
                        color=self._get_input_color(name) + "60"
                    ))

            # Create loss nodes
            for name, value in losses.items():
                if value > 0:
                    loss_id = f"loss_{stage_idx}_{name}"
                    nodes.append(SankeyNode(
                        id=loss_id,
                        label=f"{name.replace('_', ' ').title()}\n{value:.0f} kW",
                        node_type=NodeType.LOSS,
                        value_kw=value,
                        color=self._get_loss_color(name),
                        x_position=x_positions[stage_idx + 1]
                    ))
                    links.append(SankeyLink(
                        source_id=f"stage_{stage_idx}",
                        target_id=loss_id,
                        value_kw=value,
                        color=self._get_loss_color(name) + "60"
                    ))

            # Link to next stage or outputs
            if stage_idx < len(stages) - 1:
                for name, value in outputs.items():
                    links.append(SankeyLink(
                        source_id=f"stage_{stage_idx}",
                        target_id=f"stage_{stage_idx + 1}",
                        value_kw=value,
                        color="#3498DB60"
                    ))
            else:
                for name, value in outputs.items():
                    output_id = f"output_{name}"
                    nodes.append(SankeyNode(
                        id=output_id,
                        label=f"{name.replace('_', ' ').title()}\n{value:.0f} kW",
                        node_type=NodeType.OUTPUT,
                        value_kw=value,
                        color=self._get_output_color(name),
                        x_position=0.99
                    ))
                    links.append(SankeyLink(
                        source_id=f"stage_{stage_idx}",
                        target_id=output_id,
                        value_kw=value,
                        color=self._get_output_color(name) + "60"
                    ))

        # Calculate totals
        first_stage = stages[0]
        last_stage = stages[-1]
        total_input = sum(first_stage.get("inputs", {}).values())
        total_output = sum(last_stage.get("outputs", {}).values())
        total_losses = sum(
            sum(stage.get("losses", {}).values()) for stage in stages
        )
        efficiency = (total_output / total_input * 100) if total_input > 0 else 0

        # Create provenance hash
        provenance = hashlib.sha256(
            json.dumps(stages, sort_keys=True).encode()
        ).hexdigest()[:16]

        return SankeyDiagram(
            nodes=nodes,
            links=links,
            title=title,
            total_input_kw=total_input,
            total_output_kw=total_output,
            total_losses_kw=total_losses,
            efficiency_percent=efficiency,
            provenance_hash=provenance
        )

    def _get_input_color(self, name: str) -> str:
        """Get color for input node."""
        name_lower = name.lower()
        for key, color in self.COLORS["input"].items():
            if key in name_lower:
                return color
        return "#FF6B6B"  # Default fuel color

    def _get_output_color(self, name: str) -> str:
        """Get color for output node."""
        name_lower = name.lower()
        for key, color in self.COLORS["output"].items():
            if key in name_lower:
                return color
        return "#96CEB4"  # Default steam color

    def _get_loss_color(self, name: str) -> str:
        """Get color for loss node."""
        name_lower = name.lower()
        for key, color in self.COLORS["loss"].items():
            if key in name_lower:
                return color
        return "#E74C3C"  # Default loss color

    def _get_process_color(self, name: str) -> str:
        """Get color for process node."""
        name_lower = name.lower()
        for key, color in self.COLORS["process"].items():
            if key in name_lower:
                return color
        return "#3498DB"  # Default process color
