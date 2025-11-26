"""Sankey Diagram Data Generator.

This module generates Sankey diagram data for energy flow visualization
in thermal systems. The output format is compatible with Plotly and D3.js
visualization libraries.

A Sankey diagram shows energy flows from inputs through processes
to outputs and losses, with link widths proportional to flow magnitude.

Features:
    - Node definitions (inputs, processes, outputs, losses)
    - Link values (energy flows in kW)
    - Color coding by efficiency/category
    - Export format for Plotly/D3.js
    - Automatic node positioning

Author: GL-009 THERMALIQ Agent
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import hashlib
import json
from datetime import datetime


class NodeType(Enum):
    """Type of node in Sankey diagram."""
    INPUT = "input"
    PROCESS = "process"
    OUTPUT = "output"
    LOSS = "loss"
    STORAGE = "storage"


class EnergyFlowCategory(Enum):
    """Category of energy flow for color coding."""
    FUEL = "fuel"
    ELECTRICITY = "electricity"
    STEAM = "steam"
    HOT_WATER = "hot_water"
    PROCESS_HEAT = "process_heat"
    COOLING = "cooling"
    FLUE_GAS = "flue_gas"
    RADIATION = "radiation"
    CONVECTION = "convection"
    CONDUCTION = "conduction"
    UNBURNED = "unburned"
    BLOWDOWN = "blowdown"
    OTHER_LOSS = "other_loss"
    RECOVERED = "recovered"


# Color palette for categories (Plotly-compatible)
CATEGORY_COLORS: Dict[EnergyFlowCategory, str] = {
    EnergyFlowCategory.FUEL: "rgba(255, 127, 14, 0.8)",        # Orange
    EnergyFlowCategory.ELECTRICITY: "rgba(44, 160, 44, 0.8)",   # Green
    EnergyFlowCategory.STEAM: "rgba(31, 119, 180, 0.8)",        # Blue
    EnergyFlowCategory.HOT_WATER: "rgba(23, 190, 207, 0.8)",    # Cyan
    EnergyFlowCategory.PROCESS_HEAT: "rgba(188, 189, 34, 0.8)", # Yellow-green
    EnergyFlowCategory.COOLING: "rgba(148, 103, 189, 0.8)",     # Purple
    EnergyFlowCategory.FLUE_GAS: "rgba(227, 119, 194, 0.8)",    # Pink
    EnergyFlowCategory.RADIATION: "rgba(214, 39, 40, 0.8)",     # Red
    EnergyFlowCategory.CONVECTION: "rgba(255, 152, 150, 0.8)",  # Light red
    EnergyFlowCategory.CONDUCTION: "rgba(255, 187, 120, 0.8)",  # Light orange
    EnergyFlowCategory.UNBURNED: "rgba(152, 78, 163, 0.8)",     # Dark purple
    EnergyFlowCategory.BLOWDOWN: "rgba(77, 175, 74, 0.8)",      # Medium green
    EnergyFlowCategory.OTHER_LOSS: "rgba(127, 127, 127, 0.8)",  # Gray
    EnergyFlowCategory.RECOVERED: "rgba(102, 194, 165, 0.8)",   # Teal
}


@dataclass
class SankeyNode:
    """A node in the Sankey diagram.

    Attributes:
        node_id: Unique identifier for the node
        label: Display label for the node
        node_type: Type of node (input, process, output, loss)
        value_kw: Energy value at this node (kW)
        color: Node color (hex or rgba)
        x_position: Horizontal position (0-1), None for auto
        y_position: Vertical position (0-1), None for auto
        description: Optional description
    """
    node_id: str
    label: str
    node_type: NodeType
    value_kw: float
    color: Optional[str] = None
    x_position: Optional[float] = None
    y_position: Optional[float] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.node_id,
            "label": self.label,
            "type": self.node_type.value,
            "value_kw": self.value_kw
        }
        if self.color:
            result["color"] = self.color
        if self.x_position is not None:
            result["x"] = self.x_position
        if self.y_position is not None:
            result["y"] = self.y_position
        if self.description:
            result["description"] = self.description
        return result


@dataclass
class SankeyLink:
    """A link (flow) between nodes in the Sankey diagram.

    Attributes:
        source_id: ID of source node
        target_id: ID of target node
        value_kw: Energy flow value (kW)
        category: Category for color coding
        color: Link color (overrides category color)
        label: Optional label for the link
        percentage: Flow as percentage of total input
    """
    source_id: str
    target_id: str
    value_kw: float
    category: EnergyFlowCategory
    color: Optional[str] = None
    label: Optional[str] = None
    percentage: Optional[float] = None

    def get_color(self) -> str:
        """Get link color from category or override."""
        if self.color:
            return self.color
        return CATEGORY_COLORS.get(self.category, "rgba(127, 127, 127, 0.5)")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "source": self.source_id,
            "target": self.target_id,
            "value": self.value_kw,
            "category": self.category.value,
            "color": self.get_color()
        }
        if self.label:
            result["label"] = self.label
        if self.percentage is not None:
            result["percentage"] = self.percentage
        return result


@dataclass
class SankeyDiagram:
    """Complete Sankey diagram data structure.

    Attributes:
        title: Diagram title
        nodes: List of all nodes
        links: List of all links
        total_input_kw: Total energy input (kW)
        total_output_kw: Total useful output (kW)
        total_losses_kw: Total losses (kW)
        efficiency_percent: System efficiency
        provenance_hash: SHA-256 hash
        generated_timestamp: When generated
        metadata: Additional metadata
    """
    title: str
    nodes: List[SankeyNode]
    links: List[SankeyLink]
    total_input_kw: float
    total_output_kw: float
    total_losses_kw: float
    efficiency_percent: float
    provenance_hash: str
    generated_timestamp: str
    generator_version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "title": self.title,
            "nodes": [n.to_dict() for n in self.nodes],
            "links": [l.to_dict() for l in self.links],
            "summary": {
                "total_input_kw": self.total_input_kw,
                "total_output_kw": self.total_output_kw,
                "total_losses_kw": self.total_losses_kw,
                "efficiency_percent": self.efficiency_percent
            },
            "provenance_hash": self.provenance_hash,
            "generated_timestamp": self.generated_timestamp,
            "generator_version": self.generator_version,
            "metadata": self.metadata
        }

    def to_plotly_format(self) -> Dict[str, Any]:
        """Convert to Plotly Sankey trace format.

        Returns format compatible with plotly.graph_objects.Sankey

        Example:
            >>> import plotly.graph_objects as go
            >>> diagram = generator.generate_diagram(...)
            >>> data = diagram.to_plotly_format()
            >>> fig = go.Figure(go.Sankey(**data))
        """
        # Build node index mapping
        node_ids = [n.node_id for n in self.nodes]
        node_index = {nid: i for i, nid in enumerate(node_ids)}

        # Node data
        node_labels = [n.label for n in self.nodes]
        node_colors = [
            n.color or self._get_node_color(n.node_type)
            for n in self.nodes
        ]

        # Link data
        source_indices = [node_index[l.source_id] for l in self.links]
        target_indices = [node_index[l.target_id] for l in self.links]
        link_values = [l.value_kw for l in self.links]
        link_colors = [l.get_color() for l in self.links]
        link_labels = [
            l.label or f"{l.value_kw:.1f} kW ({l.percentage:.1f}%)"
            if l.percentage else f"{l.value_kw:.1f} kW"
            for l in self.links
        ]

        return {
            "node": {
                "label": node_labels,
                "color": node_colors,
                "pad": 15,
                "thickness": 20,
                "line": {"color": "black", "width": 0.5}
            },
            "link": {
                "source": source_indices,
                "target": target_indices,
                "value": link_values,
                "color": link_colors,
                "label": link_labels
            }
        }

    def to_d3_format(self) -> Dict[str, Any]:
        """Convert to D3.js Sankey format.

        Returns format compatible with d3-sankey library.
        """
        # D3 uses numeric node indices
        node_ids = [n.node_id for n in self.nodes]
        node_index = {nid: i for i, nid in enumerate(node_ids)}

        nodes = [
            {
                "name": n.label,
                "type": n.node_type.value,
                "value": n.value_kw,
                "color": n.color or self._get_node_color(n.node_type)
            }
            for n in self.nodes
        ]

        links = [
            {
                "source": node_index[l.source_id],
                "target": node_index[l.target_id],
                "value": l.value_kw,
                "color": l.get_color()
            }
            for l in self.links
        ]

        return {"nodes": nodes, "links": links}

    def _get_node_color(self, node_type: NodeType) -> str:
        """Get default color for node type."""
        colors = {
            NodeType.INPUT: "rgba(31, 119, 180, 0.8)",
            NodeType.PROCESS: "rgba(44, 160, 44, 0.8)",
            NodeType.OUTPUT: "rgba(23, 190, 207, 0.8)",
            NodeType.LOSS: "rgba(214, 39, 40, 0.8)",
            NodeType.STORAGE: "rgba(148, 103, 189, 0.8)"
        }
        return colors.get(node_type, "rgba(127, 127, 127, 0.8)")


class SankeyGenerator:
    """Sankey Diagram Data Generator.

    Generates Sankey diagram data from thermal system energy flows.
    Supports multiple output formats for visualization libraries.

    Example:
        >>> generator = SankeyGenerator()
        >>> diagram = generator.generate_from_efficiency_result(
        ...     efficiency_result=first_law_result,
        ...     title="Boiler Energy Balance"
        ... )
        >>> plotly_data = diagram.to_plotly_format()
    """

    VERSION: str = "1.0.0"
    PRECISION: int = 2

    def __init__(self, precision: int = 2) -> None:
        """Initialize the Sankey Generator.

        Args:
            precision: Decimal places for displayed values
        """
        self.precision = precision

    def generate_diagram(
        self,
        inputs: Dict[str, Tuple[float, EnergyFlowCategory]],
        outputs: Dict[str, Tuple[float, EnergyFlowCategory]],
        losses: Dict[str, Tuple[float, EnergyFlowCategory]],
        process_name: str = "Process",
        title: str = "Energy Flow Diagram"
    ) -> SankeyDiagram:
        """Generate Sankey diagram from energy flow data.

        Args:
            inputs: Dict of {name: (value_kw, category)}
            outputs: Dict of {name: (value_kw, category)}
            losses: Dict of {name: (value_kw, category)}
            process_name: Name of central process node
            title: Diagram title

        Returns:
            SankeyDiagram with nodes and links
        """
        nodes: List[SankeyNode] = []
        links: List[SankeyLink] = []

        # Calculate totals
        total_input = sum(v[0] for v in inputs.values())
        total_output = sum(v[0] for v in outputs.values())
        total_losses = sum(v[0] for v in losses.values())
        efficiency = (total_output / total_input * 100) if total_input > 0 else 0

        # Create input nodes
        y_pos = 0.1
        y_step = 0.8 / max(len(inputs), 1)
        for name, (value, category) in inputs.items():
            node_id = f"input_{name.lower().replace(' ', '_')}"
            nodes.append(SankeyNode(
                node_id=node_id,
                label=f"{name}\n{value:.1f} kW",
                node_type=NodeType.INPUT,
                value_kw=value,
                color=CATEGORY_COLORS.get(category),
                x_position=0.0,
                y_position=y_pos
            ))
            y_pos += y_step

        # Create process node
        process_id = "process_main"
        nodes.append(SankeyNode(
            node_id=process_id,
            label=f"{process_name}\n{efficiency:.1f}% eff",
            node_type=NodeType.PROCESS,
            value_kw=total_input,
            x_position=0.5,
            y_position=0.5
        ))

        # Create output nodes
        y_pos = 0.1
        y_step = 0.4 / max(len(outputs), 1)
        for name, (value, category) in outputs.items():
            node_id = f"output_{name.lower().replace(' ', '_')}"
            nodes.append(SankeyNode(
                node_id=node_id,
                label=f"{name}\n{value:.1f} kW",
                node_type=NodeType.OUTPUT,
                value_kw=value,
                color=CATEGORY_COLORS.get(category),
                x_position=1.0,
                y_position=y_pos
            ))
            y_pos += y_step

        # Create loss nodes
        y_pos = 0.6
        y_step = 0.35 / max(len(losses), 1)
        for name, (value, category) in losses.items():
            node_id = f"loss_{name.lower().replace(' ', '_')}"
            nodes.append(SankeyNode(
                node_id=node_id,
                label=f"{name}\n{value:.1f} kW",
                node_type=NodeType.LOSS,
                value_kw=value,
                color=CATEGORY_COLORS.get(category),
                x_position=1.0,
                y_position=y_pos
            ))
            y_pos += y_step

        # Create links from inputs to process
        for name, (value, category) in inputs.items():
            node_id = f"input_{name.lower().replace(' ', '_')}"
            pct = (value / total_input * 100) if total_input > 0 else 0
            links.append(SankeyLink(
                source_id=node_id,
                target_id=process_id,
                value_kw=value,
                category=category,
                percentage=pct,
                label=f"{name}: {value:.1f} kW"
            ))

        # Create links from process to outputs
        for name, (value, category) in outputs.items():
            node_id = f"output_{name.lower().replace(' ', '_')}"
            pct = (value / total_input * 100) if total_input > 0 else 0
            links.append(SankeyLink(
                source_id=process_id,
                target_id=node_id,
                value_kw=value,
                category=category,
                percentage=pct,
                label=f"{name}: {value:.1f} kW ({pct:.1f}%)"
            ))

        # Create links from process to losses
        for name, (value, category) in losses.items():
            node_id = f"loss_{name.lower().replace(' ', '_')}"
            pct = (value / total_input * 100) if total_input > 0 else 0
            links.append(SankeyLink(
                source_id=process_id,
                target_id=node_id,
                value_kw=value,
                category=category,
                percentage=pct,
                label=f"{name}: {value:.1f} kW ({pct:.1f}%)"
            ))

        # Generate provenance hash
        provenance = self._generate_provenance_hash(inputs, outputs, losses)
        timestamp = datetime.utcnow().isoformat() + "Z"

        return SankeyDiagram(
            title=title,
            nodes=nodes,
            links=links,
            total_input_kw=self._round_value(total_input),
            total_output_kw=self._round_value(total_output),
            total_losses_kw=self._round_value(total_losses),
            efficiency_percent=self._round_value(efficiency),
            provenance_hash=provenance,
            generated_timestamp=timestamp,
            generator_version=self.VERSION
        )

    def generate_boiler_sankey(
        self,
        fuel_input_kw: float,
        steam_output_kw: float,
        flue_gas_loss_kw: float,
        radiation_loss_kw: float,
        convection_loss_kw: float,
        blowdown_loss_kw: float = 0.0,
        unburned_loss_kw: float = 0.0,
        other_losses_kw: float = 0.0,
        title: str = "Boiler Energy Balance"
    ) -> SankeyDiagram:
        """Generate Sankey diagram for a boiler system.

        Convenience method for common boiler energy balance.

        Args:
            fuel_input_kw: Fuel energy input
            steam_output_kw: Steam energy output
            flue_gas_loss_kw: Flue gas heat loss
            radiation_loss_kw: Radiation heat loss
            convection_loss_kw: Convection heat loss
            blowdown_loss_kw: Blowdown loss
            unburned_loss_kw: Unburned fuel loss
            other_losses_kw: Other miscellaneous losses
            title: Diagram title

        Returns:
            SankeyDiagram for the boiler
        """
        inputs = {
            "Fuel": (fuel_input_kw, EnergyFlowCategory.FUEL)
        }

        outputs = {
            "Steam": (steam_output_kw, EnergyFlowCategory.STEAM)
        }

        losses = {
            "Flue Gas": (flue_gas_loss_kw, EnergyFlowCategory.FLUE_GAS),
            "Radiation": (radiation_loss_kw, EnergyFlowCategory.RADIATION),
            "Convection": (convection_loss_kw, EnergyFlowCategory.CONVECTION)
        }

        if blowdown_loss_kw > 0:
            losses["Blowdown"] = (blowdown_loss_kw, EnergyFlowCategory.BLOWDOWN)
        if unburned_loss_kw > 0:
            losses["Unburned Fuel"] = (unburned_loss_kw, EnergyFlowCategory.UNBURNED)
        if other_losses_kw > 0:
            losses["Other"] = (other_losses_kw, EnergyFlowCategory.OTHER_LOSS)

        return self.generate_diagram(
            inputs=inputs,
            outputs=outputs,
            losses=losses,
            process_name="Boiler",
            title=title
        )

    def generate_multi_stage_sankey(
        self,
        stages: List[Dict[str, Any]],
        title: str = "Multi-Stage Energy Flow"
    ) -> SankeyDiagram:
        """Generate Sankey diagram for multi-stage process.

        Args:
            stages: List of stage definitions, each containing:
                - name: Stage name
                - inputs: Dict of inputs
                - outputs: Dict of outputs
                - losses: Dict of losses
            title: Diagram title

        Returns:
            SankeyDiagram with all stages connected
        """
        all_nodes: List[SankeyNode] = []
        all_links: List[SankeyLink] = []

        num_stages = len(stages)
        x_step = 1.0 / (num_stages + 1)

        total_input = 0.0
        total_output = 0.0
        total_losses = 0.0

        for i, stage in enumerate(stages):
            stage_name = stage.get("name", f"Stage {i+1}")
            stage_inputs = stage.get("inputs", {})
            stage_outputs = stage.get("outputs", {})
            stage_losses = stage.get("losses", {})

            x_pos = (i + 1) * x_step

            # Process node for this stage
            stage_id = f"stage_{i}"
            stage_total = sum(v[0] for v in stage_inputs.values())
            total_input += stage_total

            all_nodes.append(SankeyNode(
                node_id=stage_id,
                label=stage_name,
                node_type=NodeType.PROCESS,
                value_kw=stage_total,
                x_position=x_pos,
                y_position=0.5
            ))

            # Input nodes (only for first stage typically)
            if i == 0:
                y_pos = 0.1
                for name, (value, category) in stage_inputs.items():
                    node_id = f"input_{name.lower().replace(' ', '_')}"
                    all_nodes.append(SankeyNode(
                        node_id=node_id,
                        label=f"{name}\n{value:.1f} kW",
                        node_type=NodeType.INPUT,
                        value_kw=value,
                        color=CATEGORY_COLORS.get(category),
                        x_position=0.0,
                        y_position=y_pos
                    ))
                    all_links.append(SankeyLink(
                        source_id=node_id,
                        target_id=stage_id,
                        value_kw=value,
                        category=category
                    ))
                    y_pos += 0.2

            # Output nodes
            y_pos = 0.1
            for name, (value, category) in stage_outputs.items():
                total_output += value
                node_id = f"output_{i}_{name.lower().replace(' ', '_')}"

                # Check if this output feeds next stage
                if i < num_stages - 1:
                    # Link to next stage
                    all_links.append(SankeyLink(
                        source_id=stage_id,
                        target_id=f"stage_{i+1}",
                        value_kw=value,
                        category=category
                    ))
                else:
                    # Final output node
                    all_nodes.append(SankeyNode(
                        node_id=node_id,
                        label=f"{name}\n{value:.1f} kW",
                        node_type=NodeType.OUTPUT,
                        value_kw=value,
                        color=CATEGORY_COLORS.get(category),
                        x_position=1.0,
                        y_position=y_pos
                    ))
                    all_links.append(SankeyLink(
                        source_id=stage_id,
                        target_id=node_id,
                        value_kw=value,
                        category=category
                    ))
                y_pos += 0.2

            # Loss nodes
            y_pos = 0.7
            for name, (value, category) in stage_losses.items():
                total_losses += value
                node_id = f"loss_{i}_{name.lower().replace(' ', '_')}"
                all_nodes.append(SankeyNode(
                    node_id=node_id,
                    label=f"{name}\n{value:.1f} kW",
                    node_type=NodeType.LOSS,
                    value_kw=value,
                    color=CATEGORY_COLORS.get(category),
                    x_position=x_pos + 0.05,
                    y_position=y_pos
                ))
                all_links.append(SankeyLink(
                    source_id=stage_id,
                    target_id=node_id,
                    value_kw=value,
                    category=category
                ))
                y_pos += 0.1

        efficiency = (total_output / total_input * 100) if total_input > 0 else 0
        provenance = self._generate_provenance_hash({}, {}, {})
        timestamp = datetime.utcnow().isoformat() + "Z"

        return SankeyDiagram(
            title=title,
            nodes=all_nodes,
            links=all_links,
            total_input_kw=total_input,
            total_output_kw=total_output,
            total_losses_kw=total_losses,
            efficiency_percent=efficiency,
            provenance_hash=provenance,
            generated_timestamp=timestamp,
            generator_version=self.VERSION,
            metadata={"num_stages": num_stages}
        )

    def export_to_json(
        self,
        diagram: SankeyDiagram,
        file_path: Optional[str] = None
    ) -> str:
        """Export Sankey diagram to JSON.

        Args:
            diagram: SankeyDiagram to export
            file_path: Optional file path to write

        Returns:
            JSON string representation
        """
        json_str = json.dumps(diagram.to_dict(), indent=2)

        if file_path:
            with open(file_path, 'w') as f:
                f.write(json_str)

        return json_str

    def _generate_provenance_hash(
        self,
        inputs: Dict,
        outputs: Dict,
        losses: Dict
    ) -> str:
        """Generate SHA-256 provenance hash."""
        data = {
            "generator": "SankeyGenerator",
            "version": self.VERSION,
            "inputs": {k: v[0] for k, v in inputs.items()},
            "outputs": {k: v[0] for k, v in outputs.items()},
            "losses": {k: v[0] for k, v in losses.items()}
        }
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _round_value(self, value: float) -> float:
        """Round value to precision."""
        decimal_value = Decimal(str(value))
        quantize_str = '0.' + '0' * self.precision
        rounded = decimal_value.quantize(
            Decimal(quantize_str),
            rounding=ROUND_HALF_UP
        )
        return float(rounded)
