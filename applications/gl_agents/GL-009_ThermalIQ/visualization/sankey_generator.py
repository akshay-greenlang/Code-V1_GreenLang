"""
ThermalIQ Sankey Diagram Generator

Generates Sankey diagrams for energy and exergy flow visualization
in thermal systems including boilers, furnaces, and heat exchangers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from pathlib import Path
import json

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class NodeType(Enum):
    """Types of nodes in thermal Sankey diagrams."""
    FUEL_INPUT = "fuel_input"
    ELECTRICITY_INPUT = "electricity_input"
    RECOVERED_HEAT = "recovered_heat"
    PROCESS_HEAT = "process_heat"
    STEAM_OUTPUT = "steam_output"
    HOT_WATER = "hot_water"
    STACK_LOSS = "stack_loss"
    RADIATION_LOSS = "radiation_loss"
    BLOWDOWN_LOSS = "blowdown_loss"
    UNACCOUNTED_LOSS = "unaccounted_loss"
    INTERMEDIATE = "intermediate"
    EXERGY_DESTRUCTION = "exergy_destruction"


class ColorScheme(Enum):
    """Color schemes for Sankey diagrams."""
    THERMAL = "thermal"  # Red-orange for energy, blue for losses
    EXERGY = "exergy"    # Green for useful, red for destruction
    GREENLANG = "greenlang"  # GreenLang brand colors
    MONOCHROME = "monochrome"  # Grayscale
    CONTRAST = "contrast"  # High contrast for accessibility


@dataclass
class SankeyNode:
    """Represents a node in the Sankey diagram."""
    id: int
    label: str
    node_type: NodeType
    value: float = 0.0
    unit: str = "kW"
    color: Optional[str] = None
    x_position: Optional[float] = None  # 0-1 for horizontal position
    y_position: Optional[float] = None  # 0-1 for vertical position

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "node_type": self.node_type.value,
            "value": self.value,
            "unit": self.unit,
            "color": self.color
        }


@dataclass
class SankeyLink:
    """Represents a link (flow) in the Sankey diagram."""
    source_id: int
    target_id: int
    value: float
    unit: str = "kW"
    label: Optional[str] = None
    color: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source_id,
            "target": self.target_id,
            "value": self.value,
            "unit": self.unit,
            "label": self.label,
            "color": self.color
        }


@dataclass
class SankeyDiagram:
    """Complete Sankey diagram data structure."""
    title: str
    nodes: List[SankeyNode]
    links: List[SankeyLink]
    unit: str = "kW"
    color_scheme: ColorScheme = ColorScheme.THERMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "nodes": [n.to_dict() for n in self.nodes],
            "links": [l.to_dict() for l in self.links],
            "unit": self.unit,
            "color_scheme": self.color_scheme.value,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @property
    def total_input(self) -> float:
        """Calculate total energy input."""
        input_types = {NodeType.FUEL_INPUT, NodeType.ELECTRICITY_INPUT, NodeType.RECOVERED_HEAT}
        return sum(n.value for n in self.nodes if n.node_type in input_types)

    @property
    def total_output(self) -> float:
        """Calculate total useful output."""
        output_types = {NodeType.PROCESS_HEAT, NodeType.STEAM_OUTPUT, NodeType.HOT_WATER}
        return sum(n.value for n in self.nodes if n.node_type in output_types)

    @property
    def total_losses(self) -> float:
        """Calculate total losses."""
        loss_types = {
            NodeType.STACK_LOSS, NodeType.RADIATION_LOSS,
            NodeType.BLOWDOWN_LOSS, NodeType.UNACCOUNTED_LOSS
        }
        return sum(n.value for n in self.nodes if n.node_type in loss_types)


class SankeyDiagramGenerator:
    """
    Generates Sankey diagrams for thermal system energy flows.

    Creates interactive visualizations showing:
    - Energy inputs (fuel, electricity, recovered heat)
    - Useful outputs (process heat, steam, hot water)
    - Losses (stack, radiation, blowdown, unaccounted)
    """

    # Default colors for different node types
    DEFAULT_COLORS = {
        ColorScheme.THERMAL: {
            NodeType.FUEL_INPUT: "rgba(255, 127, 14, 0.8)",      # Orange
            NodeType.ELECTRICITY_INPUT: "rgba(255, 187, 120, 0.8)",  # Light orange
            NodeType.RECOVERED_HEAT: "rgba(44, 160, 44, 0.8)",   # Green
            NodeType.PROCESS_HEAT: "rgba(214, 39, 40, 0.8)",     # Red
            NodeType.STEAM_OUTPUT: "rgba(255, 152, 150, 0.8)",   # Light red
            NodeType.HOT_WATER: "rgba(148, 103, 189, 0.8)",      # Purple
            NodeType.STACK_LOSS: "rgba(31, 119, 180, 0.8)",      # Blue
            NodeType.RADIATION_LOSS: "rgba(174, 199, 232, 0.8)", # Light blue
            NodeType.BLOWDOWN_LOSS: "rgba(127, 127, 127, 0.8)",  # Gray
            NodeType.UNACCOUNTED_LOSS: "rgba(199, 199, 199, 0.8)",  # Light gray
            NodeType.INTERMEDIATE: "rgba(140, 86, 75, 0.8)",     # Brown
            NodeType.EXERGY_DESTRUCTION: "rgba(227, 119, 194, 0.8)"  # Pink
        },
        ColorScheme.EXERGY: {
            NodeType.FUEL_INPUT: "rgba(44, 160, 44, 0.8)",       # Green
            NodeType.ELECTRICITY_INPUT: "rgba(0, 128, 0, 0.8)",  # Dark green
            NodeType.RECOVERED_HEAT: "rgba(152, 223, 138, 0.8)", # Light green
            NodeType.PROCESS_HEAT: "rgba(44, 160, 44, 0.8)",     # Green
            NodeType.STEAM_OUTPUT: "rgba(152, 223, 138, 0.8)",   # Light green
            NodeType.HOT_WATER: "rgba(178, 223, 138, 0.8)",      # Pale green
            NodeType.STACK_LOSS: "rgba(214, 39, 40, 0.8)",       # Red
            NodeType.RADIATION_LOSS: "rgba(255, 152, 150, 0.8)", # Light red
            NodeType.BLOWDOWN_LOSS: "rgba(255, 127, 14, 0.8)",   # Orange
            NodeType.UNACCOUNTED_LOSS: "rgba(199, 199, 199, 0.8)",
            NodeType.INTERMEDIATE: "rgba(140, 86, 75, 0.8)",
            NodeType.EXERGY_DESTRUCTION: "rgba(214, 39, 40, 0.8)"  # Red
        },
        ColorScheme.GREENLANG: {
            NodeType.FUEL_INPUT: "rgba(46, 125, 50, 0.8)",       # GreenLang primary
            NodeType.ELECTRICITY_INPUT: "rgba(76, 175, 80, 0.8)", # GreenLang secondary
            NodeType.RECOVERED_HEAT: "rgba(129, 199, 132, 0.8)", # GreenLang light
            NodeType.PROCESS_HEAT: "rgba(46, 125, 50, 0.8)",
            NodeType.STEAM_OUTPUT: "rgba(76, 175, 80, 0.8)",
            NodeType.HOT_WATER: "rgba(129, 199, 132, 0.8)",
            NodeType.STACK_LOSS: "rgba(244, 67, 54, 0.8)",       # Red
            NodeType.RADIATION_LOSS: "rgba(255, 138, 128, 0.8)", # Light red
            NodeType.BLOWDOWN_LOSS: "rgba(255, 193, 7, 0.8)",    # Amber
            NodeType.UNACCOUNTED_LOSS: "rgba(158, 158, 158, 0.8)",
            NodeType.INTERMEDIATE: "rgba(97, 97, 97, 0.8)",
            NodeType.EXERGY_DESTRUCTION: "rgba(244, 67, 54, 0.8)"
        }
    }

    def __init__(
        self,
        default_color_scheme: ColorScheme = ColorScheme.THERMAL,
        default_unit: str = "kW"
    ):
        """
        Initialize the Sankey diagram generator.

        Args:
            default_color_scheme: Default color scheme for diagrams
            default_unit: Default unit for energy values
        """
        self.default_color_scheme = default_color_scheme
        self.default_unit = default_unit
        self._custom_colors: Dict[NodeType, str] = {}

    def generate_energy_sankey(
        self,
        heat_balance: Dict[str, float],
        title: str = "Energy Flow Diagram",
        unit: str = None
    ) -> SankeyDiagram:
        """
        Generate Sankey diagram from heat balance data.

        Args:
            heat_balance: Dictionary containing:
                - fuel_input: Heat from fuel combustion (kW)
                - electricity_input: Optional electrical input (kW)
                - recovered_heat: Optional recovered heat (kW)
                - process_heat: Useful heat to process (kW)
                - steam_output: Optional steam generation (kW)
                - hot_water: Optional hot water production (kW)
                - stack_loss: Flue gas losses (kW)
                - radiation_loss: Surface radiation losses (kW)
                - blowdown_loss: Optional blowdown losses (kW)
                - unaccounted: Unaccounted losses (kW)
            title: Diagram title
            unit: Energy unit (kW, MW, GJ/hr, etc.)

        Returns:
            SankeyDiagram with energy flow visualization
        """
        unit = unit or self.default_unit
        nodes = []
        links = []
        node_id = 0

        # Create input nodes
        input_nodes = {}

        if heat_balance.get('fuel_input', 0) > 0:
            nodes.append(SankeyNode(
                id=node_id,
                label=f"Fuel Input\n{heat_balance['fuel_input']:.1f} {unit}",
                node_type=NodeType.FUEL_INPUT,
                value=heat_balance['fuel_input'],
                unit=unit,
                x_position=0.0
            ))
            input_nodes['fuel'] = node_id
            node_id += 1

        if heat_balance.get('electricity_input', 0) > 0:
            nodes.append(SankeyNode(
                id=node_id,
                label=f"Electricity\n{heat_balance['electricity_input']:.1f} {unit}",
                node_type=NodeType.ELECTRICITY_INPUT,
                value=heat_balance['electricity_input'],
                unit=unit,
                x_position=0.0
            ))
            input_nodes['electricity'] = node_id
            node_id += 1

        if heat_balance.get('recovered_heat', 0) > 0:
            nodes.append(SankeyNode(
                id=node_id,
                label=f"Recovered Heat\n{heat_balance['recovered_heat']:.1f} {unit}",
                node_type=NodeType.RECOVERED_HEAT,
                value=heat_balance['recovered_heat'],
                unit=unit,
                x_position=0.0
            ))
            input_nodes['recovered'] = node_id
            node_id += 1

        # Create intermediate node (equipment)
        total_input = sum([
            heat_balance.get('fuel_input', 0),
            heat_balance.get('electricity_input', 0),
            heat_balance.get('recovered_heat', 0)
        ])

        equipment_node_id = node_id
        nodes.append(SankeyNode(
            id=node_id,
            label="Thermal\nEquipment",
            node_type=NodeType.INTERMEDIATE,
            value=total_input,
            unit=unit,
            x_position=0.5
        ))
        node_id += 1

        # Create output nodes
        output_nodes = {}

        if heat_balance.get('process_heat', 0) > 0:
            nodes.append(SankeyNode(
                id=node_id,
                label=f"Process Heat\n{heat_balance['process_heat']:.1f} {unit}",
                node_type=NodeType.PROCESS_HEAT,
                value=heat_balance['process_heat'],
                unit=unit,
                x_position=1.0
            ))
            output_nodes['process'] = node_id
            node_id += 1

        if heat_balance.get('steam_output', 0) > 0:
            nodes.append(SankeyNode(
                id=node_id,
                label=f"Steam\n{heat_balance['steam_output']:.1f} {unit}",
                node_type=NodeType.STEAM_OUTPUT,
                value=heat_balance['steam_output'],
                unit=unit,
                x_position=1.0
            ))
            output_nodes['steam'] = node_id
            node_id += 1

        if heat_balance.get('hot_water', 0) > 0:
            nodes.append(SankeyNode(
                id=node_id,
                label=f"Hot Water\n{heat_balance['hot_water']:.1f} {unit}",
                node_type=NodeType.HOT_WATER,
                value=heat_balance['hot_water'],
                unit=unit,
                x_position=1.0
            ))
            output_nodes['hot_water'] = node_id
            node_id += 1

        # Create loss nodes
        loss_nodes = {}

        if heat_balance.get('stack_loss', 0) > 0:
            nodes.append(SankeyNode(
                id=node_id,
                label=f"Stack Loss\n{heat_balance['stack_loss']:.1f} {unit}",
                node_type=NodeType.STACK_LOSS,
                value=heat_balance['stack_loss'],
                unit=unit,
                x_position=1.0
            ))
            loss_nodes['stack'] = node_id
            node_id += 1

        if heat_balance.get('radiation_loss', 0) > 0:
            nodes.append(SankeyNode(
                id=node_id,
                label=f"Radiation Loss\n{heat_balance['radiation_loss']:.1f} {unit}",
                node_type=NodeType.RADIATION_LOSS,
                value=heat_balance['radiation_loss'],
                unit=unit,
                x_position=1.0
            ))
            loss_nodes['radiation'] = node_id
            node_id += 1

        if heat_balance.get('blowdown_loss', 0) > 0:
            nodes.append(SankeyNode(
                id=node_id,
                label=f"Blowdown Loss\n{heat_balance['blowdown_loss']:.1f} {unit}",
                node_type=NodeType.BLOWDOWN_LOSS,
                value=heat_balance['blowdown_loss'],
                unit=unit,
                x_position=1.0
            ))
            loss_nodes['blowdown'] = node_id
            node_id += 1

        if heat_balance.get('unaccounted', 0) > 0:
            nodes.append(SankeyNode(
                id=node_id,
                label=f"Unaccounted\n{heat_balance['unaccounted']:.1f} {unit}",
                node_type=NodeType.UNACCOUNTED_LOSS,
                value=heat_balance['unaccounted'],
                unit=unit,
                x_position=1.0
            ))
            loss_nodes['unaccounted'] = node_id
            node_id += 1

        # Create links: inputs to equipment
        for name, source_id in input_nodes.items():
            input_value = nodes[source_id].value
            links.append(SankeyLink(
                source_id=source_id,
                target_id=equipment_node_id,
                value=input_value,
                unit=unit,
                label=f"{input_value:.1f} {unit}"
            ))

        # Create links: equipment to outputs
        for name, target_id in output_nodes.items():
            output_value = nodes[target_id].value
            links.append(SankeyLink(
                source_id=equipment_node_id,
                target_id=target_id,
                value=output_value,
                unit=unit,
                label=f"{output_value:.1f} {unit}"
            ))

        # Create links: equipment to losses
        for name, target_id in loss_nodes.items():
            loss_value = nodes[target_id].value
            links.append(SankeyLink(
                source_id=equipment_node_id,
                target_id=target_id,
                value=loss_value,
                unit=unit,
                label=f"{loss_value:.1f} {unit}"
            ))

        # Apply colors
        self._apply_colors(nodes, self.default_color_scheme)

        return SankeyDiagram(
            title=title,
            nodes=nodes,
            links=links,
            unit=unit,
            color_scheme=self.default_color_scheme,
            metadata={
                "type": "energy_balance",
                "total_input": total_input,
                "efficiency": heat_balance.get('efficiency')
            }
        )

    def generate_exergy_sankey(
        self,
        exergy_balance: Dict[str, float],
        title: str = "Exergy Flow Diagram",
        unit: str = None
    ) -> SankeyDiagram:
        """
        Generate Sankey diagram for exergy analysis.

        Args:
            exergy_balance: Dictionary containing:
                - fuel_exergy: Chemical exergy of fuel (kW)
                - thermal_exergy_in: Thermal exergy input (kW)
                - useful_exergy: Exergy delivered to process (kW)
                - stack_exergy_loss: Exergy in flue gases (kW)
                - surface_exergy_loss: Exergy lost from surfaces (kW)
                - combustion_destruction: Exergy destroyed in combustion (kW)
                - heat_transfer_destruction: Exergy destroyed in heat transfer (kW)
                - mixing_destruction: Exergy destroyed in mixing (kW)
                - friction_destruction: Exergy destroyed by friction (kW)
            title: Diagram title
            unit: Energy unit

        Returns:
            SankeyDiagram with exergy flow visualization
        """
        unit = unit or self.default_unit
        nodes = []
        links = []
        node_id = 0

        # Input exergy
        fuel_exergy = exergy_balance.get('fuel_exergy', 0)
        thermal_exergy = exergy_balance.get('thermal_exergy_in', 0)
        total_exergy_in = fuel_exergy + thermal_exergy

        if fuel_exergy > 0:
            nodes.append(SankeyNode(
                id=node_id,
                label=f"Fuel Exergy\n{fuel_exergy:.1f} {unit}",
                node_type=NodeType.FUEL_INPUT,
                value=fuel_exergy,
                unit=unit,
                x_position=0.0
            ))
            fuel_node = node_id
            node_id += 1
        else:
            fuel_node = None

        if thermal_exergy > 0:
            nodes.append(SankeyNode(
                id=node_id,
                label=f"Thermal Exergy In\n{thermal_exergy:.1f} {unit}",
                node_type=NodeType.RECOVERED_HEAT,
                value=thermal_exergy,
                unit=unit,
                x_position=0.0
            ))
            thermal_node = node_id
            node_id += 1
        else:
            thermal_node = None

        # Process node
        process_node_id = node_id
        nodes.append(SankeyNode(
            id=node_id,
            label="Thermal\nProcess",
            node_type=NodeType.INTERMEDIATE,
            value=total_exergy_in,
            unit=unit,
            x_position=0.5
        ))
        node_id += 1

        # Useful exergy output
        useful_exergy = exergy_balance.get('useful_exergy', 0)
        if useful_exergy > 0:
            nodes.append(SankeyNode(
                id=node_id,
                label=f"Useful Exergy\n{useful_exergy:.1f} {unit}",
                node_type=NodeType.PROCESS_HEAT,
                value=useful_exergy,
                unit=unit,
                x_position=1.0
            ))
            useful_node = node_id
            node_id += 1
        else:
            useful_node = None

        # Exergy losses
        stack_loss = exergy_balance.get('stack_exergy_loss', 0)
        if stack_loss > 0:
            nodes.append(SankeyNode(
                id=node_id,
                label=f"Stack Exergy Loss\n{stack_loss:.1f} {unit}",
                node_type=NodeType.STACK_LOSS,
                value=stack_loss,
                unit=unit,
                x_position=1.0
            ))
            stack_node = node_id
            node_id += 1
        else:
            stack_node = None

        surface_loss = exergy_balance.get('surface_exergy_loss', 0)
        if surface_loss > 0:
            nodes.append(SankeyNode(
                id=node_id,
                label=f"Surface Exergy Loss\n{surface_loss:.1f} {unit}",
                node_type=NodeType.RADIATION_LOSS,
                value=surface_loss,
                unit=unit,
                x_position=1.0
            ))
            surface_node = node_id
            node_id += 1
        else:
            surface_node = None

        # Exergy destruction components
        destructions = {
            'combustion': exergy_balance.get('combustion_destruction', 0),
            'heat_transfer': exergy_balance.get('heat_transfer_destruction', 0),
            'mixing': exergy_balance.get('mixing_destruction', 0),
            'friction': exergy_balance.get('friction_destruction', 0)
        }

        destruction_nodes = {}
        for name, value in destructions.items():
            if value > 0:
                label_name = name.replace('_', ' ').title()
                nodes.append(SankeyNode(
                    id=node_id,
                    label=f"{label_name}\nDestruction\n{value:.1f} {unit}",
                    node_type=NodeType.EXERGY_DESTRUCTION,
                    value=value,
                    unit=unit,
                    x_position=1.0
                ))
                destruction_nodes[name] = node_id
                node_id += 1

        # Create links
        if fuel_node is not None:
            links.append(SankeyLink(
                source_id=fuel_node,
                target_id=process_node_id,
                value=fuel_exergy,
                unit=unit
            ))

        if thermal_node is not None:
            links.append(SankeyLink(
                source_id=thermal_node,
                target_id=process_node_id,
                value=thermal_exergy,
                unit=unit
            ))

        if useful_node is not None:
            links.append(SankeyLink(
                source_id=process_node_id,
                target_id=useful_node,
                value=useful_exergy,
                unit=unit
            ))

        if stack_node is not None:
            links.append(SankeyLink(
                source_id=process_node_id,
                target_id=stack_node,
                value=stack_loss,
                unit=unit
            ))

        if surface_node is not None:
            links.append(SankeyLink(
                source_id=process_node_id,
                target_id=surface_node,
                value=surface_loss,
                unit=unit
            ))

        for name, target_id in destruction_nodes.items():
            links.append(SankeyLink(
                source_id=process_node_id,
                target_id=target_id,
                value=destructions[name],
                unit=unit
            ))

        # Apply exergy color scheme
        self._apply_colors(nodes, ColorScheme.EXERGY)

        total_destruction = sum(destructions.values())

        return SankeyDiagram(
            title=title,
            nodes=nodes,
            links=links,
            unit=unit,
            color_scheme=ColorScheme.EXERGY,
            metadata={
                "type": "exergy_balance",
                "total_exergy_in": total_exergy_in,
                "useful_exergy": useful_exergy,
                "total_destruction": total_destruction,
                "exergy_efficiency": (useful_exergy / total_exergy_in * 100)
                    if total_exergy_in > 0 else 0
            }
        )

    def customize_colors(self, scheme: ColorScheme) -> None:
        """
        Set the color scheme for subsequent diagrams.

        Args:
            scheme: Color scheme to use
        """
        self.default_color_scheme = scheme

    def set_custom_color(self, node_type: NodeType, color: str) -> None:
        """
        Set a custom color for a specific node type.

        Args:
            node_type: Type of node to customize
            color: CSS color string (e.g., "rgba(255, 0, 0, 0.8)")
        """
        self._custom_colors[node_type] = color

    def _apply_colors(self, nodes: List[SankeyNode], scheme: ColorScheme) -> None:
        """Apply colors to nodes based on scheme."""
        colors = self.DEFAULT_COLORS.get(scheme, self.DEFAULT_COLORS[ColorScheme.THERMAL])

        for node in nodes:
            if node.node_type in self._custom_colors:
                node.color = self._custom_colors[node.node_type]
            elif node.node_type in colors:
                node.color = colors[node.node_type]
            else:
                node.color = "rgba(127, 127, 127, 0.8)"

    def export_svg(self, diagram: SankeyDiagram, path: Union[str, Path]) -> None:
        """
        Export Sankey diagram to SVG format.

        Args:
            diagram: SankeyDiagram to export
            path: Output file path
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for SVG export")

        fig = self.interactive_plotly(diagram)
        fig.write_image(str(path), format='svg')

    def export_png(
        self,
        diagram: SankeyDiagram,
        path: Union[str, Path],
        width: int = 1200,
        height: int = 800,
        scale: float = 2.0
    ) -> None:
        """
        Export Sankey diagram to PNG format.

        Args:
            diagram: SankeyDiagram to export
            path: Output file path
            width: Image width in pixels
            height: Image height in pixels
            scale: Scale factor for resolution
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for PNG export")

        fig = self.interactive_plotly(diagram)
        fig.write_image(
            str(path),
            format='png',
            width=width,
            height=height,
            scale=scale
        )

    def interactive_plotly(self, diagram: SankeyDiagram) -> "go.Figure":
        """
        Generate interactive Plotly Sankey diagram.

        Args:
            diagram: SankeyDiagram to visualize

        Returns:
            Plotly Figure with interactive Sankey diagram
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is required for interactive diagrams. "
                "Install with: pip install plotly"
            )

        # Prepare node data
        node_labels = [n.label for n in diagram.nodes]
        node_colors = [n.color or "rgba(127, 127, 127, 0.8)" for n in diagram.nodes]

        # Node positions (if specified)
        node_x = [n.x_position for n in diagram.nodes if n.x_position is not None]
        node_y = [n.y_position for n in diagram.nodes if n.y_position is not None]

        # Prepare link data
        link_sources = [l.source_id for l in diagram.links]
        link_targets = [l.target_id for l in diagram.links]
        link_values = [l.value for l in diagram.links]

        # Link colors - make semi-transparent versions of source node colors
        link_colors = []
        for link in diagram.links:
            source_color = diagram.nodes[link.source_id].color or "rgba(127, 127, 127, 0.8)"
            # Make more transparent
            if 'rgba' in source_color:
                parts = source_color.replace('rgba(', '').replace(')', '').split(',')
                if len(parts) >= 3:
                    link_colors.append(f"rgba({parts[0]},{parts[1]},{parts[2]}, 0.4)")
                else:
                    link_colors.append(source_color)
            else:
                link_colors.append(source_color)

        # Link labels for hover
        link_labels = [
            f"{diagram.nodes[l.source_id].label.split(chr(10))[0]} -> "
            f"{diagram.nodes[l.target_id].label.split(chr(10))[0]}: "
            f"{l.value:.1f} {diagram.unit}"
            for l in diagram.links
        ]

        # Create Sankey trace
        sankey_trace = go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors,
                hovertemplate="<b>%{label}</b><extra></extra>"
            ),
            link=dict(
                source=link_sources,
                target=link_targets,
                value=link_values,
                color=link_colors,
                label=link_labels,
                hovertemplate="<b>%{label}</b><extra></extra>"
            )
        )

        # Create figure
        fig = go.Figure(data=[sankey_trace])

        # Calculate efficiency for subtitle
        if diagram.total_input > 0:
            efficiency = diagram.total_output / diagram.total_input * 100
            subtitle = f"Efficiency: {efficiency:.1f}% | Total Input: {diagram.total_input:.1f} {diagram.unit}"
        else:
            subtitle = f"Total Input: {diagram.total_input:.1f} {diagram.unit}"

        fig.update_layout(
            title=dict(
                text=f"{diagram.title}<br><sub>{subtitle}</sub>",
                font=dict(size=18)
            ),
            font=dict(size=12, family="Arial"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=600,
            margin=dict(l=50, r=50, t=100, b=50),
            annotations=[
                dict(
                    text="Energy flows in " + diagram.unit,
                    x=0.5,
                    y=-0.1,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="gray")
                )
            ]
        )

        return fig

    def create_combined_diagram(
        self,
        energy_balance: Dict[str, float],
        exergy_balance: Dict[str, float],
        title: str = "Energy and Exergy Flow Comparison"
    ) -> "go.Figure":
        """
        Create a combined figure with both energy and exergy Sankey diagrams.

        Args:
            energy_balance: Energy balance data
            exergy_balance: Exergy balance data
            title: Combined diagram title

        Returns:
            Plotly Figure with side-by-side Sankey diagrams
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        from plotly.subplots import make_subplots

        # Generate individual diagrams
        energy_diagram = self.generate_energy_sankey(
            energy_balance, "Energy Balance"
        )
        exergy_diagram = self.generate_exergy_sankey(
            exergy_balance, "Exergy Balance"
        )

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Energy Balance (First Law)", "Exergy Balance (Second Law)"),
            specs=[[{"type": "sankey"}, {"type": "sankey"}]]
        )

        # Add energy Sankey
        energy_fig = self.interactive_plotly(energy_diagram)
        for trace in energy_fig.data:
            fig.add_trace(trace, row=1, col=1)

        # Add exergy Sankey
        exergy_fig = self.interactive_plotly(exergy_diagram)
        for trace in exergy_fig.data:
            fig.add_trace(trace, row=1, col=2)

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            height=700,
            showlegend=False
        )

        return fig
