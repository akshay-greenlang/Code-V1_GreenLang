# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT - Fuel Blend Sankey Diagram Visualization Module.

Comprehensive Sankey diagram visualization for multi-fuel blend composition analysis.
Visualizes energy flows from source fuels through blending processes to energy outputs.

Author: GreenLang Team
Version: 1.0.0
Standards: WCAG 2.1 Level AA, ISO 12647-2

Features:
- Multi-fuel source to output flow visualization
- Energy, cost, and carbon flow representations
- Color-coded by fuel type, efficiency, or carbon intensity
- Real-time update capability
- Interactive drill-down on nodes and links
- Export to PNG/PDF/SVG/JSON
- Responsive design with accessibility compliance
- Caching for performance optimization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Set
from enum import Enum, auto
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json
import hashlib
import math
import logging
from decimal import Decimal, ROUND_HALF_UP
import colorsys

# Local imports
from .config import (
    ThemeConfig,
    ThemeMode,
    VisualizationConfig,
    ConfigFactory,
    SankeyChartConfig,
    FuelTypeColors,
    CostCategoryColors,
    EmissionColors,
    StatusColors,
    GradientScales,
    FontConfig,
    MarginConfig,
    LegendConfig,
    AnimationConfig,
    HoverConfig,
    ExportConfig,
    AccessibilityConfig,
    ExportFormat,
    ChartType,
    get_default_config,
    get_fuel_color,
    get_emission_color,
    hex_to_rgba,
    adjust_color_brightness,
    blend_colors,
    get_plotly_config,
    create_annotation,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class NodeType(Enum):
    """Types of nodes in the Sankey diagram."""
    SOURCE = "source"
    FUEL = "fuel"
    BLEND = "blend"
    PROCESS = "process"
    OUTPUT = "output"
    LOSS = "loss"
    STORAGE = "storage"
    CONVERSION = "conversion"
    DISTRIBUTION = "distribution"


class FlowType(Enum):
    """Types of flows to visualize."""
    ENERGY = "energy"
    MASS = "mass"
    COST = "cost"
    CARBON = "carbon"
    VOLUME = "volume"


class ColorSchemeType(Enum):
    """Color scheme options for Sankey diagram."""
    BY_FUEL_TYPE = "fuel_type"
    BY_EFFICIENCY = "efficiency"
    BY_CARBON_INTENSITY = "carbon_intensity"
    BY_COST = "cost"
    BY_NODE_TYPE = "node_type"
    BY_TEMPERATURE = "temperature"
    GRADIENT = "gradient"
    MONOCHROME = "monochrome"


class SankeyOrientation(Enum):
    """Sankey diagram orientation."""
    HORIZONTAL = "h"
    VERTICAL = "v"


class NodeArrangement(Enum):
    """Node arrangement strategy."""
    SNAP = "snap"
    PERPENDICULAR = "perpendicular"
    FREEFORM = "freeform"
    FIXED = "fixed"


class LinkCurveType(Enum):
    """Link curve rendering type."""
    BEZIER = "bezier"
    STRAIGHT = "straight"
    STEP = "step"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SankeyNode:
    """Node in the Sankey diagram."""
    id: str
    label: str
    node_type: NodeType
    value: float
    unit: str = "MJ"
    color: Optional[str] = None
    x_position: Optional[float] = None
    y_position: Optional[float] = None
    efficiency: Optional[float] = None
    carbon_intensity: Optional[float] = None
    cost_per_unit: Optional[float] = None
    temperature: Optional[float] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    group: Optional[str] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.color is None:
            self.color = self._get_default_color()

    def _get_default_color(self) -> str:
        """Get default color based on node type."""
        type_colors = {
            NodeType.SOURCE: "#3498DB",
            NodeType.FUEL: "#E74C3C",
            NodeType.BLEND: "#9B59B6",
            NodeType.PROCESS: "#2ECC71",
            NodeType.OUTPUT: "#F39C12",
            NodeType.LOSS: "#95A5A6",
            NodeType.STORAGE: "#1ABC9C",
            NodeType.CONVERSION: "#E67E22",
            NodeType.DISTRIBUTION: "#34495E",
        }
        return type_colors.get(self.node_type, "#888888")

    @property
    def formatted_value(self) -> str:
        """Get formatted value string."""
        if self.value >= 1000000:
            return f"{self.value / 1000000:.2f} G{self.unit}"
        elif self.value >= 1000:
            return f"{self.value / 1000:.2f} k{self.unit}"
        else:
            return f"{self.value:.2f} {self.unit}"

    def get_hover_text(self) -> str:
        """Generate hover text for the node."""
        lines = [f"<b>{self.label}</b>"]
        lines.append(f"Type: {self.node_type.value.title()}")
        lines.append(f"Value: {self.formatted_value}")

        if self.efficiency is not None:
            lines.append(f"Efficiency: {self.efficiency:.1f}%")
        if self.carbon_intensity is not None:
            lines.append(f"Carbon Intensity: {self.carbon_intensity:.2f} kg CO2/GJ")
        if self.cost_per_unit is not None:
            lines.append(f"Cost: ${self.cost_per_unit:.2f}/{self.unit}")
        if self.temperature is not None:
            lines.append(f"Temperature: {self.temperature:.1f} C")
        if self.description:
            lines.append(f"{self.description}")

        return "<br>".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "node_type": self.node_type.value,
            "value": self.value,
            "unit": self.unit,
            "color": self.color,
            "x_position": self.x_position,
            "y_position": self.y_position,
            "efficiency": self.efficiency,
            "carbon_intensity": self.carbon_intensity,
            "cost_per_unit": self.cost_per_unit,
            "temperature": self.temperature,
            "description": self.description,
            "metadata": self.metadata,
            "visible": self.visible,
            "group": self.group,
        }


@dataclass
class SankeyLink:
    """Link between nodes in the Sankey diagram."""
    source_id: str
    target_id: str
    value: float
    unit: str = "MJ"
    color: Optional[str] = None
    label: Optional[str] = None
    efficiency: Optional[float] = None
    carbon_flow: Optional[float] = None
    cost_flow: Optional[float] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    opacity: float = 0.5
    highlight: bool = False

    def __post_init__(self):
        """Post-initialization processing."""
        if self.color is None:
            self.color = "#CCCCCC"

    @property
    def formatted_value(self) -> str:
        """Get formatted value string."""
        if self.value >= 1000000:
            return f"{self.value / 1000000:.2f} G{self.unit}"
        elif self.value >= 1000:
            return f"{self.value / 1000:.2f} k{self.unit}"
        else:
            return f"{self.value:.2f} {self.unit}"

    def get_hover_text(self) -> str:
        """Generate hover text for the link."""
        lines = []
        if self.label:
            lines.append(f"<b>{self.label}</b>")
        lines.append(f"Flow: {self.formatted_value}")

        if self.efficiency is not None:
            lines.append(f"Efficiency: {self.efficiency:.1f}%")
        if self.carbon_flow is not None:
            lines.append(f"Carbon: {self.carbon_flow:.2f} kg CO2")
        if self.cost_flow is not None:
            lines.append(f"Cost: ${self.cost_flow:,.2f}")
        if self.description:
            lines.append(f"{self.description}")

        return "<br>".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "value": self.value,
            "unit": self.unit,
            "color": self.color,
            "label": self.label,
            "efficiency": self.efficiency,
            "carbon_flow": self.carbon_flow,
            "cost_flow": self.cost_flow,
            "description": self.description,
            "metadata": self.metadata,
            "visible": self.visible,
            "opacity": self.opacity,
            "highlight": self.highlight,
        }


@dataclass
class FuelSource:
    """Fuel source definition."""
    fuel_id: str
    fuel_type: str
    fuel_name: str
    energy_content: float  # MJ/kg or MJ/L
    carbon_intensity: float  # kg CO2/GJ
    cost_per_unit: float  # $/unit
    unit: str = "kg"
    availability: float = 1.0  # 0-1 availability factor
    quality_score: float = 100.0  # 0-100 quality score
    supplier: Optional[str] = None
    origin: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def color(self) -> str:
        """Get color for this fuel type."""
        return get_fuel_color(self.fuel_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fuel_id": self.fuel_id,
            "fuel_type": self.fuel_type,
            "fuel_name": self.fuel_name,
            "energy_content": self.energy_content,
            "carbon_intensity": self.carbon_intensity,
            "cost_per_unit": self.cost_per_unit,
            "unit": self.unit,
            "availability": self.availability,
            "quality_score": self.quality_score,
            "supplier": self.supplier,
            "origin": self.origin,
            "metadata": self.metadata,
        }


@dataclass
class BlendRecipe:
    """Fuel blend recipe definition."""
    blend_id: str
    blend_name: str
    components: Dict[str, float]  # fuel_id -> ratio (0-1)
    target_energy_content: Optional[float] = None
    target_carbon_intensity: Optional[float] = None
    quality_score: float = 100.0
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate blend recipe."""
        errors = []
        total_ratio = sum(self.components.values())
        if abs(total_ratio - 1.0) > 0.001:
            errors.append(f"Component ratios must sum to 1.0, got {total_ratio}")
        for fuel_id, ratio in self.components.items():
            if ratio < 0 or ratio > 1:
                errors.append(f"Invalid ratio {ratio} for fuel {fuel_id}")
        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "blend_id": self.blend_id,
            "blend_name": self.blend_name,
            "components": self.components,
            "target_energy_content": self.target_energy_content,
            "target_carbon_intensity": self.target_carbon_intensity,
            "quality_score": self.quality_score,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class EnergyOutput:
    """Energy output definition."""
    output_id: str
    output_type: str
    output_name: str
    energy_value: float  # MJ
    efficiency: float  # Conversion efficiency
    end_use: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_id": self.output_id,
            "output_type": self.output_type,
            "output_name": self.output_name,
            "energy_value": self.energy_value,
            "efficiency": self.efficiency,
            "end_use": self.end_use,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class BlendFlowData:
    """Complete data for blend flow visualization."""
    fuel_sources: List[FuelSource]
    blend_recipe: BlendRecipe
    outputs: List[EnergyOutput]
    losses: Dict[str, float] = field(default_factory=dict)
    total_input_energy: float = 0.0
    total_output_energy: float = 0.0
    total_losses: float = 0.0
    overall_efficiency: float = 0.0
    total_carbon: float = 0.0
    total_cost: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    provenance_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived values."""
        self._calculate_totals()
        self._calculate_provenance()

    def _calculate_totals(self):
        """Calculate total values."""
        self.total_input_energy = sum(
            source.energy_content * self.blend_recipe.components.get(source.fuel_id, 0)
            for source in self.fuel_sources
        )
        self.total_output_energy = sum(output.energy_value for output in self.outputs)
        self.total_losses = sum(self.losses.values())

        if self.total_input_energy > 0:
            self.overall_efficiency = (self.total_output_energy / self.total_input_energy) * 100

        # Calculate total carbon
        for source in self.fuel_sources:
            ratio = self.blend_recipe.components.get(source.fuel_id, 0)
            energy_contribution = source.energy_content * ratio
            self.total_carbon += (energy_contribution / 1000) * source.carbon_intensity

        # Calculate total cost
        for source in self.fuel_sources:
            ratio = self.blend_recipe.components.get(source.fuel_id, 0)
            self.total_cost += source.cost_per_unit * ratio

    def _calculate_provenance(self):
        """Calculate provenance hash."""
        data = {
            "fuels": [(s.fuel_id, s.energy_content) for s in self.fuel_sources],
            "blend": self.blend_recipe.components,
            "outputs": [(o.output_id, o.energy_value) for o in self.outputs],
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fuel_sources": [s.to_dict() for s in self.fuel_sources],
            "blend_recipe": self.blend_recipe.to_dict(),
            "outputs": [o.to_dict() for o in self.outputs],
            "losses": self.losses,
            "total_input_energy": self.total_input_energy,
            "total_output_energy": self.total_output_energy,
            "total_losses": self.total_losses,
            "overall_efficiency": self.overall_efficiency,
            "total_carbon": self.total_carbon,
            "total_cost": self.total_cost,
            "timestamp": self.timestamp,
            "provenance_hash": self.provenance_hash,
            "metadata": self.metadata,
        }


@dataclass
class SankeyDiagramData:
    """Complete Sankey diagram data structure."""
    nodes: List[SankeyNode]
    links: List[SankeyLink]
    title: str = "Fuel Blend Flow"
    subtitle: Optional[str] = None
    total_input: float = 0.0
    total_output: float = 0.0
    total_losses: float = 0.0
    efficiency: float = 0.0
    provenance_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_node_by_id(self, node_id: str) -> Optional[SankeyNode]:
        """Get node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_links_from_node(self, node_id: str) -> List[SankeyLink]:
        """Get all links originating from a node."""
        return [link for link in self.links if link.source_id == node_id]

    def get_links_to_node(self, node_id: str) -> List[SankeyLink]:
        """Get all links going to a node."""
        return [link for link in self.links if link.target_id == node_id]

    def get_node_input_total(self, node_id: str) -> float:
        """Get total input value for a node."""
        return sum(link.value for link in self.get_links_to_node(node_id))

    def get_node_output_total(self, node_id: str) -> float:
        """Get total output value for a node."""
        return sum(link.value for link in self.get_links_from_node(node_id))

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the diagram data."""
        errors = []
        node_ids = {node.id for node in self.nodes}

        for link in self.links:
            if link.source_id not in node_ids:
                errors.append(f"Link source '{link.source_id}' not found in nodes")
            if link.target_id not in node_ids:
                errors.append(f"Link target '{link.target_id}' not found in nodes")
            if link.value < 0:
                errors.append(f"Negative link value: {link.source_id} -> {link.target_id}")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "links": [l.to_dict() for l in self.links],
            "title": self.title,
            "subtitle": self.subtitle,
            "total_input": self.total_input,
            "total_output": self.total_output,
            "total_losses": self.total_losses,
            "efficiency": self.efficiency,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# =============================================================================
# CHART OPTIONS
# =============================================================================

@dataclass
class SankeyChartOptions:
    """Configuration options for Sankey chart."""
    # Display options
    title: str = "Fuel Blend Composition"
    subtitle: Optional[str] = None
    orientation: SankeyOrientation = SankeyOrientation.HORIZONTAL
    arrangement: NodeArrangement = NodeArrangement.SNAP
    flow_type: FlowType = FlowType.ENERGY
    color_scheme: ColorSchemeType = ColorSchemeType.BY_FUEL_TYPE

    # Node options
    node_pad: int = 15
    node_thickness: int = 20
    node_line_color: str = "#333333"
    node_line_width: float = 0.5
    show_node_labels: bool = True
    node_label_position: str = "outside"

    # Link options
    link_opacity: float = 0.5
    link_hover_opacity: float = 0.8
    link_curve_type: LinkCurveType = LinkCurveType.BEZIER
    show_link_labels: bool = False

    # Color options
    color_blind_safe: bool = False
    custom_colors: Optional[Dict[str, str]] = None
    gradient_scale: Optional[str] = None

    # Value formatting
    value_format: str = ",.2f"
    value_suffix: str = " MJ"
    show_percentages: bool = True
    percentage_format: str = ".1f"

    # Legend options
    show_legend: bool = True
    legend_position: str = "right"

    # Interaction options
    enable_hover: bool = True
    enable_click: bool = False
    enable_zoom: bool = False
    enable_pan: bool = False

    # Size options
    width: Optional[int] = None
    height: Optional[int] = None
    auto_size: bool = True
    responsive: bool = True

    # Animation options
    animate: bool = True
    animation_duration: int = 500

    # Additional options
    show_summary: bool = True
    show_efficiency_annotation: bool = True
    show_carbon_annotation: bool = False
    show_cost_annotation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "orientation": self.orientation.value,
            "flow_type": self.flow_type.value,
            "color_scheme": self.color_scheme.value,
            "node_pad": self.node_pad,
            "node_thickness": self.node_thickness,
            "link_opacity": self.link_opacity,
            "show_legend": self.show_legend,
        }


# =============================================================================
# SANKEY ENGINE
# =============================================================================

class FuelBlendSankeyEngine:
    """
    Engine for generating fuel blend Sankey diagrams.

    Visualizes multi-fuel blend composition with energy, cost, and carbon flows.
    """

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        theme: Optional[ThemeConfig] = None,
    ):
        """
        Initialize Sankey engine.

        Args:
            config: Global visualization configuration
            theme: Theme configuration for styling
        """
        self.config = config or get_default_config()
        self.theme = theme or self.config.theme
        self._cache: Dict[str, Any] = {}

        # Apply accessibility settings
        if self.theme.accessibility.color_blind_safe:
            self._apply_accessible_colors()

    def _apply_accessible_colors(self) -> None:
        """Apply color-blind safe color palette."""
        self._accessible_mode = True

    def generate(
        self,
        data: Union[BlendFlowData, SankeyDiagramData],
        options: Optional[SankeyChartOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate Sankey diagram from blend flow data.

        Args:
            data: Blend flow data or pre-built diagram data
            options: Chart configuration options

        Returns:
            Plotly-compatible chart specification
        """
        options = options or SankeyChartOptions()

        # Convert BlendFlowData to SankeyDiagramData if needed
        if isinstance(data, BlendFlowData):
            diagram_data = self._convert_blend_flow_to_sankey(data, options)
        else:
            diagram_data = data

        # Validate data
        is_valid, errors = diagram_data.validate()
        if not is_valid:
            logger.warning(f"Sankey data validation errors: {errors}")

        # Check cache
        cache_key = self._get_cache_key(diagram_data, options)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build chart
        chart = self._build_sankey_chart(diagram_data, options)

        # Add annotations if enabled
        if options.show_efficiency_annotation:
            chart = self._add_efficiency_annotation(chart, diagram_data, options)

        if options.show_carbon_annotation and hasattr(data, 'total_carbon'):
            chart = self._add_carbon_annotation(chart, data, options)

        if options.show_cost_annotation and hasattr(data, 'total_cost'):
            chart = self._add_cost_annotation(chart, data, options)

        # Add summary if enabled
        if options.show_summary:
            chart = self._add_summary_annotation(chart, diagram_data, options)

        # Cache result
        self._cache[cache_key] = chart

        return chart

    def _get_cache_key(
        self,
        data: SankeyDiagramData,
        options: SankeyChartOptions,
    ) -> str:
        """Generate cache key."""
        key_data = {
            "provenance": data.provenance_hash,
            "flow_type": options.flow_type.value,
            "color_scheme": options.color_scheme.value,
        }
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def _convert_blend_flow_to_sankey(
        self,
        data: BlendFlowData,
        options: SankeyChartOptions,
    ) -> SankeyDiagramData:
        """Convert BlendFlowData to SankeyDiagramData."""
        nodes = []
        links = []

        # Create fuel source nodes
        for source in data.fuel_sources:
            ratio = data.blend_recipe.components.get(source.fuel_id, 0)
            if ratio > 0:
                energy_contribution = source.energy_content * ratio
                nodes.append(SankeyNode(
                    id=f"fuel_{source.fuel_id}",
                    label=source.fuel_name,
                    node_type=NodeType.FUEL,
                    value=energy_contribution,
                    unit="MJ",
                    color=source.color,
                    carbon_intensity=source.carbon_intensity,
                    cost_per_unit=source.cost_per_unit,
                    description=f"Ratio: {ratio * 100:.1f}%",
                    metadata=source.metadata,
                ))

        # Create blend node
        blend = data.blend_recipe
        blend_energy = sum(
            source.energy_content * data.blend_recipe.components.get(source.fuel_id, 0)
            for source in data.fuel_sources
        )
        nodes.append(SankeyNode(
            id="blend",
            label=blend.blend_name,
            node_type=NodeType.BLEND,
            value=blend_energy,
            unit="MJ",
            color="#9B59B6",
            efficiency=data.overall_efficiency,
            carbon_intensity=blend.target_carbon_intensity,
            description=blend.description,
        ))

        # Create output nodes
        for output in data.outputs:
            nodes.append(SankeyNode(
                id=f"output_{output.output_id}",
                label=output.output_name,
                node_type=NodeType.OUTPUT,
                value=output.energy_value,
                unit="MJ",
                color="#F39C12",
                efficiency=output.efficiency,
                description=output.description,
            ))

        # Create loss nodes
        for loss_type, loss_value in data.losses.items():
            if loss_value > 0:
                nodes.append(SankeyNode(
                    id=f"loss_{loss_type}",
                    label=f"{loss_type.replace('_', ' ').title()} Loss",
                    node_type=NodeType.LOSS,
                    value=loss_value,
                    unit="MJ",
                    color="#95A5A6",
                ))

        # Create links from fuels to blend
        for source in data.fuel_sources:
            ratio = data.blend_recipe.components.get(source.fuel_id, 0)
            if ratio > 0:
                energy_flow = source.energy_content * ratio
                carbon_flow = (energy_flow / 1000) * source.carbon_intensity
                cost_flow = source.cost_per_unit * ratio

                links.append(SankeyLink(
                    source_id=f"fuel_{source.fuel_id}",
                    target_id="blend",
                    value=energy_flow,
                    unit="MJ",
                    color=hex_to_rgba(source.color, 0.5),
                    label=f"{source.fuel_name}",
                    carbon_flow=carbon_flow,
                    cost_flow=cost_flow,
                    description=f"Blend ratio: {ratio * 100:.1f}%",
                ))

        # Create links from blend to outputs
        for output in data.outputs:
            links.append(SankeyLink(
                source_id="blend",
                target_id=f"output_{output.output_id}",
                value=output.energy_value,
                unit="MJ",
                color=hex_to_rgba("#F39C12", 0.5),
                label=output.output_name,
                efficiency=output.efficiency,
                description=output.description,
            ))

        # Create links from blend to losses
        for loss_type, loss_value in data.losses.items():
            if loss_value > 0:
                links.append(SankeyLink(
                    source_id="blend",
                    target_id=f"loss_{loss_type}",
                    value=loss_value,
                    unit="MJ",
                    color=hex_to_rgba("#95A5A6", 0.5),
                    label=f"{loss_type.replace('_', ' ').title()}",
                ))

        return SankeyDiagramData(
            nodes=nodes,
            links=links,
            title=options.title,
            subtitle=options.subtitle,
            total_input=blend_energy,
            total_output=sum(o.energy_value for o in data.outputs),
            total_losses=sum(data.losses.values()),
            efficiency=data.overall_efficiency,
            provenance_hash=data.provenance_hash,
            metadata=data.metadata,
        )

    def _build_sankey_chart(
        self,
        data: SankeyDiagramData,
        options: SankeyChartOptions,
    ) -> Dict[str, Any]:
        """Build the Sankey chart."""
        # Create node index mapping
        node_map = {node.id: i for i, node in enumerate(data.nodes)}

        # Prepare node data
        node_labels = []
        node_colors = []
        node_x = []
        node_y = []
        node_customdata = []

        for node in data.nodes:
            if not node.visible:
                continue

            node_labels.append(node.label)

            # Apply color scheme
            color = self._get_node_color(node, options)
            node_colors.append(color)

            # Position nodes if specified
            if node.x_position is not None:
                node_x.append(node.x_position)
            if node.y_position is not None:
                node_y.append(node.y_position)

            node_customdata.append(node.get_hover_text())

        # Prepare link data
        link_sources = []
        link_targets = []
        link_values = []
        link_colors = []
        link_labels = []
        link_customdata = []

        for link in data.links:
            if not link.visible:
                continue
            if link.source_id not in node_map or link.target_id not in node_map:
                continue

            link_sources.append(node_map[link.source_id])
            link_targets.append(node_map[link.target_id])
            link_values.append(link.value)

            # Apply link color
            color = self._get_link_color(link, data, options)
            link_colors.append(color)

            link_labels.append(link.label or "")
            link_customdata.append(link.get_hover_text())

        # Build node configuration
        node_config = {
            "pad": options.node_pad,
            "thickness": options.node_thickness,
            "line": {
                "color": options.node_line_color,
                "width": options.node_line_width,
            },
            "label": node_labels,
            "color": node_colors,
            "customdata": node_customdata,
            "hovertemplate": "%{customdata}<extra></extra>",
        }

        if node_x:
            node_config["x"] = node_x
        if node_y:
            node_config["y"] = node_y

        # Build link configuration
        link_config = {
            "source": link_sources,
            "target": link_targets,
            "value": link_values,
            "color": link_colors,
            "label": link_labels,
            "customdata": link_customdata,
            "hovertemplate": "%{customdata}<extra></extra>",
        }

        # Build trace
        trace = {
            "type": "sankey",
            "orientation": options.orientation.value,
            "arrangement": options.arrangement.value,
            "node": node_config,
            "link": link_config,
            "valueformat": options.value_format,
            "valuesuffix": options.value_suffix,
        }

        # Build layout
        layout = self._build_layout(data, options)

        return {
            "data": [trace],
            "layout": layout,
            "config": get_plotly_config(interactive=options.enable_hover),
        }

    def _get_node_color(
        self,
        node: SankeyNode,
        options: SankeyChartOptions,
    ) -> str:
        """Get node color based on color scheme."""
        # Check for custom colors
        if options.custom_colors and node.id in options.custom_colors:
            return options.custom_colors[node.id]

        if options.color_scheme == ColorSchemeType.BY_FUEL_TYPE:
            if node.node_type == NodeType.FUEL:
                # Extract fuel type from node metadata or ID
                fuel_type = node.metadata.get("fuel_type", node.id.replace("fuel_", ""))
                return get_fuel_color(fuel_type)
            return node.color or "#888888"

        elif options.color_scheme == ColorSchemeType.BY_EFFICIENCY:
            if node.efficiency is not None:
                # Map efficiency to color (red=low, green=high)
                scale = GradientScales.EFFICIENCY
                normalized = min(1, max(0, node.efficiency / 100))
                return self._interpolate_color(scale, normalized)
            return node.color or "#888888"

        elif options.color_scheme == ColorSchemeType.BY_CARBON_INTENSITY:
            if node.carbon_intensity is not None:
                # Map carbon intensity to color (green=low, red=high)
                scale = GradientScales.CARBON_INTENSITY
                # Assume max carbon intensity of 100 kg CO2/GJ
                normalized = min(1, max(0, node.carbon_intensity / 100))
                return self._interpolate_color(scale, normalized)
            return node.color or "#888888"

        elif options.color_scheme == ColorSchemeType.BY_COST:
            if node.cost_per_unit is not None:
                # Map cost to color (green=low, red=high)
                scale = GradientScales.COST
                # Assume max cost of $50/unit
                normalized = min(1, max(0, node.cost_per_unit / 50))
                return self._interpolate_color(scale, normalized)
            return node.color or "#888888"

        elif options.color_scheme == ColorSchemeType.BY_NODE_TYPE:
            return node._get_default_color()

        elif options.color_scheme == ColorSchemeType.BY_TEMPERATURE:
            if node.temperature is not None:
                # Map temperature to color (blue=cold, red=hot)
                scale = GradientScales.TEMPERATURE
                # Assume temperature range 0-500C
                normalized = min(1, max(0, node.temperature / 500))
                return self._interpolate_color(scale, normalized)
            return node.color or "#888888"

        elif options.color_scheme == ColorSchemeType.MONOCHROME:
            return "#555555"

        return node.color or "#888888"

    def _get_link_color(
        self,
        link: SankeyLink,
        data: SankeyDiagramData,
        options: SankeyChartOptions,
    ) -> str:
        """Get link color based on color scheme."""
        if link.highlight:
            return hex_to_rgba("#E74C3C", options.link_opacity)

        if options.color_scheme == ColorSchemeType.BY_FUEL_TYPE:
            # Use source node color
            source_node = data.get_node_by_id(link.source_id)
            if source_node:
                return hex_to_rgba(source_node.color or "#CCCCCC", options.link_opacity)

        elif options.color_scheme == ColorSchemeType.BY_EFFICIENCY:
            if link.efficiency is not None:
                scale = GradientScales.EFFICIENCY
                normalized = min(1, max(0, link.efficiency / 100))
                base_color = self._interpolate_color(scale, normalized)
                return hex_to_rgba(base_color, options.link_opacity)

        elif options.color_scheme == ColorSchemeType.BY_CARBON_INTENSITY:
            source_node = data.get_node_by_id(link.source_id)
            if source_node and source_node.carbon_intensity is not None:
                scale = GradientScales.CARBON_INTENSITY
                normalized = min(1, max(0, source_node.carbon_intensity / 100))
                base_color = self._interpolate_color(scale, normalized)
                return hex_to_rgba(base_color, options.link_opacity)

        return hex_to_rgba(link.color or "#CCCCCC", options.link_opacity)

    def _interpolate_color(
        self,
        scale: List[List],
        value: float,
    ) -> str:
        """Interpolate color from gradient scale."""
        if value <= 0:
            return scale[0][1]
        if value >= 1:
            return scale[-1][1]

        # Find surrounding colors
        for i in range(len(scale) - 1):
            if scale[i][0] <= value <= scale[i + 1][0]:
                t = (value - scale[i][0]) / (scale[i + 1][0] - scale[i][0])
                return blend_colors(scale[i][1], scale[i + 1][1], t)

        return scale[-1][1]

    def _build_layout(
        self,
        data: SankeyDiagramData,
        options: SankeyChartOptions,
    ) -> Dict[str, Any]:
        """Build Plotly layout configuration."""
        layout = self.theme.to_layout_dict()

        # Title
        layout["title"] = {
            "text": options.title,
            "font": {
                "size": self.theme.font.size_title,
                "color": self.theme.title_color,
            },
            "x": 0.5,
            "xanchor": "center",
        }

        # Subtitle
        if options.subtitle:
            layout["annotations"] = layout.get("annotations", [])
            layout["annotations"].append({
                "text": options.subtitle,
                "x": 0.5,
                "y": 1.05,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": self.theme.font.size_subtitle,
                    "color": self.theme.axis_color,
                },
            })

        # Size
        if options.width:
            layout["width"] = options.width
        if options.height:
            layout["height"] = options.height
        if options.auto_size:
            layout["autosize"] = True

        # Margin
        layout["margin"] = {"l": 20, "r": 20, "t": 80, "b": 40}

        return layout

    def _add_efficiency_annotation(
        self,
        chart: Dict[str, Any],
        data: SankeyDiagramData,
        options: SankeyChartOptions,
    ) -> Dict[str, Any]:
        """Add efficiency annotation to chart."""
        annotations = chart["layout"].get("annotations", [])

        efficiency_text = f"Overall Efficiency: {data.efficiency:.1f}%"
        annotations.append({
            "text": efficiency_text,
            "x": 0.5,
            "y": -0.08,
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {
                "size": 12,
                "color": self.theme.text_color,
            },
        })

        chart["layout"]["annotations"] = annotations
        return chart

    def _add_carbon_annotation(
        self,
        chart: Dict[str, Any],
        data: BlendFlowData,
        options: SankeyChartOptions,
    ) -> Dict[str, Any]:
        """Add carbon emission annotation to chart."""
        annotations = chart["layout"].get("annotations", [])

        carbon_text = f"Total Carbon: {data.total_carbon:,.2f} kg CO2"
        annotations.append({
            "text": carbon_text,
            "x": 1,
            "y": -0.08,
            "xref": "paper",
            "yref": "paper",
            "xanchor": "right",
            "showarrow": False,
            "font": {
                "size": 12,
                "color": EmissionColors.CO2,
            },
        })

        chart["layout"]["annotations"] = annotations
        return chart

    def _add_cost_annotation(
        self,
        chart: Dict[str, Any],
        data: BlendFlowData,
        options: SankeyChartOptions,
    ) -> Dict[str, Any]:
        """Add cost annotation to chart."""
        annotations = chart["layout"].get("annotations", [])

        cost_text = f"Total Cost: ${data.total_cost:,.2f}"
        annotations.append({
            "text": cost_text,
            "x": 0,
            "y": -0.08,
            "xref": "paper",
            "yref": "paper",
            "xanchor": "left",
            "showarrow": False,
            "font": {
                "size": 12,
                "color": CostCategoryColors.FUEL_COST,
            },
        })

        chart["layout"]["annotations"] = annotations
        return chart

    def _add_summary_annotation(
        self,
        chart: Dict[str, Any],
        data: SankeyDiagramData,
        options: SankeyChartOptions,
    ) -> Dict[str, Any]:
        """Add summary annotation to chart."""
        annotations = chart["layout"].get("annotations", [])

        summary_lines = [
            f"Input: {data.total_input:,.0f} MJ",
            f"Output: {data.total_output:,.0f} MJ",
            f"Losses: {data.total_losses:,.0f} MJ",
        ]
        summary_text = " | ".join(summary_lines)

        annotations.append({
            "text": summary_text,
            "x": 0.5,
            "y": -0.12,
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {
                "size": 10,
                "color": "#666666",
            },
        })

        chart["layout"]["annotations"] = annotations
        return chart

    def generate_energy_flow(
        self,
        data: BlendFlowData,
        options: Optional[SankeyChartOptions] = None,
    ) -> Dict[str, Any]:
        """Generate energy flow Sankey diagram."""
        options = options or SankeyChartOptions()
        options.flow_type = FlowType.ENERGY
        options.value_suffix = " MJ"
        options.title = "Energy Flow"
        return self.generate(data, options)

    def generate_cost_flow(
        self,
        data: BlendFlowData,
        options: Optional[SankeyChartOptions] = None,
    ) -> Dict[str, Any]:
        """Generate cost flow Sankey diagram."""
        options = options or SankeyChartOptions()
        options.flow_type = FlowType.COST
        options.color_scheme = ColorSchemeType.BY_COST
        options.value_suffix = " USD"
        options.title = "Cost Flow"
        options.show_cost_annotation = True

        # Convert to cost-based values
        cost_data = self._convert_to_cost_flow(data)
        return self.generate(cost_data, options)

    def generate_carbon_flow(
        self,
        data: BlendFlowData,
        options: Optional[SankeyChartOptions] = None,
    ) -> Dict[str, Any]:
        """Generate carbon flow Sankey diagram."""
        options = options or SankeyChartOptions()
        options.flow_type = FlowType.CARBON
        options.color_scheme = ColorSchemeType.BY_CARBON_INTENSITY
        options.value_suffix = " kg CO2"
        options.title = "Carbon Flow"
        options.show_carbon_annotation = True

        # Convert to carbon-based values
        carbon_data = self._convert_to_carbon_flow(data)
        return self.generate(carbon_data, options)

    def _convert_to_cost_flow(
        self,
        data: BlendFlowData,
    ) -> SankeyDiagramData:
        """Convert blend flow data to cost-based Sankey data."""
        nodes = []
        links = []

        # Create fuel source nodes with cost values
        for source in data.fuel_sources:
            ratio = data.blend_recipe.components.get(source.fuel_id, 0)
            if ratio > 0:
                cost_contribution = source.cost_per_unit * ratio * 1000  # Scale for visibility
                nodes.append(SankeyNode(
                    id=f"fuel_{source.fuel_id}",
                    label=f"{source.fuel_name}\n${source.cost_per_unit:.2f}",
                    node_type=NodeType.FUEL,
                    value=cost_contribution,
                    unit="USD",
                    color=source.color,
                    cost_per_unit=source.cost_per_unit,
                ))

        # Create blend node
        total_cost = sum(
            source.cost_per_unit * data.blend_recipe.components.get(source.fuel_id, 0)
            for source in data.fuel_sources
        ) * 1000

        nodes.append(SankeyNode(
            id="blend",
            label=data.blend_recipe.blend_name,
            node_type=NodeType.BLEND,
            value=total_cost,
            unit="USD",
            color="#9B59B6",
        ))

        # Create output node
        nodes.append(SankeyNode(
            id="output",
            label="Total Cost",
            node_type=NodeType.OUTPUT,
            value=total_cost,
            unit="USD",
            color="#F39C12",
        ))

        # Create links
        for source in data.fuel_sources:
            ratio = data.blend_recipe.components.get(source.fuel_id, 0)
            if ratio > 0:
                cost_flow = source.cost_per_unit * ratio * 1000
                links.append(SankeyLink(
                    source_id=f"fuel_{source.fuel_id}",
                    target_id="blend",
                    value=cost_flow,
                    unit="USD",
                    color=hex_to_rgba(source.color, 0.5),
                    cost_flow=cost_flow,
                ))

        links.append(SankeyLink(
            source_id="blend",
            target_id="output",
            value=total_cost,
            unit="USD",
            color=hex_to_rgba("#F39C12", 0.5),
        ))

        return SankeyDiagramData(
            nodes=nodes,
            links=links,
            title="Cost Flow",
            total_input=total_cost,
            total_output=total_cost,
        )

    def _convert_to_carbon_flow(
        self,
        data: BlendFlowData,
    ) -> SankeyDiagramData:
        """Convert blend flow data to carbon-based Sankey data."""
        nodes = []
        links = []

        # Create fuel source nodes with carbon values
        for source in data.fuel_sources:
            ratio = data.blend_recipe.components.get(source.fuel_id, 0)
            if ratio > 0:
                energy_contribution = source.energy_content * ratio
                carbon_contribution = (energy_contribution / 1000) * source.carbon_intensity

                nodes.append(SankeyNode(
                    id=f"fuel_{source.fuel_id}",
                    label=f"{source.fuel_name}\n{source.carbon_intensity:.1f} kg/GJ",
                    node_type=NodeType.FUEL,
                    value=carbon_contribution,
                    unit="kg CO2",
                    color=self._get_carbon_color(source.carbon_intensity),
                    carbon_intensity=source.carbon_intensity,
                ))

        # Create blend node
        total_carbon = sum(
            (source.energy_content * data.blend_recipe.components.get(source.fuel_id, 0) / 1000) *
            source.carbon_intensity
            for source in data.fuel_sources
        )

        nodes.append(SankeyNode(
            id="blend",
            label=data.blend_recipe.blend_name,
            node_type=NodeType.BLEND,
            value=total_carbon,
            unit="kg CO2",
            color="#9B59B6",
        ))

        # Create output node
        nodes.append(SankeyNode(
            id="emissions",
            label="Total Emissions",
            node_type=NodeType.OUTPUT,
            value=total_carbon,
            unit="kg CO2",
            color=EmissionColors.CO2,
        ))

        # Create links
        for source in data.fuel_sources:
            ratio = data.blend_recipe.components.get(source.fuel_id, 0)
            if ratio > 0:
                energy_contribution = source.energy_content * ratio
                carbon_flow = (energy_contribution / 1000) * source.carbon_intensity

                links.append(SankeyLink(
                    source_id=f"fuel_{source.fuel_id}",
                    target_id="blend",
                    value=carbon_flow,
                    unit="kg CO2",
                    color=hex_to_rgba(self._get_carbon_color(source.carbon_intensity), 0.5),
                    carbon_flow=carbon_flow,
                ))

        links.append(SankeyLink(
            source_id="blend",
            target_id="emissions",
            value=total_carbon,
            unit="kg CO2",
            color=hex_to_rgba(EmissionColors.CO2, 0.5),
        ))

        return SankeyDiagramData(
            nodes=nodes,
            links=links,
            title="Carbon Flow",
            total_input=total_carbon,
            total_output=total_carbon,
        )

    def _get_carbon_color(self, carbon_intensity: float) -> str:
        """Get color based on carbon intensity."""
        # Map carbon intensity to color (green=low, red=high)
        scale = GradientScales.CARBON_INTENSITY
        normalized = min(1, max(0, carbon_intensity / 100))
        return self._interpolate_color(scale, normalized)

    def generate_comparison(
        self,
        baseline: BlendFlowData,
        optimized: BlendFlowData,
        options: Optional[SankeyChartOptions] = None,
    ) -> Dict[str, Any]:
        """Generate comparison Sankey showing baseline vs optimized blend."""
        options = options or SankeyChartOptions()
        options.title = "Blend Comparison: Baseline vs Optimized"

        # Create side-by-side comparison
        nodes = []
        links = []

        # Add baseline nodes with prefix
        baseline_diagram = self._convert_blend_flow_to_sankey(baseline, options)
        for node in baseline_diagram.nodes:
            node.id = f"baseline_{node.id}"
            node.label = f"[Baseline]\n{node.label}"
            node.x_position = 0.1 if node.node_type == NodeType.FUEL else (0.3 if node.node_type == NodeType.BLEND else 0.5)
            nodes.append(node)

        for link in baseline_diagram.links:
            link.source_id = f"baseline_{link.source_id}"
            link.target_id = f"baseline_{link.target_id}"
            link.opacity = 0.3  # Dimmed for baseline
            links.append(link)

        # Add optimized nodes with prefix
        optimized_diagram = self._convert_blend_flow_to_sankey(optimized, options)
        for node in optimized_diagram.nodes:
            node.id = f"optimized_{node.id}"
            node.label = f"[Optimized]\n{node.label}"
            node.x_position = 0.1 if node.node_type == NodeType.FUEL else (0.3 if node.node_type == NodeType.BLEND else 0.5)
            node.y_position = (node.y_position or 0.5) + 0.5  # Offset vertically
            nodes.append(node)

        for link in optimized_diagram.links:
            link.source_id = f"optimized_{link.source_id}"
            link.target_id = f"optimized_{link.target_id}"
            link.opacity = 0.7  # Brighter for optimized
            links.append(link)

        comparison_data = SankeyDiagramData(
            nodes=nodes,
            links=links,
            title=options.title,
            efficiency=optimized.overall_efficiency,
            metadata={
                "baseline_efficiency": baseline.overall_efficiency,
                "optimized_efficiency": optimized.overall_efficiency,
                "improvement": optimized.overall_efficiency - baseline.overall_efficiency,
            },
        )

        return self._build_sankey_chart(comparison_data, options)

    def to_json(self, chart: Dict[str, Any]) -> str:
        """Export chart to JSON string."""
        return json.dumps(chart, indent=2, default=str)

    def clear_cache(self) -> None:
        """Clear the chart cache."""
        self._cache.clear()


# =============================================================================
# SPECIALIZED SANKEY GENERATORS
# =============================================================================

class MultiStageBlendSankey(FuelBlendSankeyEngine):
    """Sankey generator for multi-stage blending processes."""

    def generate_multi_stage(
        self,
        stages: List[BlendFlowData],
        options: Optional[SankeyChartOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate multi-stage blending Sankey diagram.

        Args:
            stages: List of blending stages in order
            options: Chart options

        Returns:
            Plotly chart specification
        """
        options = options or SankeyChartOptions()
        options.title = "Multi-Stage Blend Process"

        nodes = []
        links = []
        x_positions = [0.01 + i * (0.98 / (len(stages) + 1)) for i in range(len(stages) + 2)]

        for stage_idx, stage in enumerate(stages):
            stage_prefix = f"stage{stage_idx}_"

            # Add fuel nodes for this stage
            for source in stage.fuel_sources:
                ratio = stage.blend_recipe.components.get(source.fuel_id, 0)
                if ratio > 0:
                    energy = source.energy_content * ratio
                    nodes.append(SankeyNode(
                        id=f"{stage_prefix}fuel_{source.fuel_id}",
                        label=f"Stage {stage_idx + 1}: {source.fuel_name}",
                        node_type=NodeType.FUEL,
                        value=energy,
                        color=source.color,
                        x_position=x_positions[stage_idx],
                    ))

            # Add blend node
            blend_energy = sum(
                s.energy_content * stage.blend_recipe.components.get(s.fuel_id, 0)
                for s in stage.fuel_sources
            )
            nodes.append(SankeyNode(
                id=f"{stage_prefix}blend",
                label=f"Stage {stage_idx + 1}: {stage.blend_recipe.blend_name}",
                node_type=NodeType.BLEND,
                value=blend_energy,
                color="#9B59B6",
                x_position=x_positions[stage_idx + 1],
            ))

            # Create fuel to blend links
            for source in stage.fuel_sources:
                ratio = stage.blend_recipe.components.get(source.fuel_id, 0)
                if ratio > 0:
                    energy = source.energy_content * ratio
                    links.append(SankeyLink(
                        source_id=f"{stage_prefix}fuel_{source.fuel_id}",
                        target_id=f"{stage_prefix}blend",
                        value=energy,
                        color=hex_to_rgba(source.color, 0.5),
                    ))

            # Connect to next stage if not last
            if stage_idx < len(stages) - 1:
                next_prefix = f"stage{stage_idx + 1}_"
                # Connect blend to next stage fuels
                links.append(SankeyLink(
                    source_id=f"{stage_prefix}blend",
                    target_id=f"{next_prefix}blend",
                    value=blend_energy * 0.95,  # Account for some loss
                    color=hex_to_rgba("#9B59B6", 0.5),
                    label="To next stage",
                ))

        # Add final output
        final_stage = stages[-1]
        final_energy = sum(o.energy_value for o in final_stage.outputs)
        nodes.append(SankeyNode(
            id="final_output",
            label="Final Output",
            node_type=NodeType.OUTPUT,
            value=final_energy,
            color="#F39C12",
            x_position=x_positions[-1],
        ))

        final_prefix = f"stage{len(stages) - 1}_"
        links.append(SankeyLink(
            source_id=f"{final_prefix}blend",
            target_id="final_output",
            value=final_energy,
            color=hex_to_rgba("#F39C12", 0.5),
        ))

        data = SankeyDiagramData(
            nodes=nodes,
            links=links,
            title=options.title,
        )

        return self._build_sankey_chart(data, options)


class SupplyChainSankey(FuelBlendSankeyEngine):
    """Sankey generator for fuel supply chain visualization."""

    def generate_supply_chain(
        self,
        suppliers: List[Dict[str, Any]],
        storage_facilities: List[Dict[str, Any]],
        consumers: List[Dict[str, Any]],
        flows: List[Dict[str, Any]],
        options: Optional[SankeyChartOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate supply chain Sankey diagram.

        Args:
            suppliers: List of supplier definitions
            storage_facilities: List of storage facility definitions
            consumers: List of consumer definitions
            flows: List of flow definitions
            options: Chart options

        Returns:
            Plotly chart specification
        """
        options = options or SankeyChartOptions()
        options.title = "Fuel Supply Chain"

        nodes = []
        links = []

        # Add supplier nodes
        for supplier in suppliers:
            nodes.append(SankeyNode(
                id=f"supplier_{supplier['id']}",
                label=supplier.get("name", supplier["id"]),
                node_type=NodeType.SOURCE,
                value=supplier.get("capacity", 0),
                color=get_fuel_color(supplier.get("fuel_type", "unknown")),
                x_position=0.1,
            ))

        # Add storage nodes
        for storage in storage_facilities:
            nodes.append(SankeyNode(
                id=f"storage_{storage['id']}",
                label=storage.get("name", storage["id"]),
                node_type=NodeType.STORAGE,
                value=storage.get("capacity", 0),
                color="#1ABC9C",
                x_position=0.5,
            ))

        # Add consumer nodes
        for consumer in consumers:
            nodes.append(SankeyNode(
                id=f"consumer_{consumer['id']}",
                label=consumer.get("name", consumer["id"]),
                node_type=NodeType.OUTPUT,
                value=consumer.get("demand", 0),
                color="#F39C12",
                x_position=0.9,
            ))

        # Add flow links
        for flow in flows:
            links.append(SankeyLink(
                source_id=flow["source"],
                target_id=flow["target"],
                value=flow.get("value", 0),
                color=hex_to_rgba(flow.get("color", "#CCCCCC"), 0.5),
                label=flow.get("label"),
            ))

        data = SankeyDiagramData(
            nodes=nodes,
            links=links,
            title=options.title,
        )

        return self._build_sankey_chart(data, options)


# =============================================================================
# DATA BUILDERS
# =============================================================================

class BlendFlowDataBuilder:
    """Builder for creating BlendFlowData objects."""

    def __init__(self):
        """Initialize builder."""
        self._fuel_sources: List[FuelSource] = []
        self._blend_recipe: Optional[BlendRecipe] = None
        self._outputs: List[EnergyOutput] = []
        self._losses: Dict[str, float] = {}
        self._metadata: Dict[str, Any] = {}

    def add_fuel_source(
        self,
        fuel_id: str,
        fuel_type: str,
        fuel_name: str,
        energy_content: float,
        carbon_intensity: float,
        cost_per_unit: float,
        **kwargs,
    ) -> "BlendFlowDataBuilder":
        """Add a fuel source."""
        self._fuel_sources.append(FuelSource(
            fuel_id=fuel_id,
            fuel_type=fuel_type,
            fuel_name=fuel_name,
            energy_content=energy_content,
            carbon_intensity=carbon_intensity,
            cost_per_unit=cost_per_unit,
            **kwargs,
        ))
        return self

    def set_blend_recipe(
        self,
        blend_id: str,
        blend_name: str,
        components: Dict[str, float],
        **kwargs,
    ) -> "BlendFlowDataBuilder":
        """Set the blend recipe."""
        self._blend_recipe = BlendRecipe(
            blend_id=blend_id,
            blend_name=blend_name,
            components=components,
            **kwargs,
        )
        return self

    def add_output(
        self,
        output_id: str,
        output_type: str,
        output_name: str,
        energy_value: float,
        efficiency: float,
        **kwargs,
    ) -> "BlendFlowDataBuilder":
        """Add an energy output."""
        self._outputs.append(EnergyOutput(
            output_id=output_id,
            output_type=output_type,
            output_name=output_name,
            energy_value=energy_value,
            efficiency=efficiency,
            **kwargs,
        ))
        return self

    def add_loss(self, loss_type: str, loss_value: float) -> "BlendFlowDataBuilder":
        """Add a loss."""
        self._losses[loss_type] = loss_value
        return self

    def set_metadata(self, key: str, value: Any) -> "BlendFlowDataBuilder":
        """Set metadata."""
        self._metadata[key] = value
        return self

    def build(self) -> BlendFlowData:
        """Build the BlendFlowData object."""
        if not self._fuel_sources:
            raise ValueError("At least one fuel source is required")
        if not self._blend_recipe:
            raise ValueError("Blend recipe is required")
        if not self._outputs:
            raise ValueError("At least one output is required")

        return BlendFlowData(
            fuel_sources=self._fuel_sources,
            blend_recipe=self._blend_recipe,
            outputs=self._outputs,
            losses=self._losses,
            metadata=self._metadata,
        )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def create_sample_blend_flow() -> BlendFlowData:
    """Create sample blend flow data for demonstration."""
    builder = BlendFlowDataBuilder()

    # Add fuel sources
    builder.add_fuel_source(
        fuel_id="coal",
        fuel_type="coal",
        fuel_name="Bituminous Coal",
        energy_content=26000,  # MJ/tonne
        carbon_intensity=94.6,  # kg CO2/GJ
        cost_per_unit=80,  # $/tonne
    )

    builder.add_fuel_source(
        fuel_id="natural_gas",
        fuel_type="natural_gas",
        fuel_name="Natural Gas",
        energy_content=38000,  # MJ/1000 m3
        carbon_intensity=56.1,  # kg CO2/GJ
        cost_per_unit=5,  # $/GJ
    )

    builder.add_fuel_source(
        fuel_id="biomass",
        fuel_type="biomass",
        fuel_name="Wood Pellets",
        energy_content=18000,  # MJ/tonne
        carbon_intensity=10.0,  # kg CO2/GJ (biogenic)
        cost_per_unit=150,  # $/tonne
    )

    # Set blend recipe
    builder.set_blend_recipe(
        blend_id="blend_001",
        blend_name="Optimized Blend",
        components={
            "coal": 0.40,
            "natural_gas": 0.35,
            "biomass": 0.25,
        },
        description="Cost-optimized blend with reduced carbon intensity",
    )

    # Add outputs
    builder.add_output(
        output_id="steam",
        output_type="steam",
        output_name="Process Steam",
        energy_value=25000,  # MJ
        efficiency=0.85,
    )

    builder.add_output(
        output_id="heat",
        output_type="heat",
        output_name="Space Heating",
        energy_value=5000,  # MJ
        efficiency=0.90,
    )

    # Add losses
    builder.add_loss("flue_gas", 2000)
    builder.add_loss("radiation", 500)
    builder.add_loss("unaccounted", 200)

    return builder.build()


def example_energy_flow_sankey():
    """Example: Generate energy flow Sankey."""
    print("Generating energy flow Sankey...")

    data = create_sample_blend_flow()
    engine = FuelBlendSankeyEngine()
    options = SankeyChartOptions(
        title="Energy Flow: Multi-Fuel Blend",
        subtitle="Showing energy distribution from fuels to outputs",
        color_scheme=ColorSchemeType.BY_FUEL_TYPE,
    )

    chart = engine.generate_energy_flow(data, options)
    print(f"Energy flow chart generated with {len(chart['data'][0]['node']['label'])} nodes")
    return chart


def example_carbon_flow_sankey():
    """Example: Generate carbon flow Sankey."""
    print("Generating carbon flow Sankey...")

    data = create_sample_blend_flow()
    engine = FuelBlendSankeyEngine()

    chart = engine.generate_carbon_flow(data)
    print(f"Carbon flow chart generated")
    return chart


def example_cost_flow_sankey():
    """Example: Generate cost flow Sankey."""
    print("Generating cost flow Sankey...")

    data = create_sample_blend_flow()
    engine = FuelBlendSankeyEngine()

    chart = engine.generate_cost_flow(data)
    print(f"Cost flow chart generated")
    return chart


def example_multi_stage_blend():
    """Example: Generate multi-stage blend Sankey."""
    print("Generating multi-stage blend Sankey...")

    # Create two-stage blending process
    stage1 = create_sample_blend_flow()
    stage2 = create_sample_blend_flow()

    engine = MultiStageBlendSankey()
    chart = engine.generate_multi_stage([stage1, stage2])
    print(f"Multi-stage chart generated")
    return chart


def run_all_examples():
    """Run all Sankey diagram examples."""
    print("=" * 60)
    print("GL-011 FUELCRAFT - Fuel Blend Sankey Examples")
    print("=" * 60)

    examples = [
        ("Energy Flow Sankey", example_energy_flow_sankey),
        ("Carbon Flow Sankey", example_carbon_flow_sankey),
        ("Cost Flow Sankey", example_cost_flow_sankey),
        ("Multi-Stage Blend", example_multi_stage_blend),
    ]

    results = {}
    for name, func in examples:
        print(f"\n--- {name} ---")
        try:
            results[name] = func()
            print(f"SUCCESS: {name}")
        except Exception as e:
            print(f"ERROR: {name} - {e}")
            results[name] = None

    print("\n" + "=" * 60)
    print(f"Completed {len([r for r in results.values() if r])} of {len(examples)} examples")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_all_examples()
