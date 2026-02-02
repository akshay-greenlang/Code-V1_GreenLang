# -*- coding: utf-8 -*-
"""
GreenLang Sankey Diagram Generator for Process Heat Systems
=============================================================

This module provides enterprise-grade Sankey diagram generation for visualizing
energy flows in process heat systems. Designed for zero-hallucination energy
balance validation and regulatory compliance.

Key Features:
- Plotly-based interactive Sankey diagrams
- Energy balance validation (conservation of energy)
- Process heat system templates (boiler, furnace, heat recovery)
- SHA-256 provenance hashing for data integrity
- Export to HTML, PNG, and JSON formats

Example:
    >>> from greenlang.visualization.sankey_generator import (
    ...     ProcessHeatSankeyGenerator,
    ...     SankeyNode,
    ...     SankeyFlow,
    ... )
    >>> generator = ProcessHeatSankeyGenerator("Boiler Energy Balance", unit="BTU/hr")
    >>> generator.add_node(SankeyNode(id="fuel", label="Fuel Input", value=1_000_000))
    >>> generator.add_node(SankeyNode(id="steam", label="Steam Output", value=800_000))
    >>> generator.add_node(SankeyNode(id="stack", label="Stack Loss", value=150_000))
    >>> generator.add_node(SankeyNode(id="radiation", label="Radiation Loss", value=50_000))
    >>> generator.add_flow(SankeyFlow(source="fuel", target="steam", value=800_000))
    >>> generator.add_flow(SankeyFlow(source="fuel", target="stack", value=150_000))
    >>> generator.add_flow(SankeyFlow(source="fuel", target="radiation", value=50_000))
    >>> is_valid, imbalance = generator.validate_energy_balance()
    >>> assert is_valid, f"Energy imbalance: {imbalance}"
    >>> figure = generator.generate_figure()
    >>> generator.export_html("boiler_sankey.html")

Author: GreenLang Framework Team
Date: December 2024
Status: Production Ready
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Color Scheme Constants
# =============================================================================

class ProcessHeatColorScheme(str, Enum):
    """Standard color scheme for process heat Sankey diagrams."""

    FUEL_INPUT = "#FF6B35"      # Orange - Fuel/Input energy
    USEFUL_HEAT = "#2ECC71"     # Green - Useful heat output
    LOSSES = "#E74C3C"          # Red - Energy losses
    STEAM = "#3498DB"           # Blue - Steam flows
    CONDENSATE = "#1ABC9C"      # Cyan - Condensate return
    ELECTRICITY = "#9B59B6"     # Purple - Electrical energy
    WASTE_HEAT = "#F39C12"      # Amber - Waste heat
    COOLING = "#5DADE2"         # Light blue - Cooling water
    AMBIENT = "#95A5A6"         # Gray - Ambient/environment


# Color dictionary for convenience
PROCESS_HEAT_COLORS: Dict[str, str] = {
    "fuel": ProcessHeatColorScheme.FUEL_INPUT.value,
    "input": ProcessHeatColorScheme.FUEL_INPUT.value,
    "useful_heat": ProcessHeatColorScheme.USEFUL_HEAT.value,
    "process_heat": ProcessHeatColorScheme.USEFUL_HEAT.value,
    "steam": ProcessHeatColorScheme.STEAM.value,
    "steam_output": ProcessHeatColorScheme.STEAM.value,
    "loss": ProcessHeatColorScheme.LOSSES.value,
    "losses": ProcessHeatColorScheme.LOSSES.value,
    "stack_loss": ProcessHeatColorScheme.LOSSES.value,
    "flue_gas": ProcessHeatColorScheme.LOSSES.value,
    "radiation_loss": ProcessHeatColorScheme.LOSSES.value,
    "wall_loss": ProcessHeatColorScheme.LOSSES.value,
    "blowdown": ProcessHeatColorScheme.LOSSES.value,
    "condensate": ProcessHeatColorScheme.CONDENSATE.value,
    "condensate_return": ProcessHeatColorScheme.CONDENSATE.value,
    "electricity": ProcessHeatColorScheme.ELECTRICITY.value,
    "waste_heat": ProcessHeatColorScheme.WASTE_HEAT.value,
    "recovered": ProcessHeatColorScheme.USEFUL_HEAT.value,
    "rejected": ProcessHeatColorScheme.LOSSES.value,
    "cooling": ProcessHeatColorScheme.COOLING.value,
    "ambient": ProcessHeatColorScheme.AMBIENT.value,
}


def get_color_for_node_type(node_id: str) -> str:
    """
    Get appropriate color based on node ID/type.

    Args:
        node_id: Node identifier string

    Returns:
        Hex color string for the node
    """
    node_id_lower = node_id.lower()

    # Check for direct matches
    if node_id_lower in PROCESS_HEAT_COLORS:
        return PROCESS_HEAT_COLORS[node_id_lower]

    # Check for partial matches
    for key, color in PROCESS_HEAT_COLORS.items():
        if key in node_id_lower or node_id_lower in key:
            return color

    # Default to gray
    return ProcessHeatColorScheme.AMBIENT.value


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SankeyNode:
    """
    Represents a node in a Sankey diagram.

    A node represents an energy source, sink, or transformation point
    in the process heat system.

    Attributes:
        id: Unique identifier for the node
        label: Display label for the node
        value: Energy value (BTU/hr or kW)
        color: Optional hex color code (auto-assigned if not provided)
        position: Optional (x, y) position tuple for layout control

    Example:
        >>> node = SankeyNode(
        ...     id="fuel_input",
        ...     label="Natural Gas Input",
        ...     value=1_000_000,
        ...     color="#FF6B35"
        ... )
    """

    id: str
    label: str
    value: float
    color: Optional[str] = None
    position: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        """Auto-assign color based on node type if not provided."""
        if self.color is None:
            self.color = get_color_for_node_type(self.id)


@dataclass
class SankeyFlow:
    """
    Represents a flow (link) between nodes in a Sankey diagram.

    A flow represents energy transfer from one node to another.

    Attributes:
        source: Source node ID
        target: Target node ID
        value: Flow value (energy transferred)
        label: Optional display label for the flow
        color: Optional hex color code for the flow

    Example:
        >>> flow = SankeyFlow(
        ...     source="fuel_input",
        ...     target="steam_output",
        ...     value=800_000,
        ...     label="Useful Heat"
        ... )
    """

    source: str
    target: str
    value: float
    label: Optional[str] = None
    color: Optional[str] = None

    def __post_init__(self):
        """Auto-assign color based on target node type if not provided."""
        if self.color is None:
            self.color = get_color_for_node_type(self.target)


# =============================================================================
# Pydantic Validation Models
# =============================================================================

class SankeyNodeModel(BaseModel):
    """Pydantic model for SankeyNode validation."""

    id: str = Field(..., min_length=1, description="Unique node identifier")
    label: str = Field(..., min_length=1, description="Display label")
    value: float = Field(..., ge=0, description="Energy value (must be >= 0)")
    color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$", description="Hex color code")
    position_x: Optional[float] = Field(None, ge=0, le=1, description="X position (0-1)")
    position_y: Optional[float] = Field(None, ge=0, le=1, description="Y position (0-1)")

    @field_validator("id")
    @classmethod
    def validate_id_no_spaces(cls, v: str) -> str:
        """Validate that ID contains no spaces."""
        if " " in v:
            raise ValueError("Node ID must not contain spaces")
        return v


class SankeyFlowModel(BaseModel):
    """Pydantic model for SankeyFlow validation."""

    source: str = Field(..., min_length=1, description="Source node ID")
    target: str = Field(..., min_length=1, description="Target node ID")
    value: float = Field(..., gt=0, description="Flow value (must be > 0)")
    label: Optional[str] = Field(None, description="Display label")
    color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$", description="Hex color code")

    @model_validator(mode="after")
    def validate_source_not_target(self) -> "SankeyFlowModel":
        """Validate that source and target are different."""
        if self.source == self.target:
            raise ValueError("Source and target must be different nodes")
        return self


class SankeyDiagramConfig(BaseModel):
    """Configuration for Sankey diagram generation."""

    title: str = Field(..., min_length=1, description="Diagram title")
    unit: str = Field(default="BTU/hr", description="Energy unit")
    width: int = Field(default=1200, ge=400, le=4000, description="Diagram width in pixels")
    height: int = Field(default=800, ge=300, le=3000, description="Diagram height in pixels")
    font_size: int = Field(default=12, ge=8, le=24, description="Font size for labels")
    pad: int = Field(default=20, ge=5, le=50, description="Node padding")
    thickness: int = Field(default=30, ge=10, le=60, description="Node thickness")
    energy_balance_tolerance: float = Field(
        default=0.001,
        ge=0,
        le=0.1,
        description="Tolerance for energy balance validation (fraction)"
    )
    show_percentages: bool = Field(default=True, description="Show percentages in tooltips")
    show_values: bool = Field(default=True, description="Show values in tooltips")


class EnergyBalanceResult(BaseModel):
    """Result of energy balance validation."""

    is_balanced: bool = Field(..., description="Whether energy balance is valid")
    total_input: float = Field(..., ge=0, description="Total energy input")
    total_output: float = Field(..., ge=0, description="Total energy output")
    imbalance: float = Field(..., description="Energy imbalance (input - output)")
    imbalance_percentage: float = Field(..., description="Imbalance as percentage of input")
    tolerance: float = Field(..., ge=0, description="Allowed tolerance")
    validation_timestamp: datetime = Field(..., description="When validation was performed")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


# =============================================================================
# Main Generator Class
# =============================================================================

class ProcessHeatSankeyGenerator:
    """
    Generator for Process Heat Sankey diagrams with energy balance validation.

    This class provides enterprise-grade Sankey diagram generation with:
    - Zero-hallucination energy balance validation (conservation of energy)
    - Interactive Plotly-based visualizations
    - SHA-256 provenance tracking for audit trails
    - Export to multiple formats (HTML, PNG, JSON)

    Attributes:
        title: Diagram title
        unit: Energy unit (BTU/hr, kW, MW, etc.)
        config: Diagram configuration
        nodes: Dictionary of nodes by ID
        flows: List of flows

    Example:
        >>> generator = ProcessHeatSankeyGenerator("Boiler Energy Balance", unit="BTU/hr")
        >>> generator.add_node(SankeyNode(id="fuel", label="Fuel Input", value=1_000_000))
        >>> generator.add_node(SankeyNode(id="steam", label="Steam Output", value=800_000))
        >>> generator.add_flow(SankeyFlow(source="fuel", target="steam", value=800_000))
        >>> is_valid, imbalance = generator.validate_energy_balance()
        >>> figure = generator.generate_figure()
    """

    def __init__(
        self,
        title: str,
        unit: str = "BTU/hr",
        config: Optional[SankeyDiagramConfig] = None
    ):
        """
        Initialize the ProcessHeatSankeyGenerator.

        Args:
            title: Diagram title
            unit: Energy unit for display (BTU/hr, kW, MW)
            config: Optional configuration (defaults created if not provided)
        """
        self.title = title
        self.unit = unit
        self.config = config or SankeyDiagramConfig(title=title, unit=unit)

        # Node and flow storage
        self._nodes: Dict[str, SankeyNode] = {}
        self._flows: List[SankeyFlow] = []

        # Track input and output nodes for energy balance
        self._input_nodes: set = set()
        self._output_nodes: set = set()

        # Provenance tracking
        self._creation_timestamp = datetime.now(timezone.utc)
        self._modification_count = 0

        logger.info(f"ProcessHeatSankeyGenerator initialized: {title}")

    @property
    def nodes(self) -> Dict[str, SankeyNode]:
        """Get dictionary of all nodes."""
        return self._nodes.copy()

    @property
    def flows(self) -> List[SankeyFlow]:
        """Get list of all flows."""
        return self._flows.copy()

    def add_node(self, node: SankeyNode) -> None:
        """
        Add a node to the diagram.

        Args:
            node: SankeyNode to add

        Raises:
            ValueError: If node ID already exists or validation fails
        """
        # Validate using Pydantic model
        validated = SankeyNodeModel(
            id=node.id,
            label=node.label,
            value=node.value,
            color=node.color,
            position_x=node.position[0] if node.position else None,
            position_y=node.position[1] if node.position else None
        )

        if node.id in self._nodes:
            raise ValueError(f"Node with ID '{node.id}' already exists")

        self._nodes[node.id] = node
        self._modification_count += 1

        logger.debug(f"Added node: {node.id} ({node.label}) = {node.value} {self.unit}")

    def add_flow(self, flow: SankeyFlow) -> None:
        """
        Add a flow (link) between nodes.

        Args:
            flow: SankeyFlow to add

        Raises:
            ValueError: If source/target nodes don't exist or validation fails
        """
        # Validate using Pydantic model
        validated = SankeyFlowModel(
            source=flow.source,
            target=flow.target,
            value=flow.value,
            label=flow.label,
            color=flow.color
        )

        # Verify nodes exist
        if flow.source not in self._nodes:
            raise ValueError(f"Source node '{flow.source}' does not exist")
        if flow.target not in self._nodes:
            raise ValueError(f"Target node '{flow.target}' does not exist")

        self._flows.append(flow)
        self._modification_count += 1

        # Track input/output relationships
        self._input_nodes.add(flow.source)
        self._output_nodes.add(flow.target)

        logger.debug(f"Added flow: {flow.source} -> {flow.target} = {flow.value} {self.unit}")

    def mark_as_input(self, node_id: str) -> None:
        """
        Mark a node as an input node for energy balance calculation.

        Args:
            node_id: ID of the node to mark as input

        Raises:
            ValueError: If node doesn't exist
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node '{node_id}' does not exist")
        self._input_nodes.add(node_id)

    def mark_as_output(self, node_id: str) -> None:
        """
        Mark a node as an output node for energy balance calculation.

        Args:
            node_id: ID of the node to mark as output

        Raises:
            ValueError: If node doesn't exist
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node '{node_id}' does not exist")
        self._output_nodes.add(node_id)

    def validate_energy_balance(
        self,
        tolerance: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Validate that energy in equals energy out (plus losses).

        This is a ZERO-HALLUCINATION calculation - purely deterministic
        arithmetic based on flow values, no ML or LLM involved.

        The validation checks that the sum of all flows leaving input nodes
        equals the sum of all flows entering output nodes.

        Args:
            tolerance: Override default tolerance (fraction, e.g., 0.001 = 0.1%)

        Returns:
            Tuple of (is_valid, imbalance_value)

        Example:
            >>> is_valid, imbalance = generator.validate_energy_balance()
            >>> if not is_valid:
            ...     print(f"Energy imbalance detected: {imbalance} {generator.unit}")
        """
        tol = tolerance if tolerance is not None else self.config.energy_balance_tolerance

        # Calculate total input: sum of flows from input-only nodes
        # (nodes that have only outgoing flows)
        pure_inputs = self._input_nodes - self._output_nodes
        total_input = sum(
            flow.value for flow in self._flows
            if flow.source in pure_inputs
        )

        # Calculate total output: sum of flows to output-only nodes
        # (nodes that have only incoming flows)
        pure_outputs = self._output_nodes - self._input_nodes
        total_output = sum(
            flow.value for flow in self._flows
            if flow.target in pure_outputs
        )

        # Calculate imbalance
        imbalance = total_input - total_output
        imbalance_percentage = (abs(imbalance) / total_input * 100) if total_input > 0 else 0.0

        # Check if within tolerance
        is_balanced = abs(imbalance) <= (total_input * tol) if total_input > 0 else True

        logger.info(
            f"Energy balance validation: input={total_input:.2f}, "
            f"output={total_output:.2f}, imbalance={imbalance:.2f} "
            f"({imbalance_percentage:.4f}%), valid={is_balanced}"
        )

        return is_balanced, imbalance

    def get_energy_balance_result(
        self,
        tolerance: Optional[float] = None
    ) -> EnergyBalanceResult:
        """
        Get detailed energy balance validation result.

        Args:
            tolerance: Override default tolerance

        Returns:
            EnergyBalanceResult with full validation details
        """
        tol = tolerance if tolerance is not None else self.config.energy_balance_tolerance

        # Calculate totals
        pure_inputs = self._input_nodes - self._output_nodes
        pure_outputs = self._output_nodes - self._input_nodes

        total_input = sum(
            flow.value for flow in self._flows
            if flow.source in pure_inputs
        )
        total_output = sum(
            flow.value for flow in self._flows
            if flow.target in pure_outputs
        )

        imbalance = total_input - total_output
        imbalance_percentage = (abs(imbalance) / total_input * 100) if total_input > 0 else 0.0
        is_balanced = abs(imbalance) <= (total_input * tol) if total_input > 0 else True

        # Calculate provenance hash
        validation_data = {
            "total_input": total_input,
            "total_output": total_output,
            "imbalance": imbalance,
            "tolerance": tol,
            "flow_count": len(self._flows),
            "node_count": len(self._nodes)
        }
        provenance_hash = self._calculate_hash(validation_data)

        return EnergyBalanceResult(
            is_balanced=is_balanced,
            total_input=total_input,
            total_output=total_output,
            imbalance=imbalance,
            imbalance_percentage=imbalance_percentage,
            tolerance=tol,
            validation_timestamp=datetime.now(timezone.utc),
            provenance_hash=provenance_hash
        )

    def generate_figure(self) -> "go.Figure":
        """
        Generate Plotly Sankey figure.

        Returns:
            Plotly Figure object

        Raises:
            ImportError: If plotly is not installed
            ValueError: If no nodes or flows defined
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "Plotly is required for Sankey diagram generation. "
                "Install it with: pip install plotly"
            )

        if not self._nodes:
            raise ValueError("No nodes defined. Add nodes before generating figure.")
        if not self._flows:
            raise ValueError("No flows defined. Add flows before generating figure.")

        # Create node index mapping
        node_ids = list(self._nodes.keys())
        node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

        # Prepare node data
        node_labels = [self._nodes[nid].label for nid in node_ids]
        node_colors = [self._nodes[nid].color for nid in node_ids]

        # Calculate percentages for tooltips
        total_input = sum(
            flow.value for flow in self._flows
            if flow.source in (self._input_nodes - self._output_nodes)
        )

        # Prepare node customdata for tooltips
        node_customdata = []
        for nid in node_ids:
            node = self._nodes[nid]
            percentage = (node.value / total_input * 100) if total_input > 0 else 0
            node_customdata.append({
                "value": node.value,
                "unit": self.unit,
                "percentage": percentage
            })

        # Prepare flow data
        source_indices = [node_index[flow.source] for flow in self._flows]
        target_indices = [node_index[flow.target] for flow in self._flows]
        flow_values = [flow.value for flow in self._flows]
        flow_colors = [flow.color or get_color_for_node_type(flow.target) for flow in self._flows]
        flow_labels = [
            flow.label or f"{self._nodes[flow.source].label} to {self._nodes[flow.target].label}"
            for flow in self._flows
        ]

        # Create flow customdata for tooltips
        flow_customdata = []
        for flow in self._flows:
            percentage = (flow.value / total_input * 100) if total_input > 0 else 0
            flow_customdata.append({
                "value": flow.value,
                "unit": self.unit,
                "percentage": percentage,
                "source": self._nodes[flow.source].label,
                "target": self._nodes[flow.target].label
            })

        # Build hover templates
        node_hovertemplate = (
            "<b>%{label}</b><br>"
            "Value: %{customdata.value:,.0f} %{customdata.unit}<br>"
            "Percentage: %{customdata.percentage:.1f}%<br>"
            "<extra></extra>"
        ) if self.config.show_values and self.config.show_percentages else (
            "<b>%{label}</b><extra></extra>"
        )

        link_hovertemplate = (
            "<b>%{customdata.source}</b> to <b>%{customdata.target}</b><br>"
            "Value: %{customdata.value:,.0f} %{customdata.unit}<br>"
            "Percentage: %{customdata.percentage:.1f}%<br>"
            "<extra></extra>"
        ) if self.config.show_values and self.config.show_percentages else (
            "<b>%{label}</b><extra></extra>"
        )

        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=self.config.pad,
                thickness=self.config.thickness,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors,
                customdata=node_customdata,
                hovertemplate=node_hovertemplate
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=flow_values,
                color=[self._adjust_color_opacity(c, 0.6) for c in flow_colors],
                label=flow_labels,
                customdata=flow_customdata,
                hovertemplate=link_hovertemplate
            )
        )])

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{self.title}<br><sup>Unit: {self.unit}</sup>",
                font=dict(size=18)
            ),
            font=dict(size=self.config.font_size),
            width=self.config.width,
            height=self.config.height,
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        logger.info(f"Generated Sankey figure: {self.title}")
        return fig

    def export_html(self, filepath: Union[str, Path]) -> None:
        """
        Export diagram to interactive HTML file.

        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)
        fig = self.generate_figure()
        fig.write_html(str(filepath), include_plotlyjs=True, full_html=True)
        logger.info(f"Exported Sankey diagram to HTML: {filepath}")

    def export_png(
        self,
        filepath: Union[str, Path],
        scale: float = 2.0
    ) -> None:
        """
        Export diagram to PNG image.

        Args:
            filepath: Output file path
            scale: Image scale factor (default 2.0 for high resolution)

        Raises:
            ImportError: If kaleido is not installed
        """
        try:
            filepath = Path(filepath)
            fig = self.generate_figure()
            fig.write_image(str(filepath), scale=scale)
            logger.info(f"Exported Sankey diagram to PNG: {filepath}")
        except ValueError as e:
            if "kaleido" in str(e).lower():
                raise ImportError(
                    "Kaleido is required for PNG export. "
                    "Install it with: pip install kaleido"
                ) from e
            raise

    def export_json(self) -> Dict[str, Any]:
        """
        Export diagram data as JSON-serializable dictionary.

        Returns:
            Dictionary containing all diagram data with provenance hash
        """
        # Collect all data
        nodes_data = [
            {
                "id": node.id,
                "label": node.label,
                "value": node.value,
                "color": node.color,
                "position": node.position
            }
            for node in self._nodes.values()
        ]

        flows_data = [
            {
                "source": flow.source,
                "target": flow.target,
                "value": flow.value,
                "label": flow.label,
                "color": flow.color
            }
            for flow in self._flows
        ]

        # Get energy balance
        balance_result = self.get_energy_balance_result()

        # Build export data
        export_data = {
            "metadata": {
                "title": self.title,
                "unit": self.unit,
                "created_at": self._creation_timestamp.isoformat(),
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "modification_count": self._modification_count,
                "node_count": len(self._nodes),
                "flow_count": len(self._flows)
            },
            "config": self.config.model_dump(),
            "nodes": nodes_data,
            "flows": flows_data,
            "energy_balance": {
                "is_balanced": balance_result.is_balanced,
                "total_input": balance_result.total_input,
                "total_output": balance_result.total_output,
                "imbalance": balance_result.imbalance,
                "imbalance_percentage": balance_result.imbalance_percentage,
                "tolerance": balance_result.tolerance
            }
        }

        # Calculate provenance hash
        export_data["provenance_hash"] = self._calculate_hash(export_data)

        logger.info(f"Exported Sankey diagram to JSON: {self.title}")
        return export_data

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """
        Calculate SHA-256 hash of data for provenance tracking.

        Args:
            data: Data dictionary to hash

        Returns:
            Hex digest of SHA-256 hash
        """
        # Serialize data to canonical JSON
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    @staticmethod
    def _adjust_color_opacity(hex_color: str, opacity: float) -> str:
        """
        Convert hex color to RGBA with specified opacity.

        Args:
            hex_color: Hex color string (e.g., "#FF6B35")
            opacity: Opacity value (0-1)

        Returns:
            RGBA color string
        """
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {opacity})"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get diagram statistics.

        Returns:
            Dictionary of statistics
        """
        pure_inputs = self._input_nodes - self._output_nodes
        pure_outputs = self._output_nodes - self._input_nodes

        total_input = sum(
            flow.value for flow in self._flows
            if flow.source in pure_inputs
        )
        total_output = sum(
            flow.value for flow in self._flows
            if flow.target in pure_outputs
        )

        return {
            "title": self.title,
            "unit": self.unit,
            "node_count": len(self._nodes),
            "flow_count": len(self._flows),
            "input_node_count": len(pure_inputs),
            "output_node_count": len(pure_outputs),
            "total_input": total_input,
            "total_output": total_output,
            "efficiency_percentage": (total_output / total_input * 100) if total_input > 0 else 0,
            "created_at": self._creation_timestamp.isoformat(),
            "modification_count": self._modification_count
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ProcessHeatSankeyGenerator("
            f"title='{self.title}', "
            f"nodes={len(self._nodes)}, "
            f"flows={len(self._flows)})"
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_boiler_sankey(
    fuel_input: float,
    steam_output: float,
    stack_loss: float,
    radiation_loss: float,
    blowdown_loss: float,
    title: str = "Boiler Energy Balance",
    unit: str = "BTU/hr"
) -> ProcessHeatSankeyGenerator:
    """
    Create a standard boiler energy balance Sankey diagram.

    This is a ZERO-HALLUCINATION factory function - all values are passed in
    explicitly and validated for energy conservation.

    Args:
        fuel_input: Total fuel energy input
        steam_output: Useful steam energy output
        stack_loss: Stack/flue gas losses
        radiation_loss: Radiation and convection losses
        blowdown_loss: Blowdown losses
        title: Diagram title
        unit: Energy unit

    Returns:
        Configured ProcessHeatSankeyGenerator

    Raises:
        ValueError: If energy doesn't balance within tolerance

    Example:
        >>> generator = create_boiler_sankey(
        ...     fuel_input=1_000_000,
        ...     steam_output=800_000,
        ...     stack_loss=150_000,
        ...     radiation_loss=30_000,
        ...     blowdown_loss=20_000
        ... )
        >>> figure = generator.generate_figure()
    """
    # Validate energy balance
    total_output = steam_output + stack_loss + radiation_loss + blowdown_loss
    imbalance = fuel_input - total_output
    if abs(imbalance) > fuel_input * 0.001:
        raise ValueError(
            f"Energy imbalance detected: input={fuel_input}, "
            f"output={total_output}, imbalance={imbalance}"
        )

    generator = ProcessHeatSankeyGenerator(title=title, unit=unit)

    # Add nodes
    generator.add_node(SankeyNode(
        id="fuel_input",
        label="Fuel Input",
        value=fuel_input,
        color=ProcessHeatColorScheme.FUEL_INPUT.value
    ))
    generator.add_node(SankeyNode(
        id="boiler",
        label="Boiler",
        value=fuel_input,
        color=ProcessHeatColorScheme.AMBIENT.value
    ))
    generator.add_node(SankeyNode(
        id="steam_output",
        label="Steam Output",
        value=steam_output,
        color=ProcessHeatColorScheme.STEAM.value
    ))
    generator.add_node(SankeyNode(
        id="stack_loss",
        label="Stack Loss",
        value=stack_loss,
        color=ProcessHeatColorScheme.LOSSES.value
    ))
    generator.add_node(SankeyNode(
        id="radiation_loss",
        label="Radiation Loss",
        value=radiation_loss,
        color=ProcessHeatColorScheme.LOSSES.value
    ))
    generator.add_node(SankeyNode(
        id="blowdown_loss",
        label="Blowdown Loss",
        value=blowdown_loss,
        color=ProcessHeatColorScheme.LOSSES.value
    ))

    # Add flows
    generator.add_flow(SankeyFlow(
        source="fuel_input",
        target="boiler",
        value=fuel_input,
        label="Fuel Energy"
    ))
    generator.add_flow(SankeyFlow(
        source="boiler",
        target="steam_output",
        value=steam_output,
        label="Useful Heat"
    ))
    generator.add_flow(SankeyFlow(
        source="boiler",
        target="stack_loss",
        value=stack_loss,
        label="Flue Gas Loss"
    ))
    generator.add_flow(SankeyFlow(
        source="boiler",
        target="radiation_loss",
        value=radiation_loss,
        label="Radiation"
    ))
    generator.add_flow(SankeyFlow(
        source="boiler",
        target="blowdown_loss",
        value=blowdown_loss,
        label="Blowdown"
    ))

    # Mark input/output nodes
    generator.mark_as_input("fuel_input")
    generator.mark_as_output("steam_output")
    generator.mark_as_output("stack_loss")
    generator.mark_as_output("radiation_loss")
    generator.mark_as_output("blowdown_loss")

    logger.info(
        f"Created boiler Sankey: efficiency={steam_output/fuel_input*100:.1f}%"
    )

    return generator


def create_furnace_sankey(
    fuel_input: float,
    process_heat: float,
    flue_gas_loss: float,
    wall_loss: float,
    title: str = "Furnace Energy Balance",
    unit: str = "BTU/hr"
) -> ProcessHeatSankeyGenerator:
    """
    Create a furnace energy balance Sankey diagram.

    This is a ZERO-HALLUCINATION factory function for industrial furnace
    energy balance visualization.

    Args:
        fuel_input: Total fuel energy input
        process_heat: Useful heat delivered to process
        flue_gas_loss: Flue gas (exhaust) losses
        wall_loss: Wall/shell losses
        title: Diagram title
        unit: Energy unit

    Returns:
        Configured ProcessHeatSankeyGenerator

    Raises:
        ValueError: If energy doesn't balance within tolerance

    Example:
        >>> generator = create_furnace_sankey(
        ...     fuel_input=5_000_000,
        ...     process_heat=3_500_000,
        ...     flue_gas_loss=1_200_000,
        ...     wall_loss=300_000
        ... )
    """
    # Validate energy balance
    total_output = process_heat + flue_gas_loss + wall_loss
    imbalance = fuel_input - total_output
    if abs(imbalance) > fuel_input * 0.001:
        raise ValueError(
            f"Energy imbalance detected: input={fuel_input}, "
            f"output={total_output}, imbalance={imbalance}"
        )

    generator = ProcessHeatSankeyGenerator(title=title, unit=unit)

    # Add nodes
    generator.add_node(SankeyNode(
        id="fuel_input",
        label="Fuel Input",
        value=fuel_input,
        color=ProcessHeatColorScheme.FUEL_INPUT.value
    ))
    generator.add_node(SankeyNode(
        id="furnace",
        label="Furnace",
        value=fuel_input,
        color=ProcessHeatColorScheme.AMBIENT.value
    ))
    generator.add_node(SankeyNode(
        id="process_heat",
        label="Process Heat",
        value=process_heat,
        color=ProcessHeatColorScheme.USEFUL_HEAT.value
    ))
    generator.add_node(SankeyNode(
        id="flue_gas_loss",
        label="Flue Gas Loss",
        value=flue_gas_loss,
        color=ProcessHeatColorScheme.LOSSES.value
    ))
    generator.add_node(SankeyNode(
        id="wall_loss",
        label="Wall Loss",
        value=wall_loss,
        color=ProcessHeatColorScheme.LOSSES.value
    ))

    # Add flows
    generator.add_flow(SankeyFlow(
        source="fuel_input",
        target="furnace",
        value=fuel_input
    ))
    generator.add_flow(SankeyFlow(
        source="furnace",
        target="process_heat",
        value=process_heat
    ))
    generator.add_flow(SankeyFlow(
        source="furnace",
        target="flue_gas_loss",
        value=flue_gas_loss
    ))
    generator.add_flow(SankeyFlow(
        source="furnace",
        target="wall_loss",
        value=wall_loss
    ))

    # Mark input/output nodes
    generator.mark_as_input("fuel_input")
    generator.mark_as_output("process_heat")
    generator.mark_as_output("flue_gas_loss")
    generator.mark_as_output("wall_loss")

    logger.info(
        f"Created furnace Sankey: efficiency={process_heat/fuel_input*100:.1f}%"
    )

    return generator


def create_heat_recovery_sankey(
    waste_heat_sources: Dict[str, float],
    recovered_heat: float,
    rejected_heat: float,
    title: str = "Heat Recovery System",
    unit: str = "BTU/hr"
) -> ProcessHeatSankeyGenerator:
    """
    Create a heat recovery system Sankey diagram.

    This is a ZERO-HALLUCINATION factory function for heat recovery
    system visualization. Supports multiple waste heat sources.

    Args:
        waste_heat_sources: Dictionary of source_name: heat_value
        recovered_heat: Successfully recovered heat
        rejected_heat: Heat rejected to environment
        title: Diagram title
        unit: Energy unit

    Returns:
        Configured ProcessHeatSankeyGenerator

    Raises:
        ValueError: If energy doesn't balance within tolerance

    Example:
        >>> generator = create_heat_recovery_sankey(
        ...     waste_heat_sources={
        ...         "Furnace Exhaust": 500_000,
        ...         "Compressor Jacket": 200_000,
        ...         "Steam Traps": 100_000
        ...     },
        ...     recovered_heat=600_000,
        ...     rejected_heat=200_000
        ... )
    """
    # Calculate total input
    total_input = sum(waste_heat_sources.values())

    # Validate energy balance
    total_output = recovered_heat + rejected_heat
    imbalance = total_input - total_output
    if abs(imbalance) > total_input * 0.001:
        raise ValueError(
            f"Energy imbalance detected: input={total_input}, "
            f"output={total_output}, imbalance={imbalance}"
        )

    generator = ProcessHeatSankeyGenerator(title=title, unit=unit)

    # Add waste heat source nodes
    for source_name, value in waste_heat_sources.items():
        source_id = source_name.lower().replace(" ", "_")
        generator.add_node(SankeyNode(
            id=source_id,
            label=source_name,
            value=value,
            color=ProcessHeatColorScheme.WASTE_HEAT.value
        ))

    # Add heat exchanger node
    generator.add_node(SankeyNode(
        id="heat_exchanger",
        label="Heat Recovery Unit",
        value=total_input,
        color=ProcessHeatColorScheme.AMBIENT.value
    ))

    # Add output nodes
    generator.add_node(SankeyNode(
        id="recovered_heat",
        label="Recovered Heat",
        value=recovered_heat,
        color=ProcessHeatColorScheme.USEFUL_HEAT.value
    ))
    generator.add_node(SankeyNode(
        id="rejected_heat",
        label="Rejected Heat",
        value=rejected_heat,
        color=ProcessHeatColorScheme.LOSSES.value
    ))

    # Add flows from sources to heat exchanger
    for source_name, value in waste_heat_sources.items():
        source_id = source_name.lower().replace(" ", "_")
        generator.add_flow(SankeyFlow(
            source=source_id,
            target="heat_exchanger",
            value=value
        ))
        generator.mark_as_input(source_id)

    # Add flows from heat exchanger to outputs
    generator.add_flow(SankeyFlow(
        source="heat_exchanger",
        target="recovered_heat",
        value=recovered_heat
    ))
    generator.add_flow(SankeyFlow(
        source="heat_exchanger",
        target="rejected_heat",
        value=rejected_heat
    ))

    # Mark output nodes
    generator.mark_as_output("recovered_heat")
    generator.mark_as_output("rejected_heat")

    recovery_efficiency = (recovered_heat / total_input * 100) if total_input > 0 else 0
    logger.info(
        f"Created heat recovery Sankey: recovery_efficiency={recovery_efficiency:.1f}%"
    )

    return generator


def create_steam_system_sankey(
    steam_generation: float,
    steam_uses: Dict[str, float],
    condensate_return: float,
    losses: Dict[str, float],
    title: str = "Steam Distribution System",
    unit: str = "BTU/hr"
) -> ProcessHeatSankeyGenerator:
    """
    Create a steam distribution system Sankey diagram.

    This is a ZERO-HALLUCINATION factory function for steam system
    visualization including generation, uses, condensate return, and losses.

    Args:
        steam_generation: Total steam generation
        steam_uses: Dictionary of use_name: steam_value
        condensate_return: Condensate return energy
        losses: Dictionary of loss_name: loss_value
        title: Diagram title
        unit: Energy unit

    Returns:
        Configured ProcessHeatSankeyGenerator

    Raises:
        ValueError: If energy doesn't balance within tolerance

    Example:
        >>> generator = create_steam_system_sankey(
        ...     steam_generation=2_000_000,
        ...     steam_uses={
        ...         "Process Heating": 800_000,
        ...         "Building Heat": 400_000,
        ...         "Deaerator": 200_000
        ...     },
        ...     condensate_return=300_000,
        ...     losses={
        ...         "Trap Losses": 100_000,
        ...         "Distribution Losses": 150_000,
        ...         "Flash Losses": 50_000
        ...     }
        ... )
    """
    # Calculate totals
    total_uses = sum(steam_uses.values())
    total_losses = sum(losses.values())

    # Validate energy balance
    # Steam in = Steam uses + Condensate return + Losses
    total_output = total_uses + condensate_return + total_losses
    imbalance = steam_generation - total_output
    if abs(imbalance) > steam_generation * 0.001:
        raise ValueError(
            f"Energy imbalance detected: generation={steam_generation}, "
            f"total_output={total_output}, imbalance={imbalance}"
        )

    generator = ProcessHeatSankeyGenerator(title=title, unit=unit)

    # Add steam generation node
    generator.add_node(SankeyNode(
        id="steam_generation",
        label="Steam Generation",
        value=steam_generation,
        color=ProcessHeatColorScheme.STEAM.value
    ))

    # Add steam header node
    generator.add_node(SankeyNode(
        id="steam_header",
        label="Steam Header",
        value=steam_generation,
        color=ProcessHeatColorScheme.STEAM.value
    ))

    # Add flow from generation to header
    generator.add_flow(SankeyFlow(
        source="steam_generation",
        target="steam_header",
        value=steam_generation
    ))
    generator.mark_as_input("steam_generation")

    # Add steam use nodes and flows
    for use_name, value in steam_uses.items():
        use_id = use_name.lower().replace(" ", "_")
        generator.add_node(SankeyNode(
            id=use_id,
            label=use_name,
            value=value,
            color=ProcessHeatColorScheme.USEFUL_HEAT.value
        ))
        generator.add_flow(SankeyFlow(
            source="steam_header",
            target=use_id,
            value=value
        ))
        generator.mark_as_output(use_id)

    # Add condensate return node
    generator.add_node(SankeyNode(
        id="condensate_return",
        label="Condensate Return",
        value=condensate_return,
        color=ProcessHeatColorScheme.CONDENSATE.value
    ))
    generator.add_flow(SankeyFlow(
        source="steam_header",
        target="condensate_return",
        value=condensate_return
    ))
    generator.mark_as_output("condensate_return")

    # Add loss nodes and flows
    for loss_name, value in losses.items():
        loss_id = loss_name.lower().replace(" ", "_")
        generator.add_node(SankeyNode(
            id=loss_id,
            label=loss_name,
            value=value,
            color=ProcessHeatColorScheme.LOSSES.value
        ))
        generator.add_flow(SankeyFlow(
            source="steam_header",
            target=loss_id,
            value=value
        ))
        generator.mark_as_output(loss_id)

    utilization = (total_uses / steam_generation * 100) if steam_generation > 0 else 0
    logger.info(
        f"Created steam system Sankey: utilization={utilization:.1f}%"
    )

    return generator
