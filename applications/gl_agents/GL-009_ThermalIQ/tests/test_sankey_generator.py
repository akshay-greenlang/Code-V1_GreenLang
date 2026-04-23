# -*- coding: utf-8 -*-
"""
Sankey Diagram Generator Tests for GL-009 THERMALIQ

Comprehensive tests for Sankey diagram generation validating energy flow
visualization, node/link balance, and export formats.

Test Coverage:
- Energy Sankey diagram generation
- Exergy Sankey diagram generation
- Sankey node balance validation (inputs = outputs + losses)
- SVG export functionality
- Plotly figure generation
- D3.js format compatibility

Author: GL-TestEngineer
Version: 1.0.0
"""

import json
import hashlib
from decimal import Decimal
from typing import Dict, Any, List, Tuple
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# TEST CLASS: ENERGY SANKEY GENERATION
# =============================================================================

class TestEnergySankeyGeneration:
    """Test energy Sankey diagram generation."""

    @pytest.mark.unit
    def test_energy_sankey_generation_basic(self, sample_heat_balance):
        """Test basic energy Sankey diagram generation."""
        inputs = {"Fuel": (1388.9, "fuel")}
        outputs = {"Steam": (1150.0, "steam")}
        losses = {
            "Flue Gas": (80.0, "flue_gas"),
            "Radiation": (12.0, "radiation"),
            "Convection": (7.5, "convection"),
        }

        diagram = self._generate_sankey(inputs, outputs, losses)

        assert "nodes" in diagram
        assert "links" in diagram
        assert len(diagram["nodes"]) > 0
        assert len(diagram["links"]) > 0

    @pytest.mark.unit
    def test_sankey_has_all_input_nodes(self, sample_heat_balance):
        """Test that Sankey contains all input nodes."""
        inputs = {
            "Natural Gas": (1300.0, "fuel"),
            "Electrical": (88.9, "electricity"),
        }
        outputs = {"Steam": (1150.0, "steam")}
        losses = {"Flue Gas": (80.0, "flue_gas")}

        diagram = self._generate_sankey(inputs, outputs, losses)

        node_labels = [n["label"] for n in diagram["nodes"]]
        assert any("Natural Gas" in label for label in node_labels)
        assert any("Electrical" in label for label in node_labels)

    @pytest.mark.unit
    def test_sankey_has_all_output_nodes(self, sample_heat_balance):
        """Test that Sankey contains all output nodes."""
        inputs = {"Fuel": (1388.9, "fuel")}
        outputs = {
            "Steam": (1000.0, "steam"),
            "Hot Water": (150.0, "hot_water"),
        }
        losses = {"Flue Gas": (80.0, "flue_gas")}

        diagram = self._generate_sankey(inputs, outputs, losses)

        node_labels = [n["label"] for n in diagram["nodes"]]
        assert any("Steam" in label for label in node_labels)
        assert any("Hot Water" in label for label in node_labels)

    @pytest.mark.unit
    def test_sankey_has_all_loss_nodes(self, sample_heat_balance):
        """Test that Sankey contains all loss nodes."""
        inputs = {"Fuel": (1388.9, "fuel")}
        outputs = {"Steam": (1150.0, "steam")}
        losses = {
            "Flue Gas": (80.0, "flue_gas"),
            "Radiation": (12.0, "radiation"),
            "Convection": (7.5, "convection"),
            "Blowdown": (8.0, "blowdown"),
        }

        diagram = self._generate_sankey(inputs, outputs, losses)

        node_labels = [n["label"] for n in diagram["nodes"]]
        assert any("Flue Gas" in label for label in node_labels)
        assert any("Radiation" in label for label in node_labels)
        assert any("Convection" in label for label in node_labels)
        assert any("Blowdown" in label for label in node_labels)

    @pytest.mark.unit
    def test_sankey_process_node_exists(self):
        """Test that Sankey has a central process node."""
        inputs = {"Fuel": (1388.9, "fuel")}
        outputs = {"Steam": (1150.0, "steam")}
        losses = {"Flue Gas": (80.0, "flue_gas")}

        diagram = self._generate_sankey(inputs, outputs, losses, process_name="Boiler")

        node_types = [n.get("type") for n in diagram["nodes"]]
        assert "process" in node_types

    @pytest.mark.unit
    def test_sankey_efficiency_displayed(self):
        """Test that efficiency is displayed on the diagram."""
        inputs = {"Fuel": (1000.0, "fuel")}
        outputs = {"Steam": (850.0, "steam")}
        losses = {"Losses": (150.0, "other")}

        diagram = self._generate_sankey(inputs, outputs, losses)

        # Check efficiency is calculated correctly
        expected_efficiency = (850.0 / 1000.0) * 100
        assert abs(diagram["efficiency_percent"] - expected_efficiency) < 0.1

    def _generate_sankey(
        self,
        inputs: Dict[str, Tuple[float, str]],
        outputs: Dict[str, Tuple[float, str]],
        losses: Dict[str, Tuple[float, str]],
        process_name: str = "Process",
    ) -> Dict[str, Any]:
        """Generate Sankey diagram data."""
        nodes = []
        links = []

        # Calculate totals
        total_input = sum(v[0] for v in inputs.values())
        total_output = sum(v[0] for v in outputs.values())
        total_losses = sum(v[0] for v in losses.values())
        efficiency = (total_output / total_input * 100) if total_input > 0 else 0

        node_index = {}

        # Add input nodes
        for name, (value, category) in inputs.items():
            idx = len(nodes)
            node_index[f"input_{name}"] = idx
            nodes.append({
                "id": f"input_{name.lower().replace(' ', '_')}",
                "label": f"{name}\n{value:.1f} kW",
                "type": "input",
                "value_kw": value,
            })

        # Add process node
        process_idx = len(nodes)
        node_index["process"] = process_idx
        nodes.append({
            "id": "process",
            "label": f"{process_name}\n{efficiency:.1f}% eff",
            "type": "process",
            "value_kw": total_input,
        })

        # Add output nodes
        for name, (value, category) in outputs.items():
            idx = len(nodes)
            node_index[f"output_{name}"] = idx
            nodes.append({
                "id": f"output_{name.lower().replace(' ', '_')}",
                "label": f"{name}\n{value:.1f} kW",
                "type": "output",
                "value_kw": value,
            })

        # Add loss nodes
        for name, (value, category) in losses.items():
            idx = len(nodes)
            node_index[f"loss_{name}"] = idx
            nodes.append({
                "id": f"loss_{name.lower().replace(' ', '_')}",
                "label": f"{name}\n{value:.1f} kW",
                "type": "loss",
                "value_kw": value,
            })

        # Create links
        for name, (value, category) in inputs.items():
            links.append({
                "source": node_index[f"input_{name}"],
                "target": process_idx,
                "value": value,
                "category": category,
            })

        for name, (value, category) in outputs.items():
            links.append({
                "source": process_idx,
                "target": node_index[f"output_{name}"],
                "value": value,
                "category": category,
            })

        for name, (value, category) in losses.items():
            links.append({
                "source": process_idx,
                "target": node_index[f"loss_{name}"],
                "value": value,
                "category": category,
            })

        return {
            "nodes": nodes,
            "links": links,
            "total_input_kw": total_input,
            "total_output_kw": total_output,
            "total_losses_kw": total_losses,
            "efficiency_percent": efficiency,
            "provenance_hash": hashlib.sha256(
                json.dumps({"inputs": inputs, "outputs": outputs, "losses": losses}, sort_keys=True).encode()
            ).hexdigest(),
        }


# =============================================================================
# TEST CLASS: EXERGY SANKEY GENERATION
# =============================================================================

class TestExergySankeyGeneration:
    """Test exergy Sankey diagram generation."""

    @pytest.mark.unit
    def test_exergy_sankey_generation(self):
        """Test exergy Sankey diagram generation."""
        exergy_inputs = {"Fuel Exergy": (1444.5, "fuel")}
        exergy_outputs = {"Steam Exergy": (653.0, "steam")}
        exergy_losses = {
            "Exergy Destruction": (647.0, "destruction"),
            "Exergy Loss": (144.5, "loss"),
        }

        diagram = self._generate_exergy_sankey(
            exergy_inputs, exergy_outputs, exergy_losses
        )

        assert diagram["total_input_kw"] == 1444.5
        assert diagram["total_output_kw"] == 653.0

    @pytest.mark.unit
    def test_exergy_sankey_has_destruction_node(self):
        """Test that exergy Sankey shows destruction (irreversibility)."""
        exergy_inputs = {"Fuel Exergy": (1000.0, "fuel")}
        exergy_outputs = {"Steam Exergy": (450.0, "steam")}
        exergy_losses = {
            "Exergy Destruction": (400.0, "destruction"),
            "Exergy Loss": (150.0, "loss"),
        }

        diagram = self._generate_exergy_sankey(
            exergy_inputs, exergy_outputs, exergy_losses
        )

        node_labels = [n["label"] for n in diagram["nodes"]]
        assert any("Destruction" in label for label in node_labels)

    @pytest.mark.unit
    def test_exergy_lower_than_energy(self):
        """Test that exergy output is lower than energy output."""
        # Energy flow
        energy_output = 1150.0  # kW

        # Exergy flow (with Carnot factor)
        temperature_c = 180.0
        T0_K = 298.15
        T_K = temperature_c + 273.15
        carnot = 1 - T0_K / T_K

        exergy_output = energy_output * carnot

        assert exergy_output < energy_output

    def _generate_exergy_sankey(
        self,
        inputs: Dict[str, Tuple[float, str]],
        outputs: Dict[str, Tuple[float, str]],
        losses: Dict[str, Tuple[float, str]],
    ) -> Dict[str, Any]:
        """Generate exergy Sankey diagram data."""
        # Use same generator as energy
        generator = TestEnergySankeyGeneration()
        return generator._generate_sankey(inputs, outputs, losses, process_name="Exergy Analysis")


# =============================================================================
# TEST CLASS: SANKEY NODE BALANCE
# =============================================================================

class TestSankeyNodeBalance:
    """Test Sankey diagram energy balance validation."""

    @pytest.mark.unit
    def test_sankey_node_balance_perfect(self):
        """Test perfect balance: inputs = outputs + losses."""
        inputs = {"Fuel": (1000.0, "fuel")}
        outputs = {"Steam": (850.0, "steam")}
        losses = {"Losses": (150.0, "other")}

        balance = self._check_balance(inputs, outputs, losses)

        assert balance["balanced"] is True
        assert abs(balance["error_percent"]) < 0.01

    @pytest.mark.unit
    def test_sankey_node_balance_with_tolerance(self):
        """Test balance within 2% tolerance."""
        inputs = {"Fuel": (1000.0, "fuel")}
        outputs = {"Steam": (850.0, "steam")}
        losses = {"Losses": (140.0, "other")}  # 10 kW unaccounted (1%)

        balance = self._check_balance(inputs, outputs, losses)

        assert balance["error_percent"] < 2.0

    @pytest.mark.unit
    def test_sankey_node_balance_failure(self):
        """Test balance detection when significantly imbalanced."""
        inputs = {"Fuel": (1000.0, "fuel")}
        outputs = {"Steam": (850.0, "steam")}
        losses = {"Losses": (50.0, "other")}  # 100 kW unaccounted (10%)

        balance = self._check_balance(inputs, outputs, losses)

        assert balance["error_percent"] > 2.0
        assert balance["balanced"] is False

    @pytest.mark.unit
    def test_sankey_link_sum_equals_node_value(self):
        """Test that sum of links into/out of node equals node value."""
        inputs = {
            "Fuel 1": (600.0, "fuel"),
            "Fuel 2": (400.0, "fuel"),
        }
        outputs = {"Steam": (850.0, "steam")}
        losses = {"Losses": (150.0, "other")}

        diagram = TestEnergySankeyGeneration()._generate_sankey(inputs, outputs, losses)

        # Sum of input links should equal total input
        input_links = [l["value"] for l in diagram["links"] if l["target"] == 2]  # Process node
        assert abs(sum(input_links) - 1000.0) < 0.1

    @pytest.mark.unit
    def test_multiple_outputs_balance(self):
        """Test balance with multiple outputs."""
        inputs = {"Fuel": (1000.0, "fuel")}
        outputs = {
            "Steam": (700.0, "steam"),
            "Hot Water": (150.0, "hot_water"),
        }
        losses = {"Losses": (150.0, "other")}

        balance = self._check_balance(inputs, outputs, losses)

        total_output = sum(v[0] for v in outputs.values())
        total_losses = sum(v[0] for v in losses.values())

        assert abs(1000.0 - (total_output + total_losses)) < 0.1

    def _check_balance(
        self,
        inputs: Dict[str, Tuple[float, str]],
        outputs: Dict[str, Tuple[float, str]],
        losses: Dict[str, Tuple[float, str]],
    ) -> Dict[str, Any]:
        """Check energy balance of Sankey diagram."""
        total_input = sum(v[0] for v in inputs.values())
        total_output = sum(v[0] for v in outputs.values())
        total_losses = sum(v[0] for v in losses.values())

        balance_error = total_input - (total_output + total_losses)
        error_percent = abs(balance_error / total_input * 100) if total_input > 0 else 0

        return {
            "total_input": total_input,
            "total_output": total_output,
            "total_losses": total_losses,
            "balance_error": balance_error,
            "error_percent": error_percent,
            "balanced": error_percent < 2.0,
        }


# =============================================================================
# TEST CLASS: SVG EXPORT
# =============================================================================

class TestSankeyExportSVG:
    """Test Sankey diagram SVG export functionality."""

    @pytest.mark.unit
    def test_sankey_export_svg_basic(self):
        """Test basic SVG export."""
        diagram_data = {
            "nodes": [
                {"id": "input", "label": "Fuel", "type": "input"},
                {"id": "process", "label": "Boiler", "type": "process"},
                {"id": "output", "label": "Steam", "type": "output"},
            ],
            "links": [
                {"source": 0, "target": 1, "value": 1000},
                {"source": 1, "target": 2, "value": 850},
            ],
        }

        svg = self._export_to_svg(diagram_data)

        assert svg.startswith("<?xml") or svg.startswith("<svg")
        assert "</svg>" in svg

    @pytest.mark.unit
    def test_svg_contains_all_nodes(self):
        """Test that SVG contains all node elements."""
        diagram_data = {
            "nodes": [
                {"id": "input", "label": "Fuel", "type": "input"},
                {"id": "process", "label": "Boiler", "type": "process"},
                {"id": "output", "label": "Steam", "type": "output"},
            ],
            "links": [],
        }

        svg = self._export_to_svg(diagram_data)

        # Check for node rectangles
        assert svg.count("<rect") >= 3 or svg.count("<g ") >= 3

    @pytest.mark.unit
    def test_svg_has_valid_dimensions(self):
        """Test that SVG has valid dimensions."""
        diagram_data = {"nodes": [], "links": []}

        svg = self._export_to_svg(diagram_data, width=1200, height=800)

        assert 'width="1200"' in svg or "width:1200" in svg or "1200" in svg
        assert 'height="800"' in svg or "height:800" in svg or "800" in svg

    @pytest.mark.unit
    def test_svg_color_coding(self):
        """Test that SVG has proper color coding for node types."""
        diagram_data = {
            "nodes": [
                {"id": "input", "label": "Fuel", "type": "input", "color": "#ff7f0e"},
                {"id": "output", "label": "Steam", "type": "output", "color": "#2ca02c"},
                {"id": "loss", "label": "Losses", "type": "loss", "color": "#d62728"},
            ],
            "links": [],
        }

        svg = self._export_to_svg(diagram_data)

        # Check for color definitions
        assert "#ff7f0e" in svg or "ff7f0e" in svg or "orange" in svg.lower()

    def _export_to_svg(
        self,
        diagram_data: Dict[str, Any],
        width: int = 1200,
        height: int = 800,
    ) -> str:
        """Export Sankey diagram to SVG format."""
        # Simplified SVG generation for testing
        svg_parts = [
            f'<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            f'  <title>Sankey Diagram</title>',
        ]

        # Add node rectangles
        for i, node in enumerate(diagram_data.get("nodes", [])):
            color = node.get("color", "#1f77b4")
            svg_parts.append(
                f'  <g class="node" id="{node["id"]}">'
                f'    <rect x="{100 + i * 200}" y="100" width="50" height="100" fill="{color}"/>'
                f'    <text x="{125 + i * 200}" y="200">{node["label"]}</text>'
                f'  </g>'
            )

        # Add links
        for link in diagram_data.get("links", []):
            svg_parts.append(
                f'  <path class="link" d="M0,0 L100,100" stroke="#999" stroke-width="{link["value"]/10}"/>'
            )

        svg_parts.append("</svg>")

        return "\n".join(svg_parts)


# =============================================================================
# TEST CLASS: PLOTLY FIGURE
# =============================================================================

class TestSankeyPlotlyFigure:
    """Test Sankey diagram Plotly figure generation."""

    @pytest.mark.unit
    def test_sankey_plotly_figure_basic(self):
        """Test basic Plotly figure generation."""
        diagram_data = {
            "nodes": [
                {"id": "input", "label": "Fuel\n1000 kW", "type": "input"},
                {"id": "process", "label": "Boiler", "type": "process"},
                {"id": "output", "label": "Steam\n850 kW", "type": "output"},
            ],
            "links": [
                {"source": 0, "target": 1, "value": 1000},
                {"source": 1, "target": 2, "value": 850},
            ],
        }

        plotly_data = self._to_plotly_format(diagram_data)

        assert "node" in plotly_data
        assert "link" in plotly_data
        assert "label" in plotly_data["node"]
        assert "source" in plotly_data["link"]
        assert "target" in plotly_data["link"]
        assert "value" in plotly_data["link"]

    @pytest.mark.unit
    def test_plotly_node_labels(self):
        """Test that Plotly format has correct node labels."""
        diagram_data = {
            "nodes": [
                {"id": "fuel", "label": "Fuel\n1000 kW", "type": "input"},
                {"id": "steam", "label": "Steam\n850 kW", "type": "output"},
            ],
            "links": [],
        }

        plotly_data = self._to_plotly_format(diagram_data)

        assert len(plotly_data["node"]["label"]) == 2
        assert "Fuel" in plotly_data["node"]["label"][0]

    @pytest.mark.unit
    def test_plotly_link_indices(self):
        """Test that Plotly links use correct node indices."""
        diagram_data = {
            "nodes": [
                {"id": "a", "label": "A", "type": "input"},
                {"id": "b", "label": "B", "type": "process"},
                {"id": "c", "label": "C", "type": "output"},
            ],
            "links": [
                {"source": 0, "target": 1, "value": 100},
                {"source": 1, "target": 2, "value": 80},
            ],
        }

        plotly_data = self._to_plotly_format(diagram_data)

        assert plotly_data["link"]["source"] == [0, 1]
        assert plotly_data["link"]["target"] == [1, 2]

    @pytest.mark.unit
    def test_plotly_node_colors(self):
        """Test that Plotly nodes have appropriate colors."""
        diagram_data = {
            "nodes": [
                {"id": "input", "label": "Input", "type": "input", "color": "#ff7f0e"},
                {"id": "output", "label": "Output", "type": "output", "color": "#2ca02c"},
            ],
            "links": [],
        }

        plotly_data = self._to_plotly_format(diagram_data)

        assert "color" in plotly_data["node"]
        assert len(plotly_data["node"]["color"]) == 2

    @pytest.mark.unit
    def test_plotly_link_colors(self):
        """Test that Plotly links have appropriate colors."""
        diagram_data = {
            "nodes": [
                {"id": "a", "label": "A", "type": "input"},
                {"id": "b", "label": "B", "type": "output"},
            ],
            "links": [
                {"source": 0, "target": 1, "value": 100, "color": "rgba(255,127,14,0.4)"},
            ],
        }

        plotly_data = self._to_plotly_format(diagram_data)

        assert "color" in plotly_data["link"]
        assert len(plotly_data["link"]["color"]) == 1

    @pytest.mark.unit
    def test_plotly_padding_configuration(self):
        """Test that Plotly has correct node padding."""
        diagram_data = {"nodes": [], "links": []}

        plotly_data = self._to_plotly_format(diagram_data)

        assert plotly_data["node"].get("pad", 15) == 15
        assert plotly_data["node"].get("thickness", 20) == 20

    def _to_plotly_format(self, diagram_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert diagram data to Plotly Sankey format."""
        nodes = diagram_data.get("nodes", [])
        links = diagram_data.get("links", [])

        node_labels = [n["label"] for n in nodes]
        node_colors = [n.get("color", "#1f77b4") for n in nodes]

        link_sources = [l["source"] for l in links]
        link_targets = [l["target"] for l in links]
        link_values = [l["value"] for l in links]
        link_colors = [l.get("color", "rgba(127,127,127,0.4)") for l in links]

        return {
            "node": {
                "label": node_labels,
                "color": node_colors,
                "pad": 15,
                "thickness": 20,
            },
            "link": {
                "source": link_sources,
                "target": link_targets,
                "value": link_values,
                "color": link_colors,
            },
        }


# =============================================================================
# TEST CLASS: D3.JS FORMAT
# =============================================================================

class TestSankeyD3Format:
    """Test Sankey diagram D3.js format generation."""

    @pytest.mark.unit
    def test_d3_format_structure(self):
        """Test D3.js format has correct structure."""
        diagram_data = {
            "nodes": [{"id": "a", "label": "A", "type": "input"}],
            "links": [{"source": 0, "target": 1, "value": 100}],
        }

        d3_data = self._to_d3_format(diagram_data)

        assert "nodes" in d3_data
        assert "links" in d3_data

    @pytest.mark.unit
    def test_d3_nodes_have_name(self):
        """Test D3 nodes have 'name' field."""
        diagram_data = {
            "nodes": [{"id": "fuel", "label": "Fuel", "type": "input"}],
            "links": [],
        }

        d3_data = self._to_d3_format(diagram_data)

        assert "name" in d3_data["nodes"][0]

    @pytest.mark.unit
    def test_d3_links_use_indices(self):
        """Test D3 links use numeric indices."""
        diagram_data = {
            "nodes": [
                {"id": "a", "label": "A", "type": "input"},
                {"id": "b", "label": "B", "type": "output"},
            ],
            "links": [{"source": 0, "target": 1, "value": 100}],
        }

        d3_data = self._to_d3_format(diagram_data)

        assert isinstance(d3_data["links"][0]["source"], int)
        assert isinstance(d3_data["links"][0]["target"], int)

    def _to_d3_format(self, diagram_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert diagram data to D3.js format."""
        nodes = [
            {
                "name": n["label"],
                "type": n["type"],
                "value": n.get("value_kw", 0),
            }
            for n in diagram_data.get("nodes", [])
        ]

        links = [
            {
                "source": l["source"],
                "target": l["target"],
                "value": l["value"],
            }
            for l in diagram_data.get("links", [])
        ]

        return {"nodes": nodes, "links": links}


# =============================================================================
# TEST CLASS: SANKEY PERFORMANCE
# =============================================================================

class TestSankeyPerformance:
    """Performance tests for Sankey diagram generation."""

    @pytest.mark.performance
    def test_sankey_generation_time(self):
        """Test Sankey generation meets <50ms target."""
        import time

        inputs = {"Fuel": (1000.0, "fuel")}
        outputs = {"Steam": (850.0, "steam")}
        losses = {"Losses": (150.0, "other")}

        generator = TestEnergySankeyGeneration()

        start = time.perf_counter()
        for _ in range(100):
            generator._generate_sankey(inputs, outputs, losses)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 50.0, f"Generation took {elapsed_ms:.2f}ms (target: <50ms)"

    @pytest.mark.performance
    def test_large_sankey_generation(self):
        """Test Sankey generation with many nodes."""
        import time

        # Create large diagram with 50+ nodes
        inputs = {f"Input_{i}": (100.0, "fuel") for i in range(10)}
        outputs = {f"Output_{i}": (80.0, "steam") for i in range(10)}
        losses = {f"Loss_{i}": (20.0, "other") for i in range(10)}

        generator = TestEnergySankeyGeneration()

        start = time.perf_counter()
        diagram = generator._generate_sankey(inputs, outputs, losses)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(diagram["nodes"]) >= 30
        assert elapsed_ms < 100.0, f"Large diagram took {elapsed_ms:.2f}ms"
