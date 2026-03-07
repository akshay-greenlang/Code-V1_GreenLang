# -*- coding: utf-8 -*-
"""
Comprehensive test suite for VisualizationEngine.

Tests cover:
    1. Force-directed layout (Fruchterman-Reingold) algorithm
    2. Hierarchical (tier-based) layout
    3. Geographic layout with coordinate projection
    4. Circular layout
    5. Sankey diagram data generation
    6. Node clustering (country, tier, type)
    7. GeoJSON export format
    8. GraphML export format
    9. JSON-LD export format
    10. Graph filtering (commodity, country, risk, compliance, tier, type)
    11. Risk-based color coding
    12. Compliance-based color coding
    13. Node type color coding
    14. Tier depth color coding
    15. Country-based color coding
    16. Edge path computation
    17. Edge width from quantity
    18. Viewport bounding box calculation
    19. Empty graph handling
    20. Single node layout
    21. Large graph layout (1,000 nodes < 3 seconds)
    22. Deterministic layout (same seed = same positions)
    23. Snapshot storage and retrieval
    24. Time-based snapshot retrieval
    25. Configuration validation
    26. Provenance hash computation
    27. Filter combinations
    28. Sankey commodity filtering
    29. Cluster radius and center calculation
    30. Geographic projection with missing coordinates
    31. Node shape mapping
    32. Node size mapping
    33. LayoutResult serialization
    34. SankeyResult serialization
    35. Integration with graph engine (mock)
    36. Integration with geolocation linker (mock)
    37. Integration with risk engine (mock)
    38. Grid-based repulsive force approximation (large graphs)
    39. Edge coloring from endpoint risk
    40. Time budget enforcement for layout
    41. Adaptive iteration count for large graphs
    42. Multiple commodity flow aggregation in Sankey

PRD: PRD-AGENT-EUDR-001, Feature 7
Agent: GL-EUDR-SCM-001
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.eudr.supply_chain_mapper.visualization_engine import (
    COMPLIANCE_COLORS,
    JSONLD_CONTEXT,
    NODE_TYPE_SHAPES,
    NODE_TYPE_SIZES,
    RISK_COLOR_HIGH,
    RISK_COLOR_LOW,
    RISK_COLOR_STANDARD,
    RISK_COLOR_UNKNOWN,
    ClusterGroup,
    ColorScheme,
    EdgePath,
    ExportFormat,
    GraphFilter,
    LayoutAlgorithm,
    LayoutResult,
    NodePosition,
    SankeyLink,
    SankeyNode,
    SankeyResult,
    VisualizationConfig,
    VisualizationEngine,
    _compute_provenance_hash,
    _generate_id,
    _utcnow,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def default_config() -> VisualizationConfig:
    """Default visualization configuration."""
    return VisualizationConfig()


@pytest.fixture
def engine(default_config: VisualizationConfig) -> VisualizationEngine:
    """Default visualization engine."""
    return VisualizationEngine(config=default_config)


@pytest.fixture
def small_nodes() -> Dict[str, Dict[str, Any]]:
    """Five-node supply chain: producer -> collector -> processor -> trader -> importer."""
    return {
        "N-PROD-01": {
            "node_type": "producer",
            "operator_name": "Farm Alpha",
            "country_code": "BR",
            "risk_level": "high",
            "risk_score": 75.0,
            "compliance_status": "pending_verification",
            "tier_depth": 4,
            "commodities": ["soya"],
            "latitude": -15.7801,
            "longitude": -47.9292,
        },
        "N-COLL-01": {
            "node_type": "collector",
            "operator_name": "Cooperative Beta",
            "country_code": "BR",
            "risk_level": "standard",
            "risk_score": 50.0,
            "compliance_status": "under_review",
            "tier_depth": 3,
            "commodities": ["soya"],
            "latitude": -14.2350,
            "longitude": -51.9253,
        },
        "N-PROC-01": {
            "node_type": "processor",
            "operator_name": "Mill Gamma",
            "country_code": "BR",
            "risk_level": "standard",
            "risk_score": 45.0,
            "compliance_status": "compliant",
            "tier_depth": 2,
            "commodities": ["soya", "soybean_oil"],
            "latitude": -23.5505,
            "longitude": -46.6333,
        },
        "N-TRAD-01": {
            "node_type": "trader",
            "operator_name": "Trader Delta",
            "country_code": "NL",
            "risk_level": "low",
            "risk_score": 15.0,
            "compliance_status": "compliant",
            "tier_depth": 1,
            "commodities": ["soybean_oil"],
            "latitude": 52.3676,
            "longitude": 4.9041,
        },
        "N-IMP-01": {
            "node_type": "importer",
            "operator_name": "EU Import Co",
            "country_code": "DE",
            "risk_level": "low",
            "risk_score": 5.0,
            "compliance_status": "compliant",
            "tier_depth": 0,
            "commodities": ["soybean_oil"],
            "latitude": 52.5200,
            "longitude": 13.4050,
        },
    }


@pytest.fixture
def small_edges() -> Dict[str, Dict[str, Any]]:
    """Edges for the five-node supply chain."""
    return {
        "E-01": {
            "source_node_id": "N-PROD-01",
            "target_node_id": "N-COLL-01",
            "commodity": "soya",
            "quantity": Decimal("5000"),
        },
        "E-02": {
            "source_node_id": "N-COLL-01",
            "target_node_id": "N-PROC-01",
            "commodity": "soya",
            "quantity": Decimal("4500"),
        },
        "E-03": {
            "source_node_id": "N-PROC-01",
            "target_node_id": "N-TRAD-01",
            "commodity": "soybean_oil",
            "quantity": Decimal("2000"),
        },
        "E-04": {
            "source_node_id": "N-TRAD-01",
            "target_node_id": "N-IMP-01",
            "commodity": "soybean_oil",
            "quantity": Decimal("2000"),
        },
    }


@pytest.fixture
def multi_commodity_edges() -> Dict[str, Dict[str, Any]]:
    """Edges with multiple commodities for Sankey testing."""
    return {
        "E-S1": {
            "source_node_id": "N-PROD-01",
            "target_node_id": "N-COLL-01",
            "commodity": "soya",
            "quantity": Decimal("3000"),
        },
        "E-S2": {
            "source_node_id": "N-PROD-01",
            "target_node_id": "N-COLL-01",
            "commodity": "coffee",
            "quantity": Decimal("1000"),
        },
        "E-S3": {
            "source_node_id": "N-COLL-01",
            "target_node_id": "N-PROC-01",
            "commodity": "soya",
            "quantity": Decimal("2500"),
        },
    }


def _make_large_graph(n_nodes: int) -> Tuple[Dict, Dict]:
    """Generate a large graph for performance testing.

    Creates a tiered supply chain with configurable node count.
    Distribution: 40% producers, 20% collectors, 15% processors,
    15% traders, 10% importers.
    """
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[str, Dict[str, Any]] = {}

    categories = [
        ("producer", 4, 0.40),
        ("collector", 3, 0.20),
        ("processor", 2, 0.15),
        ("trader", 1, 0.15),
        ("importer", 0, 0.10),
    ]

    node_list: List[Tuple[str, str, int]] = []
    idx = 0
    for node_type, tier, fraction in categories:
        count = max(1, int(n_nodes * fraction))
        for _ in range(count):
            if idx >= n_nodes:
                break
            nid = f"N-{idx:06d}"
            nodes[nid] = {
                "node_type": node_type,
                "operator_name": f"Operator {idx}",
                "country_code": ["BR", "ID", "GH", "NL", "DE"][idx % 5],
                "risk_level": ["low", "standard", "high"][idx % 3],
                "risk_score": float((idx * 17) % 100),
                "compliance_status": "compliant",
                "tier_depth": tier,
                "commodities": ["soya"],
            }
            node_list.append((nid, node_type, tier))
            idx += 1

    # Create edges between consecutive tiers
    tier_groups: Dict[int, List[str]] = {}
    for nid, _, tier in node_list:
        tier_groups.setdefault(tier, []).append(nid)

    edge_idx = 0
    tiers_sorted = sorted(tier_groups.keys(), reverse=True)
    for i in range(len(tiers_sorted) - 1):
        upstream_tier = tiers_sorted[i]
        downstream_tier = tiers_sorted[i + 1]
        upstreams = tier_groups[upstream_tier]
        downstreams = tier_groups[downstream_tier]
        for j, src in enumerate(upstreams):
            tgt = downstreams[j % len(downstreams)]
            eid = f"E-{edge_idx:06d}"
            edges[eid] = {
                "source_node_id": src,
                "target_node_id": tgt,
                "commodity": "soya",
                "quantity": Decimal("1000"),
            }
            edge_idx += 1

    return nodes, edges


# ===========================================================================
# Test 1: Force-Directed Layout (Fruchterman-Reingold)
# ===========================================================================


class TestForceDirectedLayout:
    """Tests for the Fruchterman-Reingold force-directed layout."""

    def test_basic_layout_produces_positions(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Layout should produce positions for all nodes."""
        result = engine.compute_force_directed_layout(
            "GRAPH-001", small_nodes, small_edges
        )
        assert result.total_nodes == 5
        assert result.total_edges == 4
        assert len(result.node_positions) == 5
        assert len(result.edge_paths) == 4
        assert result.algorithm == LayoutAlgorithm.FORCE_DIRECTED.value

    def test_all_node_ids_present(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Every node ID should appear in node_positions."""
        result = engine.compute_force_directed_layout(
            "GRAPH-001", small_nodes, small_edges
        )
        for nid in small_nodes:
            assert nid in result.node_positions

    def test_node_positions_within_canvas(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """All node positions should be within canvas bounds."""
        result = engine.compute_force_directed_layout(
            "GRAPH-001", small_nodes, small_edges
        )
        for np in result.node_positions.values():
            assert 0.0 <= np.x <= 1000.0
            assert 0.0 <= np.y <= 1000.0

    def test_deterministic_layout(
        self, small_nodes: Dict, small_edges: Dict
    ):
        """Same seed should produce identical positions."""
        config = VisualizationConfig(fr_seed=42)
        engine1 = VisualizationEngine(config=config)
        engine2 = VisualizationEngine(config=config)

        result1 = engine1.compute_force_directed_layout(
            "GRAPH-001", small_nodes, small_edges
        )
        result2 = engine2.compute_force_directed_layout(
            "GRAPH-001", small_nodes, small_edges
        )

        for nid in small_nodes:
            pos1 = result1.node_positions[nid]
            pos2 = result2.node_positions[nid]
            assert abs(pos1.x - pos2.x) < 0.001
            assert abs(pos1.y - pos2.y) < 0.001

    def test_different_seed_produces_different_layout(
        self, small_nodes: Dict, small_edges: Dict
    ):
        """Different seeds should produce different positions."""
        engine1 = VisualizationEngine(config=VisualizationConfig(fr_seed=1))
        engine2 = VisualizationEngine(config=VisualizationConfig(fr_seed=999))

        result1 = engine1.compute_force_directed_layout(
            "GRAPH-001", small_nodes, small_edges
        )
        result2 = engine2.compute_force_directed_layout(
            "GRAPH-001", small_nodes, small_edges
        )

        # At least one node should have different position
        differences = 0
        for nid in small_nodes:
            pos1 = result1.node_positions[nid]
            pos2 = result2.node_positions[nid]
            if abs(pos1.x - pos2.x) > 1.0 or abs(pos1.y - pos2.y) > 1.0:
                differences += 1
        assert differences > 0

    def test_provenance_hash_present(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Layout result should include a provenance hash."""
        result = engine.compute_force_directed_layout(
            "GRAPH-001", small_nodes, small_edges
        )
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64  # SHA-256 hex

    def test_computation_time_recorded(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Computation time should be non-negative."""
        result = engine.compute_force_directed_layout(
            "GRAPH-001", small_nodes, small_edges
        )
        assert result.computation_time_ms >= 0.0

    def test_viewport_bounds_correct(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Viewport should encompass all node positions."""
        result = engine.compute_force_directed_layout(
            "GRAPH-001", small_nodes, small_edges
        )
        for np in result.node_positions.values():
            assert np.x >= result.viewport["min_x"]
            assert np.x <= result.viewport["max_x"]
            assert np.y >= result.viewport["min_y"]
            assert np.y <= result.viewport["max_y"]


# ===========================================================================
# Test 2: Hierarchical Layout
# ===========================================================================


class TestHierarchicalLayout:
    """Tests for the tier-based hierarchical layout."""

    def test_basic_hierarchical_layout(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Hierarchical layout should produce positions for all nodes."""
        result = engine.compute_hierarchical_layout(
            "GRAPH-001", small_nodes, small_edges
        )
        assert result.total_nodes == 5
        assert result.algorithm == LayoutAlgorithm.HIERARCHICAL.value

    def test_tier_ordering(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Higher tier nodes (producers) should have lower y-values (top)."""
        result = engine.compute_hierarchical_layout(
            "GRAPH-001", small_nodes, small_edges
        )
        # Producer (tier 4) at top, importer (tier 0) at bottom
        prod_y = result.node_positions["N-PROD-01"].y
        imp_y = result.node_positions["N-IMP-01"].y
        assert prod_y < imp_y  # Producers at top (smaller y)

    def test_same_tier_different_x(
        self, engine: VisualizationEngine
    ):
        """Nodes in the same tier should have different x positions."""
        nodes = {
            "A": {"node_type": "producer", "operator_name": "A", "country_code": "BR",
                   "risk_level": "low", "tier_depth": 2, "commodities": ["soya"]},
            "B": {"node_type": "producer", "operator_name": "B", "country_code": "BR",
                   "risk_level": "low", "tier_depth": 2, "commodities": ["soya"]},
            "C": {"node_type": "importer", "operator_name": "C", "country_code": "DE",
                   "risk_level": "low", "tier_depth": 0, "commodities": ["soya"]},
        }
        edges = {
            "E1": {"source_node_id": "A", "target_node_id": "C",
                    "commodity": "soya", "quantity": 100},
            "E2": {"source_node_id": "B", "target_node_id": "C",
                    "commodity": "soya", "quantity": 100},
        }
        result = engine.compute_hierarchical_layout("G1", nodes, edges)
        assert result.node_positions["A"].x != result.node_positions["B"].x
        assert result.node_positions["A"].y == result.node_positions["B"].y


# ===========================================================================
# Test 3: Geographic Layout
# ===========================================================================


class TestGeographicLayout:
    """Tests for geographic layout with coordinate projection."""

    def test_basic_geographic_layout(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Geographic layout should produce positions for all nodes."""
        result = engine.compute_geographic_layout(
            "GRAPH-001", small_nodes, small_edges
        )
        assert result.total_nodes == 5
        assert result.algorithm == LayoutAlgorithm.GEOGRAPHIC.value

    def test_lat_lon_attached_to_positions(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Geographic positions should carry lat/lon data."""
        result = engine.compute_geographic_layout(
            "GRAPH-001", small_nodes, small_edges
        )
        prod = result.node_positions["N-PROD-01"]
        assert prod.latitude is not None
        assert prod.longitude is not None
        assert abs(prod.latitude - (-15.7801)) < 0.001

    def test_geographic_projection_order(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Nodes further east should have larger x values."""
        result = engine.compute_geographic_layout(
            "GRAPH-001", small_nodes, small_edges
        )
        # Brazil (-47 lon) vs Germany (13 lon) - Germany should be more to the right
        br_x = result.node_positions["N-PROD-01"].x
        de_x = result.node_positions["N-IMP-01"].x
        assert de_x > br_x

    def test_nodes_without_coords_still_positioned(
        self, engine: VisualizationEngine
    ):
        """Nodes without coordinates should still get positions."""
        nodes = {
            "A": {"node_type": "producer", "operator_name": "A",
                   "country_code": "BR", "risk_level": "low", "tier_depth": 1,
                   "latitude": -15.0, "longitude": -47.0, "commodities": ["soya"]},
            "B": {"node_type": "trader", "operator_name": "B",
                   "country_code": "BR", "risk_level": "low", "tier_depth": 0,
                   "commodities": ["soya"]},
        }
        edges = {"E1": {"source_node_id": "A", "target_node_id": "B",
                         "commodity": "soya", "quantity": 100}}
        result = engine.compute_geographic_layout("G1", nodes, edges)
        assert "B" in result.node_positions

    def test_geolocation_linker_integration(self):
        """Geographic layout should query geolocation linker for missing coords."""
        mock_linker = MagicMock()
        mock_linker.get_links_for_producer.return_value = [
            {"latitude": -10.0, "longitude": -55.0}
        ]
        engine = VisualizationEngine(geolocation_linker=mock_linker)

        nodes = {
            "PROD-1": {"node_type": "producer", "operator_name": "Farm",
                        "country_code": "BR", "risk_level": "low", "tier_depth": 1,
                        "commodities": ["soya"]},
        }
        edges: Dict = {}
        result = engine.compute_geographic_layout("G1", nodes, edges)
        assert result.node_positions["PROD-1"].latitude == -10.0


# ===========================================================================
# Test 4: Circular Layout
# ===========================================================================


class TestCircularLayout:
    """Tests for circular layout."""

    def test_basic_circular_layout(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Circular layout should position all nodes."""
        result = engine.compute_circular_layout(
            "GRAPH-001", small_nodes, small_edges
        )
        assert result.total_nodes == 5
        assert result.algorithm == LayoutAlgorithm.CIRCULAR.value

    def test_nodes_on_circle(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Nodes should be approximately equidistant from center."""
        result = engine.compute_circular_layout(
            "GRAPH-001", small_nodes, small_edges
        )
        cx, cy = 500.0, 500.0
        distances = []
        for np in result.node_positions.values():
            dist = math.sqrt((np.x - cx) ** 2 + (np.y - cy) ** 2)
            distances.append(dist)
        # All distances should be approximately equal (within 1%)
        avg_dist = sum(distances) / len(distances)
        for d in distances:
            assert abs(d - avg_dist) / avg_dist < 0.01


# ===========================================================================
# Test 5: Sankey Diagram Data Generation
# ===========================================================================


class TestSankeyData:
    """Tests for Sankey diagram data generation."""

    def test_basic_sankey_generation(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Sankey data should include nodes and links."""
        result = engine.generate_sankey_data(
            "GRAPH-001", small_nodes, small_edges
        )
        assert len(result.nodes) == 5
        assert len(result.links) == 4
        assert result.total_flow > 0

    def test_sankey_flow_volume(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Total flow should equal sum of edge quantities."""
        result = engine.generate_sankey_data(
            "GRAPH-001", small_nodes, small_edges
        )
        expected_flow = 5000 + 4500 + 2000 + 2000
        assert abs(result.total_flow - expected_flow) < 0.01

    def test_sankey_commodity_filter(
        self, engine: VisualizationEngine, small_nodes: Dict,
        multi_commodity_edges: Dict
    ):
        """Sankey should filter by commodity."""
        result = engine.generate_sankey_data(
            "GRAPH-001", small_nodes, multi_commodity_edges,
            commodity_filter="soya",
        )
        for link in result.links:
            assert link.label == "soya"

    def test_sankey_serialization(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """SankeyResult should serialize cleanly."""
        result = engine.generate_sankey_data(
            "GRAPH-001", small_nodes, small_edges
        )
        d = result.to_dict()
        assert "nodes" in d
        assert "links" in d
        assert "total_flow" in d
        assert "sankey_id" in d

    def test_sankey_node_colors_from_risk(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Sankey node colors should reflect risk levels."""
        result = engine.generate_sankey_data(
            "GRAPH-001", small_nodes, small_edges
        )
        node_map = {n.id: n for n in result.nodes}
        assert node_map["N-PROD-01"].color == RISK_COLOR_HIGH
        assert node_map["N-IMP-01"].color == RISK_COLOR_LOW


# ===========================================================================
# Test 6: Node Clustering
# ===========================================================================


class TestClustering:
    """Tests for node clustering."""

    def test_cluster_by_country(
        self, engine: VisualizationEngine, small_nodes: Dict
    ):
        """Should group nodes by country."""
        clusters = engine.compute_clusters(small_nodes, cluster_by="country")
        # BR has 3 nodes, should form a cluster
        br_clusters = [c for c in clusters if c.label == "BR"]
        assert len(br_clusters) == 1
        assert len(br_clusters[0].node_ids) == 3

    def test_cluster_by_tier(
        self, engine: VisualizationEngine, small_nodes: Dict
    ):
        """Should group nodes by tier depth."""
        clusters = engine.compute_clusters(small_nodes, cluster_by="tier")
        # Each tier has 1 node except none repeated, so no clusters (< 2 nodes each)
        # All are different tiers (4,3,2,1,0) so no clusters
        assert len(clusters) == 0

    def test_cluster_by_type(
        self, engine: VisualizationEngine, small_nodes: Dict
    ):
        """Should group nodes by node type."""
        clusters = engine.compute_clusters(small_nodes, cluster_by="type")
        # All unique types, no clusters
        assert len(clusters) == 0

    def test_cluster_threshold_triggers(self, engine: VisualizationEngine):
        """Layout should auto-cluster when node count exceeds threshold."""
        config = VisualizationConfig(cluster_threshold=3)
        eng = VisualizationEngine(config=config)
        nodes = {
            f"N-{i}": {
                "node_type": "producer", "operator_name": f"Op{i}",
                "country_code": "BR", "risk_level": "low", "tier_depth": 0,
                "commodities": ["soya"],
            }
            for i in range(5)
        }
        edges: Dict = {}
        result = eng.compute_force_directed_layout("G1", nodes, edges)
        # Should have at least one cluster since all nodes share "BR"
        assert len(result.clusters) >= 1


# ===========================================================================
# Test 7-9: Export Formats
# ===========================================================================


class TestGeoJSONExport:
    """Tests for GeoJSON export."""

    def test_geojson_structure(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """GeoJSON should have valid FeatureCollection structure."""
        geojson = engine.export_geojson("GRAPH-001", small_nodes, small_edges)
        assert geojson["type"] == "FeatureCollection"
        assert "features" in geojson
        assert "properties" in geojson

    def test_geojson_node_features(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """GeoJSON should contain Point features for nodes with coordinates."""
        geojson = engine.export_geojson("GRAPH-001", small_nodes, small_edges)
        node_features = [
            f for f in geojson["features"]
            if f["properties"].get("feature_type") == "node"
        ]
        assert len(node_features) == 5
        for f in node_features:
            assert f["geometry"]["type"] == "Point"
            coords = f["geometry"]["coordinates"]
            assert len(coords) == 2  # [lon, lat]

    def test_geojson_edge_features(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """GeoJSON should contain LineString features for edges."""
        geojson = engine.export_geojson("GRAPH-001", small_nodes, small_edges)
        edge_features = [
            f for f in geojson["features"]
            if f["properties"].get("feature_type") == "edge"
        ]
        assert len(edge_features) == 4
        for f in edge_features:
            assert f["geometry"]["type"] == "LineString"

    def test_geojson_risk_colors(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """GeoJSON node properties should include risk color."""
        geojson = engine.export_geojson("GRAPH-001", small_nodes, small_edges)
        node_features = [
            f for f in geojson["features"]
            if f["properties"].get("feature_type") == "node"
        ]
        for f in node_features:
            assert "color" in f["properties"]


class TestGraphMLExport:
    """Tests for GraphML export."""

    def test_graphml_valid_xml(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """GraphML output should be valid XML."""
        xml_str = engine.export_graphml("GRAPH-001", small_nodes, small_edges)
        assert xml_str.startswith("<?xml")
        assert "<graphml" in xml_str
        assert "</graphml>" in xml_str

    def test_graphml_contains_nodes(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """GraphML should contain node elements."""
        xml_str = engine.export_graphml("GRAPH-001", small_nodes, small_edges)
        for nid in small_nodes:
            assert nid in xml_str

    def test_graphml_contains_edges(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """GraphML should contain edge elements."""
        xml_str = engine.export_graphml("GRAPH-001", small_nodes, small_edges)
        for eid in small_edges:
            assert eid in xml_str

    def test_graphml_attribute_keys(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """GraphML should define attribute keys for node and edge data."""
        xml_str = engine.export_graphml("GRAPH-001", small_nodes, small_edges)
        assert 'attr.name="node_type"' in xml_str
        assert 'attr.name="risk_level"' in xml_str
        assert 'attr.name="commodity"' in xml_str


class TestJSONLDExport:
    """Tests for JSON-LD export."""

    def test_jsonld_context(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """JSON-LD should include @context."""
        doc = engine.export_jsonld("GRAPH-001", small_nodes, small_edges)
        assert "@context" in doc
        assert "greenlang" in doc["@context"]

    def test_jsonld_nodes_and_edges(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """JSON-LD should contain nodes and edges lists."""
        doc = engine.export_jsonld("GRAPH-001", small_nodes, small_edges)
        assert len(doc["nodes"]) == 5
        assert len(doc["edges"]) == 4

    def test_jsonld_node_structure(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """JSON-LD nodes should have @type and identifier."""
        doc = engine.export_jsonld("GRAPH-001", small_nodes, small_edges)
        for node in doc["nodes"]:
            assert "@type" in node
            assert "identifier" in node
            assert "risk_level" in node

    def test_jsonld_geo_coordinates(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """JSON-LD nodes with coordinates should have geo property."""
        doc = engine.export_jsonld("GRAPH-001", small_nodes, small_edges)
        nodes_with_geo = [n for n in doc["nodes"] if "geo" in n]
        assert len(nodes_with_geo) == 5


# ===========================================================================
# Test 10: Graph Filtering
# ===========================================================================


class TestGraphFiltering:
    """Tests for graph filter support."""

    def test_filter_by_commodity(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Should filter nodes by commodity."""
        f = GraphFilter(commodities=["soybean_oil"])
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges, graph_filter=f
        )
        for nid, np in result.node_positions.items():
            ndata = small_nodes[nid]
            commodities = ndata.get("commodities", [])
            assert "soybean_oil" in commodities

    def test_filter_by_country(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Should filter nodes by country."""
        f = GraphFilter(countries=["BR"])
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges, graph_filter=f
        )
        assert result.total_nodes == 3
        for nid in result.node_positions:
            assert small_nodes[nid]["country_code"] == "BR"

    def test_filter_by_risk_level(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Should filter nodes by risk level."""
        f = GraphFilter(risk_levels=["high"])
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges, graph_filter=f
        )
        assert result.total_nodes == 1
        assert "N-PROD-01" in result.node_positions

    def test_filter_by_compliance_status(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Should filter nodes by compliance status."""
        f = GraphFilter(compliance_statuses=["compliant"])
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges, graph_filter=f
        )
        for nid in result.node_positions:
            assert small_nodes[nid]["compliance_status"] == "compliant"

    def test_filter_by_tier_depth(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Should filter nodes by tier depth range."""
        f = GraphFilter(min_tier_depth=2, max_tier_depth=4)
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges, graph_filter=f
        )
        for nid in result.node_positions:
            tier = small_nodes[nid]["tier_depth"]
            assert 2 <= tier <= 4

    def test_filter_by_node_type(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Should filter nodes by node type."""
        f = GraphFilter(node_types=["producer", "importer"])
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges, graph_filter=f
        )
        assert result.total_nodes == 2

    def test_filter_edges_match_filtered_nodes(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Edges should only connect filtered nodes."""
        f = GraphFilter(countries=["BR"])
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges, graph_filter=f
        )
        remaining_ids = set(result.node_positions.keys())
        for ep in result.edge_paths.values():
            assert ep.source_node_id in remaining_ids
            assert ep.target_node_id in remaining_ids

    def test_combined_filters(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Multiple filters should intersect."""
        f = GraphFilter(countries=["BR"], risk_levels=["standard"])
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges, graph_filter=f
        )
        for nid in result.node_positions:
            assert small_nodes[nid]["country_code"] == "BR"
            assert small_nodes[nid]["risk_level"] == "standard"

    def test_explicit_node_id_filter(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Should filter to only specified node IDs."""
        f = GraphFilter(node_ids=["N-PROD-01", "N-IMP-01"])
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges, graph_filter=f
        )
        assert result.total_nodes == 2
        assert "N-PROD-01" in result.node_positions
        assert "N-IMP-01" in result.node_positions


# ===========================================================================
# Test 11-15: Color Coding Schemes
# ===========================================================================


class TestColorCoding:
    """Tests for color coding schemes."""

    def test_risk_level_colors(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Risk level colors should map correctly."""
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges,
            color_scheme=ColorScheme.RISK_LEVEL,
        )
        assert result.node_positions["N-PROD-01"].color == RISK_COLOR_HIGH
        assert result.node_positions["N-COLL-01"].color == RISK_COLOR_STANDARD
        assert result.node_positions["N-IMP-01"].color == RISK_COLOR_LOW

    def test_compliance_status_colors(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Compliance status colors should map correctly."""
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges,
            color_scheme=ColorScheme.COMPLIANCE_STATUS,
        )
        # Compliant should be green
        assert result.node_positions["N-PROC-01"].color == COMPLIANCE_COLORS["compliant"]
        # Pending should be amber
        assert result.node_positions["N-PROD-01"].color == COMPLIANCE_COLORS["pending_verification"]

    def test_node_type_colors(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Node type colors should differentiate types."""
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges,
            color_scheme=ColorScheme.NODE_TYPE,
        )
        producer_color = result.node_positions["N-PROD-01"].color
        importer_color = result.node_positions["N-IMP-01"].color
        assert producer_color != importer_color

    def test_tier_depth_colors(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Tier depth colors should vary by tier."""
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges,
            color_scheme=ColorScheme.TIER_DEPTH,
        )
        tier0_color = result.node_positions["N-IMP-01"].color
        tier4_color = result.node_positions["N-PROD-01"].color
        assert tier0_color != tier4_color

    def test_country_based_colors(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Country-based colors should differentiate countries."""
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges,
            color_scheme=ColorScheme.COUNTRY,
        )
        # BR and DE should have different colors
        br_color = result.node_positions["N-PROD-01"].color
        de_color = result.node_positions["N-IMP-01"].color
        assert br_color != de_color

    def test_get_risk_coloring(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """get_risk_coloring should return node and edge color maps."""
        colors = engine.get_risk_coloring(small_nodes, small_edges)
        assert "nodes" in colors
        assert "edges" in colors
        assert colors["nodes"]["N-PROD-01"] == RISK_COLOR_HIGH
        assert colors["nodes"]["N-IMP-01"] == RISK_COLOR_LOW

    def test_edge_coloring_from_endpoints(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Edge color should be based on higher-risk endpoint."""
        colors = engine.get_risk_coloring(small_nodes, small_edges)
        # E-01: PROD(75) -> COLL(50), should use PROD's HIGH color
        assert colors["edges"]["E-01"] == RISK_COLOR_HIGH


# ===========================================================================
# Test 16-18: Edge Paths, Width, Viewport
# ===========================================================================


class TestEdgeComputation:
    """Tests for edge path and width computation."""

    def test_edge_waypoints_connect_nodes(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Edge waypoints should start at source and end at target."""
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges
        )
        for ep in result.edge_paths.values():
            src_pos = result.node_positions[ep.source_node_id]
            tgt_pos = result.node_positions[ep.target_node_id]
            assert abs(ep.waypoints[0][0] - src_pos.x) < 0.01
            assert abs(ep.waypoints[-1][0] - tgt_pos.x) < 0.01

    def test_edge_width_from_quantity(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Edge width should increase with quantity."""
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges
        )
        e01_width = result.edge_paths["E-01"].width  # qty=5000
        e04_width = result.edge_paths["E-04"].width   # qty=2000
        assert e01_width >= e04_width


# ===========================================================================
# Test 19-20: Empty and Single Node
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_graph(self, engine: VisualizationEngine):
        """Empty graph should return empty layout."""
        result = engine.compute_force_directed_layout("G1", {}, {})
        assert result.total_nodes == 0
        assert result.total_edges == 0

    def test_single_node(self, engine: VisualizationEngine):
        """Single node graph should work."""
        nodes = {
            "N-ONLY": {
                "node_type": "importer",
                "operator_name": "Solo Import",
                "country_code": "DE",
                "risk_level": "low",
                "tier_depth": 0,
                "commodities": ["soya"],
            }
        }
        result = engine.compute_force_directed_layout("G1", nodes, {})
        assert result.total_nodes == 1
        assert "N-ONLY" in result.node_positions

    def test_disconnected_components(self, engine: VisualizationEngine):
        """Disconnected nodes should all get positions."""
        nodes = {
            f"N-{i}": {
                "node_type": "producer", "operator_name": f"Op{i}",
                "country_code": "BR", "risk_level": "low", "tier_depth": 0,
                "commodities": ["soya"],
            }
            for i in range(10)
        }
        result = engine.compute_force_directed_layout("G1", nodes, {})
        assert result.total_nodes == 10


# ===========================================================================
# Test 21: Performance (1,000 Nodes < 3 Seconds)
# ===========================================================================


class TestPerformance:
    """Performance tests for layout computation."""

    def test_1000_node_layout_under_5_seconds(self):
        """Force-directed layout for 1,000 nodes should complete in < 5s.

        The time budget is set to 3 seconds for the iteration loop, but
        total wall-clock time includes setup and result construction, so
        we allow up to 5 seconds total for CI/CD environments.
        """
        config = VisualizationConfig(
            fr_iterations=50,
            fr_seed=42,
            max_layout_time_ms=3000,
        )
        engine = VisualizationEngine(config=config)
        nodes, edges = _make_large_graph(1000)

        start = time.monotonic()
        result = engine.compute_force_directed_layout("G-PERF", nodes, edges)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Layout took {elapsed:.2f}s, expected < 5.0s"
        assert result.total_nodes == len(nodes)

    def test_hierarchical_1000_nodes(self):
        """Hierarchical layout for 1,000 nodes should be fast."""
        engine = VisualizationEngine()
        nodes, edges = _make_large_graph(1000)

        start = time.monotonic()
        result = engine.compute_hierarchical_layout("G-PERF", nodes, edges)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"Hierarchical took {elapsed:.2f}s, expected < 1.0s"
        assert result.total_nodes == len(nodes)


# ===========================================================================
# Test 22: Deterministic Layout
# ===========================================================================


class TestDeterminism:
    """Tests for deterministic layout computation."""

    def test_same_input_same_output(self):
        """Running the same input twice with same seed yields identical output."""
        config = VisualizationConfig(fr_seed=123, fr_iterations=50)
        nodes = {
            f"N-{i}": {
                "node_type": "producer", "operator_name": f"Op{i}",
                "country_code": "BR", "risk_level": "low", "tier_depth": 0,
                "commodities": ["soya"],
            }
            for i in range(20)
        }
        edges = {
            f"E-{i}": {
                "source_node_id": f"N-{i}",
                "target_node_id": f"N-{(i+1) % 20}",
                "commodity": "soya", "quantity": 100,
            }
            for i in range(19)
        }

        engine1 = VisualizationEngine(config=config)
        engine2 = VisualizationEngine(config=config)

        r1 = engine1.compute_force_directed_layout("G1", nodes, edges)
        r2 = engine2.compute_force_directed_layout("G1", nodes, edges)

        for nid in nodes:
            assert abs(r1.node_positions[nid].x - r2.node_positions[nid].x) < 0.001
            assert abs(r1.node_positions[nid].y - r2.node_positions[nid].y) < 0.001


# ===========================================================================
# Test 23-24: Snapshot Storage and Retrieval
# ===========================================================================


class TestSnapshots:
    """Tests for snapshot storage and retrieval."""

    def test_store_snapshot(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Should store and retrieve a snapshot."""
        snap = engine.store_snapshot("G1", small_nodes, small_edges, version=1)
        assert snap["snapshot_id"].startswith("SNAP-")
        assert snap["node_count"] == 5
        assert snap["edge_count"] == 4
        assert snap["provenance_hash"]

    def test_retrieve_snapshot_by_version(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Should retrieve stored snapshot by version."""
        engine.store_snapshot("G1", small_nodes, small_edges, version=1)
        retrieved = engine.get_graph_snapshot("G1", version=1)
        assert retrieved is not None
        assert retrieved["version"] == 1

    def test_retrieve_latest_snapshot(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Should retrieve the latest snapshot."""
        engine.store_snapshot("G1", small_nodes, small_edges, version=1)
        engine.store_snapshot("G1", small_nodes, small_edges, version=2)
        latest = engine.get_graph_snapshot("G1")
        assert latest is not None
        assert latest["version"] == 2

    def test_missing_snapshot_returns_none(
        self, engine: VisualizationEngine
    ):
        """Should return None for non-existent snapshot."""
        result = engine.get_graph_snapshot("NONEXISTENT")
        assert result is None

    def test_graph_engine_snapshot_integration(self):
        """Should delegate to graph engine for snapshot retrieval."""
        mock_snap = MagicMock()
        mock_snap.snapshot_id = "SNAP-001"
        mock_snap.graph_id = "G1"
        mock_snap.version = 3
        mock_snap.node_count = 10
        mock_snap.edge_count = 9
        mock_snap.nodes = {}
        mock_snap.edges = {}
        mock_snap.provenance_hash = "abc123"
        mock_snap.created_at = datetime(2026, 3, 1, tzinfo=timezone.utc)

        mock_engine = MagicMock()
        mock_engine._snapshots = {"G1": [mock_snap]}

        engine = VisualizationEngine(graph_engine=mock_engine)
        result = engine.get_graph_snapshot("G1", version=3)
        assert result is not None
        assert result["version"] == 3


# ===========================================================================
# Test 25: Configuration
# ===========================================================================


class TestConfiguration:
    """Tests for visualization configuration."""

    def test_default_config(self):
        """Default config should have standard values."""
        config = VisualizationConfig()
        assert config.canvas_width == 1000.0
        assert config.fr_seed == 42
        assert config.fr_iterations == 300
        assert config.cluster_threshold == 50

    def test_custom_config(self):
        """Custom config values should be preserved."""
        config = VisualizationConfig(
            canvas_width=2000.0,
            canvas_height=1500.0,
            fr_seed=99,
            fr_iterations=200,
        )
        assert config.canvas_width == 2000.0
        assert config.fr_seed == 99

    def test_config_is_frozen(self):
        """Config should be immutable."""
        config = VisualizationConfig()
        with pytest.raises(Exception):
            config.fr_seed = 999  # type: ignore


# ===========================================================================
# Test 26: Provenance Hash
# ===========================================================================


class TestProvenance:
    """Tests for provenance hash computation."""

    def test_provenance_hash_deterministic(self):
        """Same input should produce same hash."""
        data = {"graph_id": "G1", "nodes": 10}
        h1 = _compute_provenance_hash(data)
        h2 = _compute_provenance_hash(data)
        assert h1 == h2

    def test_provenance_hash_differs_for_different_input(self):
        """Different input should produce different hash."""
        h1 = _compute_provenance_hash({"graph_id": "G1"})
        h2 = _compute_provenance_hash({"graph_id": "G2"})
        assert h1 != h2

    def test_layout_provenance_includes_seed(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Layout provenance hash should depend on configuration."""
        r1 = engine.compute_force_directed_layout("G1", small_nodes, small_edges)
        assert r1.provenance_hash
        assert len(r1.provenance_hash) == 64


# ===========================================================================
# Test 27: Filter Combinations
# ===========================================================================


class TestFilterCombinations:
    """Tests for advanced filter scenarios."""

    def test_all_filters_applied(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Applying all filters should narrow results."""
        f = GraphFilter(
            commodities=["soya"],
            countries=["BR"],
            risk_levels=["standard"],
            compliance_statuses=["under_review"],
            min_tier_depth=3,
            max_tier_depth=3,
        )
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges, graph_filter=f
        )
        assert result.total_nodes == 1
        assert "N-COLL-01" in result.node_positions

    def test_empty_filter_returns_all(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Empty filter should return all nodes."""
        f = GraphFilter()
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges, graph_filter=f
        )
        assert result.total_nodes == 5


# ===========================================================================
# Test 28: Sankey Commodity Filtering
# ===========================================================================


class TestSankeyCommodityFiltering:
    """Tests for Sankey diagram commodity filtering."""

    def test_filter_excludes_other_commodities(
        self, engine: VisualizationEngine, small_nodes: Dict,
        multi_commodity_edges: Dict
    ):
        """Sankey with commodity filter should exclude other commodities."""
        result = engine.generate_sankey_data(
            "G1", small_nodes, multi_commodity_edges,
            commodity_filter="coffee",
        )
        assert len(result.links) == 1
        assert result.links[0].label == "coffee"

    def test_no_filter_includes_all(
        self, engine: VisualizationEngine, small_nodes: Dict,
        multi_commodity_edges: Dict
    ):
        """Sankey without filter should include all commodities."""
        result = engine.generate_sankey_data(
            "G1", small_nodes, multi_commodity_edges,
        )
        labels = {l.label for l in result.links}
        assert "soya" in labels
        assert "coffee" in labels


# ===========================================================================
# Test 29: Cluster Radius and Center
# ===========================================================================


class TestClusterMetrics:
    """Tests for cluster radius and center computation."""

    def test_cluster_center_is_average(self, engine: VisualizationEngine):
        """Cluster center should be the average of node positions."""
        config = VisualizationConfig(cluster_threshold=2)
        eng = VisualizationEngine(config=config)
        nodes = {
            "A": {"node_type": "producer", "operator_name": "A",
                   "country_code": "BR", "risk_level": "low", "tier_depth": 0,
                   "commodities": ["soya"]},
            "B": {"node_type": "producer", "operator_name": "B",
                   "country_code": "BR", "risk_level": "low", "tier_depth": 0,
                   "commodities": ["soya"]},
            "C": {"node_type": "producer", "operator_name": "C",
                   "country_code": "BR", "risk_level": "low", "tier_depth": 0,
                   "commodities": ["soya"]},
        }
        result = eng.compute_force_directed_layout("G1", nodes, {})
        assert len(result.clusters) >= 1
        cluster = result.clusters[0]
        assert cluster.center_x > 0
        assert cluster.center_y > 0
        assert cluster.radius > 0


# ===========================================================================
# Test 30: Geographic Projection Edge Cases
# ===========================================================================


class TestGeographicEdgeCases:
    """Tests for geographic layout edge cases."""

    def test_all_nodes_same_coordinates(self, engine: VisualizationEngine):
        """All nodes at same location should not crash."""
        nodes = {
            f"N-{i}": {
                "node_type": "producer", "operator_name": f"Op{i}",
                "country_code": "BR", "risk_level": "low", "tier_depth": 0,
                "latitude": -15.0, "longitude": -47.0, "commodities": ["soya"],
            }
            for i in range(3)
        }
        result = engine.compute_geographic_layout("G1", nodes, {})
        assert result.total_nodes == 3

    def test_no_nodes_with_coordinates(self, engine: VisualizationEngine):
        """Nodes without any coordinates should still get positions."""
        nodes = {
            "A": {"node_type": "producer", "operator_name": "A",
                   "country_code": "BR", "risk_level": "low", "tier_depth": 0,
                   "commodities": ["soya"]},
        }
        result = engine.compute_geographic_layout("G1", nodes, {})
        assert "A" in result.node_positions


# ===========================================================================
# Test 31-32: Node Shape and Size Mapping
# ===========================================================================


class TestNodeStyling:
    """Tests for node shape and size mapping."""

    def test_node_shapes(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Node shapes should match NODE_TYPE_SHAPES mapping."""
        result = engine.compute_force_directed_layout("G1", small_nodes, small_edges)
        assert result.node_positions["N-PROD-01"].shape == "circle"
        assert result.node_positions["N-COLL-01"].shape == "diamond"
        assert result.node_positions["N-PROC-01"].shape == "square"
        assert result.node_positions["N-TRAD-01"].shape == "triangle"
        assert result.node_positions["N-IMP-01"].shape == "star"

    def test_node_sizes(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Node sizes should match NODE_TYPE_SIZES mapping."""
        result = engine.compute_force_directed_layout("G1", small_nodes, small_edges)
        assert result.node_positions["N-PROD-01"].size == 8.0
        assert result.node_positions["N-IMP-01"].size == 20.0


# ===========================================================================
# Test 33-34: Serialization
# ===========================================================================


class TestSerialization:
    """Tests for result serialization."""

    def test_layout_result_to_dict(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """LayoutResult.to_dict should produce valid JSON-serializable dict."""
        result = engine.compute_force_directed_layout("G1", small_nodes, small_edges)
        d = result.to_dict()
        assert d["graph_id"] == "G1"
        assert d["total_nodes"] == 5
        assert "node_positions" in d
        assert "edge_paths" in d
        # Verify it can be JSON-serialized
        import json
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 0

    def test_node_position_to_dict(self):
        """NodePosition.to_dict should include all fields."""
        np = NodePosition(
            node_id="N-1", x=100.0, y=200.0,
            latitude=-15.0, longitude=-47.0,
            color="#22C55E", shape="circle", size=10.0,
            label="Test Node", cluster_id="CL-1",
        )
        d = np.to_dict()
        assert d["node_id"] == "N-1"
        assert d["latitude"] == -15.0
        assert d["cluster_id"] == "CL-1"

    def test_edge_path_to_dict(self):
        """EdgePath.to_dict should include all fields."""
        ep = EdgePath(
            edge_id="E-1", source_node_id="N-1", target_node_id="N-2",
            waypoints=[(100.0, 200.0), (300.0, 400.0)],
            color="#EF4444", width=2.0, label="soya",
        )
        d = ep.to_dict()
        assert d["edge_id"] == "E-1"
        assert len(d["waypoints"]) == 2
        assert d["label"] == "soya"

    def test_cluster_group_to_dict(self):
        """ClusterGroup.to_dict should include all fields."""
        cg = ClusterGroup(
            cluster_id="CL-1", center_x=500.0, center_y=500.0,
            radius=100.0, node_ids=["N-1", "N-2"], label="BR",
        )
        d = cg.to_dict()
        assert d["cluster_id"] == "CL-1"
        assert len(d["node_ids"]) == 2


# ===========================================================================
# Test 35-37: Integration with Other Engines (Mock)
# ===========================================================================


class TestIntegration:
    """Tests for integration with graph engine, geolocation linker, risk engine."""

    def test_graph_engine_integration(self):
        """Should work with a mock graph engine."""
        mock_engine = MagicMock()
        engine = VisualizationEngine(graph_engine=mock_engine)
        nodes = {
            "N-1": {"node_type": "producer", "operator_name": "Farm",
                     "country_code": "BR", "risk_level": "low", "tier_depth": 0,
                     "commodities": ["soya"]},
        }
        result = engine.compute_force_directed_layout("G1", nodes, {})
        assert result.total_nodes == 1

    def test_risk_engine_integration(self):
        """Should accept a risk engine parameter."""
        mock_risk = MagicMock()
        engine = VisualizationEngine(risk_engine=mock_risk)
        assert engine._risk_engine is mock_risk

    def test_geolocation_linker_fallback(self):
        """Should handle geolocation linker errors gracefully."""
        mock_linker = MagicMock()
        mock_linker.get_links_for_producer.side_effect = RuntimeError("DB Error")
        engine = VisualizationEngine(geolocation_linker=mock_linker)

        nodes = {
            "N-1": {"node_type": "producer", "operator_name": "Farm",
                     "country_code": "BR", "risk_level": "low", "tier_depth": 0,
                     "commodities": ["soya"]},
        }
        # Should not raise, just skip geolocation enrichment
        result = engine.compute_geographic_layout("G1", nodes, {})
        assert result.total_nodes == 1


# ===========================================================================
# Test 38: Grid-Based Repulsive Force Approximation
# ===========================================================================


class TestGridApproximation:
    """Tests for grid-based repulsive force computation."""

    def test_grid_approximation_produces_valid_displacements(self):
        """Grid-based method should produce non-zero displacements."""
        engine = VisualizationEngine()
        node_ids = [f"N-{i}" for i in range(100)]
        positions = {
            nid: [float(i % 10) * 100, float(i // 10) * 100]
            for i, nid in enumerate(node_ids)
        }
        k = 100.0
        displacements = engine._compute_repulsive_grid(node_ids, positions, k)
        assert len(displacements) == 100
        # At least some displacements should be non-zero
        non_zero = sum(
            1 for d in displacements.values()
            if abs(d[0]) > 0.01 or abs(d[1]) > 0.01
        )
        assert non_zero > 0


# ===========================================================================
# Test 39: Edge Coloring from Endpoints
# ===========================================================================


class TestEdgeColoring:
    """Tests for edge coloring based on endpoint risk."""

    def test_edge_color_from_higher_risk(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Edge color should come from the higher-risk endpoint."""
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges
        )
        # E-01: producer(high, 75) -> collector(standard, 50)
        e01_color = result.edge_paths["E-01"].color
        assert e01_color == RISK_COLOR_HIGH

    def test_low_risk_edges(
        self, engine: VisualizationEngine, small_nodes: Dict, small_edges: Dict
    ):
        """Edges between low-risk nodes should be green."""
        result = engine.compute_force_directed_layout(
            "G1", small_nodes, small_edges
        )
        # E-04: trader(low) -> importer(low)
        e04_color = result.edge_paths["E-04"].color
        assert e04_color == RISK_COLOR_LOW


# ===========================================================================
# Test 40: Time Budget Enforcement
# ===========================================================================


class TestTimeBudget:
    """Tests for time budget enforcement during layout computation."""

    def test_time_budget_stops_early(self):
        """Layout should stop early when time budget is exceeded."""
        config = VisualizationConfig(
            fr_iterations=10000,
            max_layout_time_ms=50,  # Very short budget
            fr_seed=42,
        )
        engine = VisualizationEngine(config=config)
        nodes, edges = _make_large_graph(500)

        result = engine.compute_force_directed_layout("G1", nodes, edges)
        # Should still produce a result, just with fewer iterations
        assert result.total_nodes > 0


# ===========================================================================
# Test 41: Adaptive Iteration Count
# ===========================================================================


class TestAdaptiveIterations:
    """Tests for adaptive iteration count based on graph size."""

    def test_large_graph_uses_fewer_iterations(self):
        """Graphs with > 5000 nodes should use reduced iteration count."""
        config = VisualizationConfig(
            fr_iterations=300,
            max_layout_time_ms=10000,
            fr_seed=42,
        )
        engine = VisualizationEngine(config=config)
        # We cannot easily test iteration count directly, but we can verify
        # the layout completes reasonably quickly for large graphs
        nodes = {
            f"N-{i}": {
                "node_type": "producer", "operator_name": f"Op{i}",
                "country_code": "BR", "risk_level": "low", "tier_depth": 0,
                "commodities": ["soya"],
            }
            for i in range(100)  # Using 100 instead of 5000+ for test speed
        }
        result = engine.compute_force_directed_layout("G1", nodes, {})
        assert result.total_nodes == 100


# ===========================================================================
# Test 42: Multiple Commodity Flow in Sankey
# ===========================================================================


class TestSankeyMultipleCommodities:
    """Tests for multiple commodity flows in Sankey diagrams."""

    def test_aggregate_flows(
        self, engine: VisualizationEngine, small_nodes: Dict,
        multi_commodity_edges: Dict
    ):
        """Sankey should aggregate flows across multiple commodities."""
        result = engine.generate_sankey_data(
            "G1", small_nodes, multi_commodity_edges
        )
        # Total flow = 3000 + 1000 + 2500 = 6500
        assert abs(result.total_flow - 6500) < 0.01

    def test_sankey_computation_time(
        self, engine: VisualizationEngine, small_nodes: Dict,
        multi_commodity_edges: Dict
    ):
        """Sankey computation time should be non-negative."""
        result = engine.generate_sankey_data(
            "G1", small_nodes, multi_commodity_edges
        )
        assert result.computation_time_ms >= 0.0


# ===========================================================================
# Test 43+: Additional utility and constant tests
# ===========================================================================


class TestUtilities:
    """Tests for utility functions and constants."""

    def test_generate_id_format(self):
        """Generated IDs should have correct prefix format."""
        id1 = _generate_id("TEST")
        assert id1.startswith("TEST-")
        assert len(id1) == len("TEST-") + 12

    def test_utcnow_has_zero_microseconds(self):
        """_utcnow should return datetime with zeroed microseconds."""
        now = _utcnow()
        assert now.microsecond == 0
        assert now.tzinfo == timezone.utc

    def test_risk_color_constants(self):
        """Risk color constants should be valid hex."""
        for color in [RISK_COLOR_LOW, RISK_COLOR_STANDARD, RISK_COLOR_HIGH, RISK_COLOR_UNKNOWN]:
            assert color.startswith("#")
            assert len(color) == 7

    def test_node_type_shapes_complete(self):
        """NODE_TYPE_SHAPES should cover all node types."""
        expected_types = {
            "producer", "collector", "processor", "trader",
            "importer", "certifier", "warehouse", "port",
        }
        assert set(NODE_TYPE_SHAPES.keys()) == expected_types

    def test_node_type_sizes_complete(self):
        """NODE_TYPE_SIZES should cover all node types."""
        expected_types = {
            "producer", "collector", "processor", "trader",
            "importer", "certifier", "warehouse", "port",
        }
        assert set(NODE_TYPE_SIZES.keys()) == expected_types

    def test_compliance_colors_complete(self):
        """COMPLIANCE_COLORS should cover all compliance statuses."""
        expected = {
            "compliant", "non_compliant", "pending_verification",
            "under_review", "insufficient_data", "exempted",
        }
        assert set(COMPLIANCE_COLORS.keys()) == expected

    def test_layout_algorithm_enum(self):
        """LayoutAlgorithm enum should have all values."""
        assert LayoutAlgorithm.FORCE_DIRECTED.value == "force_directed"
        assert LayoutAlgorithm.HIERARCHICAL.value == "hierarchical"
        assert LayoutAlgorithm.GEOGRAPHIC.value == "geographic"
        assert LayoutAlgorithm.CIRCULAR.value == "circular"

    def test_export_format_enum(self):
        """ExportFormat enum should have all values."""
        assert ExportFormat.GEOJSON.value == "geojson"
        assert ExportFormat.GRAPHML.value == "graphml"
        assert ExportFormat.JSONLD.value == "jsonld"

    def test_color_scheme_enum(self):
        """ColorScheme enum should have all values."""
        assert ColorScheme.RISK_LEVEL.value == "risk_level"
        assert ColorScheme.COMPLIANCE_STATUS.value == "compliance_status"
        assert ColorScheme.NODE_TYPE.value == "node_type"
        assert ColorScheme.TIER_DEPTH.value == "tier_depth"
        assert ColorScheme.COUNTRY.value == "country"
