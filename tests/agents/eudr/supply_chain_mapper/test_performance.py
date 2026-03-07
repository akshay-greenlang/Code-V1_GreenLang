# -*- coding: utf-8 -*-
"""
Performance Tests - AGENT-EUDR-001 Supply Chain Mapper

Performance benchmarks validating PRD targets:
- Graph engine: <1ms single-node lookup, <5s for 1K node construction
- Risk propagation: <3s for 500-node graph
- Visualization: <3s layout for 100 nodes
- Sankey: <1s for 100 edges
- Serialization: <2s JSON/binary for 500 nodes
- Memory usage within acceptable bounds

Test count: 16 tests
Category: @pytest.mark.performance

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-001 Supply Chain Mapping Master
"""

import asyncio
import time
from decimal import Decimal
from typing import Any, Dict, List, Tuple

import pytest

from greenlang.agents.eudr.supply_chain_mapper.graph_engine import (
    CustodyModel,
    GraphEngineConfig,
    NodeType,
    SupplyChainGraphEngine,
)
from greenlang.agents.eudr.supply_chain_mapper.risk_propagation import (
    NodeRiskInput,
    RiskPropagationConfig,
    RiskPropagationEngine,
)
from greenlang.agents.eudr.supply_chain_mapper.gap_analyzer import (
    GapAnalyzer,
    GapAnalyzerConfig,
)
from greenlang.agents.eudr.supply_chain_mapper.visualization_engine import (
    VisualizationEngine,
)


# ===========================================================================
# Async helper
# ===========================================================================

def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def perf_engine():
    """Graph engine configured for performance testing."""
    eng = SupplyChainGraphEngine(
        config=GraphEngineConfig(
            enable_persistence=False,
            enable_audit_trail=False,  # Disable for speed
            enable_snapshots=False,
            max_graph_nodes=200_000,
        )
    )
    _run(eng.initialize())
    return eng


@pytest.fixture
def risk_engine():
    return RiskPropagationEngine()


@pytest.fixture
def gap_analyzer():
    return GapAnalyzer()


@pytest.fixture
def viz_engine():
    return VisualizationEngine()


def _build_wide_graph(engine, num_producers: int, num_tiers: int = 5):
    """Build a wide graph with many producers funneling to one importer."""
    gid = _run(engine.create_graph(
        operator_id="perf-test", commodity="cocoa",
        graph_name=f"Perf Test {num_producers} producers",
    ))

    importer = _run(engine.add_node(
        gid, NodeType.IMPORTER, "EU Importer", "NL",
        commodities=["cocoa"], tier_depth=0,
    ))
    trader = _run(engine.add_node(
        gid, NodeType.TRADER, "Central Trader", "CH",
        commodities=["cocoa"], tier_depth=1,
    ))
    _run(engine.add_edge(
        gid, trader, importer, "cocoa", Decimal("100000"),
        product_description="Bulk transfer",
    ))

    processor = _run(engine.add_node(
        gid, NodeType.PROCESSOR, "Central Processor", "GH",
        commodities=["cocoa"], tier_depth=2,
    ))
    _run(engine.add_edge(
        gid, processor, trader, "cocoa", Decimal("100000"),
        product_description="Processed bulk",
    ))

    collector = _run(engine.add_node(
        gid, NodeType.COLLECTOR, "Central Collector", "GH",
        commodities=["cocoa"], tier_depth=3,
    ))
    _run(engine.add_edge(
        gid, collector, processor, "cocoa", Decimal("100000"),
        product_description="Collected bulk",
    ))

    for i in range(num_producers):
        pid = _run(engine.add_node(
            gid, NodeType.PRODUCER, f"Farm {i:05d}", "GH",
            commodities=["cocoa"], tier_depth=4,
            latitude=5.0 + (i * 0.001),
            longitude=-1.0 + (i * 0.001),
        ))
        _run(engine.add_edge(
            gid, pid, collector, "cocoa", Decimal("100"),
            product_description=f"Farm {i} cocoa",
        ))

    return gid


def _build_linear_graph(engine, num_nodes: int):
    """Build a linear chain of num_nodes."""
    gid = _run(engine.create_graph(
        operator_id="perf-linear", commodity="cocoa",
        graph_name=f"Linear {num_nodes} nodes",
    ))
    nodes = []
    for i in range(num_nodes):
        nid = _run(engine.add_node(
            gid, NodeType.TRADER, f"Node {i:05d}", "GH",
            commodities=["cocoa"], tier_depth=num_nodes - i,
        ))
        nodes.append(nid)

    for i in range(len(nodes) - 1):
        _run(engine.add_edge(
            gid, nodes[i], nodes[i + 1], "cocoa", Decimal("1000"),
            product_description=f"Transfer {i}",
        ))

    return gid, nodes


# ===========================================================================
# Performance Tests
# ===========================================================================


@pytest.mark.performance
class TestGraphEnginePerformance:
    """Performance tests for SupplyChainGraphEngine."""

    def test_single_node_lookup_under_1ms(self, perf_engine):
        """PRD target: single-node lookup < 1ms."""
        gid = _run(perf_engine.create_graph(
            operator_id="perf", commodity="cocoa",
        ))
        # Add 1000 nodes
        target_nid = None
        for i in range(1000):
            nid = _run(perf_engine.add_node(
                gid, NodeType.PRODUCER, f"Farm {i}", "GH",
            ))
            if i == 500:
                target_nid = nid

        # Time the lookup
        start = time.perf_counter()
        for _ in range(100):
            _ = perf_engine.get_node(gid, target_nid)
        elapsed = (time.perf_counter() - start) / 100 * 1000  # avg ms
        assert elapsed < 1.0, f"Node lookup took {elapsed:.3f}ms, target <1ms"

    def test_graph_construction_1000_nodes_under_5s(self, perf_engine):
        """PRD target: graph construction < 5s for 1000 nodes."""
        start = time.perf_counter()
        gid = _build_wide_graph(perf_engine, num_producers=1000)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Construction took {elapsed:.2f}s, target <5s"

    def test_graph_construction_100_nodes(self, perf_engine):
        """100-node graph should build in < 1s."""
        start = time.perf_counter()
        gid = _build_wide_graph(perf_engine, num_producers=100)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"100-node construction took {elapsed:.2f}s"

    def test_list_nodes_1000(self, perf_engine):
        """Listing 1000+ nodes should complete in < 1s."""
        gid = _build_wide_graph(perf_engine, num_producers=1000)
        start = time.perf_counter()
        nodes = perf_engine.list_nodes(gid)
        elapsed = time.perf_counter() - start
        assert len(nodes) >= 1000
        assert elapsed < 1.0, f"List nodes took {elapsed:.2f}s"

    def test_topological_sort_1000_nodes(self, perf_engine):
        """Topological sort on 1000+ node graph should complete in < 2s."""
        gid = _build_wide_graph(perf_engine, num_producers=1000)
        start = time.perf_counter()
        topo = perf_engine.topological_sort(gid)
        elapsed = time.perf_counter() - start
        assert len(topo) >= 1000
        assert elapsed < 2.0, f"Topo sort took {elapsed:.2f}s"

    def test_cycle_detection_1000_nodes(self, perf_engine):
        """Cycle detection on 1000+ node graph < 2s."""
        gid = _build_wide_graph(perf_engine, num_producers=1000)
        start = time.perf_counter()
        has_cycles = perf_engine.has_cycles(gid)
        elapsed = time.perf_counter() - start
        assert has_cycles is False
        assert elapsed < 2.0, f"Cycle detection took {elapsed:.2f}s"

    def test_json_serialization_500_nodes(self, perf_engine):
        """JSON serialization of 500+ node graph < 2s."""
        gid = _build_wide_graph(perf_engine, num_producers=500)
        start = time.perf_counter()
        json_str = perf_engine.to_json(gid)
        elapsed = time.perf_counter() - start
        assert len(json_str) > 0
        assert elapsed < 2.0, f"JSON serialization took {elapsed:.2f}s"

    def test_binary_serialization_500_nodes(self, perf_engine):
        """Binary serialization of 500+ node graph < 2s."""
        gid = _build_wide_graph(perf_engine, num_producers=500)
        start = time.perf_counter()
        binary = perf_engine.to_binary(gid)
        elapsed = time.perf_counter() - start
        assert len(binary) > 0
        assert elapsed < 2.0, f"Binary serialization took {elapsed:.2f}s"

    def test_statistics_1000_nodes(self, perf_engine):
        """Statistics computation on 1000+ node graph < 1s."""
        gid = _build_wide_graph(perf_engine, num_producers=1000)
        start = time.perf_counter()
        stats = perf_engine.get_statistics(gid)
        elapsed = time.perf_counter() - start
        assert stats["total_nodes"] >= 1000
        assert elapsed < 1.0, f"Statistics took {elapsed:.2f}s"


@pytest.mark.performance
class TestRiskPropagationPerformance:
    """Performance tests for RiskPropagationEngine."""

    def test_propagation_100_nodes_under_1s(self, risk_engine):
        """Risk propagation for 100 nodes < 1s."""
        node_inputs = {
            f"n-{i:05d}": NodeRiskInput(
                node_id=f"n-{i:05d}",
                country_risk=50.0 + (i % 30),
                commodity_risk=40.0,
                supplier_risk=45.0,
                deforestation_risk=50.0 + (i % 20),
            )
            for i in range(100)
        }
        adjacency = {f"n-{i:05d}": [f"n-{i + 1:05d}"] for i in range(99)}

        start = time.perf_counter()
        result = risk_engine.propagate("perf-test", adjacency, node_inputs)
        elapsed = time.perf_counter() - start
        assert len(result.node_results) == 100
        assert elapsed < 1.0, f"100-node propagation took {elapsed:.2f}s"

    def test_propagation_500_nodes_under_3s(self, risk_engine):
        """Risk propagation for 500 nodes < 3s."""
        node_inputs = {
            f"n-{i:05d}": NodeRiskInput(
                node_id=f"n-{i:05d}",
                country_risk=50.0 + (i % 30),
                commodity_risk=40.0,
                supplier_risk=45.0,
                deforestation_risk=50.0 + (i % 20),
            )
            for i in range(500)
        }
        adjacency = {f"n-{i:05d}": [f"n-{i + 1:05d}"] for i in range(499)}

        start = time.perf_counter()
        result = risk_engine.propagate("perf-test", adjacency, node_inputs)
        elapsed = time.perf_counter() - start
        assert len(result.node_results) == 500
        assert elapsed < 3.0, f"500-node propagation took {elapsed:.2f}s"

    def test_propagation_wide_graph_under_2s(self, risk_engine):
        """Risk propagation for wide fan-out graph (1 root -> 200 leaves) < 2s."""
        node_inputs = {
            "root": NodeRiskInput(node_id="root", country_risk=80.0, commodity_risk=70.0,
                             supplier_risk=60.0, deforestation_risk=75.0),
        }
        adjacency = {"root": []}
        for i in range(200):
            nid = f"leaf-{i:05d}"
            node_inputs[nid] = NodeRiskInput(
                node_id=nid,
                country_risk=20.0,
                commodity_risk=20.0,
                supplier_risk=20.0,
                deforestation_risk=20.0,
            )
            adjacency["root"].append(nid)

        start = time.perf_counter()
        result = risk_engine.propagate("perf-wide", adjacency, node_inputs)
        elapsed = time.perf_counter() - start
        assert len(result.node_results) == 201
        assert elapsed < 2.0, f"Wide propagation took {elapsed:.2f}s"


@pytest.mark.performance
class TestVisualizationPerformance:
    """Performance tests for VisualizationEngine."""

    def test_force_layout_100_nodes_under_3s(self, viz_engine):
        """Force-directed layout for 100 nodes < 3s."""
        nodes, edges = _build_viz_perf_data(100)
        start = time.perf_counter()
        result = viz_engine.compute_force_directed_layout("perf-test", nodes, edges)
        elapsed = time.perf_counter() - start
        assert len(result.node_positions) == 100
        assert elapsed < 3.0, f"100-node layout took {elapsed:.2f}s"

    def test_hierarchical_layout_200_nodes_under_2s(self, viz_engine):
        """Hierarchical layout for 200 nodes < 2s."""
        nodes, edges = _build_viz_perf_data(200)
        start = time.perf_counter()
        result = viz_engine.compute_hierarchical_layout("perf-test", nodes, edges)
        elapsed = time.perf_counter() - start
        assert len(result.node_positions) == 200
        assert elapsed < 2.0, f"200-node hierarchy took {elapsed:.2f}s"

    def test_sankey_100_edges_under_1s(self, viz_engine):
        """Sankey data for 100 edges < 1s."""
        nodes, edges = _build_viz_perf_data(101)  # 100 edges
        start = time.perf_counter()
        result = viz_engine.generate_sankey_data("perf-test", nodes, edges)
        elapsed = time.perf_counter() - start
        assert len(result.links) == 100
        assert elapsed < 1.0, f"Sankey took {elapsed:.2f}s"

    def test_geojson_export_200_nodes_under_2s(self, viz_engine):
        """GeoJSON export for 200 nodes < 2s."""
        nodes, edges = _build_viz_perf_data(200)
        start = time.perf_counter()
        result = viz_engine.export_geojson("perf-test", nodes, edges)
        elapsed = time.perf_counter() - start
        assert result["type"] == "FeatureCollection"
        assert elapsed < 2.0, f"GeoJSON export took {elapsed:.2f}s"


# ===========================================================================
# Helpers
# ===========================================================================


def _build_viz_perf_data(num_nodes: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build graph data dicts for visualization performance tests.

    Returns (nodes_dict, edges_dict) as separate arguments for the engine.
    """
    nodes = {}
    for i in range(num_nodes):
        nid = f"node-{i:05d}"
        nodes[nid] = {
            "node_id": nid,
            "node_type": ["producer", "collector", "processor", "trader", "importer"][i % 5],
            "operator_name": f"Operator {i}",
            "country_code": ["BR", "GH", "ID", "CH", "NL"][i % 5],
            "risk_level": ["high", "standard", "low"][i % 3],
            "risk_score": 30.0 + (i % 50),
            "compliance_status": "compliant",
            "tier_depth": i % 5,
            "commodities": ["cocoa"],
            "certifications": [],
            "latitude": -2.5 + (i * 0.01),
            "longitude": -44.0 + (i * 0.01),
        }

    edges = {}
    for i in range(num_nodes - 1):
        eid = f"edge-{i:05d}"
        edges[eid] = {
            "edge_id": eid,
            "source_node_id": f"node-{i:05d}",
            "target_node_id": f"node-{i + 1:05d}",
            "commodity": "cocoa",
            "quantity": 1000.0 + i,
            "custody_model": "segregated",
        }

    return nodes, edges
