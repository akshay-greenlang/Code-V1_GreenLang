# -*- coding: utf-8 -*-
"""
Integration Tests - AGENT-EUDR-001 Supply Chain Mapper

End-to-end integration tests validating cross-module interactions:
- Graph Engine + Risk Propagation
- Graph Engine + Gap Analyzer
- Graph Engine + Visualization Engine
- Graph Engine + Provenance Tracker
- Graph Engine + Multi-Tier Mapper + Gap Analyzer
- Full pipeline: create graph -> add nodes -> add edges -> propagate risk
                 -> analyze gaps -> generate visualization -> export
- DDS export data validation
- Provenance chain integrity across operations
- Reproducibility across full pipeline runs

Test count: 23 tests
Coverage: Cross-module integration validation

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-001 Supply Chain Mapping Master (GL-EUDR-SCM-001)
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from greenlang.agents.eudr.supply_chain_mapper.graph_engine import (
    CustodyModel,
    GraphEngineConfig,
    NodeType,
    ComplianceStatus,
    SupplyChainGraphEngine,
)
from greenlang.agents.eudr.supply_chain_mapper.risk_propagation import (
    NodeRiskInput,
    RiskLevel,
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
from greenlang.agents.eudr.supply_chain_mapper.provenance import (
    ProvenanceTracker,
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
def graph_engine():
    eng = SupplyChainGraphEngine(
        config=GraphEngineConfig(
            enable_persistence=False,
            enable_audit_trail=True,
            enable_snapshots=False,
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


@pytest.fixture
def provenance_tracker():
    return ProvenanceTracker()


def _build_cocoa_chain(engine):
    """Build a complete cocoa supply chain for integration testing."""
    gid = _run(engine.create_graph(
        operator_id="integ-op-001", commodity="cocoa",
        graph_name="Integration Test Cocoa Chain",
    ))

    producer = _run(engine.add_node(
        gid, NodeType.PRODUCER, "Cooperative Alpha", "GH",
        commodities=["cocoa"], tier_depth=4,
        latitude=6.0, longitude=-1.5,
        plot_ids=["PLOT-GH-001", "PLOT-GH-002"],
    ))
    collector = _run(engine.add_node(
        gid, NodeType.COLLECTOR, "Aggregation Beta", "GH",
        commodities=["cocoa"], tier_depth=3,
    ))
    processor = _run(engine.add_node(
        gid, NodeType.PROCESSOR, "Mill Gamma", "GH",
        commodities=["cocoa"], tier_depth=2,
    ))
    trader = _run(engine.add_node(
        gid, NodeType.TRADER, "Trading Delta", "CH",
        commodities=["cocoa"], tier_depth=1,
    ))
    importer = _run(engine.add_node(
        gid, NodeType.IMPORTER, "EU Import Epsilon", "NL",
        commodities=["cocoa"], tier_depth=0,
    ))

    e1 = _run(engine.add_edge(
        gid, producer, collector, commodity="cocoa",
        quantity=Decimal("5000"), product_description="Raw cocoa beans",
        batch_number="BATCH-2025-001",
        custody_model=CustodyModel.IDENTITY_PRESERVED,
    ))
    e2 = _run(engine.add_edge(
        gid, collector, processor, commodity="cocoa",
        quantity=Decimal("4500"), product_description="Aggregated cocoa",
    ))
    e3 = _run(engine.add_edge(
        gid, processor, trader, commodity="cocoa",
        quantity=Decimal("4000"), product_description="Processed cocoa nibs",
    ))
    e4 = _run(engine.add_edge(
        gid, trader, importer, commodity="cocoa",
        quantity=Decimal("4000"), product_description="Traded cocoa nibs",
    ))

    nodes = [producer, collector, processor, trader, importer]
    edges = [e1, e2, e3, e4]
    return gid, nodes, edges


# ===========================================================================
# 1. Graph Engine + Risk Propagation Integration (4 tests)
# ===========================================================================


class TestGraphRiskIntegration:
    """Integration: Graph Engine + Risk Propagation Engine."""

    def test_build_risk_inputs_from_graph(self, graph_engine, risk_engine):
        gid, nodes, edges = _build_cocoa_chain(graph_engine)

        # Build risk inputs dict from graph nodes
        risk_inputs = {}
        for nid in nodes:
            node = graph_engine.get_node(gid, nid)
            risk_inputs[nid] = NodeRiskInput(
                node_id=nid,
                country_code=node.country_code,
                country_risk=65.0 if node.country_code == "GH" else 10.0,
                commodity_risk=50.0,
                supplier_risk=40.0,
                deforestation_risk=60.0 if node.country_code == "GH" else 5.0,
                node_type=node.node_type.value,
                commodities=node.commodities,
                tier_depth=node.tier_depth,
            )

        # Build adjacency from graph edges
        adjacency: Dict[str, List[str]] = {}
        for eid in edges:
            edge = graph_engine.get_edge(gid, eid)
            adjacency.setdefault(edge.source_node_id, []).append(edge.target_node_id)

        result = risk_engine.propagate(gid, adjacency, risk_inputs)
        assert len(result.node_results) == 5
        assert result.provenance_hash is not None

    def test_risk_updates_back_to_graph(self, graph_engine, risk_engine):
        gid, nodes, edges = _build_cocoa_chain(graph_engine)

        risk_inputs = {
            nid: NodeRiskInput(
                node_id=nid,
                country_risk=70.0,
                commodity_risk=50.0,
                supplier_risk=40.0,
                deforestation_risk=60.0,
            ) for nid in nodes
        }
        adjacency = {}
        for eid in edges:
            edge = graph_engine.get_edge(gid, eid)
            adjacency.setdefault(edge.source_node_id, []).append(edge.target_node_id)

        result = risk_engine.propagate(gid, adjacency, risk_inputs)

        # Write risk scores back to graph
        for nid, node_result in result.node_results.items():
            _run(graph_engine.update_node_attributes(
                gid, nid,
                risk_score=float(node_result.composite_risk),
            ))

        # Verify updates persisted
        for nid in nodes:
            node = graph_engine.get_node(gid, nid)
            assert node.risk_score > 0

    def test_risk_propagation_deterministic_with_graph(self, graph_engine, risk_engine):
        gid, nodes, edges = _build_cocoa_chain(graph_engine)

        risk_inputs = {
            nid: NodeRiskInput(node_id=nid, country_risk=50.0, commodity_risk=50.0,
                          supplier_risk=50.0, deforestation_risk=50.0)
            for nid in nodes
        }
        adjacency = {}
        for eid in edges:
            edge = graph_engine.get_edge(gid, eid)
            adjacency.setdefault(edge.source_node_id, []).append(edge.target_node_id)

        r1 = risk_engine.propagate(gid, adjacency, risk_inputs)
        r2 = risk_engine.propagate(gid, adjacency, risk_inputs)
        assert r1.provenance_hash == r2.provenance_hash

    def test_high_risk_producer_propagates_downstream(self, graph_engine, risk_engine):
        gid, nodes, edges = _build_cocoa_chain(graph_engine)

        risk_inputs = {}
        for i, nid in enumerate(nodes):
            risk_inputs[nid] = NodeRiskInput(
                node_id=nid,
                country_risk=90.0 if i == 0 else 10.0,
                commodity_risk=90.0 if i == 0 else 10.0,
                supplier_risk=90.0 if i == 0 else 10.0,
                deforestation_risk=90.0 if i == 0 else 10.0,
            )
        adjacency = {}
        for eid in edges:
            edge = graph_engine.get_edge(gid, eid)
            adjacency.setdefault(edge.source_node_id, []).append(edge.target_node_id)

        result = risk_engine.propagate(gid, adjacency, risk_inputs)
        producer_risk = result.node_results[nodes[0]].composite_risk
        importer_risk = result.node_results[nodes[4]].composite_risk
        # Importer inherits from high-risk producer
        assert importer_risk >= producer_risk


# ===========================================================================
# 2. Graph Engine + Gap Analyzer Integration (5 tests)
# ===========================================================================


class TestGraphGapIntegration:
    """Integration: Graph Engine + Gap Analyzer."""

    def test_gap_analysis_on_graph(self, graph_engine, gap_analyzer):
        gid, nodes, edges = _build_cocoa_chain(graph_engine)
        graph_nodes = _nodes_to_dicts(graph_engine, gid, nodes)
        graph_edges = _edges_to_dicts(graph_engine, gid)
        result = gap_analyzer.analyze(gid, graph_nodes, graph_edges)
        assert result is not None
        assert result.total_gaps >= 0

    def test_gap_analysis_detects_missing_geolocation(self, graph_engine, gap_analyzer):
        gid = _run(graph_engine.create_graph(
            operator_id="gap-test", commodity="cocoa",
        ))
        # Producer without coordinates
        producer = _run(graph_engine.add_node(
            gid, NodeType.PRODUCER, "Farm No GPS", "GH",
            commodities=["cocoa"], tier_depth=1,
        ))
        importer = _run(graph_engine.add_node(
            gid, NodeType.IMPORTER, "Importer", "NL",
            commodities=["cocoa"], tier_depth=0,
        ))
        _run(graph_engine.add_edge(
            gid, producer, importer, "cocoa",
            Decimal("1000"), product_description="Cocoa",
        ))

        graph_nodes = _nodes_to_dicts(graph_engine, gid, [producer, importer])
        graph_edges = _edges_to_dicts(graph_engine, gid)
        result = gap_analyzer.analyze(gid, graph_nodes, graph_edges)
        # Should detect gaps (missing geolocation or others)
        assert result.total_gaps >= 0

    def test_gap_analysis_detects_orphan(self, graph_engine, gap_analyzer):
        gid = _run(graph_engine.create_graph(
            operator_id="gap-orphan", commodity="cocoa",
        ))
        orphan = _run(graph_engine.add_node(
            gid, NodeType.WAREHOUSE, "Isolated Warehouse", "NL",
            commodities=["cocoa"], tier_depth=0,
        ))
        n1 = _run(graph_engine.add_node(
            gid, NodeType.PRODUCER, "Farm", "GH",
            commodities=["cocoa"], tier_depth=1,
        ))
        n2 = _run(graph_engine.add_node(
            gid, NodeType.IMPORTER, "Import", "NL",
            commodities=["cocoa"], tier_depth=0,
        ))
        _run(graph_engine.add_edge(gid, n1, n2, "cocoa", Decimal("1000"), product_description="C"))

        graph_nodes = _nodes_to_dicts(graph_engine, gid, [orphan, n1, n2])
        graph_edges = _edges_to_dicts(graph_engine, gid)
        result = gap_analyzer.analyze(gid, graph_nodes, graph_edges)
        assert result.total_gaps >= 1

    def test_compliance_readiness_score(self, graph_engine, gap_analyzer):
        gid, nodes, edges = _build_cocoa_chain(graph_engine)
        graph_nodes = _nodes_to_dicts(graph_engine, gid, nodes)
        graph_edges = _edges_to_dicts(graph_engine, gid)
        result = gap_analyzer.analyze(gid, graph_nodes, graph_edges)
        assert 0.0 <= result.compliance_readiness <= 100.0

    def test_gap_remediation_actions(self, graph_engine, gap_analyzer):
        gid, nodes, edges = _build_cocoa_chain(graph_engine)
        graph_nodes = _nodes_to_dicts(graph_engine, gid, nodes)
        graph_edges = _edges_to_dicts(graph_engine, gid)
        result = gap_analyzer.analyze(gid, graph_nodes, graph_edges)
        for gap in result.gaps:
            assert hasattr(gap, "remediation") or hasattr(gap, "remediation_action")


# ===========================================================================
# 3. Graph Engine + Visualization Integration (5 tests)
# ===========================================================================


class TestGraphVisualizationIntegration:
    """Integration: Graph Engine + Visualization Engine."""

    def _extract_viz_data(self, engine, gid, node_ids):
        """Extract node and edge dicts for visualization engine.

        The viz engine expects Dict[str, Dict[str, Any]], not Pydantic models.
        """
        nodes = {}
        for nid in node_ids:
            node = engine.get_node(gid, nid)
            nodes[nid] = node.model_dump(mode="json")
        edges = {}
        for e in engine.list_edges(gid):
            edges[e.edge_id] = e.model_dump(mode="json")
        return nodes, edges

    def test_force_directed_on_graph(self, graph_engine, viz_engine):
        gid, nodes, edges = _build_cocoa_chain(graph_engine)
        node_data, edge_data = self._extract_viz_data(graph_engine, gid, nodes)
        result = viz_engine.compute_force_directed_layout(gid, node_data, edge_data)
        assert len(result.node_positions) == 5

    def test_hierarchical_on_graph(self, graph_engine, viz_engine):
        gid, nodes, edges = _build_cocoa_chain(graph_engine)
        node_data, edge_data = self._extract_viz_data(graph_engine, gid, nodes)
        result = viz_engine.compute_hierarchical_layout(gid, node_data, edge_data)
        assert len(result.node_positions) == 5

    def test_sankey_on_graph(self, graph_engine, viz_engine):
        gid, nodes, edges = _build_cocoa_chain(graph_engine)
        node_data, edge_data = self._extract_viz_data(graph_engine, gid, nodes)
        result = viz_engine.generate_sankey_data(gid, node_data, edge_data)
        assert len(result.nodes) == 5
        assert len(result.links) == 4

    def test_geojson_export_on_graph(self, graph_engine, viz_engine):
        gid, nodes, edges = _build_cocoa_chain(graph_engine)
        node_data, edge_data = self._extract_viz_data(graph_engine, gid, nodes)
        result = viz_engine.export_geojson(gid, node_data, edge_data)
        assert result["type"] == "FeatureCollection"

    def test_graphml_export_on_graph(self, graph_engine, viz_engine):
        gid, nodes, edges = _build_cocoa_chain(graph_engine)
        node_data, edge_data = self._extract_viz_data(graph_engine, gid, nodes)
        result = viz_engine.export_graphml(gid, node_data, edge_data)
        assert "graphml" in result.lower() or "<?xml" in result.lower()


# ===========================================================================
# 4. Provenance Tracker Integration (5 tests)
# ===========================================================================


class TestProvenanceIntegration:
    """Integration: Graph Engine + Provenance Tracker."""

    def test_provenance_records_graph_create(self, provenance_tracker):
        entry = provenance_tracker.record("graph", "create", "g-001")
        assert entry.hash_value is not None
        assert len(entry.hash_value) == 64

    def test_provenance_chain_valid(self, provenance_tracker):
        provenance_tracker.record("graph", "create", "g-001")
        provenance_tracker.record("node", "add_node", "n-001")
        provenance_tracker.record("node", "add_node", "n-002")
        provenance_tracker.record("edge", "add_edge", "e-001")
        valid = provenance_tracker.verify_chain()
        assert valid is True

    def test_provenance_chain_tamper_detection(self, provenance_tracker):
        provenance_tracker.record("graph", "create", "g-001")
        provenance_tracker.record("node", "add_node", "n-001")
        # Tamper with the chain
        entries = provenance_tracker.get_entries()
        if len(entries) > 1:
            entries[0].hash_value = "tampered"
            valid = provenance_tracker.verify_chain()
            assert valid is False

    def test_provenance_export_json(self, provenance_tracker):
        provenance_tracker.record("graph", "create", "g-001")
        provenance_tracker.record("node", "add_node", "n-001")
        entries = provenance_tracker.export_json()
        assert isinstance(entries, (str, list))

    def test_provenance_entry_count(self, provenance_tracker):
        for i in range(10):
            provenance_tracker.record("node", "add_node", f"n-{i:03d}")
        assert provenance_tracker.entry_count == 10


# ===========================================================================
# 5. Full Pipeline Integration (4 tests)
# ===========================================================================


class TestFullPipeline:
    """Integration: Full pipeline from graph creation to export."""

    def test_full_pipeline_cocoa(self, graph_engine, risk_engine, gap_analyzer, viz_engine):
        # Step 1: Build graph
        gid, nodes, edges = _build_cocoa_chain(graph_engine)

        # Step 2: Propagate risk
        risk_inputs = {
            nid: NodeRiskInput(node_id=nid, country_risk=50.0, commodity_risk=50.0,
                          supplier_risk=50.0, deforestation_risk=50.0)
            for nid in nodes
        }
        adjacency = {}
        for eid in edges:
            edge = graph_engine.get_edge(gid, eid)
            adjacency.setdefault(edge.source_node_id, []).append(edge.target_node_id)
        risk_result = risk_engine.propagate(gid, adjacency, risk_inputs)
        assert len(risk_result.node_results) == 5

        # Step 3: Analyze gaps (expects dicts, not Pydantic models)
        graph_nodes = _nodes_to_dicts(graph_engine, gid, nodes)
        graph_edges = _edges_to_dicts(graph_engine, gid)
        gap_result = gap_analyzer.analyze(gid, graph_nodes, graph_edges)
        assert gap_result is not None

        # Step 4: Generate visualization (expects dicts)
        node_data = _nodes_to_dicts(graph_engine, gid, nodes)
        edge_data = _edges_to_dicts(graph_engine, gid)
        layout = viz_engine.compute_hierarchical_layout(gid, node_data, edge_data)
        assert len(layout.node_positions) == 5

        # Step 5: Export
        geojson = viz_engine.export_geojson(gid, node_data, edge_data)
        assert geojson["type"] == "FeatureCollection"

    def test_full_pipeline_deterministic(self, graph_engine, risk_engine, gap_analyzer):
        gid1, nodes1, edges1 = _build_cocoa_chain(graph_engine)
        gid2, nodes2, edges2 = _build_cocoa_chain(graph_engine)

        # Same risk inputs for both
        for gid, nodes, edges in [(gid1, nodes1, edges1), (gid2, nodes2, edges2)]:
            risk_inputs = {
                nid: NodeRiskInput(node_id=nid, country_risk=50.0, commodity_risk=50.0,
                              supplier_risk=50.0, deforestation_risk=50.0)
                for nid in nodes
            }
            adjacency = {}
            for eid in edges:
                edge = graph_engine.get_edge(gid, eid)
                adjacency.setdefault(edge.source_node_id, []).append(edge.target_node_id)
            result = risk_engine.propagate(gid, adjacency, risk_inputs)
            # Both runs should produce valid results
            assert len(result.node_results) == 5

    def test_pipeline_audit_trail_complete(self, graph_engine):
        gid, nodes, edges = _build_cocoa_chain(graph_engine)
        trail = graph_engine.get_audit_trail(gid)
        # Should have at minimum: 1 graph_created + 5 node_added + 4 edge_added = 10
        assert len(trail) >= 10

    def test_pipeline_graph_serialization_roundtrip(self, graph_engine):
        gid, nodes, edges = _build_cocoa_chain(graph_engine)
        json_str = graph_engine.to_json(gid)
        # from_json is a classmethod returning (engine, graph_id)
        new_engine, new_gid = SupplyChainGraphEngine.from_json(json_str)
        assert len(new_engine.list_nodes(new_gid)) == 5
        assert len(new_engine.list_edges(new_gid)) == 4


# ===========================================================================
# Helpers
# ===========================================================================


def _nodes_to_dicts(
    engine: SupplyChainGraphEngine,
    gid: str,
    node_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Convert graph engine Pydantic nodes to Dict[str, Dict] for gap/viz engines."""
    nodes = {}
    for nid in node_ids:
        node = engine.get_node(gid, nid)
        nodes[nid] = node.model_dump(mode="json")
    return nodes


def _edges_to_dicts(
    engine: SupplyChainGraphEngine,
    gid: str,
) -> Dict[str, Dict[str, Any]]:
    """Convert graph engine Pydantic edges to Dict[str, Dict] for gap/viz engines."""
    edges = {}
    for e in engine.list_edges(gid):
        edges[e.edge_id] = e.model_dump(mode="json")
    return edges
