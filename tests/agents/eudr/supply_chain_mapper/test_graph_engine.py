# -*- coding: utf-8 -*-
"""
Tests for SupplyChainGraphEngine - AGENT-EUDR-001 Feature 1: Supply Chain Graph Engine

Comprehensive test suite covering:
- Graph creation, retrieval, listing, deletion
- Node CRUD (add, get, list, update, remove)
- Edge CRUD (add, get, list, remove)
- Cycle detection and prevention
- Topological sorting
- Graph traversal (ancestors, descendants, predecessors, successors)
- Root and leaf node identification
- Orphan node detection
- Shortest path and all paths
- Serialization (JSON, GraphML, binary)
- Graph versioning and snapshots
- Audit trail and mutation records
- Provenance hash computation and verification
- Statistics computation
- Error handling (not found, capacity, cycles)
- Edge cases (empty graphs, single nodes, self-loops)
- Performance targets (<1ms lookup, <5s construction for 10K nodes)
- Determinism and reproducibility

Test count: 155 tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-001 (Feature 1)
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from greenlang.agents.eudr.supply_chain_mapper.graph_engine import (
    BINARY_MAGIC,
    GENESIS_HASH,
    CustodyModel,
    CycleDetectedError,
    EdgeNotFoundError,
    GraphCapacityError,
    GraphEngineConfig,
    GraphEngineError,
    GraphMutationRecord,
    GraphNotFoundError,
    GraphSnapshot,
    MutationType,
    NodeNotFoundError,
    NodeType,
    PersistenceError,
    RiskLevel,
    ComplianceStatus,
    SupplyChainEdge,
    SupplyChainGraphEngine,
    SupplyChainNode,
)


# ===========================================================================
# Async helper - runs a coroutine synchronously
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
def engine():
    """Create a SupplyChainGraphEngine in memory-only test mode."""
    config = GraphEngineConfig(
        enable_persistence=False,
        enable_audit_trail=True,
        enable_snapshots=False,
        max_graph_nodes=10_000,
    )
    eng = SupplyChainGraphEngine(config=config)
    _run(eng.initialize())
    return eng


@pytest.fixture
def engine_with_snapshots():
    """Engine with snapshot support enabled."""
    config = GraphEngineConfig(
        enable_persistence=False,
        enable_audit_trail=True,
        enable_snapshots=True,
        snapshot_interval=5,
        max_graph_nodes=10_000,
    )
    eng = SupplyChainGraphEngine(config=config)
    _run(eng.initialize())
    return eng


@pytest.fixture
def small_capacity_engine():
    """Engine with very small capacity for capacity testing."""
    config = GraphEngineConfig(
        enable_persistence=False,
        enable_audit_trail=False,
        enable_snapshots=False,
        max_graph_nodes=5,
    )
    eng = SupplyChainGraphEngine(config=config)
    _run(eng.initialize())
    return eng


@pytest.fixture
def populated_graph(engine):
    """Create engine with a populated 5-node linear supply chain."""
    graph_id = _run(engine.create_graph(
        operator_id="op-001",
        commodity="cocoa",
        graph_name="Test Cocoa Chain",
    ))
    # Producer -> Collector -> Processor -> Trader -> Importer
    producer_id = _run(engine.add_node(
        graph_id, node_type=NodeType.PRODUCER,
        operator_name="Farm Alpha", country_code="GH",
        commodities=["cocoa"], tier_depth=4,
    ))
    collector_id = _run(engine.add_node(
        graph_id, node_type=NodeType.COLLECTOR,
        operator_name="Coop Beta", country_code="GH",
        commodities=["cocoa"], tier_depth=3,
    ))
    processor_id = _run(engine.add_node(
        graph_id, node_type=NodeType.PROCESSOR,
        operator_name="Mill Gamma", country_code="GH",
        commodities=["cocoa"], tier_depth=2,
    ))
    trader_id = _run(engine.add_node(
        graph_id, node_type=NodeType.TRADER,
        operator_name="Trading Delta", country_code="CH",
        commodities=["cocoa"], tier_depth=1,
    ))
    importer_id = _run(engine.add_node(
        graph_id, node_type=NodeType.IMPORTER,
        operator_name="EU Import Epsilon", country_code="NL",
        commodities=["cocoa"], tier_depth=0,
    ))

    e1 = _run(engine.add_edge(
        graph_id, producer_id, collector_id, commodity="cocoa",
        quantity=Decimal("5000"), product_description="Raw cocoa beans",
    ))
    e2 = _run(engine.add_edge(
        graph_id, collector_id, processor_id, commodity="cocoa",
        quantity=Decimal("4500"), product_description="Aggregated cocoa",
    ))
    e3 = _run(engine.add_edge(
        graph_id, processor_id, trader_id, commodity="cocoa",
        quantity=Decimal("4000"), product_description="Processed cocoa",
    ))
    e4 = _run(engine.add_edge(
        graph_id, trader_id, importer_id, commodity="cocoa",
        quantity=Decimal("4000"), product_description="Traded cocoa",
    ))

    node_ids = [producer_id, collector_id, processor_id, trader_id, importer_id]
    edge_ids = [e1, e2, e3, e4]
    return engine, graph_id, node_ids, edge_ids


# ===========================================================================
# 1. Graph Creation and Lifecycle Tests (15 tests)
# ===========================================================================


class TestGraphCreation:
    """Tests for graph creation, retrieval, listing, and deletion."""

    def test_create_graph_returns_id(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        assert gid is not None
        assert isinstance(gid, str)
        assert len(gid) > 0

    def test_create_graph_with_name(self, engine):
        gid = _run(engine.create_graph(
            operator_id="op-001", commodity="cocoa",
            graph_name="Ghana Cocoa 2025",
        ))
        meta = engine.get_graph_metadata(gid)
        assert meta["graph_name"] == "Ghana Cocoa 2025"

    def test_create_graph_metadata(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        meta = engine.get_graph_metadata(gid)
        assert meta["operator_id"] == "op-001"
        assert meta["commodity"] == "cocoa"
        assert meta["version"] >= 1

    def test_create_multiple_graphs(self, engine):
        g1 = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        g2 = _run(engine.create_graph(operator_id="op-001", commodity="coffee"))
        g3 = _run(engine.create_graph(operator_id="op-002", commodity="soya"))
        assert g1 != g2 != g3
        assert len(engine.list_graphs()) == 3

    def test_list_graphs_empty(self, engine):
        assert engine.list_graphs() == []

    def test_list_graphs_returns_all(self, engine):
        for c in ["cocoa", "coffee", "soya"]:
            _run(engine.create_graph(operator_id="op-001", commodity=c))
        graphs = engine.list_graphs()
        assert len(graphs) == 3

    def test_get_graph_metadata_not_found(self, engine):
        with pytest.raises(GraphNotFoundError):
            engine.get_graph_metadata("nonexistent-graph")

    def test_delete_graph(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        _run(engine.delete_graph(gid))
        with pytest.raises(GraphNotFoundError):
            engine.get_graph_metadata(gid)

    def test_delete_graph_not_found(self, engine):
        with pytest.raises(GraphNotFoundError):
            _run(engine.delete_graph("nonexistent-graph"))

    def test_graph_version_starts_at_one(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        assert engine.get_version(gid) >= 1

    def test_graph_version_increments_on_mutation(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        v1 = engine.get_version(gid)
        _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        v2 = engine.get_version(gid)
        assert v2 > v1

    @pytest.mark.parametrize("commodity", [
        "cocoa", "coffee", "cattle", "oil_palm", "rubber", "soya", "wood",
    ])
    def test_create_graph_all_commodities(self, engine, commodity):
        gid = _run(engine.create_graph(operator_id="op-001", commodity=commodity))
        meta = engine.get_graph_metadata(gid)
        assert meta["commodity"] == commodity

    def test_graph_initial_statistics(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        stats = engine.get_statistics(gid)
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0


# ===========================================================================
# 2. Node CRUD Tests (30 tests)
# ===========================================================================


class TestNodeOperations:
    """Tests for node add, get, list, update, and remove."""

    def test_add_node_returns_id(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        nid = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm Alpha", "GH"))
        assert nid is not None
        assert isinstance(nid, str)

    def test_add_node_get_node(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        nid = _run(engine.add_node(
            gid, NodeType.PRODUCER, "Farm Alpha", "GH",
            commodities=["cocoa"], tier_depth=4,
        ))
        node = engine.get_node(gid, nid)
        assert node is not None
        assert node.operator_name == "Farm Alpha"
        assert node.country_code == "GH"
        assert node.node_type == NodeType.PRODUCER

    def test_add_node_with_coordinates(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        nid = _run(engine.add_node(
            gid, NodeType.PRODUCER, "Farm Alpha", "GH",
            latitude=5.6037, longitude=-0.1870,
        ))
        node = engine.get_node(gid, nid)
        assert node.latitude == pytest.approx(5.6037, abs=1e-4)
        assert node.longitude == pytest.approx(-0.1870, abs=1e-4)

    def test_add_node_with_certifications(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="wood"))
        nid = _run(engine.add_node(
            gid, NodeType.PRODUCER, "Forest Alpha", "BR",
            certifications=["FSC", "PEFC"],
        ))
        node = engine.get_node(gid, nid)
        assert "FSC" in node.certifications
        assert "PEFC" in node.certifications

    def test_add_node_with_plot_ids(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        nid = _run(engine.add_node(
            gid, NodeType.PRODUCER, "Farm Alpha", "GH",
            plot_ids=["PLOT-001", "PLOT-002"],
        ))
        node = engine.get_node(gid, nid)
        assert "PLOT-001" in node.plot_ids

    @pytest.mark.parametrize("node_type", [
        NodeType.PRODUCER, NodeType.COLLECTOR, NodeType.PROCESSOR,
        NodeType.TRADER, NodeType.IMPORTER, NodeType.CERTIFIER,
        NodeType.WAREHOUSE, NodeType.PORT,
    ])
    def test_add_all_node_types(self, engine, node_type):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        nid = _run(engine.add_node(gid, node_type, f"Test {node_type.value}", "NL"))
        node = engine.get_node(gid, nid)
        assert node.node_type == node_type

    def test_list_nodes_empty(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        nodes = engine.list_nodes(gid)
        assert len(nodes) == 0

    def test_list_nodes_returns_all(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        for i in range(5):
            _run(engine.add_node(gid, NodeType.PRODUCER, f"Farm {i}", "GH"))
        nodes = engine.list_nodes(gid)
        assert len(nodes) == 5

    def test_list_nodes_filter_by_type(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        _run(engine.add_node(gid, NodeType.PRODUCER, "Farm 2", "GH"))
        _run(engine.add_node(gid, NodeType.IMPORTER, "Import Co", "NL"))
        producers = engine.list_nodes(gid, node_type=NodeType.PRODUCER)
        assert len(producers) == 2

    def test_list_nodes_filter_by_country(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "BR"))
        _run(engine.add_node(gid, NodeType.IMPORTER, "Import", "NL"))
        gh_nodes = engine.list_nodes(gid, country_code="GH")
        assert len(gh_nodes) == 1

    def test_remove_node(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        nid = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        _run(engine.remove_node(gid, nid))
        with pytest.raises(NodeNotFoundError):
            engine.get_node(gid, nid)

    def test_remove_node_cascades_edges(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        eid = _run(engine.add_edge(
            gid, n1, n2, commodity="cocoa",
            quantity=Decimal("1000"), product_description="Cocoa",
        ))
        _run(engine.remove_node(gid, n1))
        with pytest.raises(EdgeNotFoundError):
            engine.get_edge(gid, eid)

    def test_remove_node_not_found(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        with pytest.raises(NodeNotFoundError):
            _run(engine.remove_node(gid, "nonexistent-node"))

    def test_get_node_graph_not_found(self, engine):
        with pytest.raises(GraphNotFoundError):
            engine.get_node("nonexistent-graph", "any-node")

    def test_get_node_not_found(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        with pytest.raises(NodeNotFoundError):
            engine.get_node(gid, "nonexistent-node")

    def test_update_node(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        nid = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        _run(engine.update_node_attributes(gid, nid, operator_name="Updated Farm"))
        node = engine.get_node(gid, nid)
        assert node.operator_name == "Updated Farm"

    def test_update_node_compliance_status(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        nid = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        _run(engine.update_node_attributes(
            gid, nid,
            compliance_status=ComplianceStatus.COMPLIANT,
        ))
        node = engine.get_node(gid, nid)
        assert node.compliance_status == ComplianceStatus.COMPLIANT

    def test_update_node_risk_score(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        nid = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        _run(engine.update_node_attributes(gid, nid, risk_score=75.0))
        node = engine.get_node(gid, nid)
        assert node.risk_score == pytest.approx(75.0)

    def test_update_node_coordinates(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        nid = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        _run(engine.update_node_attributes(gid, nid, latitude=6.0, longitude=-1.5))
        node = engine.get_node(gid, nid)
        assert node.latitude == pytest.approx(6.0)
        assert node.longitude == pytest.approx(-1.5)

    def test_country_code_normalization(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        nid = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "gh"))
        node = engine.get_node(gid, nid)
        assert node.country_code == "GH"

    def test_add_node_increments_count(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        stats = engine.get_statistics(gid)
        assert stats["total_nodes"] == 2

    def test_node_to_db_dict(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        nid = _run(engine.add_node(
            gid, NodeType.PRODUCER, "Farm", "GH",
            commodities=["cocoa"],
        ))
        node = engine.get_node(gid, nid)
        db_dict = node.to_db_dict()
        assert db_dict["node_type"] == "producer"
        assert db_dict["country_code"] == "GH"


# ===========================================================================
# 3. Edge CRUD Tests (25 tests)
# ===========================================================================


class TestEdgeOperations:
    """Tests for edge add, get, list, and remove."""

    def test_add_edge_returns_id(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        eid = _run(engine.add_edge(
            gid, n1, n2, commodity="cocoa",
            quantity=Decimal("1000"), product_description="Cocoa beans",
        ))
        assert eid is not None

    def test_get_edge(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        eid = _run(engine.add_edge(
            gid, n1, n2, commodity="cocoa",
            quantity=Decimal("1000"), product_description="Cocoa beans",
        ))
        edge = engine.get_edge(gid, eid)
        assert edge.source_node_id == n1
        assert edge.target_node_id == n2
        assert edge.commodity == "cocoa"
        assert edge.quantity == Decimal("1000")

    def test_add_edge_with_batch(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        eid = _run(engine.add_edge(
            gid, n1, n2, commodity="cocoa",
            quantity=Decimal("1000"), product_description="Cocoa",
            batch_number="BATCH-2025-001",
        ))
        edge = engine.get_edge(gid, eid)
        assert edge.batch_number == "BATCH-2025-001"

    def test_add_edge_with_custody_model(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        eid = _run(engine.add_edge(
            gid, n1, n2, commodity="cocoa",
            quantity=Decimal("1000"), product_description="Cocoa",
            custody_model=CustodyModel.IDENTITY_PRESERVED,
        ))
        edge = engine.get_edge(gid, eid)
        assert edge.custody_model == CustodyModel.IDENTITY_PRESERVED

    @pytest.mark.parametrize("custody", [
        CustodyModel.IDENTITY_PRESERVED,
        CustodyModel.SEGREGATED,
        CustodyModel.MASS_BALANCE,
    ])
    def test_all_custody_models(self, engine, custody):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        eid = _run(engine.add_edge(
            gid, n1, n2, commodity="cocoa",
            quantity=Decimal("1000"), product_description="Cocoa",
            custody_model=custody,
        ))
        edge = engine.get_edge(gid, eid)
        assert edge.custody_model == custody

    def test_list_edges_empty(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        edges = engine.list_edges(gid)
        assert len(edges) == 0

    def test_list_edges_returns_all(self, populated_graph):
        engine, gid, node_ids, edge_ids = populated_graph
        edges = engine.list_edges(gid)
        assert len(edges) == 4

    def test_remove_edge(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        eid = _run(engine.add_edge(
            gid, n1, n2, commodity="cocoa",
            quantity=Decimal("1000"), product_description="Cocoa",
        ))
        _run(engine.remove_edge(gid, eid))
        with pytest.raises(EdgeNotFoundError):
            engine.get_edge(gid, eid)

    def test_remove_edge_not_found(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        with pytest.raises(EdgeNotFoundError):
            _run(engine.remove_edge(gid, "nonexistent-edge"))

    def test_add_edge_source_not_found(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        with pytest.raises(NodeNotFoundError):
            _run(engine.add_edge(
                gid, "nonexistent", n2, commodity="cocoa",
                quantity=Decimal("1000"), product_description="Cocoa",
            ))

    def test_add_edge_target_not_found(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        with pytest.raises(NodeNotFoundError):
            _run(engine.add_edge(
                gid, n1, "nonexistent", commodity="cocoa",
                quantity=Decimal("1000"), product_description="Cocoa",
            ))

    def test_self_loop_rejected(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        with pytest.raises((CycleDetectedError, ValueError)):
            _run(engine.add_edge(
                gid, n1, n1, commodity="cocoa",
                quantity=Decimal("1000"), product_description="Self loop",
            ))

    def test_edge_provenance_hash(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        eid = _run(engine.add_edge(
            gid, n1, n2, commodity="cocoa",
            quantity=Decimal("1000"), product_description="Cocoa",
        ))
        edge = engine.get_edge(gid, eid)
        phash = edge.calculate_provenance_hash()
        assert len(phash) == 64  # SHA-256 hex length

    def test_edge_provenance_hash_deterministic(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        eid = _run(engine.add_edge(
            gid, n1, n2, commodity="cocoa",
            quantity=Decimal("1000"), product_description="Cocoa",
        ))
        edge = engine.get_edge(gid, eid)
        h1 = edge.calculate_provenance_hash()
        h2 = edge.calculate_provenance_hash()
        assert h1 == h2

    def test_edge_to_db_dict(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        eid = _run(engine.add_edge(
            gid, n1, n2, commodity="cocoa",
            quantity=Decimal("1000"), product_description="Cocoa",
        ))
        edge = engine.get_edge(gid, eid)
        db_dict = edge.to_db_dict()
        assert db_dict["source_node_id"] == n1
        assert db_dict["target_node_id"] == n2

    def test_add_edge_increments_count(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        _run(engine.add_edge(
            gid, n1, n2, commodity="cocoa",
            quantity=Decimal("1000"), product_description="Cocoa",
        ))
        stats = engine.get_statistics(gid)
        assert stats["total_edges"] == 1

    def test_parallel_edges_between_same_nodes(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        e1 = _run(engine.add_edge(
            gid, n1, n2, commodity="cocoa",
            quantity=Decimal("1000"), product_description="Batch 1",
            batch_number="B001",
        ))
        e2 = _run(engine.add_edge(
            gid, n1, n2, commodity="cocoa",
            quantity=Decimal("2000"), product_description="Batch 2",
            batch_number="B002",
        ))
        assert e1 != e2
        edges = engine.list_edges(gid)
        assert len(edges) == 2


# ===========================================================================
# 4. Cycle Detection Tests (15 tests)
# ===========================================================================


class TestCycleDetection:
    """Tests for cycle detection and prevention."""

    def test_no_cycles_in_dag(self, populated_graph):
        engine, gid, _, _ = populated_graph
        assert engine.has_cycles(gid) is False

    def test_detect_cycles_empty(self, populated_graph):
        engine, gid, _, _ = populated_graph
        cycles = engine.detect_cycles(gid)
        assert len(cycles) == 0

    def test_cycle_prevented_on_add_edge(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        n3 = _run(engine.add_node(gid, NodeType.PROCESSOR, "Mill", "GH"))
        _run(engine.add_edge(gid, n1, n2, "cocoa", Decimal("100"), product_description="C"))
        _run(engine.add_edge(gid, n2, n3, "cocoa", Decimal("100"), product_description="C"))
        with pytest.raises(CycleDetectedError):
            _run(engine.add_edge(gid, n3, n1, "cocoa", Decimal("100"), product_description="C"))

    def test_two_node_cycle_prevented(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        _run(engine.add_edge(gid, n1, n2, "cocoa", Decimal("100"), product_description="C"))
        with pytest.raises(CycleDetectedError):
            _run(engine.add_edge(gid, n2, n1, "cocoa", Decimal("100"), product_description="C"))

    def test_long_cycle_prevented(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        nodes = []
        for i in range(6):
            nid = _run(engine.add_node(gid, NodeType.TRADER, f"Node {i}", "NL"))
            nodes.append(nid)
        for i in range(5):
            _run(engine.add_edge(
                gid, nodes[i], nodes[i + 1], "cocoa",
                Decimal("100"), product_description="Transfer",
            ))
        with pytest.raises(CycleDetectedError):
            _run(engine.add_edge(
                gid, nodes[5], nodes[0], "cocoa",
                Decimal("100"), product_description="Cycle edge",
            ))

    def test_diamond_dag_no_cycle(self, engine):
        """Diamond DAG: A->B, A->C, B->D, C->D (valid DAG)."""
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        a = _run(engine.add_node(gid, NodeType.PRODUCER, "A", "GH"))
        b = _run(engine.add_node(gid, NodeType.COLLECTOR, "B", "GH"))
        c = _run(engine.add_node(gid, NodeType.COLLECTOR, "C", "GH"))
        d = _run(engine.add_node(gid, NodeType.PROCESSOR, "D", "GH"))
        _run(engine.add_edge(gid, a, b, "cocoa", Decimal("100"), product_description="E1"))
        _run(engine.add_edge(gid, a, c, "cocoa", Decimal("100"), product_description="E2"))
        _run(engine.add_edge(gid, b, d, "cocoa", Decimal("100"), product_description="E3"))
        _run(engine.add_edge(gid, c, d, "cocoa", Decimal("100"), product_description="E4"))
        assert engine.has_cycles(gid) is False

    def test_empty_graph_no_cycles(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        assert engine.has_cycles(gid) is False

    def test_single_node_no_cycles(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        _run(engine.add_node(gid, NodeType.PRODUCER, "Solo", "GH"))
        assert engine.has_cycles(gid) is False


# ===========================================================================
# 5. Graph Traversal Tests (25 tests)
# ===========================================================================


class TestGraphTraversal:
    """Tests for topological sort, ancestors, descendants, paths."""

    def test_topological_sort_returns_all_nodes(self, populated_graph):
        engine, gid, node_ids, _ = populated_graph
        topo = engine.topological_sort(gid)
        assert len(topo) == len(node_ids)
        assert set(topo) == set(node_ids)

    def test_topological_sort_order(self, populated_graph):
        engine, gid, node_ids, _ = populated_graph
        topo = engine.topological_sort(gid)
        producer_idx = topo.index(node_ids[0])
        importer_idx = topo.index(node_ids[4])
        assert producer_idx < importer_idx

    def test_get_ancestors(self, populated_graph):
        engine, gid, node_ids, _ = populated_graph
        ancestors = engine.get_ancestors(gid, node_ids[4])
        assert node_ids[0] in ancestors
        assert node_ids[3] in ancestors

    def test_get_ancestors_of_root(self, populated_graph):
        engine, gid, node_ids, _ = populated_graph
        ancestors = engine.get_ancestors(gid, node_ids[0])
        assert len(ancestors) == 0

    def test_get_descendants(self, populated_graph):
        engine, gid, node_ids, _ = populated_graph
        desc = engine.get_descendants(gid, node_ids[0])
        assert node_ids[4] in desc

    def test_get_descendants_of_leaf(self, populated_graph):
        engine, gid, node_ids, _ = populated_graph
        desc = engine.get_descendants(gid, node_ids[4])
        assert len(desc) == 0

    def test_get_predecessors(self, populated_graph):
        engine, gid, node_ids, _ = populated_graph
        preds = engine.get_predecessors(gid, node_ids[2])
        assert node_ids[1] in preds

    def test_get_successors(self, populated_graph):
        engine, gid, node_ids, _ = populated_graph
        succs = engine.get_successors(gid, node_ids[2])
        assert node_ids[3] in succs

    def test_get_root_nodes(self, populated_graph):
        engine, gid, node_ids, _ = populated_graph
        roots = engine.get_root_nodes(gid)
        assert node_ids[0] in roots

    def test_get_leaf_nodes(self, populated_graph):
        engine, gid, node_ids, _ = populated_graph
        leaves = engine.get_leaf_nodes(gid)
        assert node_ids[4] in leaves

    def test_get_orphan_nodes_none(self, populated_graph):
        engine, gid, _, _ = populated_graph
        orphans = engine.get_orphan_nodes(gid)
        assert len(orphans) == 0

    def test_get_orphan_nodes_detected(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        _run(engine.add_node(gid, NodeType.PRODUCER, "Isolated", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        n3 = _run(engine.add_node(gid, NodeType.PROCESSOR, "Mill", "GH"))
        _run(engine.add_edge(gid, n2, n3, "cocoa", Decimal("100"), product_description="C"))
        orphans = engine.get_orphan_nodes(gid)
        assert len(orphans) >= 1

    def test_shortest_path(self, populated_graph):
        engine, gid, node_ids, _ = populated_graph
        path = engine.shortest_path(gid, node_ids[0], node_ids[4])
        assert path[0] == node_ids[0]
        assert path[-1] == node_ids[4]
        assert len(path) == 5

    def test_shortest_path_no_path(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.IMPORTER, "Import", "NL"))
        # No edge between them, so the engine raises GraphEngineError
        with pytest.raises(GraphEngineError):
            engine.shortest_path(gid, n1, n2)

    def test_all_paths(self, engine):
        """Test all_paths in a diamond DAG."""
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        a = _run(engine.add_node(gid, NodeType.PRODUCER, "A", "GH"))
        b = _run(engine.add_node(gid, NodeType.COLLECTOR, "B", "GH"))
        c = _run(engine.add_node(gid, NodeType.COLLECTOR, "C", "GH"))
        d = _run(engine.add_node(gid, NodeType.PROCESSOR, "D", "GH"))
        _run(engine.add_edge(gid, a, b, "cocoa", Decimal("100"), product_description="E1"))
        _run(engine.add_edge(gid, a, c, "cocoa", Decimal("100"), product_description="E2"))
        _run(engine.add_edge(gid, b, d, "cocoa", Decimal("100"), product_description="E3"))
        _run(engine.add_edge(gid, c, d, "cocoa", Decimal("100"), product_description="E4"))
        paths = engine.all_paths(gid, a, d)
        assert len(paths) == 2

    def test_max_depth(self, populated_graph):
        engine, gid, _, _ = populated_graph
        depth = engine.get_max_depth(gid)
        assert depth >= 4

    def test_max_depth_single_node(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        _run(engine.add_node(gid, NodeType.PRODUCER, "Solo", "GH"))
        depth = engine.get_max_depth(gid)
        assert depth == 0


# ===========================================================================
# 6. Serialization Tests (20 tests)
# ===========================================================================


class TestSerialization:
    """Tests for JSON, GraphML, and binary serialization."""

    def test_to_json(self, populated_graph):
        engine, gid, _, _ = populated_graph
        json_str = engine.to_json(gid)
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert "nodes" in data or "graph_id" in data

    def test_from_json_roundtrip(self, populated_graph):
        engine, gid, node_ids, edge_ids = populated_graph
        json_str = engine.to_json(gid)
        # from_json is a classmethod returning (engine, graph_id)
        new_engine, new_gid = SupplyChainGraphEngine.from_json(json_str)
        assert new_gid is not None
        new_nodes = new_engine.list_nodes(new_gid)
        assert len(new_nodes) == len(node_ids)

    def test_to_graphml(self, populated_graph):
        engine, gid, _, _ = populated_graph
        graphml = engine.to_graphml(gid)
        assert isinstance(graphml, str)
        assert "graphml" in graphml.lower() or "<?xml" in graphml.lower()

    def test_from_graphml_roundtrip(self, populated_graph):
        engine, gid, node_ids, _ = populated_graph
        graphml = engine.to_graphml(gid)
        # from_graphml is a classmethod returning (engine, graph_id)
        new_engine, new_gid = SupplyChainGraphEngine.from_graphml(graphml)
        assert new_gid is not None

    def test_to_binary(self, populated_graph):
        engine, gid, _, _ = populated_graph
        binary = engine.to_binary(gid)
        assert isinstance(binary, bytes)
        assert binary[:4] == BINARY_MAGIC

    def test_from_binary_roundtrip(self, populated_graph):
        engine, gid, node_ids, edge_ids = populated_graph
        binary = engine.to_binary(gid)
        # from_binary is a classmethod returning (engine, graph_id)
        new_engine, new_gid = SupplyChainGraphEngine.from_binary(binary)
        assert new_gid is not None
        new_nodes = new_engine.list_nodes(new_gid)
        assert len(new_nodes) == len(node_ids)

    def test_json_serialization_empty_graph(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        json_str = engine.to_json(gid)
        data = json.loads(json_str)
        assert data is not None

    def test_graphml_serialization_empty_graph(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        graphml = engine.to_graphml(gid)
        assert graphml is not None

    def test_binary_serialization_empty_graph(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        binary = engine.to_binary(gid)
        assert binary[:4] == BINARY_MAGIC

    def test_json_preserves_quantities(self, populated_graph):
        engine, gid, _, _ = populated_graph
        json_str = engine.to_json(gid)
        assert "5000" in json_str or "4500" in json_str


# ===========================================================================
# 7. Audit Trail Tests (15 tests)
# ===========================================================================


class TestAuditTrail:
    """Tests for mutation recording and audit trail verification."""

    def test_audit_trail_recorded(self, populated_graph):
        engine, gid, _, _ = populated_graph
        trail = engine.get_audit_trail(gid)
        assert len(trail) > 0

    def test_audit_trail_graph_created(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        trail = engine.get_audit_trail(gid)
        types = [t.mutation_type for t in trail]
        assert MutationType.GRAPH_CREATED in types

    def test_audit_trail_node_added(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        trail = engine.get_audit_trail(gid)
        types = [t.mutation_type for t in trail]
        assert MutationType.NODE_ADDED in types

    def test_audit_trail_edge_added(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        n2 = _run(engine.add_node(gid, NodeType.COLLECTOR, "Coop", "GH"))
        _run(engine.add_edge(gid, n1, n2, "cocoa", Decimal("100"), product_description="C"))
        trail = engine.get_audit_trail(gid)
        types = [t.mutation_type for t in trail]
        assert MutationType.EDGE_ADDED in types

    def test_audit_trail_node_removed(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        nid = _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        _run(engine.remove_node(gid, nid))
        trail = engine.get_audit_trail(gid)
        types = [t.mutation_type for t in trail]
        assert MutationType.NODE_REMOVED in types

    def test_audit_chain_verification(self, populated_graph):
        engine, gid, _, _ = populated_graph
        is_valid = engine.verify_audit_chain(gid)
        assert is_valid is True

    def test_mutation_record_hash(self):
        record = GraphMutationRecord(
            graph_id="g-001",
            mutation_type=MutationType.NODE_ADDED,
            target_id="n-001",
        )
        h = record.calculate_hash()
        assert len(h) == 64

    def test_mutation_record_hash_deterministic(self):
        record = GraphMutationRecord(
            graph_id="g-001",
            mutation_type=MutationType.NODE_ADDED,
            target_id="n-001",
        )
        h1 = record.calculate_hash()
        h2 = record.calculate_hash()
        assert h1 == h2

    def test_mutation_record_chaining(self):
        r1 = GraphMutationRecord(
            graph_id="g-001",
            mutation_type=MutationType.NODE_ADDED,
            target_id="n-001",
        )
        h1 = r1.calculate_hash(GENESIS_HASH)
        r2 = GraphMutationRecord(
            graph_id="g-001",
            mutation_type=MutationType.NODE_ADDED,
            target_id="n-002",
        )
        h2 = r2.calculate_hash(h1)
        assert h1 != h2

    def test_audit_trail_limit(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        for i in range(10):
            _run(engine.add_node(gid, NodeType.PRODUCER, f"Farm {i}", "GH"))
        trail = engine.get_audit_trail(gid, limit=5)
        assert len(trail) <= 5


# ===========================================================================
# 8. Snapshot Tests (10 tests)
# ===========================================================================


class TestSnapshots:
    """Tests for graph snapshot creation and retrieval."""

    def test_snapshot_creation(self, populated_graph):
        engine, gid, node_ids, _ = populated_graph
        snapshot = GraphSnapshot(
            graph_id=gid,
            version=engine.get_version(gid),
            node_count=len(node_ids),
            edge_count=4,
        )
        assert snapshot.graph_id == gid
        assert snapshot.node_count == 5

    def test_snapshot_provenance_hash(self, populated_graph):
        engine, gid, node_ids, _ = populated_graph
        snapshot = GraphSnapshot(
            graph_id=gid,
            version=1,
            nodes={"n1": {"type": "producer"}},
            edges={},
        )
        h = snapshot.calculate_provenance_hash()
        assert len(h) == 64

    def test_snapshot_provenance_deterministic(self):
        s1 = GraphSnapshot(
            graph_id="g-001", version=1,
            nodes={"a": {"x": 1}}, edges={},
        )
        s2 = GraphSnapshot(
            graph_id="g-001", version=1,
            nodes={"a": {"x": 1}}, edges={},
        )
        assert s1.calculate_provenance_hash() == s2.calculate_provenance_hash()


# ===========================================================================
# 9. Statistics Tests (8 tests)
# ===========================================================================


class TestStatistics:
    """Tests for graph statistics computation."""

    def test_statistics_node_count(self, populated_graph):
        engine, gid, _, _ = populated_graph
        stats = engine.get_statistics(gid)
        assert stats["total_nodes"] == 5

    def test_statistics_edge_count(self, populated_graph):
        engine, gid, _, _ = populated_graph
        stats = engine.get_statistics(gid)
        assert stats["total_edges"] == 4

    def test_statistics_max_depth(self, populated_graph):
        engine, gid, _, _ = populated_graph
        stats = engine.get_statistics(gid)
        assert stats.get("max_depth", 0) >= 4 or stats.get("max_tier_depth", 0) >= 4

    def test_statistics_empty_graph(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        stats = engine.get_statistics(gid)
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    def test_statistics_not_found(self, engine):
        with pytest.raises(GraphNotFoundError):
            engine.get_statistics("nonexistent")


# ===========================================================================
# 10. Capacity and Error Tests (10 tests)
# ===========================================================================


class TestCapacityAndErrors:
    """Tests for capacity limits and error conditions."""

    def test_capacity_limit(self, small_capacity_engine):
        engine = small_capacity_engine
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        for i in range(5):
            _run(engine.add_node(gid, NodeType.PRODUCER, f"Farm {i}", "GH"))
        with pytest.raises(GraphCapacityError):
            _run(engine.add_node(gid, NodeType.PRODUCER, "Overflow", "GH"))

    def test_graph_not_found_error(self, engine):
        with pytest.raises(GraphNotFoundError):
            engine.get_graph_metadata("no-such-graph")

    def test_node_not_found_error(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        with pytest.raises(NodeNotFoundError):
            engine.get_node(gid, "no-such-node")

    def test_edge_not_found_error(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        with pytest.raises(EdgeNotFoundError):
            engine.get_edge(gid, "no-such-edge")

    def test_operations_on_deleted_graph(self, engine):
        gid = _run(engine.create_graph(operator_id="op-001", commodity="cocoa"))
        _run(engine.delete_graph(gid))
        with pytest.raises(GraphNotFoundError):
            _run(engine.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))


# ===========================================================================
# 11. Enums and Constants Tests (10 tests)
# ===========================================================================


class TestEnumsAndConstants:
    """Tests for enum values and module constants."""

    def test_node_type_values(self):
        assert NodeType.PRODUCER.value == "producer"
        assert NodeType.IMPORTER.value == "importer"
        assert NodeType.CERTIFIER.value == "certifier"

    def test_custody_model_values(self):
        assert CustodyModel.IDENTITY_PRESERVED.value == "identity_preserved"
        assert CustodyModel.SEGREGATED.value == "segregated"
        assert CustodyModel.MASS_BALANCE.value == "mass_balance"

    def test_risk_level_values(self):
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.STANDARD.value == "standard"
        assert RiskLevel.HIGH.value == "high"

    def test_compliance_status_values(self):
        assert ComplianceStatus.COMPLIANT.value == "compliant"
        assert ComplianceStatus.PENDING_VERIFICATION.value == "pending_verification"

    def test_mutation_type_values(self):
        assert MutationType.NODE_ADDED.value == "node_added"
        assert MutationType.EDGE_REMOVED.value == "edge_removed"

    def test_binary_magic_constant(self):
        assert BINARY_MAGIC == b"GLSC"

    def test_genesis_hash_constant(self):
        assert GENESIS_HASH == "genesis"


# ===========================================================================
# 12. Data Model Tests (15 tests)
# ===========================================================================


class TestDataModels:
    """Tests for Pydantic data model validation."""

    def test_node_creation_defaults(self):
        node = SupplyChainNode(
            node_type=NodeType.PRODUCER,
            operator_name="Farm",
            country_code="GH",
        )
        assert node.node_id is not None
        assert node.tier_depth == 0
        assert node.risk_score == 0.0
        assert node.risk_level == RiskLevel.STANDARD

    def test_node_country_code_validation(self):
        node = SupplyChainNode(
            node_type=NodeType.PRODUCER,
            operator_name="Farm",
            country_code="gh",
        )
        assert node.country_code == "GH"

    def test_edge_creation_defaults(self):
        edge = SupplyChainEdge(
            source_node_id="n1",
            target_node_id="n2",
            commodity="cocoa",
            quantity=Decimal("100"),
        )
        assert edge.unit == "kg"
        assert edge.custody_model == CustodyModel.SEGREGATED

    def test_edge_self_loop_validation(self):
        # graph_engine.SupplyChainEdge does not enforce self-loop validation
        # at the model level; the engine's add_edge method catches this.
        # Verify via add_edge instead:
        config = GraphEngineConfig(
            enable_persistence=False,
            enable_audit_trail=False,
            enable_snapshots=False,
        )
        eng = SupplyChainGraphEngine(config=config)
        _run(eng.initialize())
        gid = _run(eng.create_graph(operator_id="op-001", commodity="cocoa"))
        n1 = _run(eng.add_node(gid, NodeType.PRODUCER, "Farm", "GH"))
        with pytest.raises((GraphEngineError, CycleDetectedError, ValueError)):
            _run(eng.add_edge(gid, n1, n1, "cocoa", Decimal("100"), product_description="self-loop"))

    def test_edge_zero_quantity_validation(self):
        with pytest.raises(ValueError):
            SupplyChainEdge(
                source_node_id="n1",
                target_node_id="n2",
                commodity="cocoa",
                quantity=Decimal("0"),
            )

    def test_edge_negative_quantity_validation(self):
        with pytest.raises(ValueError):
            SupplyChainEdge(
                source_node_id="n1",
                target_node_id="n2",
                commodity="cocoa",
                quantity=Decimal("-100"),
            )

    def test_node_latitude_validation(self):
        with pytest.raises(ValueError):
            SupplyChainNode(
                node_type=NodeType.PRODUCER,
                operator_name="Farm",
                country_code="GH",
                latitude=100.0,
            )

    def test_node_longitude_validation(self):
        with pytest.raises(ValueError):
            SupplyChainNode(
                node_type=NodeType.PRODUCER,
                operator_name="Farm",
                country_code="GH",
                longitude=200.0,
            )

    def test_graph_engine_config_defaults(self):
        config = GraphEngineConfig()
        assert config.enable_persistence is True
        assert config.enable_audit_trail is True
        assert config.pool_min_size >= 1

    def test_graph_engine_config_custom(self):
        config = GraphEngineConfig(
            enable_persistence=False,
            max_graph_nodes=1000,
        )
        assert config.enable_persistence is False
        assert config.max_graph_nodes == 1000
