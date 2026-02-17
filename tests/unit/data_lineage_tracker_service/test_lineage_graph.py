# -*- coding: utf-8 -*-
"""
Unit Tests for LineageGraphEngine - AGENT-DATA-018

Tests the in-memory DAG construction, node/edge CRUD, BFS/DFS traversal,
cycle detection, topological ordering, connected components, shortest path,
subgraph extraction, snapshot, statistics, export/import, and clear
operations.

Target: 120+ tests, 12 test classes, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.data_lineage_tracker.lineage_graph import (
    LineageGraphEngine,
    VALID_EDGE_TYPES,
    _hash_dict,
)
from greenlang.data_lineage_tracker.provenance import ProvenanceTracker


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def graph() -> LineageGraphEngine:
    """Create a fresh LineageGraphEngine for each test."""
    return LineageGraphEngine()


@pytest.fixture
def provenance() -> ProvenanceTracker:
    """Create a fresh ProvenanceTracker."""
    return ProvenanceTracker()


@pytest.fixture
def graph_with_provenance(provenance) -> LineageGraphEngine:
    """Create LineageGraphEngine with explicit provenance tracker."""
    return LineageGraphEngine(provenance=provenance)


@pytest.fixture
def populated_graph(graph) -> LineageGraphEngine:
    """Create a graph with 5 nodes and 4 edges forming a DAG:

    A -> B -> D -> E
         |
         +-> C
    """
    graph.add_node("A", "raw.orders", "dataset")
    graph.add_node("B", "clean.orders", "dataset")
    graph.add_node("C", "reports.orders", "report")
    graph.add_node("D", "agg.orders", "dataset")
    graph.add_node("E", "metrics.orders", "metric")

    graph.add_edge("A", "B", edge_type="dataset_level")
    graph.add_edge("B", "C", edge_type="dataset_level")
    graph.add_edge("B", "D", edge_type="dataset_level")
    graph.add_edge("D", "E", edge_type="dataset_level")

    return graph


# ============================================================================
# TestLineageGraphInit
# ============================================================================


class TestLineageGraphInit:
    """Tests for LineageGraphEngine initialization."""

    def test_default_initialization(self, graph):
        assert len(graph) == 0
        assert graph._provenance is not None

    def test_initialization_with_provenance(self, provenance):
        # Pre-seed the tracker so __len__ > 0 (truthy), since
        # the engine uses ``provenance or ProvenanceTracker()`` and
        # ProvenanceTracker defines __len__.
        provenance.record("test", "seed", "init")
        g = LineageGraphEngine(provenance=provenance)
        assert g._provenance is provenance

    def test_creates_provenance_if_none(self, graph):
        assert isinstance(graph._provenance, ProvenanceTracker)

    def test_empty_graph_repr(self, graph):
        r = repr(graph)
        assert "nodes=0" in r
        assert "edges=0" in r

    def test_contains_returns_false_on_empty(self, graph):
        assert "nonexistent" not in graph

    def test_len_returns_zero_on_empty(self, graph):
        assert len(graph) == 0


# ============================================================================
# TestAddNode
# ============================================================================


class TestAddNode:
    """Tests for add_node method."""

    def test_add_single_node(self, graph):
        node = graph.add_node("n1", "db.table1", "dataset")
        assert node["asset_id"] == "n1"
        assert node["qualified_name"] == "db.table1"
        assert node["asset_type"] == "dataset"
        assert len(graph) == 1

    def test_add_node_with_metadata(self, graph):
        meta = {"source": "SAP", "refresh": "daily"}
        node = graph.add_node("n1", "db.table1", "dataset", metadata=meta)
        assert node["metadata"]["source"] == "SAP"
        assert node["metadata"]["refresh"] == "daily"

    def test_add_node_default_empty_metadata(self, graph):
        node = graph.add_node("n1", "db.table1", "dataset")
        assert node["metadata"] == {}

    def test_add_node_has_timestamps(self, graph):
        node = graph.add_node("n1", "db.table1", "dataset")
        assert "created_at" in node
        assert "updated_at" in node

    def test_add_duplicate_node_upserts(self, graph):
        node1 = graph.add_node("n1", "db.table1", "dataset")
        created_at = node1["created_at"]
        node2 = graph.add_node("n1", "db.table1_v2", "table")
        assert node2["qualified_name"] == "db.table1_v2"
        assert node2["created_at"] == created_at  # Preserves original creation
        assert len(graph) == 1

    def test_add_node_empty_id_raises(self, graph):
        with pytest.raises(ValueError, match="asset_id must not be empty"):
            graph.add_node("", "db.table1", "dataset")

    def test_add_node_empty_qualified_name_raises(self, graph):
        with pytest.raises(ValueError, match="qualified_name must not be empty"):
            graph.add_node("n1", "", "dataset")

    def test_add_node_empty_type_raises(self, graph):
        with pytest.raises(ValueError, match="asset_type must not be empty"):
            graph.add_node("n1", "db.table1", "")

    def test_add_multiple_nodes(self, graph):
        for i in range(10):
            graph.add_node(f"n{i}", f"db.table{i}", "dataset")
        assert len(graph) == 10

    def test_contains_after_add(self, graph):
        graph.add_node("n1", "db.table1", "dataset")
        assert "n1" in graph
        assert "n2" not in graph


# ============================================================================
# TestRemoveNode
# ============================================================================


class TestRemoveNode:
    """Tests for remove_node method."""

    def test_remove_existing_node(self, graph):
        graph.add_node("n1", "db.table1", "dataset")
        result = graph.remove_node("n1")
        assert result is True
        assert len(graph) == 0
        assert "n1" not in graph

    def test_remove_nonexistent_node(self, graph):
        result = graph.remove_node("nonexistent")
        assert result is False

    def test_remove_node_removes_connected_edges(self, populated_graph):
        g = populated_graph
        edges_before = len(g._edges)
        g.remove_node("B")
        assert "B" not in g
        # B had edges from A->B, B->C, B->D (3 edges removed)
        assert len(g._edges) == edges_before - 3

    def test_remove_leaf_node(self, populated_graph):
        g = populated_graph
        result = g.remove_node("E")
        assert result is True
        assert "E" not in g

    def test_remove_root_node(self, populated_graph):
        g = populated_graph
        result = g.remove_node("A")
        assert result is True
        assert "A" not in g


# ============================================================================
# TestAddEdge
# ============================================================================


class TestAddEdge:
    """Tests for add_edge method."""

    def test_add_edge_dataset_level(self, graph):
        graph.add_node("src", "db.source", "dataset")
        graph.add_node("tgt", "db.target", "dataset")
        edge = graph.add_edge("src", "tgt", edge_type="dataset_level")
        assert edge["source_asset_id"] == "src"
        assert edge["target_asset_id"] == "tgt"
        assert edge["edge_type"] == "dataset_level"
        assert "edge_id" in edge

    def test_add_edge_column_level(self, graph):
        graph.add_node("src", "db.source", "dataset")
        graph.add_node("tgt", "db.target", "dataset")
        edge = graph.add_edge(
            "src", "tgt",
            edge_type="column_level",
            source_field="col_a",
            target_field="col_b",
        )
        assert edge["edge_type"] == "column_level"
        assert edge["source_field"] == "col_a"
        assert edge["target_field"] == "col_b"

    def test_add_edge_default_type(self, graph):
        graph.add_node("src", "db.source", "dataset")
        graph.add_node("tgt", "db.target", "dataset")
        edge = graph.add_edge("src", "tgt")
        assert edge["edge_type"] == "dataset_level"

    def test_add_edge_with_confidence(self, graph):
        graph.add_node("src", "db.source", "dataset")
        graph.add_node("tgt", "db.target", "dataset")
        edge = graph.add_edge("src", "tgt", confidence=0.85)
        assert edge["confidence"] == 0.85

    def test_add_edge_invalid_type_raises(self, graph):
        graph.add_node("src", "db.source", "dataset")
        graph.add_node("tgt", "db.target", "dataset")
        with pytest.raises(ValueError, match="edge_type must be one of"):
            graph.add_edge("src", "tgt", edge_type="invalid_type")

    def test_add_edge_missing_source_raises(self, graph):
        graph.add_node("tgt", "db.target", "dataset")
        with pytest.raises(ValueError, match="Source node.*does not exist"):
            graph.add_edge("missing", "tgt")

    def test_add_edge_missing_target_raises(self, graph):
        graph.add_node("src", "db.source", "dataset")
        with pytest.raises(ValueError, match="Target node.*does not exist"):
            graph.add_edge("src", "missing")

    def test_add_edge_empty_source_raises(self, graph):
        with pytest.raises(ValueError, match="source_asset_id must not be empty"):
            graph.add_edge("", "tgt")

    def test_add_edge_empty_target_raises(self, graph):
        with pytest.raises(ValueError, match="target_asset_id must not be empty"):
            graph.add_edge("src", "")

    def test_add_edge_confidence_out_of_range_raises(self, graph):
        graph.add_node("src", "db.source", "dataset")
        graph.add_node("tgt", "db.target", "dataset")
        with pytest.raises(ValueError, match="confidence must be in"):
            graph.add_edge("src", "tgt", confidence=1.5)
        with pytest.raises(ValueError, match="confidence must be in"):
            graph.add_edge("src", "tgt", confidence=-0.1)

    def test_add_edge_cycle_prevention(self, graph):
        graph.add_node("A", "node_a", "dataset")
        graph.add_node("B", "node_b", "dataset")
        graph.add_edge("A", "B")
        with pytest.raises(ValueError, match="would create a cycle"):
            graph.add_edge("B", "A")

    def test_add_edge_with_transformation_id(self, graph):
        graph.add_node("src", "db.source", "dataset")
        graph.add_node("tgt", "db.target", "dataset")
        edge = graph.add_edge("src", "tgt", transformation_id="txn-001")
        assert edge["transformation_id"] == "txn-001"

    def test_add_edge_with_transformation_logic(self, graph):
        graph.add_node("src", "db.source", "dataset")
        graph.add_node("tgt", "db.target", "dataset")
        edge = graph.add_edge("src", "tgt", transformation_logic="SELECT * FROM src")
        assert edge["transformation_logic"] == "SELECT * FROM src"


# ============================================================================
# TestRemoveEdge
# ============================================================================


class TestRemoveEdge:
    """Tests for remove_edge method."""

    def test_remove_existing_edge(self, graph):
        graph.add_node("A", "node_a", "dataset")
        graph.add_node("B", "node_b", "dataset")
        edge = graph.add_edge("A", "B")
        edge_id = edge["edge_id"]
        result = graph.remove_edge(edge_id)
        assert result is True
        assert graph.get_edge(edge_id) is None

    def test_remove_nonexistent_edge(self, graph):
        result = graph.remove_edge("fake-edge-id")
        assert result is False


# ============================================================================
# TestLookupOperations
# ============================================================================


class TestLookupOperations:
    """Tests for get_node, get_edge, get_upstream_edges, get_downstream_edges."""

    def test_get_node_returns_copy(self, graph):
        graph.add_node("n1", "db.table1", "dataset")
        node = graph.get_node("n1")
        assert node is not None
        assert node["asset_id"] == "n1"
        # Mutating returned copy should not affect internal state
        node["asset_id"] = "MUTATED"
        assert graph.get_node("n1")["asset_id"] == "n1"

    def test_get_node_nonexistent(self, graph):
        assert graph.get_node("missing") is None

    def test_get_edge_returns_copy(self, populated_graph):
        g = populated_graph
        edge_id = list(g._edges.keys())[0]
        edge = g.get_edge(edge_id)
        assert edge is not None
        assert edge["edge_id"] == edge_id

    def test_get_edge_nonexistent(self, graph):
        assert graph.get_edge("missing") is None

    def test_get_upstream_edges(self, populated_graph):
        g = populated_graph
        edges = g.get_upstream_edges("B")
        assert len(edges) == 1
        assert edges[0]["source_asset_id"] == "A"

    def test_get_downstream_edges(self, populated_graph):
        g = populated_graph
        edges = g.get_downstream_edges("B")
        assert len(edges) == 2
        targets = {e["target_asset_id"] for e in edges}
        assert targets == {"C", "D"}

    def test_upstream_edges_empty_for_root(self, populated_graph):
        g = populated_graph
        edges = g.get_upstream_edges("A")
        assert len(edges) == 0

    def test_downstream_edges_empty_for_leaf(self, populated_graph):
        g = populated_graph
        edges = g.get_downstream_edges("E")
        assert len(edges) == 0


# ============================================================================
# TestTraversal
# ============================================================================


class TestTraversal:
    """Tests for traverse_backward and traverse_forward methods."""

    def test_traverse_forward_from_root(self, populated_graph):
        g = populated_graph
        result = g.traverse_forward("A", max_depth=10)
        assert result["root"] == "A"
        assert len(result["nodes"]) == 5  # A, B, C, D, E
        assert len(result["edges"]) == 4
        assert result["depth"] >= 1

    def test_traverse_forward_with_depth_limit(self, populated_graph):
        g = populated_graph
        result = g.traverse_forward("A", max_depth=1)
        node_ids = {n["asset_id"] for n in result["nodes"]}
        assert "A" in node_ids
        assert "B" in node_ids
        # C, D may not be discovered at depth 1
        assert result["depth"] <= 1

    def test_traverse_backward_from_leaf(self, populated_graph):
        g = populated_graph
        result = g.traverse_backward("E", max_depth=10)
        assert result["root"] == "E"
        node_ids = {n["asset_id"] for n in result["nodes"]}
        assert "E" in node_ids
        assert "D" in node_ids
        assert "B" in node_ids
        assert "A" in node_ids

    def test_traverse_backward_with_depth_limit(self, populated_graph):
        g = populated_graph
        result = g.traverse_backward("E", max_depth=1)
        node_ids = {n["asset_id"] for n in result["nodes"]}
        assert "E" in node_ids
        assert "D" in node_ids

    def test_traverse_nonexistent_node(self, graph):
        result = graph.traverse_forward("missing")
        assert result["root"] == "missing"
        assert result["nodes"] == []
        assert result["edges"] == []

    def test_traverse_forward_path(self, populated_graph):
        g = populated_graph
        result = g.traverse_forward("A", max_depth=10)
        assert len(result["path"]) >= 1
        assert result["path"][0] == "A"

    def test_traverse_single_node_graph(self, graph):
        graph.add_node("solo", "db.solo", "dataset")
        result = graph.traverse_forward("solo", max_depth=10)
        assert len(result["nodes"]) == 1
        assert result["depth"] == 0


# ============================================================================
# TestSubgraph
# ============================================================================


class TestSubgraph:
    """Tests for get_subgraph method."""

    def test_subgraph_center_node(self, populated_graph):
        g = populated_graph
        result = g.get_subgraph("B", depth=2)
        assert result["center"] == "B"
        node_ids = {n["asset_id"] for n in result["nodes"]}
        assert "B" in node_ids
        assert "A" in node_ids  # upstream
        assert "C" in node_ids  # downstream
        assert "D" in node_ids  # downstream

    def test_subgraph_depth_parameter(self, populated_graph):
        g = populated_graph
        result = g.get_subgraph("B", depth=1)
        node_ids = {n["asset_id"] for n in result["nodes"]}
        assert "B" in node_ids
        assert result["depth"] == 1

    def test_subgraph_deduplicates_nodes(self, populated_graph):
        g = populated_graph
        result = g.get_subgraph("B", depth=5)
        node_ids = [n["asset_id"] for n in result["nodes"]]
        assert len(node_ids) == len(set(node_ids))

    def test_subgraph_deduplicates_edges(self, populated_graph):
        g = populated_graph
        result = g.get_subgraph("B", depth=5)
        edge_ids = [e["edge_id"] for e in result["edges"]]
        assert len(edge_ids) == len(set(edge_ids))


# ============================================================================
# TestShortestPath
# ============================================================================


class TestShortestPath:
    """Tests for get_shortest_path method."""

    def test_shortest_path_direct(self, populated_graph):
        g = populated_graph
        path = g.get_shortest_path("A", "B")
        assert path == ["A", "B"]

    def test_shortest_path_multi_hop(self, populated_graph):
        g = populated_graph
        path = g.get_shortest_path("A", "E")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "E"
        # A -> B -> D -> E (length 4)
        assert len(path) == 4

    def test_shortest_path_same_node(self, populated_graph):
        g = populated_graph
        path = g.get_shortest_path("B", "B")
        assert path == ["B"]

    def test_shortest_path_no_connection(self, graph):
        graph.add_node("X", "node_x", "dataset")
        graph.add_node("Y", "node_y", "dataset")
        path = graph.get_shortest_path("X", "Y")
        assert path is None

    def test_shortest_path_reverse_direction(self, populated_graph):
        g = populated_graph
        # No path from E back to A (directed graph)
        path = g.get_shortest_path("E", "A")
        assert path is None

    def test_shortest_path_missing_node(self, graph):
        path = graph.get_shortest_path("missing1", "missing2")
        assert path is None


# ============================================================================
# TestCycleDetection
# ============================================================================


class TestCycleDetection:
    """Tests for detect_cycles method."""

    def test_no_cycles_in_dag(self, populated_graph):
        g = populated_graph
        cycles = g.detect_cycles()
        assert len(cycles) == 0

    def test_empty_graph_no_cycles(self, graph):
        cycles = graph.detect_cycles()
        assert len(cycles) == 0

    def test_single_node_no_cycles(self, graph):
        graph.add_node("solo", "db.solo", "dataset")
        cycles = graph.detect_cycles()
        assert len(cycles) == 0


# ============================================================================
# TestTopologicalOrder
# ============================================================================


class TestTopologicalOrder:
    """Tests for get_topological_order method."""

    def test_topological_order_length(self, populated_graph):
        g = populated_graph
        order = g.get_topological_order()
        assert len(order) == 5

    def test_topological_order_roots_first(self, populated_graph):
        g = populated_graph
        order = g.get_topological_order()
        a_pos = order.index("A")
        b_pos = order.index("B")
        assert a_pos < b_pos  # A must come before B

    def test_topological_order_respects_edges(self, populated_graph):
        g = populated_graph
        order = g.get_topological_order()
        d_pos = order.index("D")
        e_pos = order.index("E")
        assert d_pos < e_pos  # D must come before E

    def test_topological_order_empty_graph(self, graph):
        order = graph.get_topological_order()
        assert order == []


# ============================================================================
# TestConnectedComponents
# ============================================================================


class TestConnectedComponents:
    """Tests for get_connected_components method."""

    def test_single_component(self, populated_graph):
        g = populated_graph
        components = g.get_connected_components()
        assert len(components) == 1
        assert len(components[0]) == 5

    def test_two_disconnected_components(self, populated_graph):
        g = populated_graph
        g.add_node("X", "isolated.node", "dataset")
        g.add_node("Y", "isolated.node2", "dataset")
        g.add_edge("X", "Y")
        components = g.get_connected_components()
        assert len(components) == 2

    def test_all_orphans(self, graph):
        graph.add_node("A", "node_a", "dataset")
        graph.add_node("B", "node_b", "dataset")
        graph.add_node("C", "node_c", "dataset")
        components = graph.get_connected_components()
        assert len(components) == 3

    def test_empty_graph_components(self, graph):
        components = graph.get_connected_components()
        assert len(components) == 0


# ============================================================================
# TestRootsAndLeaves
# ============================================================================


class TestRootsAndLeaves:
    """Tests for get_roots and get_leaves methods."""

    def test_roots(self, populated_graph):
        roots = populated_graph.get_roots()
        assert "A" in roots
        assert len(roots) == 1

    def test_leaves(self, populated_graph):
        leaves = populated_graph.get_leaves()
        assert "C" in leaves
        assert "E" in leaves
        assert len(leaves) == 2

    def test_orphan_is_both_root_and_leaf(self, graph):
        graph.add_node("orphan", "db.orphan", "dataset")
        roots = graph.get_roots()
        leaves = graph.get_leaves()
        assert "orphan" in roots
        assert "orphan" in leaves


# ============================================================================
# TestDepthComputation
# ============================================================================


class TestDepthComputation:
    """Tests for compute_depth method."""

    def test_root_depth_is_zero(self, populated_graph):
        depth = populated_graph.compute_depth("A")
        assert depth == 0

    def test_leaf_depth(self, populated_graph):
        depth = populated_graph.compute_depth("E")
        assert depth == 3  # A -> B -> D -> E

    def test_mid_node_depth(self, populated_graph):
        depth = populated_graph.compute_depth("B")
        assert depth == 1  # A -> B

    def test_nonexistent_node_depth(self, graph):
        depth = graph.compute_depth("missing")
        assert depth == 0


# ============================================================================
# TestSnapshot
# ============================================================================


class TestSnapshot:
    """Tests for take_snapshot method."""

    def test_snapshot_returns_dict(self, populated_graph):
        snap = populated_graph.take_snapshot()
        assert isinstance(snap, dict)

    def test_snapshot_has_required_keys(self, populated_graph):
        snap = populated_graph.take_snapshot()
        required = [
            "snapshot_id", "name", "timestamp", "node_count",
            "edge_count", "max_depth", "connected_components",
            "orphan_count", "root_count", "leaf_count",
            "coverage_score", "graph_hash",
        ]
        for key in required:
            assert key in snap, f"Missing key: {key}"

    def test_snapshot_correct_counts(self, populated_graph):
        snap = populated_graph.take_snapshot()
        assert snap["node_count"] == 5
        assert snap["edge_count"] == 4
        assert snap["root_count"] == 1
        assert snap["leaf_count"] == 2
        assert snap["orphan_count"] == 0

    def test_snapshot_with_custom_name(self, populated_graph):
        snap = populated_graph.take_snapshot(snapshot_name="test-snapshot")
        assert snap["name"] == "test-snapshot"

    def test_snapshot_deterministic_hash(self, populated_graph):
        snap1 = populated_graph.take_snapshot()
        snap2 = populated_graph.take_snapshot()
        assert snap1["graph_hash"] == snap2["graph_hash"]

    def test_snapshot_empty_graph(self, graph):
        snap = graph.take_snapshot()
        assert snap["node_count"] == 0
        assert snap["edge_count"] == 0


# ============================================================================
# TestStatistics
# ============================================================================


class TestStatistics:
    """Tests for get_statistics method."""

    def test_statistics_keys(self, populated_graph):
        stats = populated_graph.get_statistics()
        required = [
            "total_nodes", "total_edges", "max_depth", "avg_depth",
            "connected_components", "root_count", "leaf_count", "orphan_count",
        ]
        for key in required:
            assert key in stats, f"Missing key: {key}"

    def test_statistics_correct_values(self, populated_graph):
        stats = populated_graph.get_statistics()
        assert stats["total_nodes"] == 5
        assert stats["total_edges"] == 4
        assert stats["root_count"] == 1
        assert stats["leaf_count"] == 2
        assert stats["orphan_count"] == 0
        assert stats["connected_components"] == 1

    def test_statistics_empty_graph(self, graph):
        stats = graph.get_statistics()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0


# ============================================================================
# TestExportImport
# ============================================================================


class TestExportImport:
    """Tests for export_graph and import_graph methods."""

    def test_export_graph(self, populated_graph):
        export = populated_graph.export_graph()
        assert "nodes" in export
        assert "edges" in export
        assert "metadata" in export
        assert len(export["nodes"]) == 5
        assert len(export["edges"]) == 4

    def test_import_graph(self, graph, populated_graph):
        exported = populated_graph.export_graph()
        graph.import_graph(exported)
        assert len(graph) == 5

    def test_import_clears_existing(self, populated_graph):
        g = populated_graph
        new_data = {
            "nodes": [
                {"asset_id": "X", "qualified_name": "x.node", "asset_type": "dataset"},
            ],
            "edges": [],
        }
        g.import_graph(new_data)
        assert len(g) == 1
        assert "A" not in g
        assert "X" in g

    def test_import_invalid_type_raises(self, graph):
        with pytest.raises(ValueError, match="requires a dict"):
            graph.import_graph("not_a_dict")

    def test_import_missing_nodes_key_raises(self, graph):
        with pytest.raises(ValueError, match="missing 'nodes'"):
            graph.import_graph({"edges": []})

    def test_import_missing_edges_key_raises(self, graph):
        with pytest.raises(ValueError, match="missing 'edges'"):
            graph.import_graph({"nodes": []})

    def test_import_skips_edges_with_missing_endpoints(self, graph):
        data = {
            "nodes": [
                {"asset_id": "A", "qualified_name": "a", "asset_type": "dataset"},
            ],
            "edges": [
                {
                    "edge_id": "e1",
                    "source_asset_id": "A",
                    "target_asset_id": "MISSING",
                },
            ],
        }
        graph.import_graph(data)
        assert len(graph) == 1
        assert len(graph._edges) == 0

    def test_roundtrip_preserves_graph(self, populated_graph):
        g = populated_graph
        exported = g.export_graph()
        new_graph = LineageGraphEngine()
        new_graph.import_graph(exported)
        assert len(new_graph) == len(g)
        assert len(new_graph._edges) == len(g._edges)


# ============================================================================
# TestClearGraph
# ============================================================================


class TestClearGraph:
    """Tests for clear method."""

    def test_clear_empties_graph(self, populated_graph):
        g = populated_graph
        assert len(g) > 0
        g.clear()
        assert len(g) == 0
        assert len(g._edges) == 0

    def test_clear_empty_graph(self, graph):
        graph.clear()
        assert len(graph) == 0

    def test_clear_allows_new_additions(self, populated_graph):
        g = populated_graph
        g.clear()
        g.add_node("new", "db.new", "dataset")
        assert len(g) == 1


# ============================================================================
# TestHashDict helper
# ============================================================================


class TestHashDict:
    """Tests for the module-level _hash_dict helper."""

    def test_deterministic_output(self):
        data = {"key": "value", "num": 42}
        h1 = _hash_dict(data)
        h2 = _hash_dict(data)
        assert h1 == h2
        assert len(h1) == 64

    def test_different_data_different_hash(self):
        h1 = _hash_dict({"a": 1})
        h2 = _hash_dict({"a": 2})
        assert h1 != h2

    def test_key_order_independent(self):
        h1 = _hash_dict({"b": 2, "a": 1})
        h2 = _hash_dict({"a": 1, "b": 2})
        assert h1 == h2
