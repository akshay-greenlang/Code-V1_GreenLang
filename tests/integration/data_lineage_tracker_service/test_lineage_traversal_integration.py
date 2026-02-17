# -*- coding: utf-8 -*-
"""
Integration Tests: Lineage Traversal and Impact Analysis (AGENT-DATA-018)
==========================================================================

Tests LineageGraphEngine traversal algorithms (BFS forward/backward,
shortest path, cycle detection, topological sort, connected components)
and ImpactAnalyzerEngine impact analysis (blast radius, severity scoring,
root cause analysis, dependency matrix, critical path discovery).

All tests exercise the engines against a realistic GreenLang data flow
scenario constructed by the ``populated_pipeline`` fixture:

    supplier.invoices -> pdf_extractor -> extracted_invoices ->
    excel_normalizer -> normalized_spend -> spend_categorizer ->
    categorized_spend -> emission_calculator -> scope3_emissions ->
    csrd_report

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

import pytest

from greenlang.data_lineage_tracker.lineage_graph import (
    LineageGraphEngine,
    VALID_EDGE_TYPES,
)

from tests.integration.data_lineage_tracker_service.conftest import (
    GREENLANG_ASSET_NAMES,
)


# ---------------------------------------------------------------------------
# TestLineageTraversal
# ---------------------------------------------------------------------------


class TestLineageTraversal:
    """Integration tests for lineage graph traversal and impact analysis."""

    # ================================================================== #
    # Graph structure verification
    # ================================================================== #

    def test_populated_graph_has_10_nodes(self, populated_pipeline):
        """Test that the populated pipeline creates exactly 10 nodes."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        stats = graph.get_statistics()
        assert stats["total_nodes"] == 10

    def test_populated_graph_has_9_edges(self, populated_pipeline):
        """Test that the populated pipeline creates exactly 9 edges."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        stats = graph.get_statistics()
        assert stats["total_edges"] == 9

    def test_populated_graph_single_connected_component(self, populated_pipeline):
        """Test that all 10 nodes form a single connected component."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        components = graph.get_connected_components()
        assert len(components) == 1
        assert len(components[0]) == 10

    def test_populated_graph_is_acyclic(self, populated_pipeline):
        """Test that the populated graph has no cycles (valid DAG)."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        cycles = graph.detect_cycles()
        assert cycles == []

    # ================================================================== #
    # Forward traversal tests
    # ================================================================== #

    def test_forward_traversal_from_source(self, populated_pipeline):
        """Test forward traversal from the external source reaches all 10 nodes."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        source_id = assets[0]["asset_id"]

        result = graph.traverse_forward(source_id, max_depth=20)

        assert result["root"] == source_id
        assert len(result["nodes"]) == 10  # all nodes reachable
        assert len(result["edges"]) == 9  # all edges traversed
        assert result["depth"] == 9  # linear chain of 10 nodes = depth 9

    def test_forward_traversal_from_mid_node(self, populated_pipeline):
        """Test forward traversal from a middle node in the chain."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        # data.normalized_spend is the 5th asset (index 4)
        mid_id = assets[4]["asset_id"]

        result = graph.traverse_forward(mid_id, max_depth=20)

        # From normalized_spend, we should reach: spend_categorizer,
        # categorized_spend, emission_calculator, scope3_emissions, csrd_report
        assert len(result["nodes"]) == 6  # mid + 5 downstream
        assert result["depth"] == 5

    def test_forward_traversal_from_leaf_returns_only_self(self, populated_pipeline):
        """Test forward traversal from the final leaf node returns only itself."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        leaf_id = assets[-1]["asset_id"]  # report.csrd_report

        result = graph.traverse_forward(leaf_id, max_depth=20)

        assert len(result["nodes"]) == 1  # only the leaf itself
        assert len(result["edges"]) == 0
        assert result["depth"] == 0

    def test_forward_traversal_with_depth_limit(self, populated_pipeline):
        """Test forward traversal respects depth limit."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        source_id = assets[0]["asset_id"]

        result = graph.traverse_forward(source_id, max_depth=3)

        assert result["depth"] <= 3
        # Should reach at most 4 nodes (source + 3 hops)
        assert len(result["nodes"]) <= 4

    def test_forward_traversal_nonexistent_node_returns_empty(self, populated_pipeline):
        """Test forward traversal from non-existent node returns empty result."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        result = graph.traverse_forward("nonexistent-asset-id", max_depth=10)

        assert result["root"] == "nonexistent-asset-id"
        assert len(result["nodes"]) == 0
        assert len(result["edges"]) == 0

    # ================================================================== #
    # Backward traversal tests
    # ================================================================== #

    def test_backward_traversal_full_chain(self, populated_pipeline):
        """Test backward traversal from the final report reaches all 10 nodes."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        report_id = assets[-1]["asset_id"]  # report.csrd_report

        result = graph.traverse_backward(report_id, max_depth=20)

        assert result["root"] == report_id
        assert len(result["nodes"]) == 10  # all 10 nodes reachable
        assert len(result["edges"]) == 9
        assert result["depth"] == 9

    def test_backward_traversal_from_source_returns_only_self(self, populated_pipeline):
        """Test backward traversal from the root source returns only itself."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        source_id = assets[0]["asset_id"]

        result = graph.traverse_backward(source_id, max_depth=20)

        assert len(result["nodes"]) == 1
        assert len(result["edges"]) == 0
        assert result["depth"] == 0

    def test_backward_traversal_path_order(self, populated_pipeline):
        """Test backward traversal path starts from root and follows BFS order."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        mid_id = assets[4]["asset_id"]  # data.normalized_spend

        result = graph.traverse_backward(mid_id, max_depth=20)

        # Path should start with the root asset (mid node) in BFS order
        assert result["path"][0] == mid_id
        assert len(result["path"]) == 5  # mid + 4 upstream nodes

    # ================================================================== #
    # Shortest path tests
    # ================================================================== #

    def test_shortest_path_source_to_report(self, populated_pipeline):
        """Test shortest path from external source to final report."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        source_id = assets[0]["asset_id"]
        report_id = assets[-1]["asset_id"]

        path = graph.get_shortest_path(source_id, report_id)

        assert path is not None
        assert len(path) == 10  # linear chain = all 10 nodes
        assert path[0] == source_id
        assert path[-1] == report_id

    def test_shortest_path_adjacent_nodes(self, populated_pipeline):
        """Test shortest path between directly connected nodes."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        a_id = assets[0]["asset_id"]
        b_id = assets[1]["asset_id"]

        path = graph.get_shortest_path(a_id, b_id)

        assert path is not None
        assert len(path) == 2
        assert path[0] == a_id
        assert path[1] == b_id

    def test_shortest_path_same_node(self, populated_pipeline):
        """Test shortest path from a node to itself returns single-element list."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        node_id = assets[3]["asset_id"]

        path = graph.get_shortest_path(node_id, node_id)

        assert path == [node_id]

    def test_shortest_path_reverse_direction_returns_none(self, populated_pipeline):
        """Test shortest path in reverse direction returns None (DAG is directed)."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        source_id = assets[0]["asset_id"]
        report_id = assets[-1]["asset_id"]

        # Cannot go backward in a directed graph using shortest_path
        path = graph.get_shortest_path(report_id, source_id)

        assert path is None

    def test_shortest_path_nonexistent_node_returns_none(self, populated_pipeline):
        """Test shortest path with non-existent node returns None."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        source_id = assets[0]["asset_id"]

        path = graph.get_shortest_path(source_id, "nonexistent-node")
        assert path is None

    # ================================================================== #
    # Topological ordering tests
    # ================================================================== #

    def test_topological_order_has_all_nodes(self, populated_pipeline):
        """Test topological order includes all 10 nodes."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        order = graph.get_topological_order()

        assert len(order) == 10
        # Verify all asset IDs are present
        order_set = set(order)
        for asset in assets:
            assert asset["asset_id"] in order_set

    def test_topological_order_source_before_target(self, populated_pipeline):
        """Test topological order places sources before targets."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        order = graph.get_topological_order()
        index_map = {node_id: idx for idx, node_id in enumerate(order)}

        # For a linear chain, each asset should come before the next
        for i in range(len(assets) - 1):
            src_id = assets[i]["asset_id"]
            tgt_id = assets[i + 1]["asset_id"]
            assert index_map[src_id] < index_map[tgt_id], (
                f"Topological order violation: {src_id} should come before {tgt_id}"
            )

    # ================================================================== #
    # Roots and leaves tests
    # ================================================================== #

    def test_graph_has_single_root(self, populated_pipeline):
        """Test that the linear chain has exactly one root node."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        roots = graph.get_roots()
        assert len(roots) == 1
        assert roots[0] == assets[0]["asset_id"]

    def test_graph_has_single_leaf(self, populated_pipeline):
        """Test that the linear chain has exactly one leaf node."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        leaves = graph.get_leaves()
        assert len(leaves) == 1
        assert leaves[0] == assets[-1]["asset_id"]

    # ================================================================== #
    # Depth computation tests
    # ================================================================== #

    def test_depth_of_root_is_zero(self, populated_pipeline):
        """Test that the root node has depth 0."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        depth = graph.compute_depth(assets[0]["asset_id"])
        assert depth == 0

    def test_depth_of_leaf_is_max(self, populated_pipeline):
        """Test that the leaf node has depth equal to chain length - 1."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        depth = graph.compute_depth(assets[-1]["asset_id"])
        assert depth == 9  # 10 nodes in chain, depth = 9

    def test_depth_increases_monotonically(self, populated_pipeline):
        """Test that depth increases monotonically along the linear chain."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        prev_depth = -1
        for asset in assets:
            depth = graph.compute_depth(asset["asset_id"])
            assert depth > prev_depth
            prev_depth = depth

    # ================================================================== #
    # Subgraph extraction tests
    # ================================================================== #

    def test_subgraph_from_center(self, populated_pipeline):
        """Test subgraph extraction from a center node captures both directions."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        center_id = assets[4]["asset_id"]  # data.normalized_spend

        subgraph = graph.get_subgraph(center_id, depth=3)

        assert subgraph["center"] == center_id
        assert subgraph["depth"] == 3
        assert len(subgraph["nodes"]) > 1
        assert len(subgraph["edges"]) > 0

    def test_subgraph_from_leaf_captures_upstream(self, populated_pipeline):
        """Test subgraph from leaf captures upstream nodes within depth."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph
        leaf_id = assets[-1]["asset_id"]

        subgraph = graph.get_subgraph(leaf_id, depth=2)

        assert subgraph["center"] == leaf_id
        # Should include leaf + at least 2 upstream nodes
        assert len(subgraph["nodes"]) >= 3

    # ================================================================== #
    # Graph snapshot tests
    # ================================================================== #

    def test_graph_snapshot_captures_structure(self, populated_pipeline):
        """Test that take_snapshot captures node and edge counts."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        snapshot = graph.take_snapshot()

        assert "snapshot_id" in snapshot
        assert snapshot["node_count"] == 10
        assert snapshot["edge_count"] == 9
        assert "graph_hash" in snapshot
        assert len(snapshot["graph_hash"]) == 64
        assert snapshot["orphan_count"] == 0
        assert snapshot["root_count"] == 1
        assert snapshot["leaf_count"] == 1
        assert snapshot["coverage_score"] == 1.0
        assert snapshot["max_depth"] == 9

    def test_graph_snapshot_hash_deterministic(self, populated_pipeline):
        """Test that snapshots of the same graph produce the same hash."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        snap1 = graph.take_snapshot()
        snap2 = graph.take_snapshot()

        assert snap1["graph_hash"] == snap2["graph_hash"]

    def test_graph_snapshot_hash_changes_with_mutation(self, populated_pipeline):
        """Test that the snapshot hash changes when the graph is mutated."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        snap_before = graph.take_snapshot()

        # Add a new node
        graph.add_node("new_node", "test.new_node", "dataset")

        snap_after = graph.take_snapshot()

        assert snap_before["graph_hash"] != snap_after["graph_hash"]
        assert snap_after["node_count"] == 11

    # ================================================================== #
    # Export / import tests
    # ================================================================== #

    def test_graph_export_import_roundtrip(self, populated_pipeline):
        """Test that exporting and importing the graph preserves structure."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        exported = graph.export_graph()

        assert len(exported["nodes"]) == 10
        assert len(exported["edges"]) == 9
        assert "metadata" in exported

        # Create a new graph and import
        from greenlang.data_lineage_tracker.provenance import ProvenanceTracker
        new_graph = LineageGraphEngine(provenance=ProvenanceTracker())
        new_graph.import_graph(exported)

        new_stats = new_graph.get_statistics()
        assert new_stats["total_nodes"] == 10
        assert new_stats["total_edges"] == 9

    # ================================================================== #
    # Node and edge lookup tests
    # ================================================================== #

    def test_get_node_returns_metadata(self, populated_pipeline):
        """Test that get_node returns correct metadata for existing nodes."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        node = graph.get_node(assets[0]["asset_id"])

        assert node is not None
        assert node["asset_id"] == assets[0]["asset_id"]
        assert node["qualified_name"] == "supplier.invoices"
        assert node["asset_type"] == "external_source"

    def test_get_node_nonexistent_returns_none(self, populated_pipeline):
        """Test that get_node for non-existent ID returns None."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        node = graph.get_node("does-not-exist")
        assert node is None

    def test_get_upstream_edges(self, populated_pipeline):
        """Test get_upstream_edges returns incoming edges."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        # The second asset (pdf_extractor) should have 1 incoming edge
        upstream = graph.get_upstream_edges(assets[1]["asset_id"])
        assert len(upstream) == 1
        assert upstream[0]["source_asset_id"] == assets[0]["asset_id"]

    def test_get_downstream_edges(self, populated_pipeline):
        """Test get_downstream_edges returns outgoing edges."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        # The first asset (supplier.invoices) should have 1 outgoing edge
        downstream = graph.get_downstream_edges(assets[0]["asset_id"])
        assert len(downstream) == 1
        assert downstream[0]["target_asset_id"] == assets[1]["asset_id"]

    # ================================================================== #
    # Edge validation tests
    # ================================================================== #

    def test_add_edge_invalid_type_raises(self, lineage_graph, provenance):
        """Test that adding an edge with invalid type raises ValueError."""
        lineage_graph.add_node("a", "test.a", "dataset")
        lineage_graph.add_node("b", "test.b", "dataset")

        with pytest.raises(ValueError, match="edge_type must be one of"):
            lineage_graph.add_edge("a", "b", edge_type="invalid_type")

    def test_add_edge_missing_source_raises(self, lineage_graph, provenance):
        """Test that adding an edge with non-existent source raises ValueError."""
        lineage_graph.add_node("b", "test.b", "dataset")

        with pytest.raises(ValueError, match="does not exist"):
            lineage_graph.add_edge("nonexistent", "b")

    def test_add_edge_missing_target_raises(self, lineage_graph, provenance):
        """Test that adding an edge with non-existent target raises ValueError."""
        lineage_graph.add_node("a", "test.a", "dataset")

        with pytest.raises(ValueError, match="does not exist"):
            lineage_graph.add_edge("a", "nonexistent")

    def test_add_edge_cycle_detection_prevents_cycle(self, lineage_graph, provenance):
        """Test that adding a cyclic edge is prevented."""
        lineage_graph.add_node("x", "test.x", "dataset")
        lineage_graph.add_node("y", "test.y", "dataset")
        lineage_graph.add_node("z", "test.z", "dataset")

        lineage_graph.add_edge("x", "y")
        lineage_graph.add_edge("y", "z")

        with pytest.raises(ValueError, match="cycle"):
            lineage_graph.add_edge("z", "x")

    # ================================================================== #
    # Column-level lineage tests
    # ================================================================== #

    def test_column_level_edge(self, lineage_graph, provenance):
        """Test adding a column_level edge between two nodes."""
        lineage_graph.add_node("tbl1", "schema.table1", "dataset")
        lineage_graph.add_node("tbl2", "schema.table2", "dataset")

        edge = lineage_graph.add_edge(
            "tbl1",
            "tbl2",
            edge_type="column_level",
            source_field="col_a",
            target_field="col_b",
            confidence=0.95,
        )

        assert edge["edge_type"] == "column_level"
        assert edge["source_field"] == "col_a"
        assert edge["target_field"] == "col_b"
        assert edge["confidence"] == 0.95

    # ================================================================== #
    # Multi-component graph tests
    # ================================================================== #

    def test_disconnected_graph_has_multiple_components(self, lineage_graph):
        """Test that disconnected subgraphs produce multiple components."""
        # Component 1
        lineage_graph.add_node("c1a", "comp1.a", "dataset")
        lineage_graph.add_node("c1b", "comp1.b", "dataset")
        lineage_graph.add_edge("c1a", "c1b")

        # Component 2
        lineage_graph.add_node("c2a", "comp2.a", "dataset")
        lineage_graph.add_node("c2b", "comp2.b", "dataset")
        lineage_graph.add_edge("c2a", "c2b")

        # Isolated node (Component 3)
        lineage_graph.add_node("isolated", "orphan.node", "dataset")

        components = lineage_graph.get_connected_components()
        assert len(components) == 3

    # ================================================================== #
    # Graph dunder method tests
    # ================================================================== #

    def test_graph_len(self, populated_pipeline):
        """Test that len(graph) returns node count."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        assert len(graph) == 10

    def test_graph_contains(self, populated_pipeline):
        """Test that 'in' operator checks node existence."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        assert assets[0]["asset_id"] in graph
        assert "nonexistent" not in graph

    def test_graph_repr(self, populated_pipeline):
        """Test that repr(graph) includes node and edge counts."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        repr_str = repr(graph)
        assert "nodes=10" in repr_str
        assert "edges=9" in repr_str

    # ================================================================== #
    # Node removal tests
    # ================================================================== #

    def test_remove_node_removes_connected_edges(self, lineage_graph):
        """Test that removing a node also removes all its connected edges."""
        lineage_graph.add_node("a", "test.a", "dataset")
        lineage_graph.add_node("b", "test.b", "dataset")
        lineage_graph.add_node("c", "test.c", "dataset")
        lineage_graph.add_edge("a", "b")
        lineage_graph.add_edge("b", "c")

        assert lineage_graph.remove_node("b") is True

        stats = lineage_graph.get_statistics()
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 0

    def test_remove_nonexistent_node_returns_false(self, lineage_graph):
        """Test that removing a non-existent node returns False."""
        assert lineage_graph.remove_node("nonexistent") is False

    # ================================================================== #
    # Graph clear test
    # ================================================================== #

    def test_graph_clear_empties_all(self, populated_pipeline):
        """Test that clearing the graph removes all nodes and edges."""
        pipe, assets = populated_pipeline
        graph = pipe.lineage_graph

        graph.clear()

        stats = graph.get_statistics()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    # ================================================================== #
    # Valid edge types test
    # ================================================================== #

    def test_valid_edge_types_constant(self):
        """Test that VALID_EDGE_TYPES contains expected types."""
        assert "dataset_level" in VALID_EDGE_TYPES
        assert "column_level" in VALID_EDGE_TYPES
        assert len(VALID_EDGE_TYPES) == 2
