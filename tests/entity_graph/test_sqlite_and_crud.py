# -*- coding: utf-8 -*-
"""Tests for Phase 2.3 Entity Graph hardening."""
from __future__ import annotations

from pathlib import Path

import pytest

from greenlang.entity_graph import EntityGraph
from greenlang.entity_graph.models import EntityEdge, EntityNode
from greenlang.entity_graph.types import EdgeType, NodeType


def _node(node_id, node_type, name, **kw) -> EntityNode:
    return EntityNode(node_id=node_id, node_type=node_type, name=name, **kw)


def _edge(edge_id, source, target, edge_type) -> EntityEdge:
    return EntityEdge(
        edge_id=edge_id, source_id=source, target_id=target, edge_type=edge_type
    )


# --------------------------------------------------------------------------
# NodeType hierarchy
# --------------------------------------------------------------------------


class TestNodeTypeHierarchy:
    def test_hierarchy_constants_present(self):
        assert NodeType.ORGANIZATION == "organization"
        assert NodeType.FACILITY == "facility"
        assert NodeType.ASSET == "asset"
        assert NodeType.METER == "meter"

    def test_is_valid(self):
        assert NodeType.is_valid("facility") is True
        assert NodeType.is_valid("asset") is True
        assert NodeType.is_valid("meter") is True
        assert NodeType.is_valid("unicorn") is False

    def test_all_contains_new_types(self):
        for t in ("organization", "facility", "asset", "meter"):
            assert t in NodeType.ALL

    def test_allowed_parents_mapping(self):
        assert NodeType.ALLOWED_PARENTS[NodeType.FACILITY] == (NodeType.ORGANIZATION,)
        assert NodeType.ALLOWED_PARENTS[NodeType.ASSET] == (NodeType.FACILITY,)
        assert NodeType.METER in NodeType.ALLOWED_PARENTS
        assert NodeType.ORGANIZATION in NodeType.ALLOWED_PARENTS


# --------------------------------------------------------------------------
# add_node type validation
# --------------------------------------------------------------------------


class TestAddNodeValidation:
    def test_known_type_accepted(self):
        g = EntityGraph()
        g.add_node(_node("o1", NodeType.ORGANIZATION, "Acme"))
        assert g.get_node("o1") is not None

    def test_unknown_type_rejected(self):
        g = EntityGraph()
        with pytest.raises(ValueError):
            g.add_node(_node("x1", "dragon", "Mystery"))

    def test_validation_can_be_bypassed(self):
        g = EntityGraph()
        g.add_node(_node("x1", "custom_type", "Experimental"), validate_type=False)
        assert g.get_node("x1").node_type == "custom_type"


# --------------------------------------------------------------------------
# Update + delete
# --------------------------------------------------------------------------


class TestUpdateDelete:
    def test_update_node_name(self):
        g = EntityGraph()
        g.add_node(_node("f1", NodeType.FACILITY, "Old Name"))
        updated = g.update_node("f1", name="New Name", geography="DE")
        assert updated.name == "New Name"
        assert updated.geography == "DE"

    def test_update_unknown_raises(self):
        g = EntityGraph()
        with pytest.raises(KeyError):
            g.update_node("nope", name="x")

    def test_delete_soft_memory(self):
        g = EntityGraph()
        g.add_node(_node("f1", NodeType.FACILITY, "Site A"))
        g.add_node(_node("a1", NodeType.ASSET, "Boiler"))
        g.add_edge(_edge("e1", "f1", "a1", EdgeType.OWNS))
        assert g.delete_node("f1") is True
        # Node is gone; dependent edge is purged.
        assert g.get_node("f1") is None
        assert g.get_edges("a1", direction="incoming") == []

    def test_delete_returns_false_for_unknown(self):
        g = EntityGraph()
        assert g.delete_node("nope") is False

    def test_delete_edge(self):
        g = EntityGraph()
        g.add_node(_node("o1", NodeType.ORGANIZATION, "Acme"))
        g.add_node(_node("f1", NodeType.FACILITY, "Site A"))
        g.add_edge(_edge("e1", "o1", "f1", EdgeType.OWNS))
        assert g.delete_edge("e1") is True
        assert g.get_edges("o1") == []


# --------------------------------------------------------------------------
# SQLite persistence + hierarchy
# --------------------------------------------------------------------------


class TestSQLitePersistence:
    def test_sqlite_requires_path(self):
        with pytest.raises(ValueError):
            EntityGraph(storage_backend="sqlite")

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError):
            EntityGraph(storage_backend="neo4j")

    def test_full_hierarchy_round_trip(self, tmp_path: Path):
        """org → facility → asset → meter survives save + reload."""
        db_path = tmp_path / "graph.sqlite"

        g = EntityGraph(
            graph_id="test-graph", storage_backend="sqlite", sqlite_path=db_path
        )
        try:
            g.add_node(_node("o1", NodeType.ORGANIZATION, "Acme"))
            g.add_node(_node("f1", NodeType.FACILITY, "Berlin Site", geography="DE"))
            g.add_node(_node("a1", NodeType.ASSET, "Boiler #1"))
            g.add_node(_node("m1", NodeType.METER, "Gas Meter 01"))
            g.add_edge(_edge("e1", "o1", "f1", EdgeType.OWNS))
            g.add_edge(_edge("e2", "f1", "a1", EdgeType.PART_OF))
            g.add_edge(_edge("e3", "a1", "m1", EdgeType.PART_OF))
        finally:
            g.close()

        g2 = EntityGraph(
            graph_id="test-graph", storage_backend="sqlite", sqlite_path=db_path
        )
        try:
            assert g2.get_node("o1").name == "Acme"
            assert g2.get_node("f1").geography == "DE"
            assert g2.get_node("a1").node_type == NodeType.ASSET
            assert g2.get_node("m1").node_type == NodeType.METER
            # Edges hydrated with adjacency.
            assert len(g2.get_edges("o1")) == 1
            assert g2.get_neighbors("o1")[0].node_id == "f1"
        finally:
            g2.close()

    def test_update_persists(self, tmp_path: Path):
        db_path = tmp_path / "graph.sqlite"
        g = EntityGraph(graph_id="t", storage_backend="sqlite", sqlite_path=db_path)
        try:
            g.add_node(_node("f1", NodeType.FACILITY, "Old"))
            g.update_node("f1", name="New")
        finally:
            g.close()

        g2 = EntityGraph(graph_id="t", storage_backend="sqlite", sqlite_path=db_path)
        try:
            assert g2.get_node("f1").name == "New"
        finally:
            g2.close()

    def test_soft_delete_hides_from_reload(self, tmp_path: Path):
        db_path = tmp_path / "graph.sqlite"
        g = EntityGraph(graph_id="t", storage_backend="sqlite", sqlite_path=db_path)
        try:
            g.add_node(_node("f1", NodeType.FACILITY, "Site A"))
            g.add_node(_node("f2", NodeType.FACILITY, "Site B"))
            g.delete_node("f1")
        finally:
            g.close()

        g2 = EntityGraph(graph_id="t", storage_backend="sqlite", sqlite_path=db_path)
        try:
            assert g2.get_node("f1") is None  # filtered by deleted_at
            assert g2.get_node("f2") is not None
        finally:
            g2.close()

    def test_hard_delete_removes_row(self, tmp_path: Path):
        db_path = tmp_path / "graph.sqlite"
        g = EntityGraph(graph_id="t", storage_backend="sqlite", sqlite_path=db_path)
        try:
            g.add_node(_node("f1", NodeType.FACILITY, "Site"))
            assert g.delete_node("f1", soft=False) is True
            all_rows = g.sqlite_backend.list_nodes("t", include_deleted=True)
            assert all(r["node_id"] != "f1" for r in all_rows)
        finally:
            g.close()


# --------------------------------------------------------------------------
# Migration SQL sanity
# --------------------------------------------------------------------------


class TestMigrationFile:
    def test_v441_migration_present(self):
        mig = Path("deployment/database/migrations/sql/V441__entity_graph.sql")
        assert mig.exists()
        sql = mig.read_text(encoding="utf-8")
        assert "CREATE TABLE IF NOT EXISTS entity_nodes" in sql
        assert "CREATE TABLE IF NOT EXISTS entity_edges" in sql
        assert "'asset'" in sql
        assert "'meter'" in sql
        assert "deleted_at" in sql  # soft-delete column
        assert "CONSTRAINT chk_entity_node_type" in sql
