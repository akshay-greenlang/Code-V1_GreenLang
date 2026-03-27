# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Entity Registry Engine Tests

Tests register_entity, entity lifecycle transitions, hierarchy building,
circular reference detection, entity search/filtering, tree traversal,
and provenance hash computation.

Target: 60-80 tests.
"""

import pytest
from decimal import Decimal

from engines.entity_registry_engine import (
    EntityRegistryEngine,
    EntityRecord,
    EntityHierarchy,
    EntitySearchResult,
    EntityRegistryStats,
    EntityType,
    EntityStatus,
    _compute_hash,
)


@pytest.fixture
def engine():
    """Fresh EntityRegistryEngine instance."""
    return EntityRegistryEngine()


@pytest.fixture
def populated_engine(engine, all_entity_data):
    """Engine pre-populated with parent + 3 subs + JV + associate."""
    for ed in all_entity_data:
        engine.register_entity(ed)
    return engine


class TestRegisterEntity:
    """Test entity registration."""

    def test_register_single_entity(self, engine, parent_entity_data):
        entity = engine.register_entity(parent_entity_data)
        assert entity.legal_name == "GreenTest Holdings AG"
        assert entity.entity_type == "PARENT"
        assert entity.status == "ACTIVE"

    def test_register_entity_generates_id_if_missing(self, engine):
        entity = engine.register_entity({
            "legal_name": "Test Entity",
            "entity_type": "SUBSIDIARY",
        })
        assert entity.entity_id is not None
        assert len(entity.entity_id) > 0

    def test_register_entity_with_explicit_id(self, engine):
        entity = engine.register_entity({
            "entity_id": "CUSTOM-ID-001",
            "legal_name": "Custom ID Entity",
            "entity_type": "SUBSIDIARY",
        })
        assert entity.entity_id == "CUSTOM-ID-001"

    def test_register_subsidiary_with_parent(self, engine, parent_entity_data, sub1_entity_data):
        engine.register_entity(parent_entity_data)
        sub = engine.register_entity(sub1_entity_data)
        assert sub.parent_entity_id == parent_entity_data["entity_id"]

    def test_register_entity_invalid_parent_raises(self, engine):
        with pytest.raises(ValueError, match="not found"):
            engine.register_entity({
                "legal_name": "Orphan",
                "entity_type": "SUBSIDIARY",
                "parent_entity_id": "NONEXISTENT",
            })

    def test_register_all_entity_types(self, engine):
        for etype in EntityType:
            entity = engine.register_entity({
                "legal_name": f"Entity {etype.value}",
                "entity_type": etype.value,
            })
            assert entity.entity_type == etype.value

    def test_register_entity_unknown_type_becomes_other(self, engine):
        entity = engine.register_entity({
            "legal_name": "Unknown Type",
            "entity_type": "UNKNOWN_CUSTOM",
        })
        assert entity.entity_type == "OTHER"

    def test_register_entity_invalid_status_raises(self, engine):
        with pytest.raises(ValueError, match="Invalid entity status"):
            engine.register_entity({
                "legal_name": "Bad Status",
                "entity_type": "SUBSIDIARY",
                "status": "INVALID_STATUS",
            })

    def test_register_entity_count(self, populated_engine):
        assert populated_engine.get_entity_count() == 6

    def test_register_entity_creates_timestamps(self, engine, parent_entity_data):
        entity = engine.register_entity(parent_entity_data)
        assert entity.created_at is not None
        assert entity.updated_at is not None

    def test_register_entity_with_tags(self, engine):
        entity = engine.register_entity({
            "legal_name": "Tagged Entity",
            "entity_type": "SUBSIDIARY",
            "tags": ["test", "europe"],
        })
        assert "test" in entity.tags
        assert "europe" in entity.tags

    def test_register_entity_with_metadata(self, engine):
        entity = engine.register_entity({
            "legal_name": "Meta Entity",
            "entity_type": "SUBSIDIARY",
            "metadata": {"industry": "manufacturing"},
        })
        assert entity.metadata["industry"] == "manufacturing"


class TestEntityLifecycle:
    """Test entity lifecycle transitions."""

    def test_update_entity(self, populated_engine, sub1_entity_id):
        updated = populated_engine.update_entity(sub1_entity_id, {
            "trading_name": "GreenTest Mfg",
        })
        assert updated.trading_name == "GreenTest Mfg"

    def test_update_nonexistent_entity_raises(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.update_entity("NONEXISTENT", {"legal_name": "X"})

    def test_update_immutable_fields_ignored(self, populated_engine, parent_entity_id):
        original = populated_engine.get_entity(parent_entity_id)
        populated_engine.update_entity(parent_entity_id, {
            "entity_id": "NEW-ID",
            "created_at": "2000-01-01",
        })
        updated = populated_engine.get_entity(parent_entity_id)
        assert updated.entity_id == original.entity_id

    def test_deactivate_entity_to_dormant(self, populated_engine, sub1_entity_id):
        entity = populated_engine.deactivate_entity(
            sub1_entity_id, "DORMANT", "Temporary suspension"
        )
        assert entity.status == "DORMANT"

    def test_deactivate_entity_to_divested(self, populated_engine, sub3_entity_id):
        entity = populated_engine.deactivate_entity(
            sub3_entity_id, "DIVESTED", "Sold to third party"
        )
        assert entity.status == "DIVESTED"

    def test_deactivate_entity_to_merged(self, populated_engine, sub2_entity_id):
        entity = populated_engine.deactivate_entity(
            sub2_entity_id, "MERGED", "Merged with sub1"
        )
        assert entity.status == "MERGED"

    def test_deactivate_entity_to_liquidated(self, populated_engine, jv_entity_id):
        entity = populated_engine.deactivate_entity(
            jv_entity_id, "LIQUIDATED", "JV dissolved"
        )
        assert entity.status == "LIQUIDATED"

    def test_deactivate_to_active_raises(self, populated_engine, sub1_entity_id):
        with pytest.raises(ValueError, match="Cannot deactivate to ACTIVE"):
            populated_engine.deactivate_entity(
                sub1_entity_id, "ACTIVE", "Not allowed"
            )

    def test_deactivate_nonexistent_raises(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.deactivate_entity("NONEXISTENT", "DORMANT", "reason")

    def test_deactivate_invalid_status_raises(self, populated_engine, sub1_entity_id):
        with pytest.raises(ValueError, match="Invalid status"):
            populated_engine.deactivate_entity(
                sub1_entity_id, "INVALID", "reason"
            )


class TestHierarchy:
    """Test hierarchy building and tree operations."""

    def test_build_entity_tree(self, populated_engine, parent_entity_id):
        tree = populated_engine.get_entity_tree(parent_entity_id)
        assert isinstance(tree, EntityHierarchy)
        assert tree.total_entities == 6
        assert tree.root_entity_id == parent_entity_id

    def test_entity_tree_auto_detects_root(self, populated_engine):
        tree = populated_engine.get_entity_tree()
        assert tree.root_entity_id is not None
        assert tree.total_entities == 6

    def test_entity_tree_max_depth(self, populated_engine, parent_entity_id):
        tree = populated_engine.get_entity_tree(parent_entity_id)
        assert tree.max_depth >= 1

    def test_entity_tree_type_distribution(self, populated_engine, parent_entity_id):
        tree = populated_engine.get_entity_tree(parent_entity_id)
        assert "PARENT" in tree.entity_type_distribution
        assert "SUBSIDIARY" in tree.entity_type_distribution

    def test_entity_tree_provenance_hash(self, populated_engine, parent_entity_id):
        tree = populated_engine.get_entity_tree(parent_entity_id)
        assert tree.provenance_hash != ""
        assert len(tree.provenance_hash) == 64

    def test_entity_tree_nonexistent_root_raises(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.get_entity_tree("NONEXISTENT")

    def test_entity_tree_no_root_raises(self, engine):
        engine.register_entity({
            "entity_id": "A",
            "legal_name": "A",
            "entity_type": "SUBSIDIARY",
            "parent_entity_id": "B",
        })
        engine.register_entity({
            "entity_id": "B",
            "legal_name": "B",
            "entity_type": "SUBSIDIARY",
            "parent_entity_id": "A",
        })
        with pytest.raises(ValueError, match="No root entity"):
            engine.get_entity_tree()

    def test_get_ancestors(self, populated_engine, sub1_entity_id, parent_entity_id):
        ancestors = populated_engine.get_ancestors(sub1_entity_id)
        assert len(ancestors) >= 1
        assert any(a.entity_id == parent_entity_id for a in ancestors)

    def test_get_ancestors_of_root(self, populated_engine, parent_entity_id):
        ancestors = populated_engine.get_ancestors(parent_entity_id)
        assert len(ancestors) == 0

    def test_get_descendants(self, populated_engine, parent_entity_id):
        descendants = populated_engine.get_descendants(parent_entity_id)
        assert len(descendants) == 5

    def test_get_descendants_leaf_node(self, populated_engine, associate_entity_id):
        descendants = populated_engine.get_descendants(associate_entity_id)
        assert len(descendants) == 0

    def test_get_children(self, populated_engine, parent_entity_id):
        children = populated_engine.get_children(parent_entity_id)
        assert len(children) == 5

    def test_get_children_leaf_node(self, populated_engine, sub1_entity_id):
        children = populated_engine.get_children(sub1_entity_id)
        assert len(children) == 0

    def test_get_siblings(self, populated_engine, sub1_entity_id):
        siblings = populated_engine.get_siblings(sub1_entity_id)
        assert len(siblings) == 4

    def test_get_siblings_root(self, populated_engine, parent_entity_id):
        siblings = populated_engine.get_siblings(parent_entity_id)
        assert len(siblings) == 0

    def test_get_ancestors_nonexistent_raises(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.get_ancestors("NONEXISTENT")


class TestSearchAndFilter:
    """Test entity search and filtering."""

    def test_search_all_entities(self, populated_engine):
        result = populated_engine.search_entities()
        assert result.total_results == 6

    def test_search_by_entity_type(self, populated_engine):
        result = populated_engine.search_entities({"entity_type": "SUBSIDIARY"})
        assert result.total_results == 3

    def test_search_by_entity_type_list(self, populated_engine):
        result = populated_engine.search_entities({
            "entity_type": ["SUBSIDIARY", "JOINT_VENTURE"],
        })
        assert result.total_results == 4

    def test_search_by_country(self, populated_engine):
        result = populated_engine.search_entities({"country": "DE"})
        assert result.total_results == 1

    def test_search_by_legal_name_substring(self, populated_engine):
        result = populated_engine.search_entities({"legal_name": "Manufacturing"})
        assert result.total_results == 1

    def test_search_by_status(self, populated_engine):
        result = populated_engine.search_entities({"status": "ACTIVE"})
        assert result.total_results == 6

    def test_search_by_parent_id(self, populated_engine, parent_entity_id):
        result = populated_engine.search_entities({
            "parent_entity_id": parent_entity_id,
        })
        assert result.total_results == 5

    def test_search_by_tag(self, populated_engine):
        result = populated_engine.search_entities({"tag": "manufacturing"})
        assert result.total_results >= 1

    def test_search_by_has_lei_false(self, populated_engine):
        result = populated_engine.search_entities({"has_lei": False})
        assert result.total_results == 6

    def test_search_provenance_hash(self, populated_engine):
        result = populated_engine.search_entities()
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_search_no_results(self, populated_engine):
        result = populated_engine.search_entities({"country": "XX"})
        assert result.total_results == 0
        assert len(result.entities) == 0


class TestHierarchyValidation:
    """Test hierarchy validation checks."""

    def test_validate_valid_hierarchy(self, populated_engine):
        result = populated_engine.validate_hierarchy()
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_hierarchy_root_count(self, populated_engine):
        result = populated_engine.validate_hierarchy()
        assert result["root_count"] == 1

    def test_validate_hierarchy_provenance_hash(self, populated_engine):
        result = populated_engine.validate_hierarchy()
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


class TestRegistryStats:
    """Test registry statistics."""

    def test_registry_stats_totals(self, populated_engine):
        stats = populated_engine.get_registry_stats()
        assert stats.total_entities == 6
        assert stats.active_entities == 6

    def test_registry_stats_type_distribution(self, populated_engine):
        stats = populated_engine.get_registry_stats()
        assert "SUBSIDIARY" in stats.entity_type_distribution
        assert stats.entity_type_distribution["SUBSIDIARY"] == 3

    def test_registry_stats_countries(self, populated_engine):
        stats = populated_engine.get_registry_stats()
        assert stats.countries_covered >= 4

    def test_registry_stats_top_level(self, populated_engine):
        stats = populated_engine.get_registry_stats()
        assert stats.top_level_entities == 1

    def test_registry_stats_provenance_hash(self, populated_engine):
        stats = populated_engine.get_registry_stats()
        assert len(stats.provenance_hash) == 64


class TestAccessorsAndExport:
    """Test accessor methods and export."""

    def test_get_entity(self, populated_engine, parent_entity_id):
        entity = populated_engine.get_entity(parent_entity_id)
        assert entity.entity_id == parent_entity_id

    def test_get_entity_nonexistent_raises(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.get_entity("NONEXISTENT")

    def test_get_all_entities(self, populated_engine):
        entities = populated_engine.get_all_entities()
        assert len(entities) == 6

    def test_get_active_entities(self, populated_engine):
        entities = populated_engine.get_active_entities()
        assert len(entities) == 6

    def test_get_change_log(self, populated_engine):
        log = populated_engine.get_change_log()
        assert len(log) >= 6
        assert all("event" in entry for entry in log)

    def test_export_registry(self, populated_engine):
        export = populated_engine.export_registry()
        assert export["total_entities"] == 6
        assert "entities" in export
        assert "provenance_hash" in export

    def test_import_entities(self, engine):
        entities = [
            {"entity_id": "IMP-1", "legal_name": "Import 1", "entity_type": "SUBSIDIARY"},
            {"entity_id": "IMP-2", "legal_name": "Import 2", "entity_type": "SUBSIDIARY"},
        ]
        count = engine.import_entities(entities)
        assert count == 2
        assert engine.get_entity_count() == 2

    def test_import_skips_duplicates(self, engine):
        entity = {"entity_id": "DUP-1", "legal_name": "Dup", "entity_type": "SUBSIDIARY"}
        engine.register_entity(entity)
        count = engine.import_entities([entity])
        assert count == 0
        assert engine.get_entity_count() == 1
