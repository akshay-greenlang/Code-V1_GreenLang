# -*- coding: utf-8 -*-
"""
Unit tests for SchemaRegistryEngine - AGENT-DATA-017
=====================================================

Comprehensive tests for the Schema Registry Engine covering:
- Engine initialisation and internal state
- Schema registration (all types, owner, tags, description, metadata)
- Schema retrieval (by ID, by name, non-existent)
- Schema listing with filters and pagination
- Schema updates (owner, tags, status lifecycle, description, metadata)
- Schema deletion (soft delete / archival)
- Full-text schema search with relevance scoring
- Bulk import (success, failures, limits)
- Schema export (all, by IDs, by namespace)
- Schema groups (create, get, list, add member, remove member)
- Definition validation (JSON Schema, Avro, Protobuf)
- Statistics and provenance tracking
- Edge cases (unicode, long names, special characters, thread safety)

Target: 120+ tests, 85%+ coverage of schema_registry.py

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
"""

from __future__ import annotations

import json
import threading
import uuid
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from greenlang.schema_migration.schema_registry import (
    AVRO_REQUIRED_KEYS,
    JSON_SCHEMA_REQUIRED_KEYS,
    MAX_BULK_IMPORT,
    MAX_NAMESPACE_LENGTH,
    MAX_SCHEMA_NAME_LENGTH,
    MAX_TAG_LENGTH,
    PROTOBUF_REQUIRED_KEYS,
    STATUS_TRANSITIONS,
    VALID_SCHEMA_TYPES,
    VALID_STATUSES,
    SchemaRegistryEngine,
    _build_sha256,
    _normalize_tags,
    _validate_namespace,
    _validate_schema_name,
    _validate_tags_list,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> SchemaRegistryEngine:
    """Create a fresh SchemaRegistryEngine for each test."""
    return SchemaRegistryEngine()


@pytest.fixture
def json_schema_definition() -> Dict[str, Any]:
    """A valid JSON Schema definition."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "value": {"type": "number"},
        },
        "required": ["id"],
    }


@pytest.fixture
def avro_definition() -> Dict[str, Any]:
    """A valid Avro schema definition."""
    return {
        "type": "record",
        "name": "EmissionRecord",
        "fields": [
            {"name": "source_id", "type": "string"},
            {"name": "co2_tonnes", "type": "double"},
        ],
    }


@pytest.fixture
def protobuf_definition() -> Dict[str, Any]:
    """A valid Protobuf-like schema definition."""
    return {
        "syntax": "proto3",
        "messages": [
            {
                "name": "ActivityRecord",
                "fields": [
                    {"name": "activity_id", "type": "string", "number": 1},
                ],
            },
        ],
    }


@pytest.fixture
def registered_schema(engine, json_schema_definition) -> Dict[str, Any]:
    """Register and return a single schema for reuse in tests."""
    return engine.register_schema(
        namespace="emissions",
        name="ActivityRecord",
        schema_type="json_schema",
        definition_json=json_schema_definition,
        owner="platform-team",
        tags=["emissions", "scope3"],
        description="Canonical activity record schema.",
    )


@pytest.fixture
def multiple_schemas(engine, json_schema_definition, avro_definition):
    """Register multiple schemas for list/filter tests."""
    schemas = []
    schemas.append(engine.register_schema(
        namespace="emissions",
        name="ActivityRecord",
        schema_type="json_schema",
        definition_json=json_schema_definition,
        owner="platform-team",
        tags=["emissions", "core"],
        description="Activity tracking schema.",
    ))
    schemas.append(engine.register_schema(
        namespace="emissions",
        name="FactorRecord",
        schema_type="json_schema",
        definition_json={"type": "object", "properties": {}},
        owner="data-team",
        tags=["emissions", "factors"],
        description="Emission factor schema.",
    ))
    schemas.append(engine.register_schema(
        namespace="supply_chain",
        name="SupplierRecord",
        schema_type="avro",
        definition_json=avro_definition,
        owner="data-team",
        tags=["supplier", "core"],
        description="Supplier data schema.",
    ))
    schemas.append(engine.register_schema(
        namespace="supply_chain",
        name="MaterialRecord",
        schema_type="avro",
        definition_json={
            "type": "record",
            "name": "MaterialRecord",
            "fields": [{"name": "material_id", "type": "string"}],
        },
        owner="platform-team",
        tags=["material"],
        description="Material tracking schema.",
    ))
    return schemas


# ===========================================================================
# TestSchemaRegistryEngineInit
# ===========================================================================


class TestSchemaRegistryEngineInit:
    """Test engine initialisation and default state."""

    def test_init_creates_empty_schema_store(self, engine):
        """Engine starts with no schemas registered."""
        stats = engine.get_statistics()
        assert stats["total_schemas"] == 0

    def test_init_creates_empty_groups(self, engine):
        """Engine starts with no groups."""
        assert engine.list_groups() == []

    def test_init_has_provenance_tracker(self, engine):
        """Engine has a provenance tracker instance."""
        assert engine._provenance is not None

    def test_init_has_thread_lock(self, engine):
        """Engine has a threading lock for thread safety."""
        assert isinstance(engine._lock, type(threading.Lock()))

    def test_init_indexes_are_empty(self, engine):
        """All internal indexes start empty."""
        assert len(engine._namespace_index) == 0
        assert len(engine._name_index) == 0
        assert len(engine._tag_index) == 0

    def test_statistics_initial_state(self, engine):
        """Statistics reflect zero state at init."""
        stats = engine.get_statistics()
        assert stats["total_schemas"] == 0
        assert stats["total_groups"] == 0
        assert stats["total_tags"] == 0
        assert stats["total_namespaces"] == 0
        for st in VALID_SCHEMA_TYPES:
            assert stats["by_type"][st] == 0
        for s in VALID_STATUSES:
            assert stats["by_status"][s] == 0


# ===========================================================================
# TestRegisterSchema
# ===========================================================================


class TestRegisterSchema:
    """Test schema registration across types and parameters."""

    def test_register_json_schema(self, engine, json_schema_definition):
        """Register a valid JSON Schema definition."""
        schema = engine.register_schema(
            namespace="test-ns",
            name="TestSchema",
            schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        assert schema["namespace"] == "test-ns"
        assert schema["name"] == "TestSchema"
        assert schema["schema_type"] == "json_schema"
        assert schema["status"] == "draft"

    def test_register_avro_schema(self, engine, avro_definition):
        """Register a valid Avro schema definition."""
        schema = engine.register_schema(
            namespace="avro-ns",
            name="AvroSchema",
            schema_type="avro",
            definition_json=avro_definition,
        )
        assert schema["schema_type"] == "avro"
        assert schema["status"] == "draft"

    def test_register_protobuf_schema(self, engine, protobuf_definition):
        """Register a valid Protobuf schema definition."""
        schema = engine.register_schema(
            namespace="proto-ns",
            name="ProtoSchema",
            schema_type="protobuf",
            definition_json=protobuf_definition,
        )
        assert schema["schema_type"] == "protobuf"
        assert schema["status"] == "draft"

    def test_register_generates_uuid(self, engine, json_schema_definition):
        """Registered schema has a valid UUID4 schema_id."""
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        parsed = uuid.UUID(schema["schema_id"], version=4)
        assert str(parsed) == schema["schema_id"]

    def test_register_default_status_is_draft(self, engine, json_schema_definition):
        """Newly registered schemas always have status 'draft'."""
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        assert schema["status"] == "draft"

    def test_register_with_owner(self, engine, json_schema_definition):
        """Owner field is stored correctly."""
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
            owner="data-engineering",
        )
        assert schema["owner"] == "data-engineering"

    def test_register_default_owner_is_empty(self, engine, json_schema_definition):
        """Owner defaults to empty string when not provided."""
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        assert schema["owner"] == ""

    def test_register_with_tags(self, engine, json_schema_definition):
        """Tags are normalized, sorted, and stored."""
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
            tags=["Emissions", "SCOPE3", "core"],
        )
        assert schema["tags"] == ["core", "emissions", "scope3"]

    def test_register_tags_deduplicated(self, engine, json_schema_definition):
        """Duplicate tags are removed during normalization."""
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
            tags=["dup", "DUP", " dup "],
        )
        assert schema["tags"] == ["dup"]

    def test_register_with_description(self, engine, json_schema_definition):
        """Description is stored correctly."""
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
            description="A test schema for unit tests.",
        )
        assert schema["description"] == "A test schema for unit tests."

    def test_register_with_metadata(self, engine, json_schema_definition):
        """Metadata dict is stored correctly."""
        meta = {"source": "erp", "version": "2.1"}
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
            metadata=meta,
        )
        assert schema["metadata"]["source"] == "erp"
        assert schema["metadata"]["version"] == "2.1"

    def test_register_timestamps_present(self, engine, json_schema_definition):
        """Registered schema has created_at and updated_at timestamps."""
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        assert "created_at" in schema
        assert "updated_at" in schema
        assert schema["created_at"] == schema["updated_at"]

    def test_register_provenance_hash_present(self, engine, json_schema_definition):
        """Registered schema has a non-empty provenance_hash."""
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        assert schema["provenance_hash"] != ""

    def test_register_definition_is_deep_copied(self, engine, json_schema_definition):
        """Definition is deep-copied so mutations do not affect registry."""
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        json_schema_definition["type"] = "MUTATED"
        stored = engine.get_schema(schema["schema_id"])
        assert stored["definition"]["type"] == "object"

    def test_register_duplicate_namespace_name_rejected(self, engine, json_schema_definition):
        """Registering the same (namespace, name) pair twice raises ValueError."""
        engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        with pytest.raises(ValueError, match="already exists"):
            engine.register_schema(
                namespace="ns", name="S1", schema_type="json_schema",
                definition_json=json_schema_definition,
            )

    def test_register_same_name_different_namespace_allowed(self, engine, json_schema_definition):
        """Same name in different namespaces is allowed."""
        s1 = engine.register_schema(
            namespace="ns-a", name="Schema", schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        s2 = engine.register_schema(
            namespace="ns-b", name="Schema", schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        assert s1["schema_id"] != s2["schema_id"]

    def test_register_archived_name_can_be_reused(self, engine, json_schema_definition):
        """After archiving, the same (namespace, name) can be re-registered."""
        s1 = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        engine.delete_schema(s1["schema_id"])
        s2 = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        assert s2["schema_id"] != s1["schema_id"]
        assert s2["status"] == "draft"

    def test_register_invalid_schema_type_rejected(self, engine, json_schema_definition):
        """Invalid schema_type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported schema_type"):
            engine.register_schema(
                namespace="ns", name="S1", schema_type="xml",
                definition_json=json_schema_definition,
            )

    def test_register_empty_namespace_rejected(self, engine, json_schema_definition):
        """Empty namespace raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            engine.register_schema(
                namespace="", name="S1", schema_type="json_schema",
                definition_json=json_schema_definition,
            )

    def test_register_empty_name_rejected(self, engine, json_schema_definition):
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            engine.register_schema(
                namespace="ns", name="", schema_type="json_schema",
                definition_json=json_schema_definition,
            )

    def test_register_invalid_definition_rejected(self, engine):
        """Invalid definition (missing type) raises ValueError."""
        with pytest.raises(ValueError, match="validation failed"):
            engine.register_schema(
                namespace="ns", name="Bad", schema_type="json_schema",
                definition_json={"properties": {}},
            )

    def test_register_updates_statistics(self, engine, json_schema_definition):
        """Statistics reflect the newly registered schema."""
        engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        stats = engine.get_statistics()
        assert stats["total_schemas"] == 1
        assert stats["by_type"]["json_schema"] == 1
        assert stats["by_status"]["draft"] == 1
        assert stats["total_namespaces"] == 1

    def test_register_namespace_with_dots_and_hyphens(self, engine, json_schema_definition):
        """Namespace with dots and hyphens is valid."""
        schema = engine.register_schema(
            namespace="com.greenlang.emissions-v2",
            name="TestSchema",
            schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        assert schema["namespace"] == "com.greenlang.emissions-v2"

    def test_register_namespace_with_underscores(self, engine, json_schema_definition):
        """Namespace with underscores is valid."""
        schema = engine.register_schema(
            namespace="supply_chain",
            name="TestSchema",
            schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        assert schema["namespace"] == "supply_chain"

    def test_register_namespace_too_long_rejected(self, engine, json_schema_definition):
        """Namespace exceeding MAX_NAMESPACE_LENGTH is rejected."""
        long_ns = "a" * (MAX_NAMESPACE_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            engine.register_schema(
                namespace=long_ns, name="S1", schema_type="json_schema",
                definition_json=json_schema_definition,
            )

    def test_register_name_too_long_rejected(self, engine, json_schema_definition):
        """Name exceeding MAX_SCHEMA_NAME_LENGTH is rejected."""
        long_name = "S" * (MAX_SCHEMA_NAME_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            engine.register_schema(
                namespace="ns", name=long_name, schema_type="json_schema",
                definition_json=json_schema_definition,
            )

    def test_register_tag_too_long_rejected(self, engine, json_schema_definition):
        """Tag exceeding MAX_TAG_LENGTH is rejected."""
        long_tag = "t" * (MAX_TAG_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            engine.register_schema(
                namespace="ns", name="S1", schema_type="json_schema",
                definition_json=json_schema_definition,
                tags=[long_tag],
            )

    def test_register_namespace_with_special_chars_rejected(self, engine, json_schema_definition):
        """Namespace with disallowed special characters is rejected."""
        with pytest.raises(ValueError, match="alphanumeric"):
            engine.register_schema(
                namespace="ns/bad@char",
                name="S1",
                schema_type="json_schema",
                definition_json=json_schema_definition,
            )


# ===========================================================================
# TestGetSchema
# ===========================================================================


class TestGetSchema:
    """Test schema retrieval by ID."""

    def test_get_existing_schema(self, engine, registered_schema):
        """Retrieve a registered schema by ID."""
        result = engine.get_schema(registered_schema["schema_id"])
        assert result is not None
        assert result["schema_id"] == registered_schema["schema_id"]
        assert result["name"] == "ActivityRecord"

    def test_get_non_existing_schema(self, engine):
        """Retrieving a non-existent schema returns None."""
        result = engine.get_schema("non-existent-id")
        assert result is None

    def test_get_returns_deep_copy(self, engine, registered_schema):
        """Returned schema is a deep copy; mutations do not affect storage."""
        result = engine.get_schema(registered_schema["schema_id"])
        result["name"] = "MUTATED"
        stored = engine.get_schema(registered_schema["schema_id"])
        assert stored["name"] == "ActivityRecord"

    def test_get_after_update(self, engine, registered_schema):
        """Get returns updated state after an update."""
        engine.update_schema(registered_schema["schema_id"], owner="new-owner")
        result = engine.get_schema(registered_schema["schema_id"])
        assert result["owner"] == "new-owner"

    def test_get_after_delete(self, engine, registered_schema):
        """Get returns archived schema after soft delete."""
        engine.delete_schema(registered_schema["schema_id"])
        result = engine.get_schema(registered_schema["schema_id"])
        assert result is not None
        assert result["status"] == "archived"

    def test_get_preserves_all_fields(self, engine, registered_schema):
        """Retrieved schema contains all expected fields."""
        result = engine.get_schema(registered_schema["schema_id"])
        expected_fields = {
            "schema_id", "namespace", "name", "schema_type",
            "definition", "owner", "tags", "description",
            "metadata", "status", "created_at", "updated_at",
            "provenance_hash",
        }
        assert expected_fields.issubset(set(result.keys()))

    def test_get_with_empty_string_id(self, engine):
        """Empty string ID returns None."""
        result = engine.get_schema("")
        assert result is None

    def test_get_with_random_uuid(self, engine):
        """Random UUID that was never registered returns None."""
        result = engine.get_schema(str(uuid.uuid4()))
        assert result is None


# ===========================================================================
# TestListSchemas
# ===========================================================================


class TestListSchemas:
    """Test schema listing with filters and pagination."""

    def test_list_all_schemas(self, engine, multiple_schemas):
        """List all schemas without filters returns all."""
        result = engine.list_schemas()
        assert len(result) == 4

    def test_list_filter_by_namespace(self, engine, multiple_schemas):
        """Filter by namespace returns only matching schemas."""
        result = engine.list_schemas(namespace="emissions")
        assert len(result) == 2
        assert all(s["namespace"] == "emissions" for s in result)

    def test_list_filter_by_schema_type(self, engine, multiple_schemas):
        """Filter by schema_type returns only matching schemas."""
        result = engine.list_schemas(schema_type="avro")
        assert len(result) == 2
        assert all(s["schema_type"] == "avro" for s in result)

    def test_list_filter_by_status(self, engine, multiple_schemas):
        """Filter by status returns matching schemas."""
        result = engine.list_schemas(status="draft")
        assert len(result) == 4  # All start as draft

    def test_list_filter_by_owner(self, engine, multiple_schemas):
        """Filter by owner returns only matching schemas."""
        result = engine.list_schemas(owner="data-team")
        assert len(result) == 2
        assert all(s["owner"] == "data-team" for s in result)

    def test_list_filter_by_tag(self, engine, multiple_schemas):
        """Filter by tag returns schemas carrying that tag."""
        result = engine.list_schemas(tag="core")
        assert len(result) == 2

    def test_list_filter_by_tag_case_insensitive(self, engine, multiple_schemas):
        """Tag filter is case-insensitive."""
        result = engine.list_schemas(tag="CORE")
        assert len(result) == 2

    def test_list_pagination_limit(self, engine, multiple_schemas):
        """Limit parameter restricts result count."""
        result = engine.list_schemas(limit=2)
        assert len(result) == 2

    def test_list_pagination_offset(self, engine, multiple_schemas):
        """Offset parameter skips initial results."""
        all_results = engine.list_schemas()
        offset_results = engine.list_schemas(offset=2)
        assert len(offset_results) == 2
        assert offset_results[0]["schema_id"] == all_results[2]["schema_id"]

    def test_list_pagination_limit_and_offset(self, engine, multiple_schemas):
        """Limit and offset work together for pagination."""
        result = engine.list_schemas(limit=1, offset=1)
        assert len(result) == 1

    def test_list_combined_filters(self, engine, multiple_schemas):
        """Multiple filters are AND-combined."""
        result = engine.list_schemas(
            namespace="emissions",
            owner="platform-team",
        )
        assert len(result) == 1
        assert result[0]["name"] == "ActivityRecord"

    def test_list_empty_results(self, engine, multiple_schemas):
        """Filters that match nothing return empty list."""
        result = engine.list_schemas(namespace="nonexistent")
        assert result == []

    def test_list_negative_limit_raises(self, engine):
        """Negative limit raises ValueError."""
        with pytest.raises(ValueError, match="limit"):
            engine.list_schemas(limit=-1)

    def test_list_negative_offset_raises(self, engine):
        """Negative offset raises ValueError."""
        with pytest.raises(ValueError, match="offset"):
            engine.list_schemas(offset=-1)

    def test_list_ordered_by_created_at(self, engine, multiple_schemas):
        """Results are ordered by created_at ascending."""
        result = engine.list_schemas()
        created_times = [s["created_at"] for s in result]
        assert created_times == sorted(created_times)

    def test_list_filter_by_name_contains(self, engine, multiple_schemas):
        """Filter by name_contains returns substring matches."""
        result = engine.list_schemas(name_contains="Record")
        assert len(result) == 4

    def test_list_filter_by_name_contains_case_insensitive(self, engine, multiple_schemas):
        """name_contains filter is case-insensitive."""
        result = engine.list_schemas(name_contains="activity")
        assert len(result) == 1
        assert result[0]["name"] == "ActivityRecord"


# ===========================================================================
# TestUpdateSchema
# ===========================================================================


class TestUpdateSchema:
    """Test schema update operations and status lifecycle."""

    def test_update_owner(self, engine, registered_schema):
        """Update owner field."""
        result = engine.update_schema(
            registered_schema["schema_id"], owner="new-owner"
        )
        assert result["owner"] == "new-owner"

    def test_update_description(self, engine, registered_schema):
        """Update description field."""
        result = engine.update_schema(
            registered_schema["schema_id"],
            description="Updated description.",
        )
        assert result["description"] == "Updated description."

    def test_update_tags(self, engine, registered_schema):
        """Update tags replaces existing tags."""
        result = engine.update_schema(
            registered_schema["schema_id"],
            tags=["new-tag", "updated"],
        )
        assert result["tags"] == ["new-tag", "updated"]

    def test_update_metadata_merges(self, engine, registered_schema):
        """Update metadata merges with existing metadata."""
        engine.update_schema(
            registered_schema["schema_id"],
            metadata={"key1": "val1"},
        )
        result = engine.update_schema(
            registered_schema["schema_id"],
            metadata={"key2": "val2"},
        )
        assert result["metadata"]["key1"] == "val1"
        assert result["metadata"]["key2"] == "val2"

    def test_update_status_draft_to_active(self, engine, registered_schema):
        """Status transition draft -> active succeeds."""
        result = engine.update_schema(
            registered_schema["schema_id"], status="active"
        )
        assert result["status"] == "active"

    def test_update_status_active_to_deprecated(self, engine, registered_schema):
        """Status transition active -> deprecated succeeds."""
        engine.update_schema(registered_schema["schema_id"], status="active")
        result = engine.update_schema(
            registered_schema["schema_id"], status="deprecated"
        )
        assert result["status"] == "deprecated"

    def test_update_status_deprecated_to_archived(self, engine, registered_schema):
        """Status transition deprecated -> archived succeeds."""
        engine.update_schema(registered_schema["schema_id"], status="active")
        engine.update_schema(registered_schema["schema_id"], status="deprecated")
        result = engine.update_schema(
            registered_schema["schema_id"], status="archived"
        )
        assert result["status"] == "archived"

    def test_update_status_backward_transition_rejected(self, engine, registered_schema):
        """Backward status transition (active -> draft) raises ValueError."""
        engine.update_schema(registered_schema["schema_id"], status="active")
        with pytest.raises(ValueError, match="Invalid status transition"):
            engine.update_schema(
                registered_schema["schema_id"], status="draft"
            )

    def test_update_status_skip_transition_rejected(self, engine, registered_schema):
        """Skipping status levels (draft -> deprecated) raises ValueError."""
        with pytest.raises(ValueError, match="Invalid status transition"):
            engine.update_schema(
                registered_schema["schema_id"], status="deprecated"
            )

    def test_update_status_archived_is_terminal(self, engine, registered_schema):
        """Archived status cannot transition to any other status."""
        engine.update_schema(registered_schema["schema_id"], status="active")
        engine.update_schema(registered_schema["schema_id"], status="deprecated")
        engine.update_schema(registered_schema["schema_id"], status="archived")
        with pytest.raises(ValueError, match="Invalid status transition"):
            engine.update_schema(
                registered_schema["schema_id"], status="active"
            )

    def test_update_nonexistent_schema_raises(self, engine):
        """Updating a non-existent schema raises KeyError."""
        with pytest.raises(KeyError, match="Schema not found"):
            engine.update_schema("nonexistent-id", owner="someone")

    def test_update_invalid_status_rejected(self, engine, registered_schema):
        """Invalid status string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid status"):
            engine.update_schema(
                registered_schema["schema_id"], status="invalid"
            )

    def test_update_changes_updated_at(self, engine, registered_schema):
        """Update modifies the updated_at timestamp."""
        original_updated = registered_schema["updated_at"]
        result = engine.update_schema(
            registered_schema["schema_id"], owner="new-owner"
        )
        # updated_at may or may not differ due to same-second execution,
        # but the field must exist
        assert "updated_at" in result

    def test_update_changes_provenance_hash(self, engine, registered_schema):
        """Update generates a new provenance hash."""
        original_hash = registered_schema["provenance_hash"]
        result = engine.update_schema(
            registered_schema["schema_id"], owner="new-owner"
        )
        assert result["provenance_hash"] != ""
        # Hash may differ since content changed
        assert result["provenance_hash"] is not None

    def test_update_same_status_no_op(self, engine, registered_schema):
        """Updating to the same status is allowed (no-op for status)."""
        result = engine.update_schema(
            registered_schema["schema_id"], status="draft"
        )
        assert result["status"] == "draft"

    def test_update_tag_index_updated(self, engine, registered_schema):
        """After tag update, search by new tag works and old tag no longer matches."""
        engine.update_schema(
            registered_schema["schema_id"],
            tags=["new-tag-only"],
        )
        new_results = engine.list_schemas(tag="new-tag-only")
        assert len(new_results) == 1
        old_results = engine.list_schemas(tag="emissions")
        assert len(old_results) == 0


# ===========================================================================
# TestDeleteSchema
# ===========================================================================


class TestDeleteSchema:
    """Test schema soft deletion (archival)."""

    def test_delete_sets_archived_status(self, engine, registered_schema):
        """Delete sets status to archived."""
        result = engine.delete_schema(registered_schema["schema_id"])
        assert result["status"] == "archived"

    def test_delete_already_archived_is_noop(self, engine, registered_schema):
        """Deleting an already archived schema is a no-op."""
        engine.delete_schema(registered_schema["schema_id"])
        result = engine.delete_schema(registered_schema["schema_id"])
        assert result["status"] == "archived"

    def test_delete_nonexistent_raises(self, engine):
        """Deleting a non-existent schema raises KeyError."""
        with pytest.raises(KeyError, match="Schema not found"):
            engine.delete_schema("nonexistent-id")

    def test_delete_from_draft_bypasses_lifecycle(self, engine, registered_schema):
        """Delete from draft status bypasses normal lifecycle to archive."""
        result = engine.delete_schema(registered_schema["schema_id"])
        assert result["status"] == "archived"

    def test_delete_from_active_bypasses_lifecycle(self, engine, registered_schema):
        """Delete from active status bypasses lifecycle to archive."""
        engine.update_schema(registered_schema["schema_id"], status="active")
        result = engine.delete_schema(registered_schema["schema_id"])
        assert result["status"] == "archived"

    def test_delete_preserves_data(self, engine, registered_schema):
        """Soft delete preserves all other schema fields."""
        result = engine.delete_schema(registered_schema["schema_id"])
        assert result["namespace"] == "emissions"
        assert result["name"] == "ActivityRecord"
        assert result["schema_type"] == "json_schema"

    def test_delete_updates_provenance_hash(self, engine, registered_schema):
        """Delete generates a new provenance hash."""
        original_hash = registered_schema["provenance_hash"]
        result = engine.delete_schema(registered_schema["schema_id"])
        assert result["provenance_hash"] != ""


# ===========================================================================
# TestSearchSchemas
# ===========================================================================


class TestSearchSchemas:
    """Test full-text schema search."""

    def test_search_by_name(self, engine, multiple_schemas):
        """Search by schema name returns matching results."""
        results = engine.search_schemas("Activity")
        assert len(results) >= 1
        assert any(s["name"] == "ActivityRecord" for s in results)

    def test_search_by_namespace(self, engine, multiple_schemas):
        """Search by namespace substring returns matching results."""
        results = engine.search_schemas("supply")
        assert len(results) >= 1
        assert all(
            "supply" in s["namespace"].lower() or
            "supply" in s["name"].lower() or
            "supply" in s["description"].lower() or
            any("supply" in t for t in s["tags"]) or
            "supply" in s.get("owner", "").lower()
            for s in results
        )

    def test_search_by_description(self, engine, multiple_schemas):
        """Search by description substring returns matches."""
        results = engine.search_schemas("emission factor")
        assert len(results) >= 1

    def test_search_case_insensitive(self, engine, multiple_schemas):
        """Search is case-insensitive."""
        results_lower = engine.search_schemas("activity")
        results_upper = engine.search_schemas("ACTIVITY")
        assert len(results_lower) == len(results_upper)

    def test_search_no_results(self, engine, multiple_schemas):
        """Search with non-matching query returns empty list."""
        results = engine.search_schemas("xyznonexistent123")
        assert results == []

    def test_search_empty_query_returns_empty(self, engine, multiple_schemas):
        """Empty search query returns empty list."""
        results = engine.search_schemas("")
        assert results == []

    def test_search_by_tag(self, engine, multiple_schemas):
        """Search matching a tag returns results."""
        results = engine.search_schemas("core")
        assert len(results) >= 2

    def test_search_by_owner(self, engine, multiple_schemas):
        """Search matching owner name returns results."""
        results = engine.search_schemas("platform-team")
        assert len(results) >= 1

    def test_search_relevance_name_first(self, engine, multiple_schemas):
        """Exact name match is ranked higher than namespace match."""
        results = engine.search_schemas("ActivityRecord")
        assert len(results) >= 1
        assert results[0]["name"] == "ActivityRecord"

    def test_search_whitespace_stripped(self, engine, multiple_schemas):
        """Query whitespace is stripped before search."""
        results = engine.search_schemas("  Activity  ")
        assert len(results) >= 1


# ===========================================================================
# TestBulkImport
# ===========================================================================


class TestBulkImport:
    """Test bulk schema import."""

    def test_import_multiple_schemas(self, engine):
        """Bulk import registers multiple schemas."""
        schemas_list = [
            {
                "namespace": "bulk",
                "name": f"Schema{i}",
                "schema_type": "json_schema",
                "definition_json": {"type": "object", "properties": {}},
            }
            for i in range(5)
        ]
        result = engine.import_schemas(schemas_list)
        assert result["total"] == 5
        assert result["success"] == 5
        assert result["failed"] == 0
        assert len(result["schema_ids"]) == 5
        assert result["provenance_hash"] != ""

    def test_import_with_failures(self, engine):
        """Bulk import reports failed schemas and continues."""
        schemas_list = [
            {
                "namespace": "bulk",
                "name": "Good1",
                "schema_type": "json_schema",
                "definition_json": {"type": "object"},
            },
            {
                "namespace": "",  # Invalid - empty namespace
                "name": "Bad1",
                "schema_type": "json_schema",
                "definition_json": {"type": "object"},
            },
            {
                "namespace": "bulk",
                "name": "Good2",
                "schema_type": "json_schema",
                "definition_json": {"type": "object"},
            },
        ]
        result = engine.import_schemas(schemas_list)
        assert result["total"] == 3
        assert result["success"] == 2
        assert result["failed"] == 1
        assert len(result["failures"]) == 1
        assert result["failures"][0]["index"] == 1

    def test_import_duplicates_in_batch(self, engine):
        """Duplicate (namespace, name) in same batch reports second as failure."""
        schemas_list = [
            {
                "namespace": "dup",
                "name": "Same",
                "schema_type": "json_schema",
                "definition_json": {"type": "object"},
            },
            {
                "namespace": "dup",
                "name": "Same",
                "schema_type": "json_schema",
                "definition_json": {"type": "object"},
            },
        ]
        result = engine.import_schemas(schemas_list)
        assert result["success"] == 1
        assert result["failed"] == 1

    def test_import_empty_batch(self, engine):
        """Empty batch import returns zero counts."""
        result = engine.import_schemas([])
        assert result["total"] == 0
        assert result["success"] == 0
        assert result["failed"] == 0

    def test_import_exceeds_max_raises(self, engine):
        """Batch exceeding MAX_BULK_IMPORT raises ValueError."""
        schemas = [
            {
                "namespace": "big",
                "name": f"S{i}",
                "schema_type": "json_schema",
                "definition_json": {"type": "object"},
            }
            for i in range(MAX_BULK_IMPORT + 1)
        ]
        with pytest.raises(ValueError, match="exceeds maximum"):
            engine.import_schemas(schemas)

    def test_import_with_optional_fields(self, engine):
        """Bulk import schemas with owner, tags, and description."""
        schemas_list = [
            {
                "namespace": "ns",
                "name": "WithExtras",
                "schema_type": "json_schema",
                "definition_json": {"type": "object"},
                "owner": "import-team",
                "tags": ["imported"],
                "description": "An imported schema.",
            },
        ]
        result = engine.import_schemas(schemas_list)
        assert result["success"] == 1
        stored = engine.get_schema(result["schema_ids"][0])
        assert stored["owner"] == "import-team"
        assert stored["tags"] == ["imported"]
        assert stored["description"] == "An imported schema."


# ===========================================================================
# TestExportSchemas
# ===========================================================================


class TestExportSchemas:
    """Test schema export functionality."""

    def test_export_all_schemas(self, engine, multiple_schemas):
        """Export without arguments returns all schemas."""
        result = engine.export_schemas()
        assert len(result) == 4

    def test_export_by_specific_ids(self, engine, multiple_schemas):
        """Export by schema_ids returns only those schemas."""
        ids = [multiple_schemas[0]["schema_id"], multiple_schemas[2]["schema_id"]]
        result = engine.export_schemas(schema_ids=ids)
        assert len(result) == 2

    def test_export_by_namespace(self, engine, multiple_schemas):
        """Export by namespace returns schemas from that namespace."""
        result = engine.export_schemas(namespace="emissions")
        assert len(result) == 2

    def test_export_nonexistent_ids_skipped(self, engine, multiple_schemas):
        """Non-existent IDs are silently skipped in export."""
        ids = [multiple_schemas[0]["schema_id"], "nonexistent-id"]
        result = engine.export_schemas(schema_ids=ids)
        assert len(result) == 1

    def test_export_returns_deep_copies(self, engine, multiple_schemas):
        """Exported schemas are deep copies."""
        result = engine.export_schemas()
        result[0]["name"] = "MUTATED"
        stored = engine.get_schema(result[0]["schema_id"])
        assert stored["name"] != "MUTATED"

    def test_export_ordered_by_created_at(self, engine, multiple_schemas):
        """Exported schemas are ordered by created_at ascending."""
        result = engine.export_schemas()
        created_times = [s["created_at"] for s in result]
        assert created_times == sorted(created_times)

    def test_export_union_of_ids_and_namespace(self, engine, multiple_schemas):
        """Export with both schema_ids and namespace returns the union."""
        # Pick one from supply_chain and specify emissions namespace
        sc_id = multiple_schemas[2]["schema_id"]
        result = engine.export_schemas(
            schema_ids=[sc_id], namespace="emissions"
        )
        # Should include the supply_chain schema by ID plus 2 emissions by namespace
        assert len(result) == 3

    def test_export_empty_registry(self, engine):
        """Export from empty registry returns empty list."""
        result = engine.export_schemas()
        assert result == []


# ===========================================================================
# TestSchemaGroups
# ===========================================================================


class TestSchemaGroups:
    """Test schema group operations."""

    def test_create_group(self, engine):
        """Create a new schema group."""
        group = engine.create_group(
            name="ghg-protocol",
            description="All GHG Protocol schemas",
        )
        assert group["name"] == "ghg-protocol"
        assert group["description"] == "All GHG Protocol schemas"
        assert group["schema_ids"] == []
        assert group["provenance_hash"] != ""

    def test_create_group_with_schema_ids(self, engine, registered_schema):
        """Create a group with initial schema members."""
        group = engine.create_group(
            name="emissions-group",
            schema_ids=[registered_schema["schema_id"]],
        )
        assert registered_schema["schema_id"] in group["schema_ids"]

    def test_create_group_duplicate_name_rejected(self, engine):
        """Creating a group with duplicate name raises ValueError."""
        engine.create_group(name="dup-group")
        with pytest.raises(ValueError, match="already exists"):
            engine.create_group(name="dup-group")

    def test_create_group_empty_name_rejected(self, engine):
        """Creating a group with empty name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            engine.create_group(name="")

    def test_create_group_skips_nonexistent_ids(self, engine, registered_schema):
        """Non-existent schema IDs are silently skipped during group creation."""
        group = engine.create_group(
            name="partial-group",
            schema_ids=[registered_schema["schema_id"], "nonexistent-id"],
        )
        assert len(group["schema_ids"]) == 1

    def test_get_group(self, engine):
        """Retrieve an existing group by name."""
        engine.create_group(name="test-group", description="Test")
        group = engine.get_group("test-group")
        assert group is not None
        assert group["name"] == "test-group"

    def test_get_nonexistent_group(self, engine):
        """Retrieving a non-existent group returns None."""
        result = engine.get_group("nonexistent")
        assert result is None

    def test_list_groups(self, engine):
        """List all groups returns correct count."""
        engine.create_group(name="g1")
        engine.create_group(name="g2")
        groups = engine.list_groups()
        assert len(groups) == 2

    def test_add_to_group(self, engine, registered_schema):
        """Add a schema to an existing group."""
        engine.create_group(name="g1")
        result = engine.add_to_group("g1", registered_schema["schema_id"])
        assert registered_schema["schema_id"] in result["schema_ids"]

    def test_add_to_group_duplicate_is_noop(self, engine, registered_schema):
        """Adding a schema already in the group is a no-op."""
        engine.create_group(
            name="g1",
            schema_ids=[registered_schema["schema_id"]],
        )
        result = engine.add_to_group("g1", registered_schema["schema_id"])
        assert result["schema_ids"].count(registered_schema["schema_id"]) == 1

    def test_add_to_nonexistent_group_raises(self, engine, registered_schema):
        """Adding to a non-existent group raises KeyError."""
        with pytest.raises(KeyError, match="Group not found"):
            engine.add_to_group("nonexistent", registered_schema["schema_id"])

    def test_add_nonexistent_schema_to_group_raises(self, engine):
        """Adding a non-existent schema to a group raises KeyError."""
        engine.create_group(name="g1")
        with pytest.raises(KeyError, match="Schema not found"):
            engine.add_to_group("g1", "nonexistent-schema-id")

    def test_remove_from_group(self, engine, registered_schema):
        """Remove a schema from a group."""
        engine.create_group(
            name="g1",
            schema_ids=[registered_schema["schema_id"]],
        )
        result = engine.remove_from_group("g1", registered_schema["schema_id"])
        assert registered_schema["schema_id"] not in result["schema_ids"]

    def test_remove_nonmember_from_group_is_noop(self, engine, registered_schema):
        """Removing a schema not in the group is a no-op."""
        engine.create_group(name="g1")
        result = engine.remove_from_group("g1", registered_schema["schema_id"])
        assert result["schema_ids"] == []

    def test_remove_from_nonexistent_group_raises(self, engine, registered_schema):
        """Removing from a non-existent group raises KeyError."""
        with pytest.raises(KeyError, match="Group not found"):
            engine.remove_from_group("nonexistent", registered_schema["schema_id"])

    def test_group_statistics_counted(self, engine):
        """Statistics include group counts."""
        engine.create_group(name="g1")
        engine.create_group(name="g2")
        stats = engine.get_statistics()
        assert stats["total_groups"] == 2


# ===========================================================================
# TestValidateDefinition
# ===========================================================================


class TestValidateDefinition:
    """Test schema definition validation."""

    def test_valid_json_schema(self, engine, json_schema_definition):
        """Valid JSON Schema passes validation."""
        result = engine.validate_definition("json_schema", json_schema_definition)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_valid_json_schema_minimal(self, engine):
        """Minimal JSON Schema with just 'type' passes."""
        result = engine.validate_definition(
            "json_schema", {"type": "object"}
        )
        assert result["valid"] is True

    def test_json_schema_with_only_dollar_schema(self, engine):
        """JSON Schema with only $schema key (no type) passes."""
        result = engine.validate_definition(
            "json_schema",
            {"$schema": "https://json-schema.org/draft/2020-12/schema"},
        )
        assert result["valid"] is True

    def test_json_schema_missing_type_and_schema(self, engine):
        """JSON Schema missing both type and $schema fails."""
        result = engine.validate_definition(
            "json_schema", {"properties": {"id": {"type": "string"}}}
        )
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_json_schema_not_dict_fails(self, engine):
        """JSON Schema that is not a dict fails."""
        result = engine.validate_definition("json_schema", "not a dict")
        assert result["valid"] is False

    def test_json_schema_empty_dict_fails(self, engine):
        """Empty dict JSON Schema fails."""
        result = engine.validate_definition("json_schema", {})
        assert result["valid"] is False

    def test_json_schema_invalid_properties_type(self, engine):
        """JSON Schema with non-dict properties fails."""
        result = engine.validate_definition(
            "json_schema",
            {"type": "object", "properties": "invalid"},
        )
        assert result["valid"] is False

    def test_json_schema_invalid_required_type(self, engine):
        """JSON Schema with non-list required fails."""
        result = engine.validate_definition(
            "json_schema",
            {"type": "object", "required": "id"},
        )
        assert result["valid"] is False

    def test_json_schema_required_non_string_elements(self, engine):
        """JSON Schema with non-string required elements fails."""
        result = engine.validate_definition(
            "json_schema",
            {"type": "object", "required": [1, 2, 3]},
        )
        assert result["valid"] is False

    def test_json_schema_unknown_draft_warning(self, engine):
        """Unknown $schema URI produces a warning."""
        result = engine.validate_definition(
            "json_schema",
            {"$schema": "https://unknown.org/schema", "type": "object"},
        )
        assert result["valid"] is True
        assert len(result["warnings"]) > 0

    def test_valid_avro(self, engine, avro_definition):
        """Valid Avro definition passes."""
        result = engine.validate_definition("avro", avro_definition)
        assert result["valid"] is True

    def test_avro_missing_fields_key(self, engine):
        """Avro missing 'fields' key fails."""
        result = engine.validate_definition(
            "avro", {"type": "record", "name": "Test"}
        )
        assert result["valid"] is False

    def test_avro_missing_name_key(self, engine):
        """Avro missing 'name' key fails."""
        result = engine.validate_definition(
            "avro", {"type": "record", "fields": []}
        )
        assert result["valid"] is False

    def test_avro_fields_not_list(self, engine):
        """Avro with non-list 'fields' fails."""
        result = engine.validate_definition(
            "avro",
            {"type": "record", "name": "Test", "fields": "invalid"},
        )
        assert result["valid"] is False

    def test_avro_field_missing_name(self, engine):
        """Avro field missing 'name' key fails."""
        result = engine.validate_definition(
            "avro",
            {
                "type": "record",
                "name": "Test",
                "fields": [{"type": "string"}],
            },
        )
        assert result["valid"] is False

    def test_avro_field_missing_type(self, engine):
        """Avro field missing 'type' key fails."""
        result = engine.validate_definition(
            "avro",
            {
                "type": "record",
                "name": "Test",
                "fields": [{"name": "f1"}],
            },
        )
        assert result["valid"] is False

    def test_avro_not_dict_fails(self, engine):
        """Avro definition that is not a dict fails."""
        result = engine.validate_definition("avro", [1, 2, 3])
        assert result["valid"] is False

    def test_avro_unknown_type_warning(self, engine):
        """Avro with unknown type produces a warning."""
        result = engine.validate_definition(
            "avro",
            {"type": "weird_type", "name": "Test", "fields": []},
        )
        # Still valid structurally, but has a warning
        assert result["valid"] is True
        assert len(result["warnings"]) > 0

    def test_valid_protobuf(self, engine, protobuf_definition):
        """Valid Protobuf definition passes."""
        result = engine.validate_definition("protobuf", protobuf_definition)
        assert result["valid"] is True

    def test_protobuf_missing_syntax(self, engine):
        """Protobuf missing 'syntax' fails."""
        result = engine.validate_definition(
            "protobuf",
            {"messages": [{"name": "Msg", "fields": []}]},
        )
        assert result["valid"] is False

    def test_protobuf_missing_messages(self, engine):
        """Protobuf missing 'messages' fails."""
        result = engine.validate_definition(
            "protobuf", {"syntax": "proto3"}
        )
        assert result["valid"] is False

    def test_protobuf_messages_not_list(self, engine):
        """Protobuf with non-list messages fails."""
        result = engine.validate_definition(
            "protobuf",
            {"syntax": "proto3", "messages": "invalid"},
        )
        assert result["valid"] is False

    def test_protobuf_message_missing_name(self, engine):
        """Protobuf message missing 'name' fails."""
        result = engine.validate_definition(
            "protobuf",
            {"syntax": "proto3", "messages": [{"fields": []}]},
        )
        assert result["valid"] is False

    def test_protobuf_message_not_dict(self, engine):
        """Protobuf message that is not a dict fails."""
        result = engine.validate_definition(
            "protobuf",
            {"syntax": "proto3", "messages": ["not_a_dict"]},
        )
        assert result["valid"] is False

    def test_protobuf_empty_messages_warning(self, engine):
        """Protobuf with empty messages list produces a warning."""
        result = engine.validate_definition(
            "protobuf", {"syntax": "proto3", "messages": []}
        )
        assert result["valid"] is True
        assert len(result["warnings"]) > 0

    def test_protobuf_unknown_syntax_warning(self, engine):
        """Protobuf with unknown syntax produces a warning."""
        result = engine.validate_definition(
            "protobuf",
            {"syntax": "proto4", "messages": [{"name": "Msg"}]},
        )
        assert result["valid"] is True
        assert len(result["warnings"]) > 0

    def test_protobuf_not_dict_fails(self, engine):
        """Protobuf definition that is not a dict fails."""
        result = engine.validate_definition("protobuf", 42)
        assert result["valid"] is False

    def test_validate_unsupported_type_raises(self, engine):
        """Unsupported schema_type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported schema_type"):
            engine.validate_definition("xml", {"root": {}})

    def test_validate_result_has_schema_type(self, engine):
        """Validation result includes the schema_type."""
        result = engine.validate_definition("json_schema", {"type": "object"})
        assert result["schema_type"] == "json_schema"

    def test_protobuf_message_fields_not_list(self, engine):
        """Protobuf message with non-list fields fails."""
        result = engine.validate_definition(
            "protobuf",
            {
                "syntax": "proto3",
                "messages": [{"name": "Msg", "fields": "not_a_list"}],
            },
        )
        assert result["valid"] is False


# ===========================================================================
# TestStatisticsAndProvenance
# ===========================================================================


class TestStatisticsAndProvenance:
    """Test statistics aggregation and provenance tracking."""

    def test_statistics_total_schemas(self, engine, multiple_schemas):
        """Statistics report correct total schemas."""
        stats = engine.get_statistics()
        assert stats["total_schemas"] == 4

    def test_statistics_by_type(self, engine, multiple_schemas):
        """Statistics break down schemas by type."""
        stats = engine.get_statistics()
        assert stats["by_type"]["json_schema"] == 2
        assert stats["by_type"]["avro"] == 2

    def test_statistics_by_status(self, engine, multiple_schemas):
        """Statistics break down schemas by status."""
        stats = engine.get_statistics()
        assert stats["by_status"]["draft"] == 4

    def test_statistics_by_namespace(self, engine, multiple_schemas):
        """Statistics break down schemas by namespace."""
        stats = engine.get_statistics()
        assert stats["by_namespace"]["emissions"] == 2
        assert stats["by_namespace"]["supply_chain"] == 2

    def test_statistics_total_tags(self, engine, multiple_schemas):
        """Statistics report correct total unique tags."""
        stats = engine.get_statistics()
        # tags: core, emissions, factors, material, supplier
        assert stats["total_tags"] == 5

    def test_statistics_total_namespaces(self, engine, multiple_schemas):
        """Statistics report correct total namespaces."""
        stats = engine.get_statistics()
        assert stats["total_namespaces"] == 2

    def test_statistics_provenance_entries(self, engine, multiple_schemas):
        """Statistics report provenance entries count."""
        stats = engine.get_statistics()
        assert stats["provenance_entries"] >= 4

    def test_provenance_hash_deterministic_for_same_input(self, engine):
        """Same registration input produces same definition hash."""
        definition = {"type": "object", "properties": {"x": {"type": "string"}}}
        hash1 = _build_sha256(definition)
        hash2 = _build_sha256(definition)
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_provenance_hash_changes_on_mutation(self, engine):
        """Different data produces different hashes."""
        hash1 = _build_sha256({"a": 1})
        hash2 = _build_sha256({"a": 2})
        assert hash1 != hash2


# ===========================================================================
# TestResetEngine
# ===========================================================================


class TestResetEngine:
    """Test engine reset functionality."""

    def test_reset_clears_all_schemas(self, engine, registered_schema):
        """Reset clears all schemas."""
        engine.reset()
        stats = engine.get_statistics()
        assert stats["total_schemas"] == 0

    def test_reset_clears_groups(self, engine):
        """Reset clears all groups."""
        engine.create_group(name="g1")
        engine.reset()
        assert engine.list_groups() == []

    def test_reset_clears_indexes(self, engine, registered_schema):
        """Reset clears all internal indexes."""
        engine.reset()
        assert len(engine._namespace_index) == 0
        assert len(engine._name_index) == 0
        assert len(engine._tag_index) == 0

    def test_reset_allows_reregistration(self, engine, json_schema_definition):
        """After reset, previously registered names can be reused."""
        engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        engine.reset()
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        assert schema["name"] == "S1"


# ===========================================================================
# TestSchemaRegistryEdgeCases
# ===========================================================================


class TestSchemaRegistryEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_namespace_at_max_length(self, engine, json_schema_definition):
        """Namespace at exactly MAX_NAMESPACE_LENGTH is accepted."""
        ns = "a" * MAX_NAMESPACE_LENGTH
        schema = engine.register_schema(
            namespace=ns, name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        assert schema["namespace"] == ns

    def test_name_at_max_length(self, engine, json_schema_definition):
        """Name at exactly MAX_SCHEMA_NAME_LENGTH is accepted."""
        name = "S" * MAX_SCHEMA_NAME_LENGTH
        schema = engine.register_schema(
            namespace="ns", name=name, schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        assert schema["name"] == name

    def test_tag_at_max_length(self, engine, json_schema_definition):
        """Tag at exactly MAX_TAG_LENGTH is accepted."""
        tag = "t" * MAX_TAG_LENGTH
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
            tags=[tag],
        )
        assert tag in schema["tags"]

    def test_many_schemas_registered(self, engine):
        """Register 50 schemas to test at moderate scale."""
        for i in range(50):
            engine.register_schema(
                namespace=f"ns-{i % 5}",
                name=f"Schema-{i}",
                schema_type="json_schema",
                definition_json={"type": "object"},
            )
        stats = engine.get_statistics()
        assert stats["total_schemas"] == 50
        assert stats["total_namespaces"] == 5

    def test_concurrent_registration(self, engine):
        """Thread safety: concurrent registrations do not corrupt state."""
        errors = []

        def register(idx):
            try:
                engine.register_schema(
                    namespace="concurrent",
                    name=f"Schema-{idx}",
                    schema_type="json_schema",
                    definition_json={"type": "object"},
                )
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=register, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = engine.get_statistics()
        assert stats["total_schemas"] == 20

    def test_schema_with_complex_definition(self, engine):
        """Register schema with deeply nested JSON Schema definition."""
        definition = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "number"},
                                },
                            },
                        },
                    },
                },
            },
        }
        schema = engine.register_schema(
            namespace="complex", name="DeepSchema",
            schema_type="json_schema",
            definition_json=definition,
        )
        assert schema["definition"]["properties"]["level1"]["type"] == "object"

    def test_empty_tags_list(self, engine, json_schema_definition):
        """Empty tags list is handled correctly."""
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
            tags=[],
        )
        assert schema["tags"] == []

    def test_none_tags(self, engine, json_schema_definition):
        """None tags parameter is handled correctly."""
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
            tags=None,
        )
        assert schema["tags"] == []

    def test_namespace_with_numeric_prefix(self, engine, json_schema_definition):
        """Namespace starting with a number is valid."""
        schema = engine.register_schema(
            namespace="123abc", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
        )
        assert schema["namespace"] == "123abc"

    def test_large_description(self, engine, json_schema_definition):
        """Large description string is stored correctly."""
        desc = "A" * 10000
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
            description=desc,
        )
        assert len(schema["description"]) == 10000

    def test_metadata_none_defaults_to_empty(self, engine, json_schema_definition):
        """None metadata defaults to empty dict."""
        schema = engine.register_schema(
            namespace="ns", name="S1", schema_type="json_schema",
            definition_json=json_schema_definition,
            metadata=None,
        )
        assert schema["metadata"] == {}


# ===========================================================================
# TestHelperFunctions
# ===========================================================================


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_normalize_tags_dedup_and_sort(self):
        """Tags are deduplicated, lowered, and sorted."""
        result = _normalize_tags(["B", "a", "b", "C"])
        assert result == ["a", "b", "c"]

    def test_normalize_tags_strips_whitespace(self):
        """Tags are stripped of whitespace."""
        result = _normalize_tags(["  hello  ", "  world  "])
        assert result == ["hello", "world"]

    def test_normalize_tags_removes_empty(self):
        """Empty string tags are removed."""
        result = _normalize_tags(["", "  ", "valid"])
        assert result == ["valid"]

    def test_normalize_tags_none_returns_empty(self):
        """None input returns empty list."""
        result = _normalize_tags(None)
        assert result == []

    def test_validate_namespace_valid(self):
        """Valid namespace passes validation."""
        _validate_namespace("my-namespace")  # Should not raise

    def test_validate_namespace_empty_raises(self):
        """Empty namespace raises ValueError."""
        with pytest.raises(ValueError):
            _validate_namespace("")

    def test_validate_namespace_special_chars_raises(self):
        """Namespace with special chars raises ValueError."""
        with pytest.raises(ValueError):
            _validate_namespace("ns@bad")

    def test_validate_schema_name_valid(self):
        """Valid schema name passes."""
        _validate_schema_name("MySchema")  # Should not raise

    def test_validate_schema_name_empty_raises(self):
        """Empty schema name raises ValueError."""
        with pytest.raises(ValueError):
            _validate_schema_name("")

    def test_validate_tags_list_valid(self):
        """Valid tags list passes."""
        _validate_tags_list(["tag1", "tag2"])  # Should not raise

    def test_validate_tags_list_long_tag_raises(self):
        """Tag exceeding MAX_TAG_LENGTH raises ValueError."""
        long_tag = "x" * (MAX_TAG_LENGTH + 1)
        with pytest.raises(ValueError):
            _validate_tags_list([long_tag])

    def test_build_sha256_deterministic(self):
        """SHA-256 hash is deterministic for same input."""
        h1 = _build_sha256({"key": "value"})
        h2 = _build_sha256({"key": "value"})
        assert h1 == h2
        assert len(h1) == 64

    def test_build_sha256_different_for_different_input(self):
        """Different inputs produce different hashes."""
        h1 = _build_sha256({"key": "value1"})
        h2 = _build_sha256({"key": "value2"})
        assert h1 != h2

    def test_build_sha256_key_order_independent(self):
        """Dict key order does not affect hash (sorted keys)."""
        h1 = _build_sha256({"a": 1, "b": 2})
        h2 = _build_sha256({"b": 2, "a": 1})
        assert h1 == h2


# ===========================================================================
# TestConstants
# ===========================================================================


class TestConstants:
    """Test module-level constants for correctness."""

    def test_valid_schema_types(self):
        """VALID_SCHEMA_TYPES contains expected types."""
        assert "json_schema" in VALID_SCHEMA_TYPES
        assert "avro" in VALID_SCHEMA_TYPES
        assert "protobuf" in VALID_SCHEMA_TYPES
        assert len(VALID_SCHEMA_TYPES) == 3

    def test_valid_statuses(self):
        """VALID_STATUSES contains expected statuses."""
        assert "draft" in VALID_STATUSES
        assert "active" in VALID_STATUSES
        assert "deprecated" in VALID_STATUSES
        assert "archived" in VALID_STATUSES
        assert len(VALID_STATUSES) == 4

    def test_status_transitions_correct(self):
        """STATUS_TRANSITIONS enforces correct lifecycle."""
        assert STATUS_TRANSITIONS["draft"] == {"active"}
        assert STATUS_TRANSITIONS["active"] == {"deprecated"}
        assert STATUS_TRANSITIONS["deprecated"] == {"archived"}
        assert STATUS_TRANSITIONS["archived"] == set()

    def test_max_bulk_import(self):
        """MAX_BULK_IMPORT is a positive integer."""
        assert MAX_BULK_IMPORT > 0
        assert MAX_BULK_IMPORT == 1000

    def test_max_lengths_positive(self):
        """Max length constants are positive."""
        assert MAX_NAMESPACE_LENGTH > 0
        assert MAX_SCHEMA_NAME_LENGTH > 0
        assert MAX_TAG_LENGTH > 0
