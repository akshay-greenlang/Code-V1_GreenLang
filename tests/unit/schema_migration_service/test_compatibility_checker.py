# -*- coding: utf-8 -*-
"""
Unit Tests for CompatibilityCheckerEngine - AGENT-DATA-017

Tests the CompatibilityCheckerEngine (Engine 4 of 7) for the Schema Migration
Agent. Validates Confluent Schema Registry-style compatibility semantics
including backward, forward, full, and breaking compatibility levels.

Coverage targets 85%+ across all public methods and private helpers.

Test classes (~100 tests):
    - TestCompatibilityCheckerInit (7 tests)
    - TestCheckCompatibility (16 tests)
    - TestBackwardCompatibility (16 tests)
    - TestForwardCompatibility (16 tests)
    - TestFullCompatibility (11 tests)
    - TestGetIssues (9 tests)
    - TestSuggestRemediation (9 tests)
    - TestBooleanHelpers (8 tests)
    - TestCompatibilityEdgeCases (11 tests)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
Engine: 4 of 7 - CompatibilityCheckerEngine
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.schema_migration.compatibility_checker import (
    CompatibilityCheckerEngine,
    TYPE_NARROWING,
    TYPE_WIDENING,
    _COMPATIBILITY_RULES,
)
from greenlang.schema_migration.provenance import ProvenanceTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> CompatibilityCheckerEngine:
    """Create a fresh CompatibilityCheckerEngine for each test."""
    eng = CompatibilityCheckerEngine()
    yield eng
    eng.reset()


@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Create a fresh ProvenanceTracker for injection."""
    return ProvenanceTracker()


@pytest.fixture
def engine_with_provenance(provenance_tracker) -> CompatibilityCheckerEngine:
    """Create a CompatibilityCheckerEngine with an explicit provenance tracker."""
    eng = CompatibilityCheckerEngine(provenance=provenance_tracker)
    yield eng
    eng.reset()


@pytest.fixture
def base_schema() -> Dict[str, Any]:
    """A baseline schema definition with three fields."""
    return {
        "fields": {
            "id": {"type": "integer", "required": True},
            "name": {"type": "string", "required": True},
            "email": {"type": "string", "required": False},
        }
    }


@pytest.fixture
def empty_schema() -> Dict[str, Any]:
    """An empty schema definition with no fields."""
    return {"fields": {}}


@pytest.fixture
def schema_with_enums() -> Dict[str, Any]:
    """A schema definition with an enum field."""
    return {
        "fields": {
            "status": {
                "type": "string",
                "required": True,
                "enum": ["active", "inactive"],
            },
            "id": {"type": "integer", "required": True},
        }
    }


@pytest.fixture
def schema_with_defaults() -> Dict[str, Any]:
    """A schema definition with default values."""
    return {
        "fields": {
            "id": {"type": "integer", "required": True},
            "name": {"type": "string", "required": True},
            "role": {"type": "string", "required": True, "default": "user"},
        }
    }


# ===========================================================================
# TestCompatibilityCheckerInit
# ===========================================================================


class TestCompatibilityCheckerInit:
    """Tests for CompatibilityCheckerEngine initialization."""

    def test_init_default_creates_provenance_tracker(self):
        """Engine creates its own ProvenanceTracker when none is provided."""
        engine = CompatibilityCheckerEngine()
        assert engine._provenance is not None
        assert isinstance(engine._provenance, ProvenanceTracker)

    def test_init_with_custom_provenance(self, provenance_tracker):
        """Engine uses the provided ProvenanceTracker instance."""
        engine = CompatibilityCheckerEngine(provenance=provenance_tracker)
        assert engine._provenance is provenance_tracker

    def test_init_empty_checks_store(self):
        """Engine starts with an empty checks store."""
        engine = CompatibilityCheckerEngine()
        assert engine._checks == {}

    def test_init_statistics_zero(self):
        """All statistics are zero on initialization."""
        engine = CompatibilityCheckerEngine()
        stats = engine.get_statistics()
        assert stats["total_checks"] == 0
        assert stats["full_compatible"] == 0
        assert stats["backward_compatible"] == 0
        assert stats["forward_compatible"] == 0
        assert stats["breaking"] == 0
        assert stats["total_issues_found"] == 0

    def test_init_has_lock(self):
        """Engine has a threading lock for thread safety."""
        engine = CompatibilityCheckerEngine()
        assert isinstance(engine._lock, type(threading.Lock()))

    def test_init_stored_checks_count_zero(self):
        """Stored checks count is zero at initialization."""
        engine = CompatibilityCheckerEngine()
        stats = engine.get_statistics()
        assert stats["stored_checks"] == 0

    def test_init_provenance_entries_zero(self):
        """Provenance entries count is zero at initialization."""
        engine = CompatibilityCheckerEngine()
        stats = engine.get_statistics()
        assert stats["provenance_entries"] == 0


# ===========================================================================
# TestCheckCompatibility
# ===========================================================================


class TestCheckCompatibility:
    """Tests for the main check_compatibility entry point."""

    def test_identical_schemas_are_fully_compatible(self, engine, base_schema):
        """Identical schemas produce full compatibility."""
        result = engine.check_compatibility(base_schema, base_schema)
        assert result["compatibility_level"] == "full"
        assert result["backward_compatible"] is True
        assert result["forward_compatible"] is True
        assert result["issues"] == []
        assert result["change_count"] == 0

    def test_check_returns_all_expected_keys(self, engine, base_schema):
        """Check result contains all documented keys."""
        result = engine.check_compatibility(base_schema, base_schema)
        expected_keys = {
            "check_id",
            "compatibility_level",
            "backward_compatible",
            "forward_compatible",
            "issues",
            "recommendations",
            "provenance_hash",
            "checked_at",
            "source_field_count",
            "target_field_count",
            "change_count",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_check_id_is_uuid(self, engine, base_schema):
        """check_id is a valid UUID4 string."""
        result = engine.check_compatibility(base_schema, base_schema)
        check_id = result["check_id"]
        assert isinstance(check_id, str)
        assert len(check_id) == 36  # UUID4 format with dashes

    def test_provenance_hash_is_sha256(self, engine, base_schema):
        """Provenance hash is a 64-character hex SHA-256 string."""
        result = engine.check_compatibility(base_schema, base_schema)
        assert len(result["provenance_hash"]) == 64
        # Validate hex characters
        int(result["provenance_hash"], 16)

    def test_backward_compatible_add_optional_field(self, engine, base_schema):
        """Adding an optional field is backward compatible."""
        target = copy.deepcopy(base_schema)
        target["fields"]["age"] = {"type": "integer", "required": False}
        result = engine.check_compatibility(base_schema, target)
        assert result["backward_compatible"] is True
        assert result["compatibility_level"] == "full"

    def test_backward_incompatible_remove_required_field(self, engine, base_schema):
        """Removing a required field breaks backward compatibility."""
        target = copy.deepcopy(base_schema)
        del target["fields"]["name"]
        result = engine.check_compatibility(base_schema, target)
        assert result["backward_compatible"] is False
        assert result["compatibility_level"] in ("forward", "breaking")

    def test_forward_incompatible_add_enum_value(self, engine, schema_with_enums):
        """Adding an enum value breaks forward compatibility."""
        target = copy.deepcopy(schema_with_enums)
        target["fields"]["status"]["enum"] = ["active", "inactive", "pending"]
        result = engine.check_compatibility(schema_with_enums, target)
        assert result["forward_compatible"] is False

    def test_breaking_change_type_narrowing(self, engine, base_schema):
        """Type narrowing (number -> integer) is breaking."""
        source = {"fields": {"value": {"type": "number", "required": True}}}
        target = {"fields": {"value": {"type": "integer", "required": True}}}
        result = engine.check_compatibility(source, target)
        assert result["compatibility_level"] == "breaking"
        assert result["backward_compatible"] is False
        assert result["forward_compatible"] is False

    def test_field_counts_reported(self, engine, base_schema):
        """Source and target field counts are correctly reported."""
        target = copy.deepcopy(base_schema)
        target["fields"]["extra"] = {"type": "string", "required": False}
        result = engine.check_compatibility(base_schema, target)
        assert result["source_field_count"] == 3
        assert result["target_field_count"] == 4

    def test_check_is_stored(self, engine, base_schema):
        """Check result is stored and retrievable by check_id."""
        result = engine.check_compatibility(base_schema, base_schema)
        stored = engine.get_check(result["check_id"])
        assert stored is not None
        assert stored["check_id"] == result["check_id"]

    def test_statistics_updated_after_check(self, engine, base_schema):
        """Statistics are updated after a compatibility check."""
        engine.check_compatibility(base_schema, base_schema)
        stats = engine.get_statistics()
        assert stats["total_checks"] == 1
        assert stats["full_compatible"] == 1

    def test_invalid_source_definition_not_dict(self, engine, base_schema):
        """TypeError raised when source_definition is not a dict."""
        with pytest.raises(TypeError, match="source_definition must be a dict"):
            engine.check_compatibility("not_a_dict", base_schema)

    def test_invalid_target_definition_not_dict(self, engine, base_schema):
        """TypeError raised when target_definition is not a dict."""
        with pytest.raises(TypeError, match="target_definition must be a dict"):
            engine.check_compatibility(base_schema, 42)

    def test_missing_fields_key_source(self, engine):
        """ValueError raised when source_definition has no 'fields' key."""
        with pytest.raises(ValueError, match="must contain a 'fields' key"):
            engine.check_compatibility({"name": "test"}, {"fields": {}})

    def test_missing_fields_key_target(self, engine):
        """ValueError raised when target_definition has no 'fields' key."""
        with pytest.raises(ValueError, match="must contain a 'fields' key"):
            engine.check_compatibility({"fields": {}}, {"name": "test"})

    def test_precomputed_changes_bypass_auto_diff(self, engine, base_schema):
        """When changes are passed, auto-diff is skipped."""
        changes = [
            {
                "change_type": "add_optional_field",
                "field_path": "synthetic_field",
            }
        ]
        result = engine.check_compatibility(base_schema, base_schema, changes=changes)
        assert result["change_count"] == 1
        assert result["compatibility_level"] == "full"


# ===========================================================================
# TestBackwardCompatibility
# ===========================================================================


class TestBackwardCompatibility:
    """Tests for check_backward_compatibility and backward-specific rules."""

    def test_add_optional_field_is_backward_compatible(self, engine, base_schema):
        """Adding an optional field passes backward compatibility."""
        target = copy.deepcopy(base_schema)
        target["fields"]["tag"] = {"type": "string", "required": False}
        result = engine.check_backward_compatibility(base_schema, target)
        assert result["compatible"] is True
        assert result["issues"] == []

    def test_add_field_with_default_is_backward_compatible(self, engine, base_schema):
        """Adding a required field with a default is backward compatible."""
        target = copy.deepcopy(base_schema)
        target["fields"]["role"] = {
            "type": "string",
            "required": True,
            "default": "user",
        }
        result = engine.check_backward_compatibility(base_schema, target)
        assert result["compatible"] is True

    def test_reorder_fields_is_backward_compatible(self, engine):
        """Field reordering has no backward compatibility impact."""
        source = {
            "fields": {
                "a": {"type": "string", "required": True},
                "b": {"type": "integer", "required": True},
            }
        }
        # Same fields, reorder check via pre-computed changes
        changes = [{"change_type": "reorder_fields", "field_path": "a"}]
        result = engine.check_backward_compatibility(source, source, changes=changes)
        assert result["compatible"] is True

    def test_change_description_is_backward_compatible(self, engine):
        """Changing a field description has no backward compatibility impact."""
        changes = [
            {"change_type": "change_description", "field_path": "name"}
        ]
        source = {"fields": {"name": {"type": "string", "required": True}}}
        result = engine.check_backward_compatibility(source, source, changes=changes)
        assert result["compatible"] is True

    def test_change_default_value_is_backward_compatible(self, engine, schema_with_defaults):
        """Changing a default value is backward compatible."""
        target = copy.deepcopy(schema_with_defaults)
        target["fields"]["role"]["default"] = "admin"
        result = engine.check_backward_compatibility(schema_with_defaults, target)
        assert result["compatible"] is True

    def test_type_widening_int_to_number_is_backward_compatible(self, engine):
        """Widening integer -> number is backward compatible."""
        source = {"fields": {"val": {"type": "integer", "required": True}}}
        target = {"fields": {"val": {"type": "number", "required": True}}}
        result = engine.check_backward_compatibility(source, target)
        assert result["compatible"] is True

    def test_type_widening_int_to_string_is_backward_compatible(self, engine):
        """Widening integer -> string is backward compatible."""
        source = {"fields": {"val": {"type": "integer", "required": True}}}
        target = {"fields": {"val": {"type": "string", "required": True}}}
        result = engine.check_backward_compatibility(source, target)
        assert result["compatible"] is True

    def test_remove_required_field_fails_backward(self, engine, base_schema):
        """Removing a required field breaks backward compatibility."""
        target = copy.deepcopy(base_schema)
        del target["fields"]["name"]
        result = engine.check_backward_compatibility(base_schema, target)
        assert result["compatible"] is False
        assert len(result["issues"]) >= 1

    def test_remove_optional_field_is_backward_compatible(self, engine, base_schema):
        """Removing an optional field is backward compatible."""
        target = copy.deepcopy(base_schema)
        del target["fields"]["email"]  # email is optional (required=False)
        result = engine.check_backward_compatibility(base_schema, target)
        assert result["compatible"] is True

    def test_change_required_type_fails_backward(self, engine):
        """Changing a field type incompatibly fails backward check."""
        source = {"fields": {"data": {"type": "string", "required": True}}}
        target = {"fields": {"data": {"type": "object", "required": True}}}
        result = engine.check_backward_compatibility(source, target)
        assert result["compatible"] is False

    def test_add_required_field_without_default_fails_backward(self, engine, base_schema):
        """Adding a required field without a default fails backward check."""
        target = copy.deepcopy(base_schema)
        target["fields"]["new_required"] = {
            "type": "string",
            "required": True,
        }
        result = engine.check_backward_compatibility(base_schema, target)
        assert result["compatible"] is False

    def test_narrow_enum_values_fails_backward(self, engine, schema_with_enums):
        """Removing an enum value fails backward compatibility."""
        target = copy.deepcopy(schema_with_enums)
        target["fields"]["status"]["enum"] = ["active"]
        result = engine.check_backward_compatibility(schema_with_enums, target)
        assert result["compatible"] is False

    def test_make_required_optional_is_backward_compatible(self, engine):
        """Making a required field optional is backward compatible."""
        source = {"fields": {"x": {"type": "string", "required": True}}}
        target = {"fields": {"x": {"type": "string", "required": False}}}
        result = engine.check_backward_compatibility(source, target)
        assert result["compatible"] is True

    def test_make_optional_required_fails_backward(self, engine):
        """Making an optional field required (no default) fails backward check."""
        source = {"fields": {"x": {"type": "string", "required": False}}}
        target = {"fields": {"x": {"type": "string", "required": True}}}
        result = engine.check_backward_compatibility(source, target)
        assert result["compatible"] is False

    def test_type_narrowing_fails_backward(self, engine):
        """Type narrowing (number -> integer) fails backward compatibility."""
        source = {"fields": {"val": {"type": "number", "required": True}}}
        target = {"fields": {"val": {"type": "integer", "required": True}}}
        result = engine.check_backward_compatibility(source, target)
        assert result["compatible"] is False

    def test_checked_changes_count(self, engine, base_schema):
        """checked_changes count matches the number of diff items."""
        target = copy.deepcopy(base_schema)
        target["fields"]["extra1"] = {"type": "string", "required": False}
        target["fields"]["extra2"] = {"type": "integer", "required": False}
        result = engine.check_backward_compatibility(base_schema, target)
        assert result["checked_changes"] >= 2


# ===========================================================================
# TestForwardCompatibility
# ===========================================================================


class TestForwardCompatibility:
    """Tests for check_forward_compatibility and forward-specific rules."""

    def test_remove_optional_field_is_forward_compatible(self, engine, base_schema):
        """Removing an optional field passes forward compatibility."""
        target = copy.deepcopy(base_schema)
        del target["fields"]["email"]
        result = engine.check_forward_compatibility(base_schema, target)
        assert result["compatible"] is True

    def test_add_optional_field_is_forward_compatible(self, engine, base_schema):
        """Adding an optional field passes forward compatibility."""
        target = copy.deepcopy(base_schema)
        target["fields"]["tag"] = {"type": "string", "required": False}
        result = engine.check_forward_compatibility(base_schema, target)
        assert result["compatible"] is True

    def test_add_enum_values_fails_forward(self, engine, schema_with_enums):
        """Adding enum values fails forward compatibility."""
        target = copy.deepcopy(schema_with_enums)
        target["fields"]["status"]["enum"] = ["active", "inactive", "pending"]
        result = engine.check_forward_compatibility(schema_with_enums, target)
        assert result["compatible"] is False

    def test_remove_enum_value_is_forward_compatible(self, engine, schema_with_enums):
        """Removing an enum value is forward compatible."""
        target = copy.deepcopy(schema_with_enums)
        target["fields"]["status"]["enum"] = ["active"]
        result = engine.check_forward_compatibility(schema_with_enums, target)
        assert result["compatible"] is True

    def test_add_required_field_fails_forward(self, engine, base_schema):
        """Adding a required field (with default) fails forward compatibility."""
        target = copy.deepcopy(base_schema)
        target["fields"]["new_req"] = {
            "type": "string",
            "required": True,
            "default": "n/a",
        }
        result = engine.check_forward_compatibility(base_schema, target)
        assert result["compatible"] is False

    def test_type_widening_fails_forward(self, engine):
        """Type widening (int -> number) fails forward compatibility."""
        source = {"fields": {"val": {"type": "integer", "required": True}}}
        target = {"fields": {"val": {"type": "number", "required": True}}}
        result = engine.check_forward_compatibility(source, target)
        assert result["compatible"] is False

    def test_description_change_is_forward_compatible(self, engine):
        """Description changes pass forward compatibility."""
        changes = [{"change_type": "change_description", "field_path": "name"}]
        source = {"fields": {"name": {"type": "string", "required": True}}}
        result = engine.check_forward_compatibility(source, source, changes=changes)
        assert result["compatible"] is True

    def test_reorder_fields_is_forward_compatible(self, engine):
        """Field reordering passes forward compatibility."""
        changes = [{"change_type": "reorder_fields", "field_path": "a"}]
        source = {"fields": {"a": {"type": "string", "required": True}}}
        result = engine.check_forward_compatibility(source, source, changes=changes)
        assert result["compatible"] is True

    def test_make_required_optional_fails_forward(self, engine):
        """Making a required field optional fails forward compatibility."""
        source = {"fields": {"x": {"type": "string", "required": True}}}
        target = {"fields": {"x": {"type": "string", "required": False}}}
        result = engine.check_forward_compatibility(source, target)
        assert result["compatible"] is False

    def test_remove_required_field_fails_forward(self, engine, base_schema):
        """Removing a required field fails forward compatibility."""
        target = copy.deepcopy(base_schema)
        del target["fields"]["id"]
        result = engine.check_forward_compatibility(base_schema, target)
        assert result["compatible"] is False

    def test_type_narrowing_fails_forward(self, engine):
        """Type narrowing (number -> integer) fails forward compatibility."""
        source = {"fields": {"val": {"type": "number", "required": True}}}
        target = {"fields": {"val": {"type": "integer", "required": True}}}
        result = engine.check_forward_compatibility(source, target)
        assert result["compatible"] is False

    def test_make_optional_required_fails_forward(self, engine):
        """Making optional required fails forward compatibility."""
        source = {"fields": {"x": {"type": "string", "required": False}}}
        target = {"fields": {"x": {"type": "string", "required": True}}}
        result = engine.check_forward_compatibility(source, target)
        assert result["compatible"] is False

    def test_change_default_value_is_forward_compatible(self, engine, schema_with_defaults):
        """Changing a default value passes forward compatibility."""
        target = copy.deepcopy(schema_with_defaults)
        target["fields"]["role"]["default"] = "admin"
        result = engine.check_forward_compatibility(schema_with_defaults, target)
        assert result["compatible"] is True

    def test_add_required_field_without_default_fails_forward(self, engine, base_schema):
        """Adding a required field without a default fails forward compatibility."""
        target = copy.deepcopy(base_schema)
        target["fields"]["mandatory"] = {"type": "string", "required": True}
        result = engine.check_forward_compatibility(base_schema, target)
        assert result["compatible"] is False

    def test_incompatible_type_change_fails_forward(self, engine):
        """Incompatible type change fails forward compatibility."""
        source = {"fields": {"data": {"type": "string", "required": True}}}
        target = {"fields": {"data": {"type": "array", "required": True}}}
        result = engine.check_forward_compatibility(source, target)
        assert result["compatible"] is False

    def test_forward_checked_changes_count(self, engine, base_schema):
        """checked_changes count is reported in forward result."""
        result = engine.check_forward_compatibility(base_schema, base_schema)
        assert result["checked_changes"] == 0


# ===========================================================================
# TestFullCompatibility
# ===========================================================================


class TestFullCompatibility:
    """Tests for check_full_compatibility."""

    def test_identical_schemas_are_fully_compatible(self, engine, base_schema):
        """Identical schemas are fully compatible."""
        result = engine.check_full_compatibility(base_schema, base_schema)
        assert result["compatible"] is True
        assert result["level"] == "full"

    def test_description_only_changes_are_fully_compatible(self, engine):
        """Description-only changes are fully compatible."""
        source = {"fields": {"x": {"type": "string", "required": True}}}
        changes = [{"change_type": "change_description", "field_path": "x"}]
        result = engine.check_full_compatibility(source, source, changes=changes)
        assert result["compatible"] is True
        assert result["level"] == "full"

    def test_reorder_only_is_fully_compatible(self, engine):
        """Field reordering only is fully compatible."""
        source = {"fields": {"a": {"type": "string", "required": True}}}
        changes = [{"change_type": "reorder_fields", "field_path": "a"}]
        result = engine.check_full_compatibility(source, source, changes=changes)
        assert result["compatible"] is True
        assert result["level"] == "full"

    def test_add_optional_field_is_fully_compatible(self, engine, base_schema):
        """Adding an optional field is fully compatible."""
        target = copy.deepcopy(base_schema)
        target["fields"]["tag"] = {"type": "string", "required": False}
        result = engine.check_full_compatibility(base_schema, target)
        assert result["compatible"] is True
        assert result["level"] == "full"

    def test_remove_optional_field_is_fully_compatible(self, engine, base_schema):
        """Removing an optional field is fully compatible."""
        target = copy.deepcopy(base_schema)
        del target["fields"]["email"]
        result = engine.check_full_compatibility(base_schema, target)
        assert result["compatible"] is True
        assert result["level"] == "full"

    def test_backward_only_change_not_fully_compatible(self, engine):
        """Type widening (backward-only) is not fully compatible."""
        source = {"fields": {"val": {"type": "integer", "required": True}}}
        target = {"fields": {"val": {"type": "number", "required": True}}}
        result = engine.check_full_compatibility(source, target)
        assert result["compatible"] is False
        assert result["level"] == "backward"

    def test_forward_only_change_not_fully_compatible(self, engine, schema_with_enums):
        """Removing an enum value (forward-only) is not fully compatible."""
        target = copy.deepcopy(schema_with_enums)
        target["fields"]["status"]["enum"] = ["active"]
        result = engine.check_full_compatibility(schema_with_enums, target)
        assert result["compatible"] is False
        assert result["level"] == "forward"

    def test_breaking_change_not_fully_compatible(self, engine, base_schema):
        """Removing a required field (breaking) is not fully compatible."""
        target = copy.deepcopy(base_schema)
        del target["fields"]["name"]
        result = engine.check_full_compatibility(base_schema, target)
        assert result["compatible"] is False
        assert result["level"] == "breaking"

    def test_full_result_contains_backward_and_forward_results(self, engine, base_schema):
        """Full compatibility result includes backward_result and forward_result."""
        result = engine.check_full_compatibility(base_schema, base_schema)
        assert "backward_result" in result
        assert "forward_result" in result
        assert result["backward_result"]["compatible"] is True
        assert result["forward_result"]["compatible"] is True

    def test_full_issues_are_deduplicated(self, engine):
        """Issues from backward and forward checks are deduplicated."""
        source = {"fields": {"val": {"type": "number", "required": True}}}
        target = {"fields": {"val": {"type": "integer", "required": True}}}
        result = engine.check_full_compatibility(source, target)
        # Breaking change: both backward and forward fail, but deduplicated
        field_paths = [i["field_path"] for i in result["issues"]]
        # Each field_path + issue_type pair should appear at most twice
        # (once backward, once forward) but deduplicated by _deduplicate_issues
        # in the parent check_full_compatibility
        assert len(result["issues"]) <= 2

    def test_full_change_default_value_is_fully_compatible(self, engine, schema_with_defaults):
        """Changing a default value is fully compatible."""
        target = copy.deepcopy(schema_with_defaults)
        target["fields"]["role"]["default"] = "admin"
        result = engine.check_full_compatibility(schema_with_defaults, target)
        assert result["compatible"] is True
        assert result["level"] == "full"


# ===========================================================================
# TestGetIssues (via check_backward/forward which generate issues)
# ===========================================================================


class TestGetIssues:
    """Tests for issue generation and structure."""

    def test_no_issues_for_compatible_change(self, engine, base_schema):
        """Compatible schemas produce no issues."""
        target = copy.deepcopy(base_schema)
        target["fields"]["tag"] = {"type": "string", "required": False}
        result = engine.check_compatibility(base_schema, target)
        assert result["issues"] == []

    def test_issues_for_incompatible_change(self, engine, base_schema):
        """Incompatible schemas produce at least one issue."""
        target = copy.deepcopy(base_schema)
        del target["fields"]["name"]
        result = engine.check_compatibility(base_schema, target)
        assert len(result["issues"]) >= 1

    def test_issue_has_field_path(self, engine, base_schema):
        """Each issue includes a field_path."""
        target = copy.deepcopy(base_schema)
        del target["fields"]["name"]
        result = engine.check_compatibility(base_schema, target)
        for issue in result["issues"]:
            assert "field_path" in issue
            assert issue["field_path"] == "name"

    def test_issue_has_description(self, engine, base_schema):
        """Each issue includes a human-readable description."""
        target = copy.deepcopy(base_schema)
        del target["fields"]["name"]
        result = engine.check_compatibility(base_schema, target)
        for issue in result["issues"]:
            assert "description" in issue
            assert len(issue["description"]) > 0

    def test_issue_has_severity(self, engine, base_schema):
        """Each issue includes a severity field."""
        target = copy.deepcopy(base_schema)
        del target["fields"]["name"]
        result = engine.check_compatibility(base_schema, target)
        for issue in result["issues"]:
            assert "severity" in issue
            assert issue["severity"] == "breaking"

    def test_issue_has_issue_type(self, engine, base_schema):
        """Each issue includes an issue_type field."""
        target = copy.deepcopy(base_schema)
        del target["fields"]["name"]
        result = engine.check_compatibility(base_schema, target)
        for issue in result["issues"]:
            assert "issue_type" in issue
            assert issue["issue_type"] in (
                "backward_incompatible",
                "forward_incompatible",
            )

    def test_issue_has_change_type(self, engine, base_schema):
        """Each issue includes the change_type that triggered it."""
        target = copy.deepcopy(base_schema)
        del target["fields"]["name"]
        result = engine.check_compatibility(base_schema, target)
        for issue in result["issues"]:
            assert "change_type" in issue
            assert issue["change_type"] == "remove_required_field"

    def test_multiple_issues_for_multiple_breaking_changes(self, engine, base_schema):
        """Multiple breaking changes produce multiple issues."""
        target = copy.deepcopy(base_schema)
        del target["fields"]["name"]
        del target["fields"]["id"]
        result = engine.check_compatibility(base_schema, target)
        # At least 2 issues (one per removed required field)
        assert len(result["issues"]) >= 2

    def test_issues_are_deduplicated_across_directions(self, engine):
        """Same field issue from backward and forward checks is deduplicated."""
        source = {"fields": {"val": {"type": "number", "required": True}}}
        target = {"fields": {"val": {"type": "boolean", "required": True}}}
        result = engine.check_compatibility(source, target)
        # The type change on 'val' produces backward + forward issues
        # They should be deduplicated by (field_path, issue_type)
        field_issues = [i for i in result["issues"] if i["field_path"] == "val"]
        unique_types = set((i["field_path"], i["issue_type"]) for i in field_issues)
        assert len(field_issues) == len(unique_types)


# ===========================================================================
# TestSuggestRemediation
# ===========================================================================


class TestSuggestRemediation:
    """Tests for generate_remediation."""

    def test_empty_issues_produce_no_remediation(self, engine):
        """No remediation suggestions for empty issues list."""
        result = engine.generate_remediation([])
        assert result == []

    def test_remediation_for_remove_required_field(self, engine):
        """Remediation suggested for removing a required field."""
        issues = [
            {
                "field_path": "name",
                "change_type": "remove_required_field",
                "issue_type": "backward_incompatible",
                "severity": "breaking",
                "description": "Removing required field",
                "remediation": "Deprecate first",
            }
        ]
        rems = engine.generate_remediation(issues)
        assert len(rems) == 1
        assert rems[0]["field_path"] == "name"
        assert rems[0]["change_type"] == "remove_required_field"
        assert rems[0]["priority"] == "high"
        assert len(rems[0]["suggestion"]) > 0

    def test_remediation_for_rename_field(self, engine):
        """Remediation suggested for renaming a field (add alias approach)."""
        issues = [
            {
                "field_path": "dept",
                "change_type": "rename_field",
                "issue_type": "backward_incompatible",
                "severity": "breaking",
                "description": "Renaming field",
                "remediation": "Use alias",
            }
        ]
        rems = engine.generate_remediation(issues)
        assert len(rems) == 1
        assert "alias" in rems[0]["suggestion"].lower()

    def test_remediation_for_type_narrowing(self, engine):
        """Remediation suggested for type narrowing."""
        issues = [
            {
                "field_path": "value",
                "change_type": "type_narrowing",
                "issue_type": "backward_incompatible",
                "severity": "breaking",
                "description": "Type narrowing",
                "remediation": "Keep old type",
            }
        ]
        rems = engine.generate_remediation(issues)
        assert len(rems) == 1
        assert rems[0]["priority"] == "high"

    def test_remediation_for_make_optional_required(self, engine):
        """Remediation for making an optional field required (add default)."""
        issues = [
            {
                "field_path": "role",
                "change_type": "make_optional_required",
                "issue_type": "backward_incompatible",
                "severity": "breaking",
                "description": "Making optional required",
                "remediation": "Add default",
            }
        ]
        rems = engine.generate_remediation(issues)
        assert len(rems) == 1
        assert "default" in rems[0]["suggestion"].lower()

    def test_remediation_for_add_required_without_default(self, engine):
        """Remediation for adding required field without default."""
        issues = [
            {
                "field_path": "new_field",
                "change_type": "add_required_field_without_default",
                "issue_type": "backward_incompatible",
                "severity": "breaking",
                "description": "Adding required without default",
                "remediation": "Add default value",
            }
        ]
        rems = engine.generate_remediation(issues)
        assert len(rems) == 1
        assert "default" in rems[0]["suggestion"].lower()

    def test_remediation_deduplicates(self, engine):
        """Duplicate issues produce only one remediation suggestion."""
        issue = {
            "field_path": "name",
            "change_type": "remove_required_field",
            "issue_type": "backward_incompatible",
            "severity": "breaking",
            "description": "Removing field",
            "remediation": "Deprecate",
        }
        rems = engine.generate_remediation([issue, issue])
        assert len(rems) == 1

    def test_remediation_uses_generic_for_unknown_change(self, engine):
        """Unknown change types get generic remediation suggestion."""
        issues = [
            {
                "field_path": "unknown_field",
                "change_type": "some_unknown_type",
                "issue_type": "backward_incompatible",
                "severity": "breaking",
                "description": "Unknown change",
                "remediation": "",
            }
        ]
        rems = engine.generate_remediation(issues)
        assert len(rems) == 1
        assert "intermediate schema version" in rems[0]["suggestion"].lower()

    def test_non_breaking_issues_skipped(self, engine):
        """Non-breaking issues do not produce remediation suggestions."""
        issues = [
            {
                "field_path": "desc",
                "change_type": "change_description",
                "issue_type": "info",
                "severity": "non_breaking",
                "description": "Description changed",
                "remediation": "",
            }
        ]
        rems = engine.generate_remediation(issues)
        assert rems == []

    def test_remediation_result_has_correct_keys(self, engine):
        """Remediation suggestion dict contains all expected keys."""
        issues = [
            {
                "field_path": "val",
                "change_type": "type_narrowing",
                "issue_type": "backward_incompatible",
                "severity": "breaking",
                "description": "Narrowing",
                "remediation": "Keep old type",
            }
        ]
        rems = engine.generate_remediation(issues)
        expected_keys = {"field_path", "change_type", "issue_type", "suggestion", "priority"}
        assert expected_keys == set(rems[0].keys())


# ===========================================================================
# TestBooleanHelpers
# ===========================================================================


class TestBooleanHelpers:
    """Tests for is_type_widening, is_type_narrowing, and level comparison helpers."""

    def test_is_type_widening_integer_to_number(self, engine):
        """integer -> number is a widening."""
        assert engine.is_type_widening("integer", "number") is True

    def test_is_type_widening_integer_to_string(self, engine):
        """integer -> string is a widening."""
        assert engine.is_type_widening("integer", "string") is True

    def test_is_type_widening_number_to_integer_is_false(self, engine):
        """number -> integer is NOT a widening."""
        assert engine.is_type_widening("number", "integer") is False

    def test_is_type_narrowing_number_to_integer(self, engine):
        """number -> integer is a narrowing."""
        assert engine.is_type_narrowing("number", "integer") is True

    def test_is_type_narrowing_string_to_boolean(self, engine):
        """string -> boolean is a narrowing."""
        assert engine.is_type_narrowing("string", "boolean") is True

    def test_is_type_narrowing_integer_to_number_is_false(self, engine):
        """integer -> number is NOT a narrowing."""
        assert engine.is_type_narrowing("integer", "number") is False

    def test_compare_compatibility_levels_full_vs_backward(self, engine):
        """Comparing full and backward returns backward (stricter)."""
        result = engine.compare_compatibility_levels("full", "backward")
        assert result == "backward"

    def test_compare_compatibility_levels_breaking_vs_forward(self, engine):
        """Comparing breaking and forward returns breaking (stricter)."""
        result = engine.compare_compatibility_levels("breaking", "forward")
        assert result == "breaking"


# ===========================================================================
# TestCompatibilityEdgeCases
# ===========================================================================


class TestCompatibilityEdgeCases:
    """Tests for edge cases and corner scenarios."""

    def test_empty_schemas_are_fully_compatible(self, engine, empty_schema):
        """Two empty schemas are fully compatible."""
        result = engine.check_compatibility(empty_schema, empty_schema)
        assert result["compatibility_level"] == "full"
        assert result["issues"] == []

    def test_identical_complex_schemas(self, engine):
        """Identical complex schemas with nested fields are fully compatible."""
        schema = {
            "fields": {
                "user_id": {"type": "integer", "required": True},
                "name": {"type": "string", "required": True},
                "tags": {"type": "array", "required": False},
                "metadata": {"type": "object", "required": False},
                "active": {"type": "boolean", "required": True, "default": True},
            }
        }
        result = engine.check_compatibility(schema, schema)
        assert result["compatibility_level"] == "full"

    def test_array_type_change_is_breaking(self, engine):
        """Changing a field from array to string is breaking."""
        source = {"fields": {"tags": {"type": "array", "required": False}}}
        target = {"fields": {"tags": {"type": "string", "required": False}}}
        result = engine.check_compatibility(source, target)
        assert result["compatibility_level"] == "breaking"

    def test_object_to_string_is_breaking(self, engine):
        """Changing a field from object to string is breaking."""
        source = {"fields": {"meta": {"type": "object", "required": False}}}
        target = {"fields": {"meta": {"type": "string", "required": False}}}
        result = engine.check_compatibility(source, target)
        assert result["compatibility_level"] == "breaking"

    def test_enum_added_and_removed_simultaneously(self, engine):
        """Adding and removing enum values simultaneously."""
        source = {
            "fields": {
                "status": {
                    "type": "string",
                    "required": True,
                    "enum": ["a", "b", "c"],
                }
            }
        }
        target = {
            "fields": {
                "status": {
                    "type": "string",
                    "required": True,
                    "enum": ["a", "b", "d"],
                }
            }
        }
        result = engine.check_compatibility(source, target)
        # Both adding 'd' (backward-only) and removing 'c' (forward-only) happen
        assert result["issues"] != [] or result["compatibility_level"] != "full"

    def test_boolean_to_integer_widening(self, engine):
        """boolean -> integer is a widening (0/1 encoding)."""
        assert engine.is_type_widening("boolean", "integer") is True

    def test_integer_to_boolean_narrowing(self, engine):
        """integer -> boolean is a narrowing."""
        assert engine.is_type_narrowing("integer", "boolean") is True

    def test_schema_with_many_fields(self, engine):
        """Schema with 50 fields: add one optional field is still fully compatible."""
        fields = {
            f"field_{i}": {"type": "string", "required": False}
            for i in range(50)
        }
        source = {"fields": fields}
        target_fields = copy.deepcopy(fields)
        target_fields["field_50"] = {"type": "string", "required": False}
        target = {"fields": target_fields}
        result = engine.check_compatibility(source, target)
        assert result["compatibility_level"] == "full"
        assert result["source_field_count"] == 50
        assert result["target_field_count"] == 51

    def test_validate_transition_full_satisfies_backward(self, engine):
        """Full compatibility satisfies a backward requirement."""
        result = engine.validate_transition("full", "backward")
        assert result["allowed"] is True

    def test_validate_transition_backward_does_not_satisfy_full(self, engine):
        """Backward compatibility does not satisfy a full requirement."""
        result = engine.validate_transition("backward", "full")
        assert result["allowed"] is False

    def test_compare_levels_invalid_level_raises(self, engine):
        """Invalid level string raises ValueError."""
        with pytest.raises(ValueError):
            engine.compare_compatibility_levels("invalid", "full")


# ===========================================================================
# TestDetermineCompatibilityLevel
# ===========================================================================


class TestDetermineCompatibilityLevel:
    """Tests for the determine_compatibility_level truth table."""

    def test_both_true_is_full(self, engine):
        """backward=True, forward=True -> full."""
        level = engine.determine_compatibility_level(
            {"compatible": True}, {"compatible": True}
        )
        assert level == "full"

    def test_backward_only_is_backward(self, engine):
        """backward=True, forward=False -> backward."""
        level = engine.determine_compatibility_level(
            {"compatible": True}, {"compatible": False}
        )
        assert level == "backward"

    def test_forward_only_is_forward(self, engine):
        """backward=False, forward=True -> forward."""
        level = engine.determine_compatibility_level(
            {"compatible": False}, {"compatible": True}
        )
        assert level == "forward"

    def test_both_false_is_breaking(self, engine):
        """backward=False, forward=False -> breaking."""
        level = engine.determine_compatibility_level(
            {"compatible": False}, {"compatible": False}
        )
        assert level == "breaking"


# ===========================================================================
# TestAssessFieldChange
# ===========================================================================


class TestAssessFieldChange:
    """Tests for assess_field_change rule lookups."""

    def test_add_optional_field_assessment(self, engine):
        """add_optional_field is assessed as fully compatible."""
        change = {"change_type": "add_optional_field", "field_path": "tag"}
        assessment = engine.assess_field_change(change)
        assert assessment["backward_compatible"] is True
        assert assessment["forward_compatible"] is True
        assert assessment["level"] == "full"

    def test_remove_required_field_assessment(self, engine):
        """remove_required_field is assessed as breaking."""
        change = {"change_type": "remove_required_field", "field_path": "name"}
        assessment = engine.assess_field_change(change)
        assert assessment["backward_compatible"] is False
        assert assessment["forward_compatible"] is False
        assert assessment["level"] == "breaking"

    def test_retyped_widening_resolves_correctly(self, engine):
        """Retyped change from integer->number resolves as type_widening."""
        change = {
            "change_type": "retyped",
            "field_path": "val",
            "old_type": "integer",
            "new_type": "number",
        }
        assessment = engine.assess_field_change(change)
        assert assessment["change_type"] == "type_widening"
        assert assessment["backward_compatible"] is True
        assert assessment["forward_compatible"] is False

    def test_retyped_narrowing_resolves_correctly(self, engine):
        """Retyped change from number->integer resolves as type_narrowing."""
        change = {
            "change_type": "retyped",
            "field_path": "val",
            "old_type": "number",
            "new_type": "integer",
        }
        assessment = engine.assess_field_change(change)
        assert assessment["change_type"] == "type_narrowing"
        assert assessment["level"] == "breaking"

    def test_retyped_incompatible_resolves_correctly(self, engine):
        """Retyped change from string->object resolves as change_type_incompatible."""
        change = {
            "change_type": "retyped",
            "field_path": "data",
            "old_type": "string",
            "new_type": "object",
        }
        assessment = engine.assess_field_change(change)
        assert assessment["change_type"] == "change_type_incompatible"
        assert assessment["level"] == "breaking"

    def test_make_optional_required_with_default_resolved(self, engine):
        """make_optional_required with has_default=True resolves to backward-only."""
        change = {
            "change_type": "make_optional_required",
            "field_path": "role",
            "has_default": True,
        }
        assessment = engine.assess_field_change(change)
        assert assessment["backward_compatible"] is True
        assert assessment["forward_compatible"] is False
        assert assessment["level"] == "backward"

    def test_unknown_change_type_treated_as_breaking(self, engine):
        """Unknown change types are treated as breaking."""
        change = {"change_type": "totally_unknown", "field_path": "x"}
        assessment = engine.assess_field_change(change)
        assert assessment["level"] == "breaking"
        assert assessment["backward_compatible"] is False
        assert assessment["forward_compatible"] is False

    def test_add_enum_value_is_backward_only(self, engine):
        """add_enum_value is backward compatible only."""
        change = {"change_type": "add_enum_value", "field_path": "status"}
        assessment = engine.assess_field_change(change)
        assert assessment["backward_compatible"] is True
        assert assessment["forward_compatible"] is False
        assert assessment["level"] == "backward"

    def test_remove_enum_value_is_forward_only(self, engine):
        """remove_enum_value is forward compatible only."""
        change = {"change_type": "remove_enum_value", "field_path": "status"}
        assessment = engine.assess_field_change(change)
        assert assessment["backward_compatible"] is False
        assert assessment["forward_compatible"] is True
        assert assessment["level"] == "forward"


# ===========================================================================
# TestGetCheckAndListChecks
# ===========================================================================


class TestGetCheckAndListChecks:
    """Tests for get_check, list_checks, and stored check retrieval."""

    def test_get_check_returns_none_for_unknown_id(self, engine):
        """get_check returns None for an unknown check_id."""
        assert engine.get_check("nonexistent-id") is None

    def test_get_check_returns_stored_result(self, engine, base_schema):
        """get_check returns the stored result after a compatibility check."""
        result = engine.check_compatibility(base_schema, base_schema)
        stored = engine.get_check(result["check_id"])
        assert stored is not None
        assert stored["compatibility_level"] == "full"

    def test_list_checks_empty_initially(self, engine):
        """list_checks returns empty list before any checks."""
        checks = engine.list_checks()
        assert checks == []

    def test_list_checks_returns_all_checks(self, engine, base_schema):
        """list_checks returns all stored checks."""
        engine.check_compatibility(base_schema, base_schema)
        engine.check_compatibility(base_schema, base_schema)
        checks = engine.list_checks()
        assert len(checks) == 2

    def test_list_checks_pagination(self, engine, base_schema):
        """list_checks supports limit and offset pagination."""
        for _ in range(5):
            engine.check_compatibility(base_schema, base_schema)
        page = engine.list_checks(limit=2, offset=1)
        assert len(page) == 2

    def test_list_checks_invalid_limit_raises(self, engine):
        """list_checks raises ValueError for limit < 1."""
        with pytest.raises(ValueError, match="limit must be >= 1"):
            engine.list_checks(limit=0)

    def test_list_checks_invalid_offset_raises(self, engine):
        """list_checks raises ValueError for negative offset."""
        with pytest.raises(ValueError, match="offset must be >= 0"):
            engine.list_checks(offset=-1)


# ===========================================================================
# TestResetAndStatistics
# ===========================================================================


class TestResetAndStatistics:
    """Tests for reset and statistics tracking."""

    def test_reset_clears_checks(self, engine, base_schema):
        """reset() clears all stored checks."""
        engine.check_compatibility(base_schema, base_schema)
        assert engine.get_statistics()["total_checks"] == 1
        engine.reset()
        assert engine.get_statistics()["total_checks"] == 0
        assert engine.get_statistics()["stored_checks"] == 0

    def test_statistics_track_all_levels(self, engine):
        """Statistics correctly track different compatibility levels."""
        # Full compatible
        source_a = {"fields": {"x": {"type": "string", "required": False}}}
        engine.check_compatibility(source_a, source_a)

        # Backward only (type widening)
        source_b = {"fields": {"val": {"type": "integer", "required": True}}}
        target_b = {"fields": {"val": {"type": "number", "required": True}}}
        engine.check_compatibility(source_b, target_b)

        # Breaking (type narrowing)
        source_c = {"fields": {"val": {"type": "number", "required": True}}}
        target_c = {"fields": {"val": {"type": "integer", "required": True}}}
        engine.check_compatibility(source_c, target_c)

        stats = engine.get_statistics()
        assert stats["total_checks"] == 3
        assert stats["full_compatible"] == 1
        assert stats["backward_compatible"] == 1
        assert stats["breaking"] == 1

    def test_statistics_track_total_issues(self, engine):
        """Statistics accumulate total issues found across checks."""
        source = {"fields": {"name": {"type": "string", "required": True}}}
        target = {"fields": {}}
        engine.check_compatibility(source, target)
        stats = engine.get_statistics()
        assert stats["total_issues_found"] >= 1


# ===========================================================================
# TestProvenanceIntegration
# ===========================================================================


class TestProvenanceIntegration:
    """Tests for provenance tracking integration."""

    def test_provenance_recorded_after_check(self, engine_with_provenance, provenance_tracker, base_schema):
        """A provenance entry is recorded after each compatibility check."""
        initial_count = provenance_tracker.entry_count
        engine_with_provenance.check_compatibility(base_schema, base_schema)
        assert provenance_tracker.entry_count > initial_count

    def test_provenance_hash_deterministic(self, engine, base_schema):
        """Same inputs produce the same provenance hash (via _compute_check_hash)."""
        # We test the private _compute_check_hash directly for determinism
        check_id = "test-check-id"
        level = "full"
        issues: List[Dict[str, Any]] = []
        hash1 = engine._compute_check_hash(check_id, base_schema, base_schema, level, issues)
        hash2 = engine._compute_check_hash(check_id, base_schema, base_schema, level, issues)
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_provenance_hash_changes_with_different_level(self, engine, base_schema):
        """Different levels produce different provenance hashes."""
        check_id = "test-check-id"
        issues: List[Dict[str, Any]] = []
        hash_full = engine._compute_check_hash(check_id, base_schema, base_schema, "full", issues)
        hash_breaking = engine._compute_check_hash(check_id, base_schema, base_schema, "breaking", issues)
        assert hash_full != hash_breaking


# ===========================================================================
# TestCompatibilityRules
# ===========================================================================


class TestCompatibilityRules:
    """Tests for get_compatibility_rules and rule matrix access."""

    def test_get_compatibility_rules_returns_deep_copy(self, engine):
        """get_compatibility_rules returns a deep copy that is not the original."""
        rules = engine.get_compatibility_rules()
        rules["add_optional_field"]["level"] = "MUTATED"
        original = engine.get_compatibility_rules()
        assert original["add_optional_field"]["level"] == "full"

    def test_all_rules_have_backward_key(self, engine):
        """All rules in the matrix have a 'backward' boolean key."""
        rules = engine.get_compatibility_rules()
        for rule_name, rule in rules.items():
            assert "backward" in rule, f"Rule '{rule_name}' missing 'backward'"
            assert isinstance(rule["backward"], bool)

    def test_all_rules_have_forward_key(self, engine):
        """All rules in the matrix have a 'forward' boolean key."""
        rules = engine.get_compatibility_rules()
        for rule_name, rule in rules.items():
            assert "forward" in rule, f"Rule '{rule_name}' missing 'forward'"
            assert isinstance(rule["forward"], bool)

    def test_all_rules_have_level_key(self, engine):
        """All rules in the matrix have a 'level' key."""
        rules = engine.get_compatibility_rules()
        for rule_name, rule in rules.items():
            assert "level" in rule, f"Rule '{rule_name}' missing 'level'"
            assert rule["level"] in ("full", "backward", "forward", "breaking")


# ===========================================================================
# TestTypeWideningNarrowingMatrices
# ===========================================================================


class TestTypeWideningNarrowingMatrices:
    """Tests for the TYPE_WIDENING and TYPE_NARROWING module constants."""

    def test_type_widening_contains_integer_to_number(self):
        """TYPE_WIDENING maps (integer, number) -> True."""
        assert TYPE_WIDENING[("integer", "number")] is True

    def test_type_widening_contains_boolean_to_string(self):
        """TYPE_WIDENING maps (boolean, string) -> True."""
        assert TYPE_WIDENING[("boolean", "string")] is True

    def test_type_narrowing_contains_number_to_integer(self):
        """TYPE_NARROWING maps (number, integer) -> True."""
        assert TYPE_NARROWING[("number", "integer")] is True

    def test_type_narrowing_contains_string_to_integer(self):
        """TYPE_NARROWING maps (string, integer) -> True."""
        assert TYPE_NARROWING[("string", "integer")] is True

    def test_widening_and_narrowing_are_disjoint(self):
        """No (old_type, new_type) pair appears in both WIDENING and NARROWING."""
        widening_keys = set(TYPE_WIDENING.keys())
        narrowing_keys = set(TYPE_NARROWING.keys())
        assert widening_keys.isdisjoint(narrowing_keys)


# ===========================================================================
# TestValidateTransition
# ===========================================================================


class TestValidateTransition:
    """Tests for validate_transition policy enforcement."""

    def test_full_satisfies_full(self, engine):
        """Full level satisfies full requirement."""
        result = engine.validate_transition("full", "full")
        assert result["allowed"] is True

    def test_full_satisfies_backward(self, engine):
        """Full level satisfies backward requirement."""
        result = engine.validate_transition("full", "backward")
        assert result["allowed"] is True

    def test_full_satisfies_forward(self, engine):
        """Full level satisfies forward requirement."""
        result = engine.validate_transition("full", "forward")
        assert result["allowed"] is True

    def test_full_satisfies_breaking(self, engine):
        """Full level satisfies breaking requirement."""
        result = engine.validate_transition("full", "breaking")
        assert result["allowed"] is True

    def test_backward_does_not_satisfy_full(self, engine):
        """Backward level does not satisfy full requirement."""
        result = engine.validate_transition("backward", "full")
        assert result["allowed"] is False

    def test_breaking_does_not_satisfy_backward(self, engine):
        """Breaking level does not satisfy backward requirement."""
        result = engine.validate_transition("breaking", "backward")
        assert result["allowed"] is False

    def test_validate_transition_invalid_current_raises(self, engine):
        """Invalid current_level raises ValueError."""
        with pytest.raises(ValueError):
            engine.validate_transition("invalid", "full")

    def test_validate_transition_invalid_required_raises(self, engine):
        """Invalid required_level raises ValueError."""
        with pytest.raises(ValueError):
            engine.validate_transition("full", "invalid")

    def test_validate_transition_result_keys(self, engine):
        """Validate transition result contains expected keys."""
        result = engine.validate_transition("full", "backward")
        assert set(result.keys()) == {"allowed", "current_level", "required_level", "reason"}
        assert result["current_level"] == "full"
        assert result["required_level"] == "backward"


# ===========================================================================
# TestAutoDiff
# ===========================================================================


class TestAutoDiff:
    """Tests for the _auto_diff internal method."""

    def test_auto_diff_detects_added_field(self, engine, base_schema):
        """Auto-diff detects a new field in the target."""
        target = copy.deepcopy(base_schema)
        target["fields"]["tag"] = {"type": "string", "required": False}
        changes = engine._auto_diff(base_schema, target)
        change_types = [c["change_type"] for c in changes]
        assert "add_optional_field" in change_types

    def test_auto_diff_detects_removed_required_field(self, engine, base_schema):
        """Auto-diff detects a removed required field."""
        target = copy.deepcopy(base_schema)
        del target["fields"]["name"]
        changes = engine._auto_diff(base_schema, target)
        change_types = [c["change_type"] for c in changes]
        assert "remove_required_field" in change_types

    def test_auto_diff_detects_removed_optional_field(self, engine, base_schema):
        """Auto-diff detects a removed optional field."""
        target = copy.deepcopy(base_schema)
        del target["fields"]["email"]
        changes = engine._auto_diff(base_schema, target)
        change_types = [c["change_type"] for c in changes]
        assert "remove_optional_field" in change_types

    def test_auto_diff_detects_type_change(self, engine):
        """Auto-diff detects a type change as 'retyped'."""
        source = {"fields": {"val": {"type": "integer", "required": True}}}
        target = {"fields": {"val": {"type": "number", "required": True}}}
        changes = engine._auto_diff(source, target)
        change_types = [c["change_type"] for c in changes]
        assert "retyped" in change_types

    def test_auto_diff_detects_required_flag_change(self, engine):
        """Auto-diff detects required -> optional flag change."""
        source = {"fields": {"x": {"type": "string", "required": True}}}
        target = {"fields": {"x": {"type": "string", "required": False}}}
        changes = engine._auto_diff(source, target)
        change_types = [c["change_type"] for c in changes]
        assert "make_required_optional" in change_types

    def test_auto_diff_detects_default_change(self, engine):
        """Auto-diff detects default value change."""
        source = {"fields": {"x": {"type": "string", "required": True, "default": "a"}}}
        target = {"fields": {"x": {"type": "string", "required": True, "default": "b"}}}
        changes = engine._auto_diff(source, target)
        change_types = [c["change_type"] for c in changes]
        assert "change_default_value" in change_types

    def test_auto_diff_detects_enum_added(self, engine, schema_with_enums):
        """Auto-diff detects added enum values."""
        target = copy.deepcopy(schema_with_enums)
        target["fields"]["status"]["enum"] = ["active", "inactive", "pending"]
        changes = engine._auto_diff(schema_with_enums, target)
        change_types = [c["change_type"] for c in changes]
        assert "add_enum_value" in change_types

    def test_auto_diff_detects_enum_removed(self, engine, schema_with_enums):
        """Auto-diff detects removed enum values."""
        target = copy.deepcopy(schema_with_enums)
        target["fields"]["status"]["enum"] = ["active"]
        changes = engine._auto_diff(schema_with_enums, target)
        change_types = [c["change_type"] for c in changes]
        assert "remove_enum_value" in change_types

    def test_auto_diff_empty_to_empty(self, engine, empty_schema):
        """Auto-diff of two empty schemas produces no changes."""
        changes = engine._auto_diff(empty_schema, empty_schema)
        assert changes == []

    def test_auto_diff_add_required_with_default(self, engine):
        """Auto-diff classifies added required field with default correctly."""
        source = {"fields": {}}
        target = {"fields": {"role": {"type": "string", "required": True, "default": "user"}}}
        changes = engine._auto_diff(source, target)
        change_types = [c["change_type"] for c in changes]
        assert "add_required_field_with_default" in change_types

    def test_auto_diff_add_required_without_default(self, engine):
        """Auto-diff classifies added required field without default correctly."""
        source = {"fields": {}}
        target = {"fields": {"name": {"type": "string", "required": True}}}
        changes = engine._auto_diff(source, target)
        change_types = [c["change_type"] for c in changes]
        assert "add_required_field_without_default" in change_types
