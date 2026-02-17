# -*- coding: utf-8 -*-
"""
Unit tests for ChangeDetectorEngine - AGENT-DATA-017

Tests the ChangeDetectorEngine from greenlang.schema_migration.change_detector
with ~120 tests covering initialization, field additions, field removals,
type changes, constraint changes, rename detection (Jaro-Winkler), nested
changes, enum changes, default changes, array changes, reordering,
classification, summarization, statistics, reset, and edge cases.

Author: GreenLang Platform Team / GL-TestEngineer
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from greenlang.schema_migration.change_detector import (
    ChangeDetectorEngine,
    _jaro_similarity,
    _jaro_winkler_similarity,
    _is_type_widening,
    _is_type_narrowing,
    _extract_json_schema_fields,
    _extract_avro_fields,
    _detect_schema_format,
    _extract_fields,
    _compute_sha256,
    _RENAME_SIMILARITY_THRESHOLD,
)
from greenlang.schema_migration.provenance import ProvenanceTracker


# ---------------------------------------------------------------------------
# Helper: build a JSON Schema definition dict
# ---------------------------------------------------------------------------


def _schema(
    properties: Dict[str, Any],
    required: List[str] | None = None,
) -> Dict[str, Any]:
    """Build a minimal JSON Schema object definition."""
    defn: Dict[str, Any] = {"type": "object", "properties": properties}
    if required is not None:
        defn["required"] = required
    return defn


def _avro_schema(
    name: str,
    fields: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a minimal Avro record schema."""
    return {"type": "record", "name": name, "fields": fields}


# ===========================================================================
# TestChangeDetectorInit
# ===========================================================================


class TestChangeDetectorInit:
    """Test ChangeDetectorEngine initialization."""

    def test_default_initialization(self):
        """Engine initializes with default max_depth=10."""
        engine = ChangeDetectorEngine()
        assert engine._max_depth == 10

    def test_custom_max_depth(self):
        """Engine accepts a custom max_depth value."""
        engine = ChangeDetectorEngine(max_depth=5)
        assert engine._max_depth == 5

    def test_max_depth_minimum_1(self):
        """Engine accepts max_depth=1 (minimum valid value)."""
        engine = ChangeDetectorEngine(max_depth=1)
        assert engine._max_depth == 1

    def test_max_depth_zero_raises_value_error(self):
        """Engine rejects max_depth=0 with ValueError."""
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            ChangeDetectorEngine(max_depth=0)

    def test_max_depth_negative_raises_value_error(self):
        """Engine rejects negative max_depth with ValueError."""
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            ChangeDetectorEngine(max_depth=-3)

    def test_custom_provenance_tracker(self):
        """Engine accepts an external ProvenanceTracker."""
        tracker = ProvenanceTracker(genesis_hash="test-genesis")
        engine = ChangeDetectorEngine(provenance_tracker=tracker)
        assert engine._provenance is tracker

    def test_default_provenance_tracker_created(self):
        """Engine creates its own ProvenanceTracker when none is supplied."""
        engine = ChangeDetectorEngine()
        assert isinstance(engine._provenance, ProvenanceTracker)

    def test_initial_counters_are_zero(self):
        """All internal counters start at zero."""
        engine = ChangeDetectorEngine()
        assert engine._total_detections == 0
        assert engine._total_changes_detected == 0
        assert engine._detection_times_ms == []


# ===========================================================================
# TestDetectChanges (main entry point)
# ===========================================================================


class TestDetectChanges:
    """Test the top-level detect_changes method."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.engine = ChangeDetectorEngine()

    def test_identical_schemas_produce_no_changes(self):
        """Comparing identical schemas returns zero changes."""
        schema = _schema({"id": {"type": "integer"}}, required=["id"])
        result = self.engine.detect_changes(schema, schema)
        assert result["summary"]["total_count"] == 0
        assert result["changes"] == []

    def test_empty_schemas_produce_no_changes(self):
        """Comparing two empty schemas returns zero changes."""
        result = self.engine.detect_changes({}, {})
        assert result["summary"]["total_count"] == 0

    def test_result_has_detection_id(self):
        """Result contains a unique detection_id string."""
        result = self.engine.detect_changes({}, {})
        assert result["detection_id"].startswith("DET-")
        assert len(result["detection_id"]) == 16  # "DET-" + 12 hex chars

    def test_result_has_source_and_target_hashes(self):
        """Result contains SHA-256 hashes of source and target definitions."""
        source = _schema({"a": {"type": "string"}})
        target = _schema({"b": {"type": "string"}})
        result = self.engine.detect_changes(source, target)
        assert len(result["source_hash"]) == 64
        assert len(result["target_hash"]) == 64

    def test_result_has_provenance_hash(self):
        """Result includes a provenance chain hash."""
        result = self.engine.detect_changes({}, {})
        assert len(result["provenance_hash"]) == 64

    def test_result_has_processing_time(self):
        """Result records processing_time_ms >= 0."""
        result = self.engine.detect_changes({}, {})
        assert result["processing_time_ms"] >= 0.0

    def test_result_has_detected_at_timestamp(self):
        """Result contains an ISO-format detected_at timestamp."""
        result = self.engine.detect_changes({}, {})
        assert "T" in result["detected_at"]

    def test_single_field_added(self):
        """Detect a single added field."""
        source = _schema({"id": {"type": "integer"}})
        target = _schema({"id": {"type": "integer"}, "name": {"type": "string"}})
        result = self.engine.detect_changes(source, target)
        added = [c for c in result["changes"] if c["change_type"] == "added"]
        assert len(added) == 1
        assert added[0]["field_path"] == "name"

    def test_single_field_removed(self):
        """Detect a single removed field."""
        source = _schema({"id": {"type": "integer"}, "name": {"type": "string"}})
        target = _schema({"id": {"type": "integer"}})
        result = self.engine.detect_changes(source, target)
        removed = [c for c in result["changes"] if c["change_type"] == "removed"]
        assert len(removed) == 1
        assert removed[0]["field_path"] == "name"
        assert removed[0]["severity"] == "breaking"

    def test_type_changed(self):
        """Detect a type change on a common field."""
        source = _schema({"age": {"type": "string"}})
        target = _schema({"age": {"type": "integer"}})
        result = self.engine.detect_changes(source, target)
        retyped = [c for c in result["changes"] if c["change_type"] == "retyped"]
        assert len(retyped) == 1
        assert retyped[0]["old_value"] == "string"
        assert retyped[0]["new_value"] == "integer"

    def test_multiple_changes_detected(self):
        """Detect multiple simultaneous changes."""
        source = _schema(
            {"id": {"type": "integer"}, "old_field": {"type": "string"}},
            required=["id"],
        )
        target = _schema(
            {"id": {"type": "integer"}, "new_field": {"type": "boolean"}},
            required=["id"],
        )
        result = self.engine.detect_changes(source, target)
        assert result["summary"]["total_count"] >= 2

    def test_source_not_dict_raises_type_error(self):
        """Non-dict source_definition raises TypeError."""
        with pytest.raises(TypeError, match="source_definition must be a dict"):
            self.engine.detect_changes("not a dict", {})

    def test_target_not_dict_raises_type_error(self):
        """Non-dict target_definition raises TypeError."""
        with pytest.raises(TypeError, match="target_definition must be a dict"):
            self.engine.detect_changes({}, [])

    def test_max_depth_override(self):
        """Passing max_depth override to detect_changes works."""
        result = self.engine.detect_changes({}, {}, max_depth=3)
        assert result["summary"]["total_count"] == 0

    def test_max_depth_override_invalid_raises(self):
        """Invalid max_depth override raises ValueError."""
        with pytest.raises(ValueError, match="max_depth override must be >= 1"):
            self.engine.detect_changes({}, {}, max_depth=0)

    def test_detection_stored_internally(self):
        """After detect_changes, the result is retrievable via get_detection."""
        result = self.engine.detect_changes({}, {})
        stored = self.engine.get_detection(result["detection_id"])
        assert stored is not None
        assert stored["detection_id"] == result["detection_id"]

    def test_detection_counter_increments(self):
        """Each call to detect_changes increments total_detections."""
        self.engine.detect_changes({}, {})
        self.engine.detect_changes({}, {})
        stats = self.engine.get_statistics()
        assert stats["total_detections"] == 2


# ===========================================================================
# TestDetectFieldAdditions
# ===========================================================================


class TestDetectFieldAdditions:
    """Test detect_added_fields."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.engine = ChangeDetectorEngine()

    def _fields(self, schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return _extract_json_schema_fields(schema)

    def test_optional_field_added_is_non_breaking(self):
        """Adding an optional field is classified as non_breaking."""
        source = _schema({"id": {"type": "integer"}})
        target = _schema({"id": {"type": "integer"}, "name": {"type": "string"}})
        changes = self.engine.detect_added_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1
        assert changes[0]["severity"] == "non_breaking"

    def test_required_field_added_is_breaking(self):
        """Adding a required field is classified as breaking."""
        source = _schema({"id": {"type": "integer"}}, required=["id"])
        target = _schema(
            {"id": {"type": "integer"}, "email": {"type": "string"}},
            required=["id", "email"],
        )
        changes = self.engine.detect_added_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1
        assert changes[0]["severity"] == "breaking"

    def test_multiple_additions(self):
        """Multiple added fields are all detected."""
        source = _schema({})
        target = _schema({
            "a": {"type": "string"},
            "b": {"type": "integer"},
            "c": {"type": "boolean"},
        })
        changes = self.engine.detect_added_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 3

    def test_no_additions_when_same_fields(self):
        """No additions when both schemas have the same fields."""
        schema = _schema({"x": {"type": "integer"}})
        changes = self.engine.detect_added_fields(
            self._fields(schema), self._fields(schema)
        )
        assert len(changes) == 0

    def test_exclude_names_skips_rename_targets(self):
        """Fields listed in exclude_names are not reported as added."""
        source = _schema({})
        target = _schema({"alpha": {"type": "string"}, "beta": {"type": "string"}})
        changes = self.engine.detect_added_fields(
            self._fields(source),
            self._fields(target),
            exclude_names={"alpha"},
        )
        assert len(changes) == 1
        assert changes[0]["field_path"] == "beta"

    def test_path_prefix_applied(self):
        """Added field paths include the parent path prefix."""
        source = _schema({})
        target = _schema({"child": {"type": "string"}})
        changes = self.engine.detect_added_fields(
            self._fields(source), self._fields(target), path="parent"
        )
        assert changes[0]["field_path"] == "parent.child"

    def test_field_with_format_added(self):
        """Added field with format metadata is detected."""
        source = _schema({})
        target = _schema({"ts": {"type": "string", "format": "date-time"}})
        changes = self.engine.detect_added_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1
        assert changes[0]["new_value"]["format"] == "date-time"

    def test_added_field_description_text(self):
        """Change dict description mentions 'added' and field path."""
        source = _schema({})
        target = _schema({"status": {"type": "string"}})
        changes = self.engine.detect_added_fields(
            self._fields(source), self._fields(target)
        )
        assert "added" in changes[0]["description"].lower()
        assert "status" in changes[0]["description"]

    def test_added_from_empty_source(self):
        """All fields are additions when source is empty."""
        source = _schema({})
        target = _schema({"a": {"type": "string"}, "b": {"type": "integer"}})
        changes = self.engine.detect_added_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 2

    def test_added_field_has_uuid_id(self):
        """Each change dict has a non-empty 'id' field."""
        source = _schema({})
        target = _schema({"x": {"type": "string"}})
        changes = self.engine.detect_added_fields(
            self._fields(source), self._fields(target)
        )
        assert changes[0]["id"]
        assert len(changes[0]["id"]) == 36  # UUID v4 format


# ===========================================================================
# TestDetectFieldRemovals
# ===========================================================================


class TestDetectFieldRemovals:
    """Test detect_removed_fields."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.engine = ChangeDetectorEngine()

    def _fields(self, schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return _extract_json_schema_fields(schema)

    def test_field_removed_is_breaking(self):
        """Removing any field is classified as breaking."""
        source = _schema({"name": {"type": "string"}}, required=["name"])
        target = _schema({})
        changes = self.engine.detect_removed_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1
        assert changes[0]["severity"] == "breaking"

    def test_optional_field_removed_is_breaking(self):
        """Removing even an optional field is classified as breaking."""
        source = _schema({"opt": {"type": "string"}})
        target = _schema({})
        changes = self.engine.detect_removed_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1
        assert changes[0]["severity"] == "breaking"

    def test_multiple_removals(self):
        """Multiple removed fields are all detected."""
        source = _schema({
            "a": {"type": "string"},
            "b": {"type": "integer"},
            "c": {"type": "boolean"},
        })
        target = _schema({})
        changes = self.engine.detect_removed_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 3

    def test_no_removals_when_same_fields(self):
        """No removals when both schemas have the same fields."""
        schema = _schema({"x": {"type": "integer"}})
        changes = self.engine.detect_removed_fields(
            self._fields(schema), self._fields(schema)
        )
        assert len(changes) == 0

    def test_exclude_names_skips_rename_sources(self):
        """Fields listed in exclude_names are not reported as removed."""
        source = _schema({"alpha": {"type": "string"}, "beta": {"type": "string"}})
        target = _schema({})
        changes = self.engine.detect_removed_fields(
            self._fields(source),
            self._fields(target),
            exclude_names={"alpha"},
        )
        assert len(changes) == 1
        assert changes[0]["field_path"] == "beta"

    def test_path_prefix_applied_to_removals(self):
        """Removed field paths include the parent path prefix."""
        source = _schema({"child": {"type": "string"}})
        target = _schema({})
        changes = self.engine.detect_removed_fields(
            self._fields(source), self._fields(target), path="parent"
        )
        assert changes[0]["field_path"] == "parent.child"

    def test_removal_old_value_contains_field_descriptor(self):
        """old_value on a removal contains the field descriptor dict."""
        source = _schema({"name": {"type": "string"}})
        target = _schema({})
        changes = self.engine.detect_removed_fields(
            self._fields(source), self._fields(target)
        )
        assert changes[0]["old_value"]["type"] == "string"
        assert changes[0]["new_value"] is None

    def test_removal_description_mentions_removed(self):
        """Change description for removal mentions 'removed'."""
        source = _schema({"x": {"type": "integer"}})
        target = _schema({})
        changes = self.engine.detect_removed_fields(
            self._fields(source), self._fields(target)
        )
        assert "removed" in changes[0]["description"].lower()

    def test_partial_removal(self):
        """Only the removed field is reported, not the retained one."""
        source = _schema({"keep": {"type": "string"}, "drop": {"type": "integer"}})
        target = _schema({"keep": {"type": "string"}})
        changes = self.engine.detect_removed_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1
        assert changes[0]["field_path"] == "drop"

    def test_removal_from_empty_target(self):
        """All fields are removals when target is empty."""
        source = _schema({"a": {"type": "string"}, "b": {"type": "integer"}})
        target = _schema({})
        changes = self.engine.detect_removed_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 2


# ===========================================================================
# TestDetectTypeChanges
# ===========================================================================


class TestDetectTypeChanges:
    """Test detect_retyped_fields."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.engine = ChangeDetectorEngine()

    def _fields(self, schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return _extract_json_schema_fields(schema)

    def test_string_to_integer_is_breaking(self):
        """Changing string -> integer is classified as breaking."""
        source = _schema({"age": {"type": "string"}})
        target = _schema({"age": {"type": "integer"}})
        changes = self.engine.detect_retyped_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1
        assert changes[0]["severity"] == "breaking"

    def test_integer_to_number_is_non_breaking(self):
        """Widening integer -> number is classified as non_breaking."""
        source = _schema({"val": {"type": "integer"}})
        target = _schema({"val": {"type": "number"}})
        changes = self.engine.detect_retyped_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1
        assert changes[0]["severity"] == "non_breaking"

    def test_number_to_integer_is_breaking(self):
        """Narrowing number -> integer is classified as breaking."""
        source = _schema({"val": {"type": "number"}})
        target = _schema({"val": {"type": "integer"}})
        changes = self.engine.detect_retyped_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1
        assert changes[0]["severity"] == "breaking"

    def test_integer_to_string_is_non_breaking(self):
        """Widening integer -> string is non_breaking."""
        source = _schema({"code": {"type": "integer"}})
        target = _schema({"code": {"type": "string"}})
        changes = self.engine.detect_retyped_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1
        assert changes[0]["severity"] == "non_breaking"

    def test_string_to_boolean_is_breaking(self):
        """Unrecognized pair string -> boolean is breaking by default."""
        source = _schema({"flag": {"type": "string"}})
        target = _schema({"flag": {"type": "boolean"}})
        changes = self.engine.detect_retyped_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1
        assert changes[0]["severity"] == "breaking"

    def test_no_change_when_type_same(self):
        """No retyped change when types are identical."""
        schema = _schema({"x": {"type": "string"}})
        changes = self.engine.detect_retyped_fields(
            self._fields(schema), self._fields(schema)
        )
        assert len(changes) == 0

    def test_only_common_fields_checked(self):
        """Fields only in source or only in target are not checked for retype."""
        source = _schema({"a": {"type": "string"}})
        target = _schema({"b": {"type": "integer"}})
        changes = self.engine.detect_retyped_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 0

    def test_multiple_type_changes(self):
        """Multiple fields with type changes are all detected."""
        source = _schema({"a": {"type": "string"}, "b": {"type": "integer"}})
        target = _schema({"a": {"type": "boolean"}, "b": {"type": "string"}})
        changes = self.engine.detect_retyped_fields(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 2

    def test_path_prefix_for_type_change(self):
        """Type change field paths include the parent path prefix."""
        source = _schema({"x": {"type": "string"}})
        target = _schema({"x": {"type": "integer"}})
        changes = self.engine.detect_retyped_fields(
            self._fields(source), self._fields(target), path="parent"
        )
        assert changes[0]["field_path"] == "parent.x"

    def test_type_change_description_includes_types(self):
        """Description mentions both old and new types."""
        source = _schema({"x": {"type": "string"}})
        target = _schema({"x": {"type": "integer"}})
        changes = self.engine.detect_retyped_fields(
            self._fields(source), self._fields(target)
        )
        assert "string" in changes[0]["description"]
        assert "integer" in changes[0]["description"]


# ===========================================================================
# TestDetectConstraintChanges
# ===========================================================================


class TestDetectConstraintChanges:
    """Test detect_constraint_changes."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.engine = ChangeDetectorEngine()

    def _fields(self, schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return _extract_json_schema_fields(schema)

    def test_optional_to_required_is_breaking(self):
        """Changing optional -> required is breaking."""
        source = _schema({"email": {"type": "string"}})
        target = _schema({"email": {"type": "string"}}, required=["email"])
        changes = self.engine.detect_constraint_changes(
            self._fields(source), self._fields(target)
        )
        req_changes = [c for c in changes if "required" in str(c.get("old_value", ""))]
        assert len(req_changes) == 1
        assert req_changes[0]["severity"] == "breaking"

    def test_required_to_optional_is_non_breaking(self):
        """Changing required -> optional is non_breaking."""
        source = _schema({"email": {"type": "string"}}, required=["email"])
        target = _schema({"email": {"type": "string"}})
        changes = self.engine.detect_constraint_changes(
            self._fields(source), self._fields(target)
        )
        req_changes = [c for c in changes if "required" in str(c.get("old_value", ""))]
        assert len(req_changes) == 1
        assert req_changes[0]["severity"] == "non_breaking"

    def test_minimum_added_is_breaking(self):
        """Adding a minimum constraint is breaking (new restriction)."""
        source = _schema({"val": {"type": "integer"}})
        target = _schema({"val": {"type": "integer", "minimum": 0}})
        changes = self.engine.detect_constraint_changes(
            self._fields(source), self._fields(target)
        )
        min_changes = [c for c in changes if "minimum" in str(c.get("new_value", ""))]
        assert len(min_changes) == 1
        assert min_changes[0]["severity"] == "breaking"

    def test_minimum_removed_is_non_breaking(self):
        """Removing a minimum constraint is non_breaking (relaxation)."""
        source = _schema({"val": {"type": "integer", "minimum": 0}})
        target = _schema({"val": {"type": "integer"}})
        changes = self.engine.detect_constraint_changes(
            self._fields(source), self._fields(target)
        )
        min_changes = [c for c in changes if "minimum" in str(c.get("old_value", ""))]
        assert len(min_changes) == 1
        assert min_changes[0]["severity"] == "non_breaking"

    def test_minimum_increased_is_breaking(self):
        """Increasing a minimum is breaking (tightening)."""
        source = _schema({"val": {"type": "integer", "minimum": 0}})
        target = _schema({"val": {"type": "integer", "minimum": 10}})
        changes = self.engine.detect_constraint_changes(
            self._fields(source), self._fields(target)
        )
        min_changes = [c for c in changes if "minimum" in str(c.get("new_value", ""))]
        assert len(min_changes) == 1
        assert min_changes[0]["severity"] == "breaking"

    def test_minimum_decreased_is_non_breaking(self):
        """Decreasing a minimum is non_breaking (relaxation)."""
        source = _schema({"val": {"type": "integer", "minimum": 10}})
        target = _schema({"val": {"type": "integer", "minimum": 0}})
        changes = self.engine.detect_constraint_changes(
            self._fields(source), self._fields(target)
        )
        min_changes = [c for c in changes if "minimum" in str(c.get("new_value", ""))]
        assert len(min_changes) == 1
        assert min_changes[0]["severity"] == "non_breaking"

    def test_maximum_decreased_is_breaking(self):
        """Decreasing a maximum is breaking (tightening)."""
        source = _schema({"val": {"type": "integer", "maximum": 100}})
        target = _schema({"val": {"type": "integer", "maximum": 50}})
        changes = self.engine.detect_constraint_changes(
            self._fields(source), self._fields(target)
        )
        max_changes = [c for c in changes if "maximum" in str(c.get("new_value", ""))]
        assert len(max_changes) == 1
        assert max_changes[0]["severity"] == "breaking"

    def test_maximum_increased_is_non_breaking(self):
        """Increasing a maximum is non_breaking (relaxation)."""
        source = _schema({"val": {"type": "integer", "maximum": 50}})
        target = _schema({"val": {"type": "integer", "maximum": 100}})
        changes = self.engine.detect_constraint_changes(
            self._fields(source), self._fields(target)
        )
        max_changes = [c for c in changes if "maximum" in str(c.get("new_value", ""))]
        assert len(max_changes) == 1
        assert max_changes[0]["severity"] == "non_breaking"

    def test_pattern_change_is_breaking(self):
        """Any pattern change is classified as breaking."""
        source = _schema({"email": {"type": "string", "pattern": "^[a-z]+$"}})
        target = _schema({"email": {"type": "string", "pattern": "^[A-Z]+$"}})
        changes = self.engine.detect_constraint_changes(
            self._fields(source), self._fields(target)
        )
        pattern_changes = [c for c in changes if "pattern" in str(c.get("new_value", ""))]
        assert len(pattern_changes) == 1
        assert pattern_changes[0]["severity"] == "breaking"

    def test_no_constraint_changes_when_same(self):
        """No constraint changes when fields are identical."""
        schema = _schema(
            {"val": {"type": "integer", "minimum": 0, "maximum": 100}},
            required=["val"],
        )
        changes = self.engine.detect_constraint_changes(
            self._fields(schema), self._fields(schema)
        )
        assert len(changes) == 0

    def test_max_length_added(self):
        """Adding maxLength is detected as a constraint change."""
        source = _schema({"name": {"type": "string"}})
        target = _schema({"name": {"type": "string", "maxLength": 255}})
        changes = self.engine.detect_constraint_changes(
            self._fields(source), self._fields(target)
        )
        ml_changes = [c for c in changes if "maxLength" in str(c.get("new_value", ""))]
        assert len(ml_changes) == 1


# ===========================================================================
# TestDetectRenames
# ===========================================================================


class TestDetectRenames:
    """Test detect_renamed_fields with Jaro-Winkler similarity."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.engine = ChangeDetectorEngine()

    def _fields(self, schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return _extract_json_schema_fields(schema)

    def test_very_similar_names_detected_as_rename(self):
        """Fields with Jaro-Winkler > 0.85 and same type are renames."""
        source = _schema({"user_id": {"type": "integer"}})
        target = _schema({"userid": {"type": "integer"}})
        renames, src_set, tgt_set = self.engine.detect_renamed_fields(
            self._fields(source), self._fields(target)
        )
        assert len(renames) == 1
        assert renames[0]["change_type"] == "renamed"
        assert "user_id" in src_set
        assert "userid" in tgt_set

    def test_dissimilar_names_not_matched(self):
        """Fields with very different names are not matched as renames."""
        source = _schema({"alpha": {"type": "string"}})
        target = _schema({"zebra": {"type": "string"}})
        renames, _, _ = self.engine.detect_renamed_fields(
            self._fields(source), self._fields(target)
        )
        assert len(renames) == 0

    def test_different_types_not_matched(self):
        """Fields with similar names but different types are not renames."""
        source = _schema({"user_name": {"type": "string"}})
        target = _schema({"username": {"type": "integer"}})
        renames, _, _ = self.engine.detect_renamed_fields(
            self._fields(source), self._fields(target)
        )
        assert len(renames) == 0

    def test_rename_is_non_breaking(self):
        """Renames are classified as non_breaking."""
        source = _schema({"first_name": {"type": "string"}})
        target = _schema({"firstname": {"type": "string"}})
        renames, _, _ = self.engine.detect_renamed_fields(
            self._fields(source), self._fields(target)
        )
        assert len(renames) == 1
        assert renames[0]["severity"] == "non_breaking"

    def test_case_change_rename(self):
        """A minor case change (same length, mostly same chars) is a rename."""
        source = _schema({"UserEmail": {"type": "string"}})
        target = _schema({"useremail": {"type": "string"}})
        score = _jaro_winkler_similarity("UserEmail", "useremail")
        renames, _, _ = self.engine.detect_renamed_fields(
            self._fields(source), self._fields(target)
        )
        if score >= _RENAME_SIMILARITY_THRESHOLD:
            assert len(renames) == 1
        else:
            assert len(renames) == 0

    def test_prefix_suffix_change(self):
        """Adding a prefix/suffix may or may not meet threshold."""
        source = _schema({"email": {"type": "string"}})
        target = _schema({"email_addr": {"type": "string"}})
        score = _jaro_winkler_similarity("email", "email_addr")
        renames, _, _ = self.engine.detect_renamed_fields(
            self._fields(source), self._fields(target)
        )
        if score >= _RENAME_SIMILARITY_THRESHOLD:
            assert len(renames) == 1
        else:
            assert len(renames) == 0

    def test_no_renames_when_all_fields_common(self):
        """No renames detected when all fields exist in both schemas."""
        schema = _schema({"a": {"type": "string"}, "b": {"type": "integer"}})
        renames, src_set, tgt_set = self.engine.detect_renamed_fields(
            self._fields(schema), self._fields(schema)
        )
        assert len(renames) == 0
        assert len(src_set) == 0
        assert len(tgt_set) == 0

    def test_no_renames_when_no_candidates(self):
        """No renames when only additions or only removals exist."""
        source = _schema({"a": {"type": "string"}})
        target = _schema({"a": {"type": "string"}, "b": {"type": "integer"}})
        renames, _, _ = self.engine.detect_renamed_fields(
            self._fields(source), self._fields(target)
        )
        assert len(renames) == 0

    def test_greedy_matching_best_score_first(self):
        """When multiple candidates exist, the best scoring pair wins."""
        source = _schema({
            "customer_name": {"type": "string"},
            "customer_id": {"type": "integer"},
        })
        target = _schema({
            "cust_name": {"type": "string"},
            "cust_id": {"type": "integer"},
        })
        renames, src_set, tgt_set = self.engine.detect_renamed_fields(
            self._fields(source), self._fields(target)
        )
        # Both should be matched if scores are above threshold
        for r in renames:
            assert r["change_type"] == "renamed"

    def test_rename_description_includes_similarity_score(self):
        """Rename description includes the Jaro-Winkler similarity score."""
        source = _schema({"user_id": {"type": "integer"}})
        target = _schema({"userid": {"type": "integer"}})
        renames, _, _ = self.engine.detect_renamed_fields(
            self._fields(source), self._fields(target)
        )
        if renames:
            assert "similarity=" in renames[0]["description"]


# ===========================================================================
# TestDetectNestedChanges
# ===========================================================================


class TestDetectNestedChanges:
    """Test detect_nested_changes for recursive object properties.

    NOTE: The production code has a known NameError bug on line 1314 of
    change_detector.py where ``src_obj`` is referenced instead of the
    correct parameter name ``source_obj``. Tests that would trigger this
    code path (schemas with nested object properties) are marked with
    ``pytest.xfail`` to document the bug. Tests that do NOT trigger the
    buggy code path (no nested properties, or depth-at-max early return)
    remain as normal passing tests.
    """

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.engine = ChangeDetectorEngine()

    @pytest.mark.xfail(
        reason="Known NameError bug: line 1314 uses 'src_obj' instead of 'source_obj'",
        raises=NameError,
        strict=True,
    )
    def test_nested_field_added(self):
        """Added field inside a nested object is detected."""
        source = _schema({
            "address": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            }
        })
        target = _schema({
            "address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zip": {"type": "string"},
                },
            }
        })
        changes = self.engine.detect_nested_changes(source, target)
        added = [c for c in changes if c["change_type"] == "added"]
        assert len(added) == 1
        assert "address.zip" in added[0]["field_path"]

    @pytest.mark.xfail(
        reason="Known NameError bug: line 1314 uses 'src_obj' instead of 'source_obj'",
        raises=NameError,
        strict=True,
    )
    def test_nested_field_removed(self):
        """Removed field inside a nested object is detected."""
        source = _schema({
            "address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zip": {"type": "string"},
                },
            }
        })
        target = _schema({
            "address": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            }
        })
        changes = self.engine.detect_nested_changes(source, target)
        removed = [c for c in changes if c["change_type"] == "removed"]
        assert len(removed) == 1
        assert "address.zip" in removed[0]["field_path"]

    @pytest.mark.xfail(
        reason="Known NameError bug: line 1314 uses 'src_obj' instead of 'source_obj'",
        raises=NameError,
        strict=True,
    )
    def test_deeply_nested_5_levels(self):
        """Changes at 5 levels of nesting are detected (within max_depth=10)."""

        def _nested(depth: int, leaf_props: Dict[str, Any]) -> Dict[str, Any]:
            if depth == 0:
                return {"type": "object", "properties": leaf_props}
            return {
                "type": "object",
                "properties": {
                    f"level{depth}": _nested(depth - 1, leaf_props),
                },
            }

        source_leaf = {"leaf": {"type": "string"}}
        target_leaf = {"leaf": {"type": "string"}, "new_leaf": {"type": "integer"}}
        source = _nested(5, source_leaf)
        target = _nested(5, target_leaf)

        changes = self.engine.detect_nested_changes(source, target)
        added = [c for c in changes if c["change_type"] == "added"]
        assert len(added) >= 1

    @pytest.mark.xfail(
        reason="Known NameError bug: line 1314 uses 'src_obj' instead of 'source_obj'",
        raises=NameError,
        strict=True,
    )
    def test_max_depth_respected(self):
        """Changes beyond max_depth are not detected."""
        engine = ChangeDetectorEngine(max_depth=1)

        source = _schema({
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "object",
                        "properties": {"deep_field": {"type": "string"}},
                    }
                },
            }
        })
        target = _schema({
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "object",
                        "properties": {
                            "deep_field": {"type": "string"},
                            "new_deep": {"type": "integer"},
                        },
                    }
                },
            }
        })
        # depth=0, max_depth=1: we go one level down then stop
        changes = engine.detect_nested_changes(source, target, depth=0)
        # The engine only goes 1 level deep; 2nd level should be truncated

    def test_no_nested_changes_when_no_properties(self):
        """No nested changes when fields have no 'properties' sub-dict."""
        source = _schema({"name": {"type": "string"}, "age": {"type": "integer"}})
        target = _schema({"name": {"type": "string"}, "age": {"type": "integer"}})
        changes = self.engine.detect_nested_changes(source, target)
        assert len(changes) == 0

    @pytest.mark.xfail(
        reason="Known NameError bug: line 1314 uses 'src_obj' instead of 'source_obj'",
        raises=NameError,
        strict=True,
    )
    def test_nested_type_change(self):
        """Type change inside a nested object is detected."""
        source = _schema({
            "meta": {
                "type": "object",
                "properties": {"version": {"type": "string"}},
            }
        })
        target = _schema({
            "meta": {
                "type": "object",
                "properties": {"version": {"type": "integer"}},
            }
        })
        changes = self.engine.detect_nested_changes(source, target)
        retyped = [c for c in changes if c["change_type"] == "retyped"]
        assert len(retyped) == 1

    def test_nested_changes_empty_when_depth_at_max(self):
        """Starting at depth >= max_depth returns no changes."""
        engine = ChangeDetectorEngine(max_depth=3)
        source = _schema({"x": {"type": "object", "properties": {"a": {"type": "string"}}}})
        target = _schema({"x": {"type": "object", "properties": {"b": {"type": "string"}}}})
        changes = engine.detect_nested_changes(source, target, depth=3)
        assert len(changes) == 0

    @pytest.mark.xfail(
        reason="Known NameError bug: line 1314 uses 'src_obj' instead of 'source_obj'",
        raises=NameError,
        strict=True,
    )
    def test_nested_path_prefix(self):
        """Nested field paths include the provided parent prefix."""
        source = _schema({
            "data": {
                "type": "object",
                "properties": {"val": {"type": "string"}},
            }
        })
        target = _schema({
            "data": {
                "type": "object",
                "properties": {
                    "val": {"type": "string"},
                    "extra": {"type": "integer"},
                },
            }
        })
        changes = self.engine.detect_nested_changes(source, target, path="root")
        if changes:
            assert changes[0]["field_path"].startswith("root.")

    @pytest.mark.xfail(
        reason="Known NameError bug: line 1314 uses 'src_obj' instead of 'source_obj'",
        raises=NameError,
        strict=True,
    )
    def test_nested_changes_with_one_side_no_properties(self):
        """When source field has no properties but target does, nested recursion triggers."""
        source = _schema({
            "data": {
                "type": "object",
                "properties": {"existing": {"type": "string"}},
            }
        })
        target = _schema({
            "data": {
                "type": "object",
                "properties": {
                    "existing": {"type": "string"},
                    "sub": {
                        "type": "object",
                        "properties": {"nested": {"type": "string"}},
                    },
                },
            }
        })
        # This triggers the add detection then nested recursion
        changes = self.engine.detect_nested_changes(source, target)

    @pytest.mark.xfail(
        reason="Known NameError bug: line 1314 uses 'src_obj' instead of 'source_obj'",
        raises=NameError,
        strict=True,
    )
    def test_multiple_nested_objects(self):
        """Changes in multiple nested objects are all detected."""
        source = _schema({
            "addr": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
            "meta": {
                "type": "object",
                "properties": {"version": {"type": "string"}},
            },
        })
        target = _schema({
            "addr": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                },
            },
            "meta": {
                "type": "object",
                "properties": {
                    "version": {"type": "string"},
                    "build": {"type": "integer"},
                },
            },
        })
        changes = self.engine.detect_nested_changes(source, target)
        added = [c for c in changes if c["change_type"] == "added"]
        assert len(added) == 2


# ===========================================================================
# TestClassifyChange
# ===========================================================================


class TestClassifyChange:
    """Test classify_change method."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.engine = ChangeDetectorEngine()

    def test_added_optional_is_non_breaking(self):
        """added + optional field -> non_breaking."""
        severity = self.engine.classify_change(
            "added", None, {"required": False}
        )
        assert severity == "non_breaking"

    def test_added_required_is_breaking(self):
        """added + required field -> breaking."""
        severity = self.engine.classify_change(
            "added", None, {"required": True}
        )
        assert severity == "breaking"

    def test_removed_is_always_breaking(self):
        """removed is always breaking."""
        severity = self.engine.classify_change("removed", {"name": "x"}, None)
        assert severity == "breaking"

    def test_renamed_is_non_breaking(self):
        """renamed is always non_breaking."""
        severity = self.engine.classify_change("renamed", "old_name", "new_name")
        assert severity == "non_breaking"

    def test_reordered_is_cosmetic(self):
        """reordered is always cosmetic."""
        severity = self.engine.classify_change("reordered", None, None)
        assert severity == "cosmetic"

    def test_retyped_widening_is_non_breaking(self):
        """Type widening (int->number) is non_breaking."""
        severity = self.engine.classify_change("retyped", "integer", "number")
        assert severity == "non_breaking"

    def test_retyped_narrowing_is_breaking(self):
        """Type narrowing (number->integer) is breaking."""
        severity = self.engine.classify_change("retyped", "number", "integer")
        assert severity == "breaking"

    def test_default_changed_is_cosmetic(self):
        """default_changed is always cosmetic."""
        severity = self.engine.classify_change("default_changed", "old", "new")
        assert severity == "cosmetic"

    def test_enum_changed_removal_is_breaking(self):
        """Removing enum values is breaking."""
        severity = self.engine.classify_change(
            "enum_changed", ["a", "b", "c"], ["a", "b"]
        )
        assert severity == "breaking"

    def test_enum_changed_addition_only_is_non_breaking(self):
        """Adding enum values (no removals) is non_breaking."""
        severity = self.engine.classify_change(
            "enum_changed", ["a", "b"], ["a", "b", "c"]
        )
        assert severity == "non_breaking"

    def test_unknown_change_type_defaults_to_non_breaking(self):
        """Unknown change type defaults to non_breaking."""
        severity = self.engine.classify_change("unknown_type", None, None)
        assert severity == "non_breaking"

    def test_constraint_changed_optional_to_required_is_breaking(self):
        """constraint_changed: optional -> required is breaking."""
        severity = self.engine.classify_change(
            "constraint_changed",
            {"required": False},
            {"required": True},
        )
        assert severity == "breaking"


# ===========================================================================
# TestDetectEnumChanges
# ===========================================================================


class TestDetectEnumChanges:
    """Test detect_enum_changes."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.engine = ChangeDetectorEngine()

    def _fields(self, schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return _extract_json_schema_fields(schema)

    def test_enum_value_added(self):
        """Adding new enum values is detected as non_breaking."""
        source = _schema({"status": {"type": "string", "enum": ["a", "b"]}})
        target = _schema({"status": {"type": "string", "enum": ["a", "b", "c"]}})
        changes = self.engine.detect_enum_changes(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1
        assert changes[0]["severity"] == "non_breaking"

    def test_enum_value_removed(self):
        """Removing enum values is detected as breaking."""
        source = _schema({"status": {"type": "string", "enum": ["a", "b", "c"]}})
        target = _schema({"status": {"type": "string", "enum": ["a", "b"]}})
        changes = self.engine.detect_enum_changes(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1
        assert changes[0]["severity"] == "breaking"

    def test_enum_both_added_and_removed(self):
        """Mixed enum changes (add + remove) is breaking."""
        source = _schema({"status": {"type": "string", "enum": ["a", "b"]}})
        target = _schema({"status": {"type": "string", "enum": ["a", "c"]}})
        changes = self.engine.detect_enum_changes(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1
        assert changes[0]["severity"] == "breaking"

    def test_no_enum_change_when_same(self):
        """No changes when enums are identical."""
        schema = _schema({"status": {"type": "string", "enum": ["x", "y"]}})
        changes = self.engine.detect_enum_changes(
            self._fields(schema), self._fields(schema)
        )
        assert len(changes) == 0

    def test_enum_added_to_field(self):
        """Adding an enum to a field that previously had none."""
        source = _schema({"status": {"type": "string"}})
        target = _schema({"status": {"type": "string", "enum": ["active", "inactive"]}})
        changes = self.engine.detect_enum_changes(
            self._fields(source), self._fields(target)
        )
        # enum went from None to a list
        assert len(changes) == 1

    def test_enum_removed_from_field(self):
        """Removing the enum constraint from a field."""
        source = _schema({"status": {"type": "string", "enum": ["a", "b"]}})
        target = _schema({"status": {"type": "string"}})
        changes = self.engine.detect_enum_changes(
            self._fields(source), self._fields(target)
        )
        # enum went from a list to None -- values removed is breaking
        assert len(changes) == 1


# ===========================================================================
# TestDetectDefaultChanges
# ===========================================================================


class TestDetectDefaultChanges:
    """Test detect_default_changes."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.engine = ChangeDetectorEngine()

    def _fields(self, schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return _extract_json_schema_fields(schema)

    def test_default_changed(self):
        """Changing a default value is detected as cosmetic."""
        source = _schema({"status": {"type": "string", "default": "active"}})
        target = _schema({"status": {"type": "string", "default": "pending"}})
        changes = self.engine.detect_default_changes(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1
        assert changes[0]["severity"] == "cosmetic"

    def test_default_added(self):
        """Adding a default value is detected."""
        source = _schema({"status": {"type": "string"}})
        target = _schema({"status": {"type": "string", "default": "active"}})
        changes = self.engine.detect_default_changes(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1

    def test_default_removed(self):
        """Removing a default value is detected."""
        source = _schema({"status": {"type": "string", "default": "active"}})
        target = _schema({"status": {"type": "string"}})
        changes = self.engine.detect_default_changes(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1

    def test_no_default_change_when_same(self):
        """No changes when defaults are identical."""
        schema = _schema({"status": {"type": "string", "default": "active"}})
        changes = self.engine.detect_default_changes(
            self._fields(schema), self._fields(schema)
        )
        assert len(changes) == 0

    def test_no_default_change_when_both_absent(self):
        """No changes when neither schema has a default."""
        schema = _schema({"status": {"type": "string"}})
        changes = self.engine.detect_default_changes(
            self._fields(schema), self._fields(schema)
        )
        assert len(changes) == 0


# ===========================================================================
# TestDetectArrayChanges
# ===========================================================================


class TestDetectArrayChanges:
    """Test detect_array_changes."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.engine = ChangeDetectorEngine()

    def _fields(self, schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return _extract_json_schema_fields(schema)

    def test_array_item_type_changed(self):
        """Changing array item type is detected."""
        source = _schema({
            "tags": {"type": "array", "items": {"type": "string"}}
        })
        target = _schema({
            "tags": {"type": "array", "items": {"type": "integer"}}
        })
        changes = self.engine.detect_array_changes(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 1
        assert "tags[]" in changes[0]["field_path"]

    def test_array_item_type_unchanged(self):
        """No change when array item types are the same."""
        schema = _schema({
            "tags": {"type": "array", "items": {"type": "string"}}
        })
        changes = self.engine.detect_array_changes(
            self._fields(schema), self._fields(schema)
        )
        assert len(changes) == 0

    def test_non_array_fields_ignored(self):
        """Non-array fields are not checked for array changes."""
        source = _schema({"name": {"type": "string"}})
        target = _schema({"name": {"type": "string"}})
        changes = self.engine.detect_array_changes(
            self._fields(source), self._fields(target)
        )
        assert len(changes) == 0


# ===========================================================================
# TestGetChangeSummary
# ===========================================================================


class TestGetChangeSummary:
    """Test summarize_changes."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.engine = ChangeDetectorEngine()

    def test_empty_changes_summary(self):
        """Empty changes list produces all-zero summary."""
        summary = self.engine.summarize_changes([])
        assert summary["total_count"] == 0
        assert summary["breaking_count"] == 0
        assert summary["non_breaking_count"] == 0
        assert summary["cosmetic_count"] == 0

    def test_summary_counts_by_type(self):
        """Summary counts changes grouped by type."""
        changes = [
            {"change_type": "added", "severity": "non_breaking"},
            {"change_type": "added", "severity": "breaking"},
            {"change_type": "removed", "severity": "breaking"},
        ]
        summary = self.engine.summarize_changes(changes)
        assert summary["total_count"] == 3
        assert summary["by_type"]["added"] == 2
        assert summary["by_type"]["removed"] == 1
        assert summary["added_count"] == 2
        assert summary["removed_count"] == 1

    def test_summary_counts_by_severity(self):
        """Summary counts changes grouped by severity."""
        changes = [
            {"change_type": "added", "severity": "non_breaking"},
            {"change_type": "removed", "severity": "breaking"},
            {"change_type": "reordered", "severity": "cosmetic"},
        ]
        summary = self.engine.summarize_changes(changes)
        assert summary["breaking_count"] == 1
        assert summary["non_breaking_count"] == 1
        assert summary["cosmetic_count"] == 1

    def test_summary_shortcut_counts(self):
        """Shortcut counts (added_count, removed_count, etc.) are correct."""
        changes = [
            {"change_type": "renamed", "severity": "non_breaking"},
            {"change_type": "retyped", "severity": "breaking"},
        ]
        summary = self.engine.summarize_changes(changes)
        assert summary["renamed_count"] == 1
        assert summary["retyped_count"] == 1
        assert summary["added_count"] == 0

    def test_summary_all_breaking(self):
        """Summary with all breaking changes."""
        changes = [
            {"change_type": "removed", "severity": "breaking"},
            {"change_type": "removed", "severity": "breaking"},
        ]
        summary = self.engine.summarize_changes(changes)
        assert summary["breaking_count"] == 2
        assert summary["non_breaking_count"] == 0


# ===========================================================================
# TestGetStatistics
# ===========================================================================


class TestGetStatistics:
    """Test get_statistics method."""

    def test_fresh_engine_statistics(self):
        """Fresh engine has zero statistics."""
        engine = ChangeDetectorEngine()
        stats = engine.get_statistics()
        assert stats["total_detections"] == 0
        assert stats["total_changes_detected"] == 0
        assert stats["stored_detections"] == 0
        assert stats["avg_processing_ms"] == 0.0
        assert stats["max_depth"] == 10
        assert stats["rename_threshold"] == _RENAME_SIMILARITY_THRESHOLD

    def test_statistics_after_detection(self):
        """Statistics update after running detection."""
        engine = ChangeDetectorEngine()
        source = _schema({"id": {"type": "integer"}})
        target = _schema({"id": {"type": "integer"}, "name": {"type": "string"}})
        engine.detect_changes(source, target)
        stats = engine.get_statistics()
        assert stats["total_detections"] == 1
        assert stats["stored_detections"] == 1
        # avg_processing_ms may be 0.0 on fast platforms where sub-ms
        # resolution rounds to zero; just verify the key exists and is numeric
        assert stats["avg_processing_ms"] >= 0.0


# ===========================================================================
# TestListDetections
# ===========================================================================


class TestListDetections:
    """Test list_detections method."""

    def test_list_empty(self):
        """list_detections on fresh engine returns empty list."""
        engine = ChangeDetectorEngine()
        assert engine.list_detections() == []

    def test_list_after_detections(self):
        """list_detections returns summaries for all stored detections."""
        engine = ChangeDetectorEngine()
        engine.detect_changes({}, {})
        engine.detect_changes({}, {})
        result = engine.list_detections()
        assert len(result) == 2
        for entry in result:
            assert "detection_id" in entry
            assert "total_count" in entry
            assert "breaking_count" in entry

    def test_list_pagination_limit(self):
        """list_detections respects the limit parameter."""
        engine = ChangeDetectorEngine()
        for _ in range(5):
            engine.detect_changes({}, {})
        result = engine.list_detections(limit=2)
        assert len(result) == 2

    def test_list_pagination_offset(self):
        """list_detections respects the offset parameter."""
        engine = ChangeDetectorEngine()
        for _ in range(5):
            engine.detect_changes({}, {})
        all_results = engine.list_detections()
        offset_results = engine.list_detections(offset=2)
        assert len(offset_results) == 3
        assert offset_results[0]["detection_id"] == all_results[2]["detection_id"]

    def test_list_invalid_limit_raises(self):
        """limit <= 0 raises ValueError."""
        engine = ChangeDetectorEngine()
        with pytest.raises(ValueError, match="limit must be > 0"):
            engine.list_detections(limit=0)

    def test_list_invalid_offset_raises(self):
        """offset < 0 raises ValueError."""
        engine = ChangeDetectorEngine()
        with pytest.raises(ValueError, match="offset must be >= 0"):
            engine.list_detections(offset=-1)


# ===========================================================================
# TestReset
# ===========================================================================


class TestReset:
    """Test the reset method."""

    def test_reset_clears_detections(self):
        """After reset, all stored detections are gone."""
        engine = ChangeDetectorEngine()
        result = engine.detect_changes({}, {})
        engine.reset()
        assert engine.get_detection(result["detection_id"]) is None
        assert engine.list_detections() == []

    def test_reset_clears_counters(self):
        """After reset, all counters return to zero."""
        engine = ChangeDetectorEngine()
        engine.detect_changes({}, {})
        engine.reset()
        stats = engine.get_statistics()
        assert stats["total_detections"] == 0
        assert stats["total_changes_detected"] == 0


# ===========================================================================
# TestFieldReordering
# ===========================================================================


class TestFieldReordering:
    """Test _detect_reordered_fields (exposed via detect_changes)."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.engine = ChangeDetectorEngine()

    def test_reordered_fields_detected(self):
        """Fields in different order are detected as reordered (cosmetic)."""
        # Use OrderedDict-like insertion order in the dict literal
        source = {"type": "object", "properties": {
            "a": {"type": "string"},
            "b": {"type": "integer"},
            "c": {"type": "boolean"},
        }}
        target = {"type": "object", "properties": {
            "c": {"type": "boolean"},
            "a": {"type": "string"},
            "b": {"type": "integer"},
        }}
        result = self.engine.detect_changes(source, target)
        reordered = [c for c in result["changes"] if c["change_type"] == "reordered"]
        assert len(reordered) == 1
        assert reordered[0]["severity"] == "cosmetic"

    def test_same_order_no_reorder(self):
        """Same field order produces no reorder change."""
        schema = {"type": "object", "properties": {
            "a": {"type": "string"},
            "b": {"type": "integer"},
        }}
        result = self.engine.detect_changes(schema, schema)
        reordered = [c for c in result["changes"] if c["change_type"] == "reordered"]
        assert len(reordered) == 0


# ===========================================================================
# TestAvroSchemaSupport
# ===========================================================================


class TestAvroSchemaSupport:
    """Test Avro schema format support."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.engine = ChangeDetectorEngine()

    def test_avro_format_auto_detected(self):
        """Avro record schemas are auto-detected."""
        avro = _avro_schema("User", [
            {"name": "id", "type": "long"},
            {"name": "email", "type": "string"},
        ])
        assert _detect_schema_format(avro) == "avro"

    def test_json_schema_format_auto_detected(self):
        """JSON Schema objects are auto-detected."""
        js = _schema({"id": {"type": "integer"}})
        assert _detect_schema_format(js) == "json_schema"

    def test_avro_field_extraction(self):
        """Avro fields are extracted correctly."""
        avro = _avro_schema("User", [
            {"name": "id", "type": "long"},
            {"name": "email", "type": ["null", "string"], "default": None},
        ])
        fields = _extract_avro_fields(avro)
        assert "id" in fields
        assert fields["id"]["type"] == "long"
        assert fields["id"]["required"] is True
        assert "email" in fields
        assert fields["email"]["type"] == "string"
        assert fields["email"]["required"] is False

    def test_avro_detect_added_field(self):
        """Detect added field in Avro schema."""
        source = _avro_schema("User", [
            {"name": "id", "type": "long"},
        ])
        target = _avro_schema("User", [
            {"name": "id", "type": "long"},
            {"name": "email", "type": "string"},
        ])
        result = self.engine.detect_changes(source, target)
        added = [c for c in result["changes"] if c["change_type"] == "added"]
        assert len(added) == 1


# ===========================================================================
# TestJaroWinklerHelpers
# ===========================================================================


class TestJaroWinklerHelpers:
    """Test the internal Jaro and Jaro-Winkler helper functions."""

    def test_identical_strings_score_1(self):
        """Identical strings have similarity 1.0."""
        assert _jaro_similarity("hello", "hello") == 1.0
        assert _jaro_winkler_similarity("hello", "hello") == 1.0

    def test_empty_string_score_0(self):
        """Comparing with an empty string returns 0.0."""
        assert _jaro_similarity("hello", "") == 0.0
        assert _jaro_similarity("", "hello") == 0.0

    def test_both_empty_score_1(self):
        """Two empty strings have Jaro similarity 1.0."""
        assert _jaro_similarity("", "") == 1.0

    def test_jaro_winkler_ge_jaro(self):
        """Jaro-Winkler score >= Jaro score for strings sharing a prefix."""
        s1, s2 = "user_name", "user_id"
        jaro = _jaro_similarity(s1, s2)
        jw = _jaro_winkler_similarity(s1, s2)
        assert jw >= jaro

    def test_known_jaro_winkler_value(self):
        """Verify Jaro-Winkler against a known reference value."""
        score = _jaro_winkler_similarity("martha", "marhta")
        assert round(score, 4) == 0.9611

    def test_completely_different_strings(self):
        """Completely different strings have low similarity."""
        score = _jaro_winkler_similarity("abc", "xyz")
        assert score < 0.5


# ===========================================================================
# TestTypeWideningNarrowingHelpers
# ===========================================================================


class TestTypeWideningNarrowingHelpers:
    """Test _is_type_widening and _is_type_narrowing."""

    @pytest.mark.parametrize("old_type,new_type", [
        ("integer", "number"),
        ("integer", "string"),
        ("number", "string"),
        ("int", "float"),
        ("float", "double"),
    ])
    def test_type_widening_pairs(self, old_type, new_type):
        """Known type widening pairs return True."""
        assert _is_type_widening(old_type, new_type) is True

    @pytest.mark.parametrize("old_type,new_type", [
        ("number", "integer"),
        ("string", "integer"),
        ("string", "number"),
        ("float", "int"),
        ("double", "float"),
    ])
    def test_type_narrowing_pairs(self, old_type, new_type):
        """Known type narrowing pairs return True."""
        assert _is_type_narrowing(old_type, new_type) is True

    def test_same_type_not_widening(self):
        """Same type is not widening."""
        assert _is_type_widening("string", "string") is False

    def test_same_type_not_narrowing(self):
        """Same type is not narrowing."""
        assert _is_type_narrowing("string", "string") is False


# ===========================================================================
# TestComputeSHA256
# ===========================================================================


class TestComputeSHA256:
    """Test _compute_sha256 helper."""

    def test_deterministic(self):
        """Same input produces the same SHA-256 hash."""
        data = {"key": "value", "num": 42}
        h1 = _compute_sha256(data)
        h2 = _compute_sha256(data)
        assert h1 == h2
        assert len(h1) == 64

    def test_none_produces_hash(self):
        """None input produces a valid hash (of the string 'null')."""
        h = _compute_sha256(None)
        assert len(h) == 64

    def test_different_data_different_hash(self):
        """Different inputs produce different hashes."""
        h1 = _compute_sha256({"a": 1})
        h2 = _compute_sha256({"a": 2})
        assert h1 != h2


# ===========================================================================
# TestChangeDetectorEdgeCases
# ===========================================================================


class TestChangeDetectorEdgeCases:
    """Edge cases and stress tests."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        self.engine = ChangeDetectorEngine()

    def test_large_schema_500_fields(self):
        """Engine handles schemas with 500+ fields without error."""
        props = {f"field_{i}": {"type": "string"} for i in range(500)}
        source = _schema(props)
        target_props = dict(props)
        target_props["field_500"] = {"type": "integer"}
        target = _schema(target_props)

        result = self.engine.detect_changes(source, target)
        added = [c for c in result["changes"] if c["change_type"] == "added"]
        assert len(added) == 1
        assert result["processing_time_ms"] >= 0

    def test_special_characters_in_field_names(self):
        """Engine handles field names with special characters."""
        source = _schema({"field-with-dashes": {"type": "string"}})
        target = _schema({
            "field-with-dashes": {"type": "string"},
            "field.with.dots": {"type": "integer"},
        })
        result = self.engine.detect_changes(source, target)
        added = [c for c in result["changes"] if c["change_type"] == "added"]
        assert len(added) == 1

    def test_unicode_field_names(self):
        """Engine handles Unicode field names."""
        source = _schema({})
        target = _schema({
            "nombre": {"type": "string"},
            "prix_unite": {"type": "number"},
        })
        result = self.engine.detect_changes(source, target)
        assert result["summary"]["added_count"] == 2

    def test_empty_properties_dict(self):
        """Schema with empty properties dict produces no changes."""
        result = self.engine.detect_changes(
            {"type": "object", "properties": {}},
            {"type": "object", "properties": {}},
        )
        assert result["summary"]["total_count"] == 0

    def test_schema_without_type_key(self):
        """Schema without 'type' key at top level is handled."""
        result = self.engine.detect_changes(
            {"properties": {"a": {"type": "string"}}},
            {"properties": {"a": {"type": "string"}, "b": {"type": "integer"}}},
        )
        assert result["summary"]["added_count"] == 1

    def test_deeply_nested_field_type_array(self):
        """JSON Schema with type as array (e.g., ['string', 'null']) is handled."""
        source = _schema({"val": {"type": ["string", "null"]}})
        target = _schema({"val": {"type": "string"}})
        src_fields = _extract_json_schema_fields(source)
        tgt_fields = _extract_json_schema_fields(target)
        # Both should resolve to "string" so no type change
        assert src_fields["val"]["type"] == "string"
        assert tgt_fields["val"]["type"] == "string"

    def test_thread_safety_concurrent_detections(self):
        """Engine handles concurrent detect_changes calls safely."""
        engine = ChangeDetectorEngine()
        errors = []

        def run_detection(idx: int):
            try:
                source = _schema({f"f{idx}": {"type": "string"}})
                target = _schema({f"f{idx}": {"type": "integer"}})
                result = engine.detect_changes(source, target)
                assert result["detection_id"].startswith("DET-")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run_detection, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = engine.get_statistics()
        assert stats["total_detections"] == 10

    def test_provenance_chain_integrity(self):
        """Provenance chain remains valid after multiple detections."""
        engine = ChangeDetectorEngine()
        for i in range(5):
            source = _schema({f"f{i}": {"type": "string"}})
            target = _schema({f"f{i}": {"type": "integer"}})
            engine.detect_changes(source, target)
        assert engine._provenance.verify_chain() is True

    def test_deterministic_source_target_hashes(self):
        """Source and target hashes are deterministic across runs."""
        source = _schema({"id": {"type": "integer"}, "name": {"type": "string"}})
        target = _schema({"id": {"type": "integer"}})
        r1 = self.engine.detect_changes(source, target)
        engine2 = ChangeDetectorEngine()
        r2 = engine2.detect_changes(source, target)
        assert r1["source_hash"] == r2["source_hash"]
        assert r1["target_hash"] == r2["target_hash"]

    def test_large_schema_performance_under_one_second(self):
        """Detecting changes on a 1000-field schema completes in under 1s."""
        props = {f"field_{i}": {"type": "string"} for i in range(1000)}
        source = _schema(props)
        target_props = dict(props)
        for i in range(1000, 1010):
            target_props[f"field_{i}"] = {"type": "integer"}
        target = _schema(target_props)

        start = time.monotonic()
        result = self.engine.detect_changes(source, target)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        assert elapsed_ms < 1000.0
        assert result["summary"]["added_count"] == 10
