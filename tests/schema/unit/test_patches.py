# -*- coding: utf-8 -*-
"""
Unit Tests for JSON Patch Generator (RFC 6902)

Tests for greenlang.schema.suggestions.patches module implementing
JSON Patch operations per RFC 6902 and JSON Pointer utilities per RFC 6901.

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

import copy
import pytest
from typing import Any, Dict, List

from greenlang.schema.suggestions.patches import (
    # Exceptions
    JSONPointerError,
    PatchApplicationError,
    # Enums
    PatchOp,
    # Models
    JSONPatchOperation,
    PatchSequence,
    # Generator
    PatchGenerator,
    # JSON Pointer utilities
    parse_json_pointer,
    build_json_pointer,
    get_value_at_pointer,
    set_value_at_pointer,
    remove_value_at_pointer,
    pointer_parent,
    pointer_last_segment,
    escape_json_pointer_token,
    unescape_json_pointer_token,
    # Patch application
    apply_patch,
    apply_patch_sequence,
    # Patch validation
    validate_patch,
    validate_patch_syntax,
    # Convenience functions
    create_add_patch,
    create_remove_patch,
    create_replace_patch,
    create_move_patch,
)


# =============================================================================
# JSON Pointer Utilities Tests (RFC 6901)
# =============================================================================

class TestJSONPointerParsing:
    """Tests for JSON Pointer parsing and building per RFC 6901."""

    def test_parse_empty_pointer(self):
        """Empty string represents the root document."""
        assert parse_json_pointer("") == []

    def test_parse_simple_pointer(self):
        """Parse a simple single-segment pointer."""
        assert parse_json_pointer("/foo") == ["foo"]

    def test_parse_nested_pointer(self):
        """Parse a multi-segment pointer."""
        assert parse_json_pointer("/foo/bar/baz") == ["foo", "bar", "baz"]

    def test_parse_array_index_pointer(self):
        """Parse pointer with array indices."""
        assert parse_json_pointer("/items/0/name") == ["items", "0", "name"]

    def test_parse_escaped_tilde(self):
        """Parse pointer with escaped tilde (~0)."""
        assert parse_json_pointer("/a~0b") == ["a~b"]

    def test_parse_escaped_slash(self):
        """Parse pointer with escaped slash (~1)."""
        assert parse_json_pointer("/a~1b") == ["a/b"]

    def test_parse_complex_escaping(self):
        """Parse pointer with multiple escape sequences."""
        assert parse_json_pointer("/a~0b~1c") == ["a~b/c"]

    def test_parse_empty_segment(self):
        """Parse pointer with empty segment."""
        assert parse_json_pointer("/foo//bar") == ["foo", "", "bar"]

    def test_parse_invalid_pointer_no_slash(self):
        """Pointer not starting with / should raise error."""
        with pytest.raises(JSONPointerError):
            parse_json_pointer("foo/bar")

    def test_build_empty_segments(self):
        """Empty segments list produces empty string."""
        assert build_json_pointer([]) == ""

    def test_build_simple_pointer(self):
        """Build a simple single-segment pointer."""
        assert build_json_pointer(["foo"]) == "/foo"

    def test_build_nested_pointer(self):
        """Build a multi-segment pointer."""
        assert build_json_pointer(["foo", "bar"]) == "/foo/bar"

    def test_build_with_special_chars(self):
        """Build pointer with characters that need escaping."""
        assert build_json_pointer(["a/b", "c~d"]) == "/a~1b/c~0d"

    def test_roundtrip_simple(self):
        """Parse and build should be inverse operations."""
        pointer = "/foo/bar/baz"
        assert build_json_pointer(parse_json_pointer(pointer)) == pointer

    def test_roundtrip_with_escapes(self):
        """Roundtrip with escape sequences."""
        pointer = "/a~0b/c~1d"
        assert build_json_pointer(parse_json_pointer(pointer)) == pointer


class TestEscapeUnescape:
    """Tests for escape/unescape functions."""

    def test_escape_tilde(self):
        """Tilde should be escaped as ~0."""
        assert escape_json_pointer_token("a~b") == "a~0b"

    def test_escape_slash(self):
        """Slash should be escaped as ~1."""
        assert escape_json_pointer_token("a/b") == "a~1b"

    def test_escape_both(self):
        """Both tilde and slash should be escaped correctly."""
        assert escape_json_pointer_token("a~b/c") == "a~0b~1c"

    def test_unescape_tilde(self):
        """~0 should be unescaped to tilde."""
        assert unescape_json_pointer_token("a~0b") == "a~b"

    def test_unescape_slash(self):
        """~1 should be unescaped to slash."""
        assert unescape_json_pointer_token("a~1b") == "a/b"

    def test_unescape_both(self):
        """Both ~0 and ~1 should be unescaped correctly."""
        assert unescape_json_pointer_token("a~0b~1c") == "a~b/c"

    def test_roundtrip_escape_unescape(self):
        """Escape and unescape should be inverse operations."""
        original = "path/with~special"
        escaped = escape_json_pointer_token(original)
        assert unescape_json_pointer_token(escaped) == original


class TestGetValueAtPointer:
    """Tests for get_value_at_pointer function."""

    @pytest.fixture
    def sample_doc(self) -> Dict[str, Any]:
        """Sample document for testing."""
        return {
            "foo": "bar",
            "nested": {"a": 1, "b": {"c": 2}},
            "array": [10, 20, {"x": 30}],
            "empty": {},
            "null_val": None,
        }

    def test_get_root(self, sample_doc):
        """Empty pointer returns the whole document."""
        exists, value = get_value_at_pointer(sample_doc, "")
        assert exists is True
        assert value == sample_doc

    def test_get_top_level(self, sample_doc):
        """Get a top-level field."""
        exists, value = get_value_at_pointer(sample_doc, "/foo")
        assert exists is True
        assert value == "bar"

    def test_get_nested_object(self, sample_doc):
        """Get a nested object field."""
        exists, value = get_value_at_pointer(sample_doc, "/nested/b/c")
        assert exists is True
        assert value == 2

    def test_get_array_element(self, sample_doc):
        """Get an array element by index."""
        exists, value = get_value_at_pointer(sample_doc, "/array/1")
        assert exists is True
        assert value == 20

    def test_get_nested_in_array(self, sample_doc):
        """Get a field inside an array element."""
        exists, value = get_value_at_pointer(sample_doc, "/array/2/x")
        assert exists is True
        assert value == 30

    def test_get_null_value(self, sample_doc):
        """Null values should exist and return None."""
        exists, value = get_value_at_pointer(sample_doc, "/null_val")
        assert exists is True
        assert value is None

    def test_get_nonexistent_field(self, sample_doc):
        """Non-existent field should return (False, None)."""
        exists, value = get_value_at_pointer(sample_doc, "/nonexistent")
        assert exists is False
        assert value is None

    def test_get_nonexistent_nested(self, sample_doc):
        """Non-existent nested field should return (False, None)."""
        exists, value = get_value_at_pointer(sample_doc, "/nested/x/y")
        assert exists is False
        assert value is None

    def test_get_invalid_array_index(self, sample_doc):
        """Invalid array index should return (False, None)."""
        exists, value = get_value_at_pointer(sample_doc, "/array/99")
        assert exists is False

    def test_get_array_dash(self, sample_doc):
        """Array dash index should return (False, None) for reading."""
        exists, value = get_value_at_pointer(sample_doc, "/array/-")
        assert exists is False


class TestSetValueAtPointer:
    """Tests for set_value_at_pointer function."""

    @pytest.fixture
    def sample_doc(self) -> Dict[str, Any]:
        """Sample document for testing."""
        return {"foo": "bar", "nested": {"a": 1}}

    def test_set_existing_field(self, sample_doc):
        """Set an existing field."""
        result = set_value_at_pointer(sample_doc, "/foo", "baz")
        assert result["foo"] == "baz"
        # Original should be unchanged
        assert sample_doc["foo"] == "bar"

    def test_set_new_field(self, sample_doc):
        """Set a new field."""
        result = set_value_at_pointer(sample_doc, "/new", 123)
        assert result["new"] == 123
        assert "new" not in sample_doc

    def test_set_nested_field(self, sample_doc):
        """Set a nested field."""
        result = set_value_at_pointer(sample_doc, "/nested/a", 99)
        assert result["nested"]["a"] == 99

    def test_set_creates_intermediate(self, sample_doc):
        """Setting a deep path creates intermediate objects."""
        result = set_value_at_pointer(sample_doc, "/x/y/z", "deep")
        assert result["x"]["y"]["z"] == "deep"

    def test_set_array_element(self):
        """Set an array element."""
        doc = {"arr": [1, 2, 3]}
        result = set_value_at_pointer(doc, "/arr/1", 99)
        assert result["arr"] == [1, 99, 3]

    def test_set_array_append(self):
        """Append to array using dash index."""
        doc = {"arr": [1, 2]}
        result = set_value_at_pointer(doc, "/arr/-", 3)
        assert result["arr"] == [1, 2, 3]

    def test_original_not_modified(self, sample_doc):
        """Original document should not be modified."""
        original = copy.deepcopy(sample_doc)
        set_value_at_pointer(sample_doc, "/foo", "changed")
        assert sample_doc == original


class TestRemoveValueAtPointer:
    """Tests for remove_value_at_pointer function."""

    def test_remove_field(self):
        """Remove a field from object."""
        doc = {"a": 1, "b": 2}
        result = remove_value_at_pointer(doc, "/a")
        assert result == {"b": 2}
        assert doc == {"a": 1, "b": 2}  # Original unchanged

    def test_remove_nested_field(self):
        """Remove a nested field."""
        doc = {"outer": {"inner": 1, "other": 2}}
        result = remove_value_at_pointer(doc, "/outer/inner")
        assert result == {"outer": {"other": 2}}

    def test_remove_array_element(self):
        """Remove an array element."""
        doc = {"arr": [1, 2, 3]}
        result = remove_value_at_pointer(doc, "/arr/1")
        assert result["arr"] == [1, 3]

    def test_remove_root_raises(self):
        """Cannot remove root document."""
        with pytest.raises(JSONPointerError):
            remove_value_at_pointer({"a": 1}, "")

    def test_remove_nonexistent_raises(self):
        """Removing non-existent path raises error."""
        with pytest.raises(JSONPointerError):
            remove_value_at_pointer({"a": 1}, "/b")


class TestPointerUtilities:
    """Tests for pointer utility functions."""

    def test_pointer_parent_nested(self):
        """Get parent of nested pointer."""
        assert pointer_parent("/foo/bar/baz") == "/foo/bar"

    def test_pointer_parent_single(self):
        """Single segment pointer parent is empty."""
        assert pointer_parent("/foo") == ""

    def test_pointer_parent_root(self):
        """Root pointer has no parent."""
        assert pointer_parent("") == ""

    def test_pointer_last_segment(self):
        """Get last segment of pointer."""
        assert pointer_last_segment("/foo/bar") == "bar"

    def test_pointer_last_segment_single(self):
        """Last segment of single-segment pointer."""
        assert pointer_last_segment("/foo") == "foo"

    def test_pointer_last_segment_root(self):
        """Root pointer has no last segment."""
        assert pointer_last_segment("") is None


# =============================================================================
# JSON Patch Operation Tests
# =============================================================================

class TestJSONPatchOperation:
    """Tests for JSONPatchOperation Pydantic model."""

    def test_create_add_operation(self):
        """Create an add operation."""
        op = JSONPatchOperation(op=PatchOp.ADD, path="/foo", value=42)
        assert op.op == PatchOp.ADD
        assert op.path == "/foo"
        assert op.value == 42

    def test_create_remove_operation(self):
        """Create a remove operation."""
        op = JSONPatchOperation(op=PatchOp.REMOVE, path="/foo")
        assert op.op == PatchOp.REMOVE
        assert op.path == "/foo"

    def test_create_replace_operation(self):
        """Create a replace operation."""
        op = JSONPatchOperation(op=PatchOp.REPLACE, path="/foo", value="new")
        assert op.op == PatchOp.REPLACE
        assert op.value == "new"

    def test_create_move_operation(self):
        """Create a move operation."""
        op = JSONPatchOperation(op=PatchOp.MOVE, path="/new", from_="/old")
        assert op.op == PatchOp.MOVE
        assert op.from_ == "/old"

    def test_create_copy_operation(self):
        """Create a copy operation."""
        op = JSONPatchOperation(op=PatchOp.COPY, path="/dest", from_="/src")
        assert op.op == PatchOp.COPY

    def test_create_test_operation(self):
        """Create a test operation."""
        op = JSONPatchOperation(op=PatchOp.TEST, path="/version", value="1.0")
        assert op.op == PatchOp.TEST
        assert op.value == "1.0"

    def test_move_requires_from(self):
        """Move operation requires from field."""
        with pytest.raises(ValueError):
            JSONPatchOperation(op=PatchOp.MOVE, path="/new")

    def test_copy_requires_from(self):
        """Copy operation requires from field."""
        with pytest.raises(ValueError):
            JSONPatchOperation(op=PatchOp.COPY, path="/dest")

    def test_invalid_path_raises(self):
        """Invalid path should raise ValueError."""
        with pytest.raises(ValueError):
            JSONPatchOperation(op=PatchOp.ADD, path="invalid", value=1)

    def test_to_rfc6902_add(self):
        """Convert add operation to RFC 6902 format."""
        op = JSONPatchOperation(op=PatchOp.ADD, path="/foo", value=42)
        assert op.to_rfc6902() == {"op": "add", "path": "/foo", "value": 42}

    def test_to_rfc6902_remove(self):
        """Convert remove operation to RFC 6902 format."""
        op = JSONPatchOperation(op=PatchOp.REMOVE, path="/foo")
        assert op.to_rfc6902() == {"op": "remove", "path": "/foo"}

    def test_to_rfc6902_move(self):
        """Convert move operation to RFC 6902 format."""
        op = JSONPatchOperation(op=PatchOp.MOVE, path="/new", from_="/old")
        assert op.to_rfc6902() == {"op": "move", "path": "/new", "from": "/old"}

    def test_is_additive(self):
        """Test is_additive method."""
        add_op = JSONPatchOperation(op=PatchOp.ADD, path="/x", value=1)
        copy_op = JSONPatchOperation(op=PatchOp.COPY, path="/y", from_="/x")
        remove_op = JSONPatchOperation(op=PatchOp.REMOVE, path="/x")

        assert add_op.is_additive() is True
        assert copy_op.is_additive() is True
        assert remove_op.is_additive() is False

    def test_is_destructive(self):
        """Test is_destructive method."""
        remove_op = JSONPatchOperation(op=PatchOp.REMOVE, path="/x")
        add_op = JSONPatchOperation(op=PatchOp.ADD, path="/x", value=1)

        assert remove_op.is_destructive() is True
        assert add_op.is_destructive() is False

    def test_is_test(self):
        """Test is_test method."""
        test_op = JSONPatchOperation(op=PatchOp.TEST, path="/x", value=1)
        add_op = JSONPatchOperation(op=PatchOp.ADD, path="/x", value=1)

        assert test_op.is_test() is True
        assert add_op.is_test() is False


class TestPatchSequence:
    """Tests for PatchSequence model."""

    def test_empty_sequence(self):
        """Create empty patch sequence."""
        seq = PatchSequence()
        assert len(seq) == 0
        assert bool(seq) is False

    def test_sequence_with_operations(self):
        """Create sequence with operations."""
        ops = [
            JSONPatchOperation(op=PatchOp.ADD, path="/a", value=1),
            JSONPatchOperation(op=PatchOp.ADD, path="/b", value=2),
        ]
        seq = PatchSequence(operations=ops)
        assert len(seq) == 2
        assert bool(seq) is True

    def test_to_json_patch(self):
        """Convert sequence to JSON Patch format."""
        ops = [
            JSONPatchOperation(op=PatchOp.ADD, path="/a", value=1),
            JSONPatchOperation(op=PatchOp.REMOVE, path="/b"),
        ]
        seq = PatchSequence(operations=ops)
        result = seq.to_json_patch()
        assert result == [
            {"op": "add", "path": "/a", "value": 1},
            {"op": "remove", "path": "/b"},
        ]

    def test_add_operation(self):
        """Add operation to sequence."""
        seq = PatchSequence()
        new_op = JSONPatchOperation(op=PatchOp.ADD, path="/x", value=1)
        new_seq = seq.add(new_op)
        assert len(new_seq) == 1
        assert len(seq) == 0  # Original unchanged

    def test_extend_operations(self):
        """Extend sequence with multiple operations."""
        seq = PatchSequence(operations=[
            JSONPatchOperation(op=PatchOp.ADD, path="/a", value=1)
        ])
        new_ops = [
            JSONPatchOperation(op=PatchOp.ADD, path="/b", value=2),
            JSONPatchOperation(op=PatchOp.ADD, path="/c", value=3),
        ]
        new_seq = seq.extend(new_ops)
        assert len(new_seq) == 3

    def test_get_tests(self):
        """Get only test operations."""
        ops = [
            JSONPatchOperation(op=PatchOp.TEST, path="/v", value=1),
            JSONPatchOperation(op=PatchOp.REPLACE, path="/v", value=2),
            JSONPatchOperation(op=PatchOp.TEST, path="/x", value=3),
        ]
        seq = PatchSequence(operations=ops)
        tests = seq.get_tests()
        assert len(tests) == 2
        assert all(op.is_test() for op in tests)

    def test_get_mutations(self):
        """Get only mutation operations."""
        ops = [
            JSONPatchOperation(op=PatchOp.TEST, path="/v", value=1),
            JSONPatchOperation(op=PatchOp.REPLACE, path="/v", value=2),
        ]
        seq = PatchSequence(operations=ops)
        mutations = seq.get_mutations()
        assert len(mutations) == 1
        assert mutations[0].op == PatchOp.REPLACE

    def test_affected_paths(self):
        """Get all affected paths."""
        ops = [
            JSONPatchOperation(op=PatchOp.ADD, path="/a", value=1),
            JSONPatchOperation(op=PatchOp.MOVE, path="/c", from_="/b"),
        ]
        seq = PatchSequence(operations=ops)
        paths = seq.affected_paths()
        assert set(paths) == {"/a", "/b", "/c"}

    def test_compute_hash(self):
        """Hash should be consistent."""
        ops = [JSONPatchOperation(op=PatchOp.ADD, path="/a", value=1)]
        seq1 = PatchSequence(operations=ops)
        seq2 = PatchSequence(operations=ops)
        assert seq1.compute_hash() == seq2.compute_hash()

    def test_iteration(self):
        """Sequence should be iterable."""
        ops = [
            JSONPatchOperation(op=PatchOp.ADD, path="/a", value=1),
            JSONPatchOperation(op=PatchOp.ADD, path="/b", value=2),
        ]
        seq = PatchSequence(operations=ops)
        collected = list(seq)
        assert len(collected) == 2


# =============================================================================
# Patch Generator Tests
# =============================================================================

class TestPatchGenerator:
    """Tests for PatchGenerator class."""

    @pytest.fixture
    def generator(self) -> PatchGenerator:
        """Create a patch generator."""
        return PatchGenerator()

    def test_generate_add(self, generator):
        """Generate add operation."""
        op = generator.generate_add("/energy", 100)
        assert op.op == PatchOp.ADD
        assert op.path == "/energy"
        assert op.value == 100

    def test_generate_add_complex_value(self, generator):
        """Generate add with complex value."""
        value = {"value": 100, "unit": "kWh"}
        op = generator.generate_add("/energy", value)
        assert op.value == value

    def test_generate_remove(self, generator):
        """Generate remove operation."""
        op = generator.generate_remove("/deprecated")
        assert op.op == PatchOp.REMOVE
        assert op.path == "/deprecated"

    def test_generate_replace_with_test(self, generator):
        """Generate replace with test precondition."""
        ops = generator.generate_replace("/name", "new", old_value="old")
        assert len(ops) == 2
        assert ops[0].op == PatchOp.TEST
        assert ops[0].value == "old"
        assert ops[1].op == PatchOp.REPLACE
        assert ops[1].value == "new"

    def test_generate_replace_without_test(self, generator):
        """Generate replace without test."""
        ops = generator.generate_replace("/name", "new", include_test=False)
        assert len(ops) == 1
        assert ops[0].op == PatchOp.REPLACE

    def test_generate_move(self, generator):
        """Generate move operation."""
        ops = generator.generate_move("/old", "/new")
        assert len(ops) == 1
        assert ops[0].op == PatchOp.MOVE
        assert ops[0].from_ == "/old"
        assert ops[0].path == "/new"

    def test_generate_copy(self, generator):
        """Generate copy operation."""
        op = generator.generate_copy("/source", "/dest")
        assert op.op == PatchOp.COPY
        assert op.from_ == "/source"
        assert op.path == "/dest"

    def test_generate_test(self, generator):
        """Generate test operation."""
        op = generator.generate_test("/version", "1.0")
        assert op.op == PatchOp.TEST
        assert op.value == "1.0"

    def test_generate_add_default(self, generator):
        """Generate add default operation."""
        ops = generator.generate_add_default("/config", {"timeout": 30})
        assert len(ops) == 1
        assert ops[0].op == PatchOp.ADD

    def test_generate_type_coercion(self, generator):
        """Generate type coercion operations."""
        ops = generator.generate_type_coercion("/count", "42", 42)
        assert len(ops) == 2
        assert ops[0].op == PatchOp.TEST
        assert ops[0].value == "42"
        assert ops[1].op == PatchOp.REPLACE
        assert ops[1].value == 42

    def test_generate_unit_conversion(self, generator):
        """Generate unit conversion operations."""
        original = {"value": 1000, "unit": "Wh"}
        converted = {"value": 1, "unit": "kWh"}
        ops = generator.generate_unit_conversion("/energy", original, converted)
        assert len(ops) == 2
        assert ops[0].value == original
        assert ops[1].value == converted

    def test_generate_field_rename(self, generator):
        """Generate field rename operation."""
        ops = generator.generate_field_rename("/old_name", "/new_name")
        assert len(ops) == 1
        assert ops[0].op == PatchOp.MOVE

    def test_generate_batch(self, generator):
        """Generate batch of operations."""
        specs = [
            {"type": "add", "path": "/a", "value": 1},
            {"type": "remove", "path": "/b"},
            {"type": "test", "path": "/c", "value": 3},
        ]
        seq = generator.generate_batch(specs)
        assert len(seq) == 3


# =============================================================================
# Patch Application Tests
# =============================================================================

class TestApplyPatch:
    """Tests for apply_patch function."""

    def test_apply_add(self):
        """Apply add operation."""
        doc = {"a": 1}
        patch = [JSONPatchOperation(op=PatchOp.ADD, path="/b", value=2)]
        result = apply_patch(doc, patch)
        assert result == {"a": 1, "b": 2}
        assert doc == {"a": 1}  # Original unchanged

    def test_apply_add_nested(self):
        """Apply add to nested path."""
        doc = {"outer": {"inner": 1}}
        patch = [JSONPatchOperation(op=PatchOp.ADD, path="/outer/new", value=2)]
        result = apply_patch(doc, patch)
        assert result["outer"]["new"] == 2

    def test_apply_add_array_element(self):
        """Apply add to array."""
        doc = {"arr": [1, 2]}
        patch = [JSONPatchOperation(op=PatchOp.ADD, path="/arr/-", value=3)]
        result = apply_patch(doc, patch)
        assert result["arr"] == [1, 2, 3]

    def test_apply_remove(self):
        """Apply remove operation."""
        doc = {"a": 1, "b": 2}
        patch = [JSONPatchOperation(op=PatchOp.REMOVE, path="/a")]
        result = apply_patch(doc, patch)
        assert result == {"b": 2}

    def test_apply_remove_nonexistent_fails(self):
        """Remove non-existent path should fail."""
        doc = {"a": 1}
        patch = [JSONPatchOperation(op=PatchOp.REMOVE, path="/b")]
        with pytest.raises(PatchApplicationError):
            apply_patch(doc, patch)

    def test_apply_replace(self):
        """Apply replace operation."""
        doc = {"name": "old"}
        patch = [JSONPatchOperation(op=PatchOp.REPLACE, path="/name", value="new")]
        result = apply_patch(doc, patch)
        assert result["name"] == "new"

    def test_apply_replace_nonexistent_fails(self):
        """Replace non-existent path should fail."""
        doc = {"a": 1}
        patch = [JSONPatchOperation(op=PatchOp.REPLACE, path="/b", value=2)]
        with pytest.raises(PatchApplicationError):
            apply_patch(doc, patch)

    def test_apply_move(self):
        """Apply move operation."""
        doc = {"old": "value", "other": 1}
        patch = [JSONPatchOperation(op=PatchOp.MOVE, path="/new", from_="/old")]
        result = apply_patch(doc, patch)
        assert result == {"new": "value", "other": 1}
        assert "old" not in result

    def test_apply_copy(self):
        """Apply copy operation."""
        doc = {"source": "value"}
        patch = [JSONPatchOperation(op=PatchOp.COPY, path="/dest", from_="/source")]
        result = apply_patch(doc, patch)
        assert result == {"source": "value", "dest": "value"}

    def test_apply_test_success(self):
        """Test operation succeeds when value matches."""
        doc = {"version": "1.0"}
        patch = [
            JSONPatchOperation(op=PatchOp.TEST, path="/version", value="1.0"),
            JSONPatchOperation(op=PatchOp.REPLACE, path="/version", value="2.0"),
        ]
        result = apply_patch(doc, patch)
        assert result["version"] == "2.0"

    def test_apply_test_failure(self):
        """Test operation fails when value doesn't match."""
        doc = {"version": "1.0"}
        patch = [
            JSONPatchOperation(op=PatchOp.TEST, path="/version", value="2.0"),
            JSONPatchOperation(op=PatchOp.REPLACE, path="/version", value="3.0"),
        ]
        with pytest.raises(PatchApplicationError):
            apply_patch(doc, patch)

    def test_apply_test_nonexistent_path(self):
        """Test operation fails when path doesn't exist."""
        doc = {"a": 1}
        patch = [JSONPatchOperation(op=PatchOp.TEST, path="/b", value=2)]
        with pytest.raises(PatchApplicationError):
            apply_patch(doc, patch)

    def test_apply_multiple_operations(self):
        """Apply sequence of operations."""
        doc = {"a": 1, "b": 2}
        patch = [
            JSONPatchOperation(op=PatchOp.ADD, path="/c", value=3),
            JSONPatchOperation(op=PatchOp.REMOVE, path="/a"),
            JSONPatchOperation(op=PatchOp.REPLACE, path="/b", value=20),
        ]
        result = apply_patch(doc, patch)
        assert result == {"b": 20, "c": 3}

    def test_apply_patch_sequence(self):
        """Apply PatchSequence object."""
        doc = {"x": 1}
        seq = PatchSequence(operations=[
            JSONPatchOperation(op=PatchOp.ADD, path="/y", value=2)
        ])
        result = apply_patch_sequence(doc, seq)
        assert result == {"x": 1, "y": 2}

    def test_deep_equality_test(self):
        """Test operation uses deep equality."""
        doc = {"data": {"nested": [1, 2, 3]}}
        patch = [JSONPatchOperation(
            op=PatchOp.TEST,
            path="/data",
            value={"nested": [1, 2, 3]}
        )]
        # Should not raise
        apply_patch(doc, patch)


# =============================================================================
# Patch Validation Tests
# =============================================================================

class TestValidatePatch:
    """Tests for patch validation functions."""

    def test_validate_valid_patch(self):
        """Valid patch should return empty error list."""
        doc = {"a": 1}
        patch = [JSONPatchOperation(op=PatchOp.ADD, path="/b", value=2)]
        errors = validate_patch(doc, patch)
        assert errors == []

    def test_validate_invalid_patch(self):
        """Invalid patch should return errors."""
        doc = {"a": 1}
        patch = [JSONPatchOperation(op=PatchOp.REMOVE, path="/b")]
        errors = validate_patch(doc, patch)
        assert len(errors) > 0

    def test_validate_syntax_valid(self):
        """Valid syntax should return empty error list."""
        patch = [JSONPatchOperation(op=PatchOp.ADD, path="/a", value=1)]
        errors = validate_patch_syntax(patch)
        assert errors == []


# =============================================================================
# Convenience Functions Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_add_patch(self):
        """Create add patch document."""
        patch = create_add_patch("/foo", 42)
        assert patch == [{"op": "add", "path": "/foo", "value": 42}]

    def test_create_remove_patch(self):
        """Create remove patch document."""
        patch = create_remove_patch("/foo")
        assert patch == [{"op": "remove", "path": "/foo"}]

    def test_create_replace_patch_with_test(self):
        """Create replace patch with test."""
        patch = create_replace_patch("/name", "new", "old")
        assert len(patch) == 2
        assert patch[0]["op"] == "test"
        assert patch[1]["op"] == "replace"

    def test_create_replace_patch_without_test(self):
        """Create replace patch without test."""
        patch = create_replace_patch("/name", "new")
        assert len(patch) == 1
        assert patch[0]["op"] == "replace"

    def test_create_move_patch(self):
        """Create move patch document."""
        patch = create_move_patch("/old", "/new")
        assert patch == [{"op": "move", "path": "/new", "from": "/old"}]


# =============================================================================
# RFC 6902 Compliance Tests
# =============================================================================

class TestRFC6902Compliance:
    """Tests for RFC 6902 compliance."""

    def test_rfc_example_add_object_member(self):
        """RFC 6902 Section 4.1 example: add object member."""
        doc = {"foo": "bar"}
        patch = [JSONPatchOperation(op=PatchOp.ADD, path="/baz", value="qux")]
        result = apply_patch(doc, patch)
        assert result == {"foo": "bar", "baz": "qux"}

    def test_rfc_example_add_array_element(self):
        """RFC 6902 Section 4.1 example: add array element."""
        doc = {"foo": ["bar", "baz"]}
        patch = [JSONPatchOperation(op=PatchOp.ADD, path="/foo/1", value="qux")]
        result = apply_patch(doc, patch)
        assert result == {"foo": ["bar", "qux", "baz"]}

    def test_rfc_example_remove(self):
        """RFC 6902 Section 4.2 example: remove."""
        doc = {"baz": "qux", "foo": "bar"}
        patch = [JSONPatchOperation(op=PatchOp.REMOVE, path="/baz")]
        result = apply_patch(doc, patch)
        assert result == {"foo": "bar"}

    def test_rfc_example_replace(self):
        """RFC 6902 Section 4.3 example: replace."""
        doc = {"baz": "qux", "foo": "bar"}
        patch = [JSONPatchOperation(op=PatchOp.REPLACE, path="/baz", value="boo")]
        result = apply_patch(doc, patch)
        assert result == {"baz": "boo", "foo": "bar"}

    def test_rfc_example_move(self):
        """RFC 6902 Section 4.4 example: move."""
        doc = {"foo": {"bar": "baz", "waldo": "fred"}, "qux": {"corge": "grault"}}
        patch = [JSONPatchOperation(
            op=PatchOp.MOVE,
            path="/qux/thud",
            from_="/foo/waldo"
        )]
        result = apply_patch(doc, patch)
        assert result == {
            "foo": {"bar": "baz"},
            "qux": {"corge": "grault", "thud": "fred"}
        }

    def test_rfc_example_copy(self):
        """RFC 6902 Section 4.5 example: copy."""
        doc = {"foo": {"bar": "baz", "waldo": "fred"}, "qux": {"corge": "grault"}}
        patch = [JSONPatchOperation(
            op=PatchOp.COPY,
            path="/qux/thud",
            from_="/foo/waldo"
        )]
        result = apply_patch(doc, patch)
        assert result == {
            "foo": {"bar": "baz", "waldo": "fred"},
            "qux": {"corge": "grault", "thud": "fred"}
        }

    def test_rfc_example_test_success(self):
        """RFC 6902 Section 4.6 example: test success."""
        doc = {"baz": "qux", "foo": ["a", 2, "c"]}
        patch = [
            JSONPatchOperation(op=PatchOp.TEST, path="/baz", value="qux"),
            JSONPatchOperation(op=PatchOp.TEST, path="/foo/1", value=2),
        ]
        # Should not raise
        apply_patch(doc, patch)

    def test_rfc_example_test_failure(self):
        """RFC 6902 Section 4.6 example: test failure."""
        doc = {"baz": "qux"}
        patch = [JSONPatchOperation(op=PatchOp.TEST, path="/baz", value="bar")]
        with pytest.raises(PatchApplicationError):
            apply_patch(doc, patch)


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_patch(self):
        """Empty patch should return unchanged document."""
        doc = {"a": 1}
        result = apply_patch(doc, [])
        assert result == {"a": 1}

    def test_null_value(self):
        """Handle null values correctly."""
        doc = {}
        patch = [JSONPatchOperation(op=PatchOp.ADD, path="/null_val", value=None)]
        result = apply_patch(doc, patch)
        assert result == {"null_val": None}

    def test_deeply_nested_operation(self):
        """Handle deeply nested paths."""
        doc = {"a": {"b": {"c": {"d": 1}}}}
        patch = [JSONPatchOperation(
            op=PatchOp.REPLACE,
            path="/a/b/c/d",
            value=2
        )]
        result = apply_patch(doc, patch)
        assert result["a"]["b"]["c"]["d"] == 2

    def test_unicode_path(self):
        """Handle unicode in paths."""
        doc = {}
        patch = [JSONPatchOperation(
            op=PatchOp.ADD,
            path="/unicode_key",
            value="test"
        )]
        result = apply_patch(doc, patch)
        assert "unicode_key" in result

    def test_special_chars_in_keys(self):
        """Handle special characters in keys."""
        doc = {}
        # Key with tilde and slash
        key = "a/b~c"
        path = "/" + escape_json_pointer_token(key)
        patch = [JSONPatchOperation(op=PatchOp.ADD, path=path, value=1)]
        result = apply_patch(doc, patch)
        assert result[key] == 1

    def test_empty_string_key(self):
        """Handle empty string key."""
        doc = {"": "empty key value"}
        exists, value = get_value_at_pointer(doc, "/")
        assert exists is True
        assert value == "empty key value"

    def test_operation_order_matters(self):
        """Operations should be applied in order."""
        doc = {"x": 1}
        patch = [
            JSONPatchOperation(op=PatchOp.ADD, path="/y", value=2),
            JSONPatchOperation(op=PatchOp.REPLACE, path="/y", value=3),
        ]
        result = apply_patch(doc, patch)
        assert result["y"] == 3

    def test_test_precondition_prevents_replace(self):
        """Failed test should prevent subsequent operations."""
        doc = {"x": 1}
        patch = [
            JSONPatchOperation(op=PatchOp.TEST, path="/x", value=999),  # Fails
            JSONPatchOperation(op=PatchOp.REPLACE, path="/x", value=2),
        ]
        with pytest.raises(PatchApplicationError):
            apply_patch(doc, patch)
        # Original should be unchanged due to atomic failure
        assert doc["x"] == 1

    def test_array_bounds_checking(self):
        """Array index out of bounds should fail."""
        doc = {"arr": [1, 2, 3]}
        patch = [JSONPatchOperation(op=PatchOp.REPLACE, path="/arr/10", value=99)]
        with pytest.raises(PatchApplicationError):
            apply_patch(doc, patch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
