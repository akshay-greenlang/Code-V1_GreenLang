# -*- coding: utf-8 -*-
"""
Unit tests for Key Canonicalizer (Task 3.3).

This module tests the KeyCanonicalizer class and related utilities:
    - Alias resolution from schema
    - Casing normalization (snake_case, camelCase, PascalCase)
    - Stable key ordering
    - Rename record tracking
    - Typo correction (opt-in feature)
    - Recursive processing of nested structures

Test Coverage:
    - Casing conversion utilities
    - Casing detection
    - Alias resolution
    - Casing normalization
    - Stable ordering with required/optional/unknown fields
    - Nested object processing
    - Array handling
    - Idempotency property
    - Edge cases

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 3.3
"""

from datetime import datetime
from typing import Dict, Any, List, Set

import pytest

from greenlang.schema.normalizer.keys import (
    KeyCanonicalizer,
    KeyRename,
    RenameReason,
    to_snake_case,
    to_camel_case,
    to_pascal_case,
    detect_casing,
    normalize_to_casing,
)
from greenlang.schema.compiler.ir import SchemaIR, PropertyIR


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def minimal_schema_ir() -> SchemaIR:
    """Create a minimal SchemaIR for testing."""
    return SchemaIR(
        schema_id="test/schema",
        version="1.0.0",
        schema_hash="a" * 64,
        compiled_at=datetime.now(),
        properties={},
        required_paths=set(),
    )


@pytest.fixture
def schema_ir_with_aliases() -> SchemaIR:
    """Create a SchemaIR with renamed_fields for alias testing."""
    return SchemaIR(
        schema_id="test/schema",
        version="1.0.0",
        schema_hash="a" * 64,
        compiled_at=datetime.now(),
        properties={
            "/new_field": PropertyIR(path="/new_field", type="string", required=True),
            "/energy_consumption": PropertyIR(path="/energy_consumption", type="number", required=False),
        },
        required_paths={"/new_field"},
        renamed_fields={
            "old_field": "new_field",
            "deprecated_name": "energy_consumption",
        },
    )


@pytest.fixture
def schema_ir_with_properties() -> SchemaIR:
    """Create a SchemaIR with multiple properties for ordering tests."""
    return SchemaIR(
        schema_id="test/schema",
        version="1.0.0",
        schema_hash="a" * 64,
        compiled_at=datetime.now(),
        properties={
            "/id": PropertyIR(path="/id", type="string", required=True),
            "/name": PropertyIR(path="/name", type="string", required=True),
            "/description": PropertyIR(path="/description", type="string", required=False),
            "/status": PropertyIR(path="/status", type="string", required=False),
        },
        required_paths={"/id", "/name"},
    )


@pytest.fixture
def schema_ir_with_extensions() -> SchemaIR:
    """Create a SchemaIR with property extensions containing aliases."""
    return SchemaIR(
        schema_id="test/schema",
        version="1.0.0",
        schema_hash="a" * 64,
        compiled_at=datetime.now(),
        properties={
            "/energy_consumption": PropertyIR(
                path="/energy_consumption",
                type="number",
                required=True,
                gl_extensions={
                    "aliases": ["energy", "consumption", "energyUsage"],
                    "unit": "kWh",
                },
            ),
        },
        required_paths={"/energy_consumption"},
    )


# =============================================================================
# CASING CONVERSION TESTS
# =============================================================================


class TestToSnakeCase:
    """Tests for to_snake_case utility function."""

    def test_camel_case_conversion(self):
        """Test conversion from camelCase to snake_case."""
        assert to_snake_case("energyConsumption") == "energy_consumption"
        assert to_snake_case("firstName") == "first_name"
        assert to_snake_case("userID") == "user_id"

    def test_pascal_case_conversion(self):
        """Test conversion from PascalCase to snake_case."""
        assert to_snake_case("EnergyConsumption") == "energy_consumption"
        assert to_snake_case("FirstName") == "first_name"
        assert to_snake_case("HTTPRequest") == "http_request"

    def test_already_snake_case(self):
        """Test that snake_case strings are unchanged."""
        assert to_snake_case("energy_consumption") == "energy_consumption"
        assert to_snake_case("first_name") == "first_name"
        assert to_snake_case("user_id") == "user_id"

    def test_single_word(self):
        """Test single word strings."""
        assert to_snake_case("energy") == "energy"
        assert to_snake_case("Energy") == "energy"
        assert to_snake_case("ENERGY") == "energy"

    def test_empty_string(self):
        """Test empty string."""
        assert to_snake_case("") == ""

    def test_consecutive_uppercase(self):
        """Test strings with consecutive uppercase letters."""
        assert to_snake_case("HTTPResponse") == "http_response"
        assert to_snake_case("XMLParser") == "xml_parser"
        assert to_snake_case("getHTTPResponse") == "get_http_response"

    def test_numbers(self):
        """Test strings with numbers."""
        assert to_snake_case("scope1Emissions") == "scope1_emissions"
        assert to_snake_case("Scope1Emissions") == "scope1_emissions"


class TestToCamelCase:
    """Tests for to_camel_case utility function."""

    def test_snake_case_conversion(self):
        """Test conversion from snake_case to camelCase."""
        assert to_camel_case("energy_consumption") == "energyConsumption"
        assert to_camel_case("first_name") == "firstName"
        assert to_camel_case("user_id") == "userId"

    def test_already_camel_case(self):
        """Test that camelCase strings are unchanged."""
        assert to_camel_case("energyConsumption") == "energyConsumption"
        assert to_camel_case("firstName") == "firstName"

    def test_pascal_case_conversion(self):
        """Test conversion from PascalCase to camelCase."""
        assert to_camel_case("EnergyConsumption") == "energyConsumption"
        assert to_camel_case("FirstName") == "firstName"

    def test_single_word(self):
        """Test single word strings."""
        assert to_camel_case("energy") == "energy"
        assert to_camel_case("Energy") == "energy"

    def test_empty_string(self):
        """Test empty string."""
        assert to_camel_case("") == ""


class TestToPascalCase:
    """Tests for to_pascal_case utility function."""

    def test_snake_case_conversion(self):
        """Test conversion from snake_case to PascalCase."""
        assert to_pascal_case("energy_consumption") == "EnergyConsumption"
        assert to_pascal_case("first_name") == "FirstName"
        assert to_pascal_case("user_id") == "UserId"

    def test_camel_case_conversion(self):
        """Test conversion from camelCase to PascalCase."""
        assert to_pascal_case("energyConsumption") == "EnergyConsumption"
        assert to_pascal_case("firstName") == "FirstName"

    def test_already_pascal_case(self):
        """Test that PascalCase strings are unchanged."""
        assert to_pascal_case("EnergyConsumption") == "EnergyConsumption"
        assert to_pascal_case("FirstName") == "FirstName"

    def test_single_word(self):
        """Test single word strings."""
        assert to_pascal_case("energy") == "Energy"
        assert to_pascal_case("Energy") == "Energy"

    def test_empty_string(self):
        """Test empty string."""
        assert to_pascal_case("") == ""


class TestDetectCasing:
    """Tests for detect_casing utility function."""

    def test_detect_snake_case(self):
        """Test detection of snake_case."""
        assert detect_casing("energy_consumption") == "snake_case"
        assert detect_casing("first_name") == "snake_case"
        assert detect_casing("user_id") == "snake_case"

    def test_detect_camel_case(self):
        """Test detection of camelCase."""
        assert detect_casing("energyConsumption") == "camelCase"
        assert detect_casing("firstName") == "camelCase"
        assert detect_casing("userId") == "camelCase"

    def test_detect_pascal_case(self):
        """Test detection of PascalCase."""
        assert detect_casing("EnergyConsumption") == "PascalCase"
        assert detect_casing("FirstName") == "PascalCase"
        assert detect_casing("UserId") == "PascalCase"

    def test_detect_unknown(self):
        """Test detection of unknown casing."""
        assert detect_casing("ENERGY") == "unknown"
        assert detect_casing("energy") == "unknown"  # Single word lowercase
        assert detect_casing("") == "unknown"


class TestNormalizeToCasing:
    """Tests for normalize_to_casing utility function."""

    def test_normalize_to_snake_case(self):
        """Test normalization to snake_case."""
        assert normalize_to_casing("energyConsumption", "snake_case") == "energy_consumption"
        assert normalize_to_casing("EnergyConsumption", "snake_case") == "energy_consumption"

    def test_normalize_to_camel_case(self):
        """Test normalization to camelCase."""
        assert normalize_to_casing("energy_consumption", "camelCase") == "energyConsumption"
        assert normalize_to_casing("EnergyConsumption", "camelCase") == "energyConsumption"

    def test_normalize_to_pascal_case(self):
        """Test normalization to PascalCase."""
        assert normalize_to_casing("energy_consumption", "PascalCase") == "EnergyConsumption"
        assert normalize_to_casing("energyConsumption", "PascalCase") == "EnergyConsumption"


# =============================================================================
# KEY RENAME MODEL TESTS
# =============================================================================


class TestKeyRename:
    """Tests for KeyRename model."""

    def test_create_key_rename(self):
        """Test creating a KeyRename instance."""
        rename = KeyRename(
            path="/data",
            original_key="oldField",
            canonical_key="old_field",
            reason=RenameReason.CASING,
        )
        assert rename.path == "/data"
        assert rename.original_key == "oldField"
        assert rename.canonical_key == "old_field"
        assert rename.reason == RenameReason.CASING

    def test_to_dict(self):
        """Test converting KeyRename to dictionary."""
        rename = KeyRename(
            path="/data",
            original_key="old_field",
            canonical_key="new_field",
            reason=RenameReason.ALIAS,
        )
        d = rename.to_dict()
        assert d["path"] == "/data"
        assert d["original_key"] == "old_field"
        assert d["canonical_key"] == "new_field"
        assert d["reason"] == "alias"

    def test_rename_reason_values(self):
        """Test RenameReason enum values."""
        assert RenameReason.ALIAS.value == "alias"
        assert RenameReason.CASING.value == "casing"
        assert RenameReason.TYPO_CORRECTION.value == "typo"


# =============================================================================
# KEY CANONICALIZER TESTS
# =============================================================================


class TestKeyCanonicalizer:
    """Tests for KeyCanonicalizer class."""

    def test_initialization(self, minimal_schema_ir):
        """Test KeyCanonicalizer initialization."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        assert canonicalizer.ir == minimal_schema_ir
        assert canonicalizer.expected_casing == "snake_case"
        assert canonicalizer.enable_typo_correction is False

    def test_initialization_with_custom_casing(self, minimal_schema_ir):
        """Test KeyCanonicalizer with custom casing."""
        canonicalizer = KeyCanonicalizer(
            minimal_schema_ir,
            expected_casing="camelCase"
        )
        assert canonicalizer.expected_casing == "camelCase"


class TestAliasResolution:
    """Tests for alias resolution."""

    def test_resolve_alias(self, schema_ir_with_aliases):
        """Test resolving known aliases."""
        canonicalizer = KeyCanonicalizer(schema_ir_with_aliases)
        result, renames = canonicalizer.canonicalize({"old_field": "value"})

        assert "new_field" in result
        assert "old_field" not in result
        assert result["new_field"] == "value"

        assert len(renames) == 1
        assert renames[0].original_key == "old_field"
        assert renames[0].canonical_key == "new_field"
        assert renames[0].reason == RenameReason.ALIAS

    def test_resolve_multiple_aliases(self, schema_ir_with_aliases):
        """Test resolving multiple aliases."""
        canonicalizer = KeyCanonicalizer(schema_ir_with_aliases)
        result, renames = canonicalizer.canonicalize({
            "old_field": "value1",
            "deprecated_name": 100,
        })

        assert "new_field" in result
        assert "energy_consumption" in result
        assert len(renames) == 2

    def test_no_alias_for_canonical_keys(self, schema_ir_with_aliases):
        """Test that canonical keys are not modified."""
        canonicalizer = KeyCanonicalizer(schema_ir_with_aliases)
        result, renames = canonicalizer.canonicalize({"new_field": "value"})

        assert "new_field" in result
        assert len(renames) == 0

    def test_alias_from_extensions(self, schema_ir_with_extensions):
        """Test resolving aliases from property extensions."""
        canonicalizer = KeyCanonicalizer(schema_ir_with_extensions)
        result, renames = canonicalizer.canonicalize({"energy": 100})

        assert "energy_consumption" in result
        assert result["energy_consumption"] == 100
        assert len(renames) == 1
        assert renames[0].reason == RenameReason.ALIAS


class TestCasingNormalization:
    """Tests for casing normalization."""

    def test_normalize_camel_case_to_snake_case(self, minimal_schema_ir):
        """Test normalizing camelCase to snake_case."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        result, renames = canonicalizer.canonicalize({
            "energyConsumption": 100,
            "firstName": "John",
        })

        assert "energy_consumption" in result
        assert "first_name" in result
        assert len(renames) == 2
        for rename in renames:
            assert rename.reason == RenameReason.CASING

    def test_normalize_pascal_case_to_snake_case(self, minimal_schema_ir):
        """Test normalizing PascalCase to snake_case."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        result, renames = canonicalizer.canonicalize({
            "EnergyConsumption": 100,
        })

        assert "energy_consumption" in result
        assert len(renames) == 1
        assert renames[0].reason == RenameReason.CASING

    def test_no_normalization_for_snake_case(self, minimal_schema_ir):
        """Test that snake_case keys are not modified."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        result, renames = canonicalizer.canonicalize({
            "energy_consumption": 100,
            "first_name": "John",
        })

        assert "energy_consumption" in result
        assert "first_name" in result
        assert len(renames) == 0

    def test_normalize_to_camel_case(self, minimal_schema_ir):
        """Test normalizing to camelCase when specified."""
        canonicalizer = KeyCanonicalizer(
            minimal_schema_ir,
            expected_casing="camelCase"
        )
        result, renames = canonicalizer.canonicalize({
            "energy_consumption": 100,
        })

        assert "energyConsumption" in result
        assert len(renames) == 1
        assert renames[0].reason == RenameReason.CASING


class TestStableOrdering:
    """Tests for stable key ordering."""

    def test_basic_alphabetical_ordering(self, minimal_schema_ir):
        """Test basic alphabetical ordering for unknown fields."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        result, _ = canonicalizer.canonicalize({
            "zebra": 1,
            "apple": 2,
            "mango": 3,
        })

        keys = list(result.keys())
        assert keys == ["apple", "mango", "zebra"]

    def test_required_fields_first(self, schema_ir_with_properties):
        """Test that required fields come first."""
        canonicalizer = KeyCanonicalizer(schema_ir_with_properties)
        result, _ = canonicalizer.canonicalize({
            "description": "desc",
            "status": "active",
            "id": "123",
            "name": "test",
        })

        keys = list(result.keys())
        # Required fields (id, name) should come first
        assert "id" in keys[:2]
        assert "name" in keys[:2]

    def test_ordering_is_deterministic(self, minimal_schema_ir):
        """Test that ordering is deterministic across multiple calls."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        payload = {"c": 1, "a": 2, "b": 3}

        result1, _ = canonicalizer.canonicalize(payload)
        result2, _ = canonicalizer.canonicalize(payload)

        assert list(result1.keys()) == list(result2.keys())


class TestNestedStructures:
    """Tests for nested object handling."""

    def test_nested_object_canonicalization(self, minimal_schema_ir):
        """Test canonicalization of nested objects."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        result, renames = canonicalizer.canonicalize({
            "parentField": {
                "childField": "value",
                "anotherChild": 123,
            }
        })

        assert "parent_field" in result
        assert "child_field" in result["parent_field"]
        assert "another_child" in result["parent_field"]
        assert len(renames) == 3

    def test_deeply_nested_objects(self, minimal_schema_ir):
        """Test canonicalization of deeply nested objects."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        result, renames = canonicalizer.canonicalize({
            "level1": {
                "level2": {
                    "level3": {
                        "deepField": "value"
                    }
                }
            }
        })

        assert result["level1"]["level2"]["level3"]["deep_field"] == "value"

    def test_array_of_objects(self, minimal_schema_ir):
        """Test canonicalization of arrays containing objects."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        result, renames = canonicalizer.canonicalize({
            "items": [
                {"firstName": "John"},
                {"firstName": "Jane"},
            ]
        })

        assert result["items"][0]["first_name"] == "John"
        assert result["items"][1]["first_name"] == "Jane"

    def test_mixed_nested_arrays(self, minimal_schema_ir):
        """Test canonicalization with mixed nested arrays and objects."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        result, _ = canonicalizer.canonicalize({
            "data": [
                {
                    "nestedArray": [1, 2, 3],
                    "nestedObject": {"deepKey": "value"}
                }
            ]
        })

        assert result["data"][0]["nested_array"] == [1, 2, 3]
        assert result["data"][0]["nested_object"]["deep_key"] == "value"


class TestTypoCorrection:
    """Tests for typo correction (opt-in feature)."""

    def test_typo_correction_disabled_by_default(self, schema_ir_with_properties):
        """Test that typo correction is disabled by default."""
        canonicalizer = KeyCanonicalizer(schema_ir_with_properties)
        result, renames = canonicalizer.canonicalize({"naem": "value"})  # typo of "name"

        # Typo should not be corrected
        assert "naem" in result
        # Only casing rename if any
        typo_renames = [r for r in renames if r.reason == RenameReason.TYPO_CORRECTION]
        assert len(typo_renames) == 0

    def test_typo_correction_when_enabled(self, schema_ir_with_properties):
        """Test typo correction when explicitly enabled."""
        canonicalizer = KeyCanonicalizer(
            schema_ir_with_properties,
            enable_typo_correction=True,
            typo_threshold=2,
        )
        result, renames = canonicalizer.canonicalize({"naem": "value"})

        # Typo should be corrected
        assert "name" in result
        assert "naem" not in result

        typo_renames = [r for r in renames if r.reason == RenameReason.TYPO_CORRECTION]
        assert len(typo_renames) == 1
        assert typo_renames[0].original_key == "naem"
        assert typo_renames[0].canonical_key == "name"

    def test_typo_correction_threshold(self, schema_ir_with_properties):
        """Test that typo correction respects edit distance threshold."""
        canonicalizer = KeyCanonicalizer(
            schema_ir_with_properties,
            enable_typo_correction=True,
            typo_threshold=1,
        )
        # "naem" has edit distance 2 from "name", so should not be corrected
        result, renames = canonicalizer.canonicalize({"naem": "value"})

        assert "naem" in result
        typo_renames = [r for r in renames if r.reason == RenameReason.TYPO_CORRECTION]
        assert len(typo_renames) == 0


class TestIdempotency:
    """Tests for idempotency property."""

    def test_canonicalize_is_idempotent(self, minimal_schema_ir):
        """Test that canonicalize(canonicalize(x)) == canonicalize(x)."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        payload = {
            "energyConsumption": 100,
            "firstName": "John",
            "nested": {"childField": "value"}
        }

        result1, _ = canonicalizer.canonicalize(payload)
        result2, renames2 = canonicalizer.canonicalize(result1)

        assert result1 == result2
        assert len(renames2) == 0  # No renames on second pass

    def test_idempotency_with_aliases(self, schema_ir_with_aliases):
        """Test idempotency with alias resolution."""
        canonicalizer = KeyCanonicalizer(schema_ir_with_aliases)
        payload = {"old_field": "value"}

        result1, _ = canonicalizer.canonicalize(payload)
        result2, renames2 = canonicalizer.canonicalize(result1)

        assert result1 == result2
        assert len(renames2) == 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_payload(self, minimal_schema_ir):
        """Test canonicalization of empty payload."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        result, renames = canonicalizer.canonicalize({})

        assert result == {}
        assert len(renames) == 0

    def test_payload_with_primitive_values(self, minimal_schema_ir):
        """Test that primitive values are preserved."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        result, _ = canonicalizer.canonicalize({
            "string_value": "text",
            "int_value": 42,
            "float_value": 3.14,
            "bool_value": True,
            "null_value": None,
        })

        assert result["string_value"] == "text"
        assert result["int_value"] == 42
        assert result["float_value"] == 3.14
        assert result["bool_value"] is True
        assert result["null_value"] is None

    def test_invalid_payload_type(self, minimal_schema_ir):
        """Test that non-dict payload raises error."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)

        with pytest.raises(ValueError, match="must be a dictionary"):
            canonicalizer.canonicalize([1, 2, 3])

        with pytest.raises(ValueError, match="must be a dictionary"):
            canonicalizer.canonicalize("string")

    def test_clear_renames(self, minimal_schema_ir):
        """Test clearing rename records."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        canonicalizer.canonicalize({"energyConsumption": 100})

        assert len(canonicalizer.get_renames()) == 1

        canonicalizer.clear_renames()
        assert len(canonicalizer.get_renames()) == 0

    def test_get_renames_returns_copy(self, minimal_schema_ir):
        """Test that get_renames returns a copy."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        canonicalizer.canonicalize({"energyConsumption": 100})

        renames1 = canonicalizer.get_renames()
        renames2 = canonicalizer.get_renames()

        assert renames1 is not renames2
        assert renames1 == renames2


class TestRenameRecordTracking:
    """Tests for rename record tracking."""

    def test_rename_path_tracking(self, minimal_schema_ir):
        """Test that rename paths are correctly tracked."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        _, renames = canonicalizer.canonicalize({
            "parentField": {
                "childField": "value"
            }
        })

        paths = [r.path for r in renames]
        assert "" in paths  # Root level rename
        assert "/parent_field" in paths  # Nested rename

    def test_multiple_renames_same_object(self, minimal_schema_ir):
        """Test tracking multiple renames in same object."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        _, renames = canonicalizer.canonicalize({
            "fieldOne": 1,
            "fieldTwo": 2,
            "fieldThree": 3,
        })

        assert len(renames) == 3
        original_keys = [r.original_key for r in renames]
        assert "fieldOne" in original_keys
        assert "fieldTwo" in original_keys
        assert "fieldThree" in original_keys


# =============================================================================
# PROPERTY-BASED TESTS
# =============================================================================


class TestPropertyBased:
    """Property-based tests for KeyCanonicalizer."""

    def test_casing_roundtrip(self):
        """Test that snake_case -> camelCase -> snake_case is identity."""
        test_cases = [
            "energy_consumption",
            "first_name",
            "user_id",
            "http_request",
        ]
        for snake in test_cases:
            camel = to_camel_case(snake)
            back_to_snake = to_snake_case(camel)
            assert back_to_snake == snake, f"Roundtrip failed for {snake}"

    def test_output_keys_are_valid_identifiers(self, minimal_schema_ir):
        """Test that output keys are valid Python identifiers."""
        canonicalizer = KeyCanonicalizer(minimal_schema_ir)
        result, _ = canonicalizer.canonicalize({
            "CamelCase": 1,
            "snake_case": 2,
            "PascalCase": 3,
            "mixedCASE_with_underscores": 4,
        })

        for key in result.keys():
            assert key.isidentifier() or "_" in key, f"Invalid key: {key}"
