# -*- coding: utf-8 -*-
"""
Unit Tests for NormalizationEngine - GL-FOUND-X-002 Task 3.4

This module provides comprehensive unit tests for the normalization engine,
covering all normalization steps and validating the key properties:
- Determinism: Same input always produces same output
- Idempotency: normalize(normalize(x)) == normalize(x)
- Non-destructiveness: No semantic information is lost

Test Categories:
    - Basic normalization tests
    - Key canonicalization tests
    - Default application tests
    - Type coercion tests
    - Unit canonicalization tests
    - Meta block tests
    - Property tests (idempotency, determinism)

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 3.4
"""

from datetime import datetime, timezone
from typing import Any, Dict

import pytest

from greenlang.schema.normalizer.engine import (
    NormalizationEngine,
    NormalizationMeta,
    NormalizationResult,
    normalize,
    is_normalization_idempotent,
    _remove_meta_block,
)
from greenlang.schema.normalizer.coercions import CoercionRecord
from greenlang.schema.normalizer.canonicalizer import ConversionRecord
from greenlang.schema.normalizer.keys import KeyRename, RenameReason
from greenlang.schema.compiler.ir import (
    SchemaIR,
    PropertyIR,
    UnitSpecIR,
)
from greenlang.schema.units.catalog import UnitCatalog
from greenlang.schema.models.config import (
    ValidationOptions,
    CoercionPolicy,
    ValidationProfile,
)
from greenlang.schema.models.schema_ref import SchemaRef


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def unit_catalog() -> UnitCatalog:
    """Create a unit catalog with standard units."""
    return UnitCatalog()


@pytest.fixture
def default_options() -> ValidationOptions:
    """Create default validation options."""
    return ValidationOptions()


@pytest.fixture
def strict_options() -> ValidationOptions:
    """Create strict validation options (no coercion)."""
    return ValidationOptions.strict()


@pytest.fixture
def permissive_options() -> ValidationOptions:
    """Create permissive validation options."""
    return ValidationOptions.permissive()


@pytest.fixture
def minimal_ir() -> SchemaIR:
    """Create a minimal SchemaIR for testing."""
    return SchemaIR(
        schema_id="test/minimal",
        version="1.0.0",
        schema_hash="a" * 64,
        compiled_at=datetime.now(timezone.utc),
        properties={},
        required_paths=set(),
    )


@pytest.fixture
def simple_ir() -> SchemaIR:
    """Create a simple SchemaIR with a few properties."""
    return SchemaIR(
        schema_id="test/simple",
        version="1.0.0",
        schema_hash="b" * 64,
        compiled_at=datetime.now(timezone.utc),
        properties={
            "/name": PropertyIR(path="/name", type="string", required=True),
            "/age": PropertyIR(path="/age", type="integer", required=False),
            "/active": PropertyIR(path="/active", type="boolean", required=False, has_default=True, default_value=True),
        },
        required_paths={"/name"},
    )


@pytest.fixture
def ir_with_units() -> SchemaIR:
    """Create a SchemaIR with unit specifications."""
    return SchemaIR(
        schema_id="test/units",
        version="1.0.0",
        schema_hash="c" * 64,
        compiled_at=datetime.now(timezone.utc),
        properties={
            "/energy": PropertyIR(path="/energy", type="object", required=True),
            "/energy/value": PropertyIR(path="/energy/value", type="number", required=True),
            "/energy/unit": PropertyIR(path="/energy/unit", type="string", required=True),
        },
        required_paths={"/energy"},
        unit_specs={
            "/energy": UnitSpecIR(
                path="/energy",
                dimension="energy",
                canonical="kWh",
                allowed=["kWh", "MWh", "GWh", "Wh"],
            ),
        },
    )


@pytest.fixture
def ir_with_aliases() -> SchemaIR:
    """Create a SchemaIR with field aliases."""
    return SchemaIR(
        schema_id="test/aliases",
        version="1.0.0",
        schema_hash="d" * 64,
        compiled_at=datetime.now(timezone.utc),
        properties={
            "/energy_consumption": PropertyIR(path="/energy_consumption", type="number", required=True),
        },
        required_paths={"/energy_consumption"},
        renamed_fields={
            "Energy": "energy_consumption",
            "energyConsumption": "energy_consumption",
            "power_usage": "energy_consumption",
        },
    )


@pytest.fixture
def ir_with_defaults() -> SchemaIR:
    """Create a SchemaIR with default values."""
    return SchemaIR(
        schema_id="test/defaults",
        version="1.0.0",
        schema_hash="e" * 64,
        compiled_at=datetime.now(timezone.utc),
        properties={
            "/name": PropertyIR(path="/name", type="string", required=True),
            "/status": PropertyIR(
                path="/status",
                type="string",
                required=False,
                has_default=True,
                default_value="active",
            ),
            "/count": PropertyIR(
                path="/count",
                type="integer",
                required=False,
                has_default=True,
                default_value=0,
            ),
            "/tags": PropertyIR(
                path="/tags",
                type="array",
                required=False,
                has_default=True,
                default_value=[],
            ),
        },
        required_paths={"/name"},
    )


# =============================================================================
# TEST CLASS: NormalizationMeta
# =============================================================================


class TestNormalizationMeta:
    """Tests for NormalizationMeta model."""

    def test_default_values(self):
        """Test that default values are applied correctly."""
        meta = NormalizationMeta()

        assert meta.schema_ref is None
        assert meta.normalized_at is not None
        assert meta.coercions == []
        assert meta.conversions == []
        assert meta.renames == []
        assert meta.defaults_applied == []
        assert meta.provenance_hash is None

    def test_has_changes_empty(self):
        """Test has_changes returns False when no changes."""
        meta = NormalizationMeta()
        assert meta.has_changes() is False

    def test_has_changes_with_coercions(self):
        """Test has_changes returns True when coercions present."""
        meta = NormalizationMeta(
            coercions=[
                CoercionRecord(
                    path="/test",
                    original_value="42",
                    original_type="string",
                    coerced_value=42,
                    coerced_type="integer",
                    reversible=True,
                    coercion_type="string_to_integer",
                )
            ]
        )
        assert meta.has_changes() is True

    def test_has_changes_with_renames(self):
        """Test has_changes returns True when renames present."""
        meta = NormalizationMeta(
            renames=[
                KeyRename(
                    path="/",
                    original_key="OldName",
                    canonical_key="old_name",
                    reason=RenameReason.CASING,
                )
            ]
        )
        assert meta.has_changes() is True

    def test_has_changes_with_defaults(self):
        """Test has_changes returns True when defaults applied."""
        meta = NormalizationMeta(defaults_applied=["/optional_field"])
        assert meta.has_changes() is True

    def test_to_dict(self):
        """Test to_dict produces valid dictionary."""
        meta = NormalizationMeta(
            schema_ref=SchemaRef(schema_id="test", version="1.0.0"),
            defaults_applied=["/status"],
            provenance_hash="abc123",
        )
        result = meta.to_dict()

        assert "normalized_at" in result
        assert result["schema_ref"] == "gl://schemas/test@1.0.0"
        assert result["defaults_applied"] == ["/status"]
        assert result["provenance_hash"] == "abc123"


# =============================================================================
# TEST CLASS: NormalizationResult
# =============================================================================


class TestNormalizationResult:
    """Tests for NormalizationResult model."""

    def test_basic_creation(self):
        """Test basic NormalizationResult creation."""
        result = NormalizationResult(
            normalized={"key": "value"},
            meta=NormalizationMeta(),
            is_modified=False,
        )

        assert result.normalized == {"key": "value"}
        assert result.is_modified is False

    def test_count_properties(self):
        """Test count property methods."""
        meta = NormalizationMeta(
            coercions=[
                CoercionRecord(
                    path="/a", original_value="1", original_type="string",
                    coerced_value=1, coerced_type="integer", reversible=True,
                    coercion_type="string_to_integer"
                ),
                CoercionRecord(
                    path="/b", original_value="2", original_type="string",
                    coerced_value=2, coerced_type="integer", reversible=True,
                    coercion_type="string_to_integer"
                ),
            ],
            defaults_applied=["/c", "/d", "/e"],
        )
        result = NormalizationResult(
            normalized={},
            meta=meta,
            is_modified=True,
        )

        assert result.coercion_count == 2
        assert result.conversion_count == 0
        assert result.rename_count == 0
        assert result.default_count == 3


# =============================================================================
# TEST CLASS: NormalizationEngine Basic
# =============================================================================


class TestNormalizationEngineBasic:
    """Basic tests for NormalizationEngine."""

    def test_initialization(self, minimal_ir, unit_catalog, default_options):
        """Test engine initialization."""
        engine = NormalizationEngine(minimal_ir, unit_catalog, default_options)

        assert engine.ir is minimal_ir
        assert engine.catalog is unit_catalog
        assert engine.options is default_options

    def test_normalize_empty_payload(self, minimal_ir, unit_catalog, default_options):
        """Test normalizing an empty payload."""
        engine = NormalizationEngine(minimal_ir, unit_catalog, default_options)
        result = engine.normalize({})

        assert result.normalized == {}
        assert result.is_modified is False

    def test_normalize_simple_payload(self, simple_ir, unit_catalog, default_options):
        """Test normalizing a simple payload."""
        engine = NormalizationEngine(simple_ir, unit_catalog, default_options)
        result = engine.normalize({"name": "Test"})

        assert "name" in result.normalized
        assert result.normalized["name"] == "Test"

    def test_normalize_preserves_original(self, simple_ir, unit_catalog, default_options):
        """Test that normalization does not modify original payload."""
        engine = NormalizationEngine(simple_ir, unit_catalog, default_options)
        original = {"name": "Test", "age": 25}
        original_copy = {"name": "Test", "age": 25}

        engine.normalize(original)

        assert original == original_copy

    def test_normalize_rejects_non_dict(self, minimal_ir, unit_catalog, default_options):
        """Test that non-dict payloads are rejected."""
        engine = NormalizationEngine(minimal_ir, unit_catalog, default_options)

        with pytest.raises(ValueError, match="must be a dictionary"):
            engine.normalize("not a dict")  # type: ignore

        with pytest.raises(ValueError, match="must be a dictionary"):
            engine.normalize([1, 2, 3])  # type: ignore

    def test_normalize_disabled(self, simple_ir, unit_catalog):
        """Test normalization when disabled in options."""
        options = ValidationOptions(normalize=False)
        engine = NormalizationEngine(simple_ir, unit_catalog, options)
        payload = {"name": "Test"}

        result = engine.normalize(payload)

        assert result.is_modified is False
        assert result.normalized == payload


# =============================================================================
# TEST CLASS: Default Application
# =============================================================================


class TestDefaultApplication:
    """Tests for default value application."""

    def test_apply_missing_defaults(self, ir_with_defaults, unit_catalog, default_options):
        """Test that missing optional fields get defaults applied."""
        engine = NormalizationEngine(ir_with_defaults, unit_catalog, default_options)
        result = engine.normalize({"name": "Test"})

        # Defaults should be applied
        assert result.normalized.get("status") == "active"
        assert result.normalized.get("count") == 0
        assert result.normalized.get("tags") == []
        assert result.is_modified is True
        assert len(result.meta.defaults_applied) == 3

    def test_no_override_existing_values(self, ir_with_defaults, unit_catalog, default_options):
        """Test that existing values are not overridden by defaults."""
        engine = NormalizationEngine(ir_with_defaults, unit_catalog, default_options)
        result = engine.normalize({
            "name": "Test",
            "status": "inactive",
            "count": 42,
        })

        assert result.normalized.get("status") == "inactive"
        assert result.normalized.get("count") == 42
        # Only tags default should be applied
        assert "/tags" in result.meta.defaults_applied
        assert "/status" not in result.meta.defaults_applied
        assert "/count" not in result.meta.defaults_applied

    def test_deep_copy_defaults(self, ir_with_defaults, unit_catalog, default_options):
        """Test that mutable defaults are deep copied."""
        engine = NormalizationEngine(ir_with_defaults, unit_catalog, default_options)

        result1 = engine.normalize({"name": "Test1"})
        result2 = engine.normalize({"name": "Test2"})

        # Mutate the first result's tags
        result1.normalized["tags"].append("tag1")

        # Second result should be unaffected
        assert result2.normalized["tags"] == []


# =============================================================================
# TEST CLASS: Type Coercion
# =============================================================================


class TestTypeCoercion:
    """Tests for type coercion during normalization."""

    def test_string_to_integer_coercion(self, unit_catalog, default_options):
        """Test coercing string to integer."""
        ir = SchemaIR(
            schema_id="test/coercion",
            version="1.0.0",
            schema_hash="f" * 64,
            compiled_at=datetime.now(timezone.utc),
            properties={
                "/count": PropertyIR(path="/count", type="integer", required=True),
            },
            required_paths={"/count"},
        )
        engine = NormalizationEngine(ir, unit_catalog, default_options)
        result = engine.normalize({"count": "42"})

        assert result.normalized["count"] == 42
        assert len(result.meta.coercions) == 1

    def test_string_to_number_coercion(self, unit_catalog, default_options):
        """Test coercing string to number."""
        ir = SchemaIR(
            schema_id="test/coercion",
            version="1.0.0",
            schema_hash="f" * 64,
            compiled_at=datetime.now(timezone.utc),
            properties={
                "/value": PropertyIR(path="/value", type="number", required=True),
            },
            required_paths={"/value"},
        )
        engine = NormalizationEngine(ir, unit_catalog, default_options)
        result = engine.normalize({"value": "3.14"})

        assert result.normalized["value"] == 3.14
        assert len(result.meta.coercions) == 1

    def test_string_to_boolean_coercion(self, unit_catalog, default_options):
        """Test coercing string to boolean."""
        ir = SchemaIR(
            schema_id="test/coercion",
            version="1.0.0",
            schema_hash="f" * 64,
            compiled_at=datetime.now(timezone.utc),
            properties={
                "/active": PropertyIR(path="/active", type="boolean", required=True),
            },
            required_paths={"/active"},
        )
        engine = NormalizationEngine(ir, unit_catalog, default_options)
        result = engine.normalize({"active": "true"})

        assert result.normalized["active"] is True
        assert len(result.meta.coercions) == 1

    def test_coercion_disabled(self, unit_catalog, strict_options):
        """Test that coercion is disabled in strict mode."""
        ir = SchemaIR(
            schema_id="test/coercion",
            version="1.0.0",
            schema_hash="f" * 64,
            compiled_at=datetime.now(timezone.utc),
            properties={
                "/count": PropertyIR(path="/count", type="integer", required=True),
            },
            required_paths={"/count"},
        )
        engine = NormalizationEngine(ir, unit_catalog, strict_options)
        result = engine.normalize({"count": "42"})

        # Value should remain a string since coercion is disabled
        assert result.normalized["count"] == "42"
        assert len(result.meta.coercions) == 0


# =============================================================================
# TEST CLASS: Key Canonicalization
# =============================================================================


class TestKeyCanonicalization:
    """Tests for key canonicalization."""

    def test_alias_resolution(self, ir_with_aliases, unit_catalog, default_options):
        """Test that aliases are resolved to canonical names."""
        engine = NormalizationEngine(ir_with_aliases, unit_catalog, default_options)
        result = engine.normalize({"energyConsumption": 100})

        assert "energy_consumption" in result.normalized
        assert result.normalized["energy_consumption"] == 100
        assert len(result.meta.renames) >= 1

    def test_casing_normalization(self, minimal_ir, unit_catalog, default_options):
        """Test that casing is normalized to snake_case."""
        engine = NormalizationEngine(minimal_ir, unit_catalog, default_options)
        result = engine.normalize({"EnergyValue": 100, "totalEmissions": 50})

        # Keys should be in snake_case
        assert "energy_value" in result.normalized
        assert "total_emissions" in result.normalized


# =============================================================================
# TEST CLASS: Unit Canonicalization
# =============================================================================


class TestUnitCanonicalization:
    """Tests for unit canonicalization."""

    def test_unit_conversion(self, ir_with_units, unit_catalog, default_options):
        """Test that units are converted to canonical form."""
        engine = NormalizationEngine(ir_with_units, unit_catalog, default_options)
        result = engine.normalize({
            "energy": {"value": 1000, "unit": "Wh"}
        })

        # Value should be converted from Wh to kWh
        assert result.normalized["energy"]["value"] == 1.0
        assert result.normalized["energy"]["unit"] == "kWh"
        assert len(result.meta.conversions) == 1

    def test_no_conversion_needed(self, ir_with_units, unit_catalog, default_options):
        """Test that canonical units are not converted."""
        engine = NormalizationEngine(ir_with_units, unit_catalog, default_options)
        result = engine.normalize({
            "energy": {"value": 100, "unit": "kWh"}
        })

        # Value should remain unchanged
        assert result.normalized["energy"]["value"] == 100
        assert result.normalized["energy"]["unit"] == "kWh"
        assert len(result.meta.conversions) == 0


# =============================================================================
# TEST CLASS: Meta Block
# =============================================================================


class TestMetaBlock:
    """Tests for _meta block generation."""

    def test_meta_block_when_changes(self, ir_with_defaults, unit_catalog, default_options):
        """Test that _meta block is added when changes are made."""
        engine = NormalizationEngine(ir_with_defaults, unit_catalog, default_options)
        result = engine.normalize({"name": "Test"})

        assert "_meta" in result.normalized
        assert "normalized_at" in result.normalized["_meta"]
        assert "defaults_applied" in result.normalized["_meta"]

    def test_no_meta_block_when_no_changes(self, minimal_ir, unit_catalog, default_options):
        """Test that _meta block is not added when no changes."""
        engine = NormalizationEngine(minimal_ir, unit_catalog, default_options)
        result = engine.normalize({"key": "value"})

        assert "_meta" not in result.normalized

    def test_provenance_hash_included(self, ir_with_defaults, unit_catalog, default_options):
        """Test that provenance hash is included in meta."""
        engine = NormalizationEngine(ir_with_defaults, unit_catalog, default_options)
        result = engine.normalize({"name": "Test"})

        assert result.meta.provenance_hash is not None
        assert len(result.meta.provenance_hash) == 64  # SHA-256 hex length


# =============================================================================
# TEST CLASS: Idempotency
# =============================================================================


class TestIdempotency:
    """Tests for normalization idempotency."""

    def test_idempotent_simple(self, simple_ir, unit_catalog, default_options):
        """Test that normalization is idempotent for simple payloads."""
        assert is_normalization_idempotent(
            {"name": "Test", "age": 25},
            simple_ir,
            unit_catalog,
            default_options,
        )

    def test_idempotent_with_defaults(self, ir_with_defaults, unit_catalog, default_options):
        """Test idempotency when defaults are applied."""
        assert is_normalization_idempotent(
            {"name": "Test"},
            ir_with_defaults,
            unit_catalog,
            default_options,
        )

    def test_idempotent_with_coercion(self, unit_catalog, default_options):
        """Test idempotency when coercion is applied."""
        ir = SchemaIR(
            schema_id="test/coercion",
            version="1.0.0",
            schema_hash="g" * 64,
            compiled_at=datetime.now(timezone.utc),
            properties={
                "/value": PropertyIR(path="/value", type="integer", required=True),
            },
            required_paths={"/value"},
        )
        assert is_normalization_idempotent(
            {"value": "42"},
            ir,
            unit_catalog,
            default_options,
        )


# =============================================================================
# TEST CLASS: Determinism
# =============================================================================


class TestDeterminism:
    """Tests for normalization determinism."""

    def test_deterministic_output(self, ir_with_defaults, unit_catalog, default_options):
        """Test that same input produces same output."""
        engine = NormalizationEngine(ir_with_defaults, unit_catalog, default_options)
        payload = {"name": "Test"}

        result1 = engine.normalize(payload)
        result2 = engine.normalize(payload)

        # Exclude timestamp comparison
        normalized1 = _remove_meta_block(result1.normalized)
        normalized2 = _remove_meta_block(result2.normalized)

        assert normalized1 == normalized2

    def test_stable_key_ordering(self, minimal_ir, unit_catalog, default_options):
        """Test that key ordering is stable."""
        engine = NormalizationEngine(minimal_ir, unit_catalog, default_options)

        # Create payload with keys in random order
        payload = {"z": 1, "a": 2, "m": 3}
        result = engine.normalize(payload)

        # Keys should be in consistent order
        keys = list(result.normalized.keys())
        assert keys == sorted(keys)


# =============================================================================
# TEST CLASS: Convenience Function
# =============================================================================


class TestConvenienceFunction:
    """Tests for the normalize convenience function."""

    def test_normalize_function_basic(self, simple_ir):
        """Test basic usage of normalize function."""
        result = normalize({"name": "Test"}, simple_ir)

        assert result.normalized["name"] == "Test"
        assert result.meta.schema_ref is not None
        assert result.meta.schema_ref.schema_id == "test/simple"

    def test_normalize_function_with_options(self, simple_ir, strict_options):
        """Test normalize function with custom options."""
        result = normalize(
            {"name": "Test"},
            simple_ir,
            options=strict_options,
        )

        assert result.normalized["name"] == "Test"


# =============================================================================
# TEST CLASS: Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_remove_meta_block_basic(self):
        """Test _remove_meta_block with simple payload."""
        payload = {
            "key": "value",
            "_meta": {"normalized_at": "2024-01-01"},
        }
        result = _remove_meta_block(payload)

        assert "key" in result
        assert "_meta" not in result

    def test_remove_meta_block_nested(self):
        """Test _remove_meta_block with nested objects."""
        payload = {
            "outer": {
                "inner": "value",
                "_meta": {"info": "nested"},
            },
            "_meta": {"info": "root"},
        }
        result = _remove_meta_block(payload)

        assert "_meta" not in result
        assert "_meta" not in result["outer"]
        assert result["outer"]["inner"] == "value"

    def test_remove_meta_block_in_arrays(self):
        """Test _remove_meta_block with arrays."""
        payload = {
            "items": [
                {"value": 1, "_meta": {"index": 0}},
                {"value": 2, "_meta": {"index": 1}},
            ],
            "_meta": {"info": "root"},
        }
        result = _remove_meta_block(payload)

        assert "_meta" not in result
        assert len(result["items"]) == 2
        for item in result["items"]:
            assert "_meta" not in item


# =============================================================================
# TEST CLASS: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_deeply_nested_payload(self, minimal_ir, unit_catalog, default_options):
        """Test handling of deeply nested payloads."""
        engine = NormalizationEngine(minimal_ir, unit_catalog, default_options)

        # Create deeply nested payload
        payload: Dict[str, Any] = {"level": 0}
        current = payload
        for i in range(50):
            current["nested"] = {"level": i + 1}
            current = current["nested"]

        # Should not raise
        result = engine.normalize(payload)
        assert result.normalized is not None

    def test_large_array_payload(self, minimal_ir, unit_catalog, default_options):
        """Test handling of large array payloads."""
        engine = NormalizationEngine(minimal_ir, unit_catalog, default_options)

        payload = {"items": list(range(1000))}
        result = engine.normalize(payload)

        assert len(result.normalized["items"]) == 1000

    def test_special_characters_in_keys(self, minimal_ir, unit_catalog, default_options):
        """Test handling of special characters in keys."""
        engine = NormalizationEngine(minimal_ir, unit_catalog, default_options)

        payload = {
            "key-with-dashes": 1,
            "key.with.dots": 2,
            "key with spaces": 3,
        }
        result = engine.normalize(payload)

        # Should normalize without errors
        assert result.normalized is not None

    def test_null_values(self, minimal_ir, unit_catalog, default_options):
        """Test handling of null values."""
        engine = NormalizationEngine(minimal_ir, unit_catalog, default_options)

        payload = {"key": None, "nested": {"value": None}}
        result = engine.normalize(payload)

        assert result.normalized["key"] is None
        assert result.normalized["nested"]["value"] is None

    def test_empty_string_values(self, simple_ir, unit_catalog, default_options):
        """Test handling of empty string values."""
        engine = NormalizationEngine(simple_ir, unit_catalog, default_options)

        payload = {"name": ""}
        result = engine.normalize(payload)

        assert result.normalized["name"] == ""

    def test_unicode_values(self, simple_ir, unit_catalog, default_options):
        """Test handling of unicode values."""
        engine = NormalizationEngine(simple_ir, unit_catalog, default_options)

        payload = {"name": "test unicode name"}
        result = engine.normalize(payload)

        assert result.normalized["name"] == "test unicode name"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
