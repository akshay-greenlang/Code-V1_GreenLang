# -*- coding: utf-8 -*-
"""
Comprehensive test suite for GL-FOUND-X-002 Task 1.4: Schema Compiler (AST -> IR).

Tests cover:
- IR model creation and validation
- Schema compilation pipeline
- Property flattening for O(1) lookup
- Constraint compilation (numeric, string, array)
- Pattern compilation with ReDoS safety analysis
- Unit specification extraction
- Rule binding extraction
- Deprecation indexing
- Enum extraction
- Schema hash computation (SHA-256)
- Determinism and reproducibility

Target: 85%+ code coverage

Author: GreenLang Framework Team
Date: January 2026
GL-FOUND-X-002: Schema Compiler & Validator
"""

import hashlib
import json
import pytest
from datetime import datetime
from copy import deepcopy

from greenlang.schema.compiler.ir import (
    COMPILER_VERSION,
    CompiledPattern,
    NumericConstraintIR,
    StringConstraintIR,
    ArrayConstraintIR,
    UnitSpecIR,
    RuleBindingIR,
    PropertyIR,
    DeprecationInfoIR,
    SchemaIR,
    CompilationResult,
    CompilationError,
)
from greenlang.schema.compiler.compiler import (
    SchemaCompiler,
    GL_UNIT_KEY,
    GL_DIMENSION_KEY,
    GL_RULES_KEY,
    GL_ALIASES_KEY,
    GL_DEPRECATED_KEY,
    GL_RENAMED_FROM_KEY,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def compiler():
    """Create SchemaCompiler instance."""
    return SchemaCompiler()


@pytest.fixture
def simple_schema():
    """Simple JSON Schema for basic tests."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 100},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "active": {"type": "boolean"},
        },
        "required": ["name"],
    }


@pytest.fixture
def complex_schema():
    """Complex JSON Schema with nested structures and GreenLang extensions."""
    return {
        "type": "object",
        "properties": {
            "organization": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "facilities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "pattern": "^FAC-[0-9]{4}$"},
                                "energy_consumption": {
                                    "type": "number",
                                    "minimum": 0,
                                    "$unit": {
                                        "dimension": "energy",
                                        "canonical": "kWh",
                                        "allowed": ["kWh", "MWh", "GJ"],
                                    },
                                },
                                "emissions": {
                                    "type": "number",
                                    "$unit": {
                                        "dimension": "mass_co2e",
                                        "canonical": "kgCO2e",
                                        "allowed": ["kgCO2e", "tCO2e"],
                                    },
                                },
                            },
                            "required": ["id"],
                        },
                        "minItems": 1,
                        "maxItems": 100,
                    },
                },
                "required": ["name"],
            },
            "status": {"type": "string", "enum": ["active", "inactive", "pending"]},
        },
        "required": ["organization"],
        "$rules": [
            {
                "rule_id": "facility_emissions_check",
                "severity": "warning",
                "check": {"gt": ["$emissions", 0]},
                "message": "Emissions should be positive",
            }
        ],
    }


@pytest.fixture
def schema_with_deprecations():
    """Schema with deprecated and renamed fields."""
    return {
        "type": "object",
        "properties": {
            "new_field": {
                "type": "string",
                "$renamed_from": "old_field",
            },
            "legacy_field": {
                "type": "number",
                "$deprecated": {
                    "since_version": "2.0.0",
                    "message": "Use 'modern_field' instead",
                    "replacement": "/modern_field",
                    "removal_version": "3.0.0",
                },
            },
            "modern_field": {"type": "number"},
        },
    }


# ==============================================================================
# Test Class 1: IR Model Tests - CompiledPattern
# ==============================================================================

class TestCompiledPatternIR:
    """Tests for CompiledPattern IR model."""

    def test_create_safe_pattern(self):
        """Test creating a safe pattern."""
        pattern = CompiledPattern(
            pattern="^[a-zA-Z0-9]+$",
            complexity_score=0.1,
            is_safe=True,
            timeout_ms=100,
            is_re2_compatible=True,
        )

        assert pattern.pattern == "^[a-zA-Z0-9]+$"
        assert pattern.is_safe is True
        assert pattern.complexity_score == 0.1
        assert pattern.is_re2_compatible is True

    def test_create_unsafe_pattern(self):
        """Test creating an unsafe pattern with vulnerability info."""
        pattern = CompiledPattern(
            pattern="(a+)+",
            complexity_score=0.9,
            is_safe=False,
            timeout_ms=100,
            is_re2_compatible=False,
            vulnerability_type="nested_quantifier",
            recommendation="Avoid nested quantifiers",
        )

        assert pattern.is_safe is False
        assert pattern.vulnerability_type == "nested_quantifier"
        assert pattern.recommendation is not None

    def test_get_compiled_safe(self):
        """Test getting compiled regex for safe pattern."""
        pattern = CompiledPattern(
            pattern="^[a-z]+$",
            is_safe=True,
        )

        compiled = pattern.get_compiled()
        assert compiled is not None
        assert compiled.match("hello") is not None
        assert compiled.match("HELLO") is None

    def test_get_compiled_unsafe(self):
        """Test getting compiled regex for unsafe pattern returns None."""
        pattern = CompiledPattern(
            pattern="(a+)+",
            is_safe=False,
        )

        compiled = pattern.get_compiled()
        assert compiled is None

    def test_pattern_immutable(self):
        """Test that CompiledPattern is immutable."""
        pattern = CompiledPattern(pattern="^test$", is_safe=True)

        with pytest.raises(Exception):
            pattern.pattern = "new_pattern"


# ==============================================================================
# Test Class 2: IR Model Tests - Constraint IRs
# ==============================================================================

class TestConstraintIRs:
    """Tests for constraint IR models."""

    def test_numeric_constraint_basic(self):
        """Test basic numeric constraint."""
        constraint = NumericConstraintIR(
            path="/age",
            minimum=0,
            maximum=150,
        )

        assert constraint.path == "/age"
        assert constraint.minimum == 0
        assert constraint.maximum == 150
        assert constraint.has_constraints() is True

    def test_numeric_constraint_exclusive(self):
        """Test exclusive numeric constraints."""
        constraint = NumericConstraintIR(
            path="/temperature",
            exclusive_minimum=0.0,
            exclusive_maximum=100.0,
        )

        assert constraint.exclusive_minimum == 0.0
        assert constraint.exclusive_maximum == 100.0

    def test_numeric_constraint_effective_minimum(self):
        """Test get_effective_minimum method."""
        # With exclusive
        constraint1 = NumericConstraintIR(
            path="/val",
            exclusive_minimum=5.0,
        )
        result1 = constraint1.get_effective_minimum()
        assert result1 == (5.0, True)

        # With inclusive
        constraint2 = NumericConstraintIR(
            path="/val",
            minimum=10.0,
        )
        result2 = constraint2.get_effective_minimum()
        assert result2 == (10.0, False)

        # No minimum
        constraint3 = NumericConstraintIR(path="/val")
        result3 = constraint3.get_effective_minimum()
        assert result3 is None

    def test_numeric_constraint_multiple_of(self):
        """Test multipleOf constraint."""
        constraint = NumericConstraintIR(
            path="/step",
            multiple_of=0.5,
        )

        assert constraint.multiple_of == 0.5
        assert constraint.has_constraints() is True

    def test_string_constraint_basic(self):
        """Test basic string constraint."""
        constraint = StringConstraintIR(
            path="/name",
            min_length=1,
            max_length=100,
        )

        assert constraint.path == "/name"
        assert constraint.min_length == 1
        assert constraint.max_length == 100
        assert constraint.has_constraints() is True

    def test_string_constraint_with_pattern(self):
        """Test string constraint with pattern."""
        constraint = StringConstraintIR(
            path="/email",
            pattern="^[a-z]+@[a-z]+\\.[a-z]+$",
            pattern_compiled=CompiledPattern(
                pattern="^[a-z]+@[a-z]+\\.[a-z]+$",
                is_safe=True,
            ),
        )

        assert constraint.has_pattern() is True
        assert constraint.pattern_compiled is not None

    def test_string_constraint_with_format(self):
        """Test string constraint with format."""
        constraint = StringConstraintIR(
            path="/date",
            format="date",
        )

        assert constraint.has_format() is True
        assert constraint.format == "date"

    def test_array_constraint_basic(self):
        """Test basic array constraint."""
        constraint = ArrayConstraintIR(
            path="/items",
            min_items=1,
            max_items=100,
            unique_items=True,
        )

        assert constraint.path == "/items"
        assert constraint.min_items == 1
        assert constraint.max_items == 100
        assert constraint.unique_items is True
        assert constraint.has_constraints() is True

    def test_array_constraint_no_constraints(self):
        """Test array constraint with no constraints."""
        constraint = ArrayConstraintIR(path="/items")

        assert constraint.has_constraints() is False


# ==============================================================================
# Test Class 3: IR Model Tests - UnitSpecIR and RuleBindingIR
# ==============================================================================

class TestUnitAndRuleIRs:
    """Tests for unit specification and rule binding IRs."""

    def test_unit_spec_basic(self):
        """Test basic unit specification."""
        unit_spec = UnitSpecIR(
            path="/energy",
            dimension="energy",
            canonical="kWh",
            allowed=["kWh", "MWh", "GJ"],
        )

        assert unit_spec.path == "/energy"
        assert unit_spec.dimension == "energy"
        assert unit_spec.canonical == "kWh"
        assert len(unit_spec.allowed) == 3

    def test_unit_spec_is_unit_allowed(self):
        """Test is_unit_allowed method."""
        unit_spec = UnitSpecIR(
            path="/energy",
            dimension="energy",
            canonical="kWh",
            allowed=["kWh", "MWh"],
        )

        assert unit_spec.is_unit_allowed("kWh") is True
        assert unit_spec.is_unit_allowed("GJ") is False

    def test_unit_spec_no_restrictions(self):
        """Test unit spec with no allowed restrictions."""
        unit_spec = UnitSpecIR(
            path="/value",
            dimension="any",
            canonical="unit",
            allowed=[],
        )

        assert unit_spec.is_unit_allowed("anything") is True

    def test_unit_spec_get_allowed_set(self):
        """Test get_allowed_set returns frozen set."""
        unit_spec = UnitSpecIR(
            path="/mass",
            dimension="mass",
            canonical="kg",
            allowed=["kg", "g", "mg"],
        )

        allowed_set = unit_spec.get_allowed_set()
        assert isinstance(allowed_set, frozenset)
        assert "kg" in allowed_set

    def test_rule_binding_basic(self):
        """Test basic rule binding."""
        rule = RuleBindingIR(
            rule_id="test_rule",
            severity="error",
            check={"gt": ["$value", 0]},
            message="Value must be positive",
        )

        assert rule.rule_id == "test_rule"
        assert rule.severity == "error"
        assert rule.is_blocking() is True

    def test_rule_binding_warning(self):
        """Test warning rule binding."""
        rule = RuleBindingIR(
            rule_id="soft_check",
            severity="warning",
            check={"lt": ["$value", 1000]},
            message="Value seems high",
        )

        assert rule.is_blocking() is False

    def test_rule_binding_with_when(self):
        """Test rule binding with conditional."""
        rule = RuleBindingIR(
            rule_id="conditional_rule",
            severity="error",
            when={"exists": "$optional_field"},
            check={"gt": ["$optional_field", 0]},
            message="If provided, must be positive",
        )

        assert rule.when is not None

    def test_rule_binding_severity_validation(self):
        """Test rule binding validates severity."""
        with pytest.raises(ValueError):
            RuleBindingIR(
                rule_id="bad_rule",
                severity="invalid_severity",
                check={},
                message="test",
            )


# ==============================================================================
# Test Class 4: IR Model Tests - PropertyIR and SchemaIR
# ==============================================================================

class TestPropertyAndSchemaIRs:
    """Tests for PropertyIR and SchemaIR models."""

    def test_property_ir_basic(self):
        """Test basic PropertyIR."""
        prop = PropertyIR(
            path="/name",
            type="string",
            required=True,
            has_default=False,
        )

        assert prop.path == "/name"
        assert prop.type == "string"
        assert prop.required is True

    def test_property_ir_with_default(self):
        """Test PropertyIR with default value."""
        prop = PropertyIR(
            path="/count",
            type="integer",
            required=False,
            has_default=True,
            default_value=0,
        )

        assert prop.has_default is True
        assert prop.default_value == 0

    def test_property_ir_with_extensions(self):
        """Test PropertyIR with GreenLang extensions."""
        prop = PropertyIR(
            path="/energy",
            type="number",
            required=True,
            gl_extensions={
                "unit": {"dimension": "energy", "canonical": "kWh"},
            },
        )

        assert prop.gl_extensions is not None
        assert "unit" in prop.gl_extensions

    def test_schema_ir_basic(self):
        """Test basic SchemaIR creation."""
        ir = SchemaIR(
            schema_id="test/schema",
            version="1.0.0",
            schema_hash="a" * 64,
            compiled_at=datetime.utcnow(),
        )

        assert ir.schema_id == "test/schema"
        assert ir.version == "1.0.0"
        assert ir.compiler_version == COMPILER_VERSION

    def test_schema_ir_lookup_methods(self):
        """Test SchemaIR lookup methods."""
        ir = SchemaIR(
            schema_id="test",
            version="1.0.0",
            schema_hash="a" * 64,
            compiled_at=datetime.utcnow(),
            properties={
                "/name": PropertyIR(path="/name", type="string", required=True),
            },
            required_paths={"/name"},
            numeric_constraints={
                "/age": NumericConstraintIR(path="/age", minimum=0),
            },
            enums={
                "/status": ["active", "inactive"],
            },
        )

        assert ir.get_property("/name") is not None
        assert ir.get_property("/nonexistent") is None
        assert ir.is_required("/name") is True
        assert ir.is_required("/age") is False
        assert ir.get_numeric_constraint("/age") is not None
        assert ir.get_enum("/status") == ["active", "inactive"]

    def test_schema_ir_get_stats(self):
        """Test SchemaIR get_stats method."""
        ir = SchemaIR(
            schema_id="test",
            version="1.0.0",
            schema_hash="a" * 64,
            compiled_at=datetime.utcnow(),
            properties={
                "/a": PropertyIR(path="/a", type="string"),
                "/b": PropertyIR(path="/b", type="number"),
            },
            required_paths={"/a"},
        )

        stats = ir.get_stats()
        assert stats["properties"] == 2
        assert stats["required_paths"] == 1


# ==============================================================================
# Test Class 5: Compiler Tests - Basic Compilation
# ==============================================================================

class TestSchemaCompilerBasic:
    """Basic compilation tests."""

    def test_compile_simple_schema(self, compiler, simple_schema):
        """Test compiling a simple schema."""
        result = compiler.compile(simple_schema, "test/simple", "1.0.0")

        assert result.success is True
        assert result.ir is not None
        assert result.ir.schema_id == "test/simple"
        assert result.ir.version == "1.0.0"
        assert len(result.errors) == 0

    def test_compile_time_tracked(self, compiler, simple_schema):
        """Test that compile time is tracked."""
        result = compiler.compile(simple_schema, "test", "1.0.0")

        assert result.compile_time_ms >= 0

    def test_compile_warnings_collected(self, compiler):
        """Test that compilation warnings are collected."""
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "minimum": 100,
                    "maximum": 10,  # Invalid: min > max
                },
            },
        }

        result = compiler.compile(schema, "test", "1.0.0")

        assert result.success is True
        assert len(result.warnings) > 0

    def test_compile_from_dict(self, compiler):
        """Test compiling from dictionary."""
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = compiler.compile(schema, "test", "1.0.0")

        assert result.success is True

    def test_compile_from_string(self, compiler):
        """Test compiling from JSON string."""
        schema_str = '{"type": "object", "properties": {"x": {"type": "string"}}}'
        result = compiler.compile(schema_str, "test", "1.0.0")

        assert result.success is True


# ==============================================================================
# Test Class 6: Compiler Tests - Schema Hash
# ==============================================================================

class TestSchemaCompilerHash:
    """Tests for schema hash computation."""

    def test_schema_hash_length(self, compiler, simple_schema):
        """Test schema hash is 64 characters (SHA-256 hex)."""
        result = compiler.compile(simple_schema, "test", "1.0.0")

        assert len(result.ir.schema_hash) == 64

    def test_schema_hash_deterministic(self, compiler, simple_schema):
        """Test schema hash is deterministic."""
        result1 = compiler.compile(simple_schema, "test", "1.0.0")
        result2 = compiler.compile(simple_schema, "test", "1.0.0")

        assert result1.ir.schema_hash == result2.ir.schema_hash

    def test_schema_hash_changes_with_content(self, compiler, simple_schema):
        """Test schema hash changes when content changes."""
        result1 = compiler.compile(simple_schema, "test", "1.0.0")

        modified_schema = deepcopy(simple_schema)
        modified_schema["properties"]["email"] = {"type": "string"}
        result2 = compiler.compile(modified_schema, "test", "1.0.0")

        assert result1.ir.schema_hash != result2.ir.schema_hash

    def test_schema_hash_independent_of_metadata(self, compiler, simple_schema):
        """Test schema hash is independent of compile metadata."""
        result1 = compiler.compile(simple_schema, "test1", "1.0.0")
        result2 = compiler.compile(simple_schema, "test2", "2.0.0")

        # Hash should be same because content is same
        assert result1.ir.schema_hash == result2.ir.schema_hash

    def test_schema_hash_order_independent(self, compiler):
        """Test schema hash is order-independent due to key sorting."""
        schema1 = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "number"}}}
        schema2 = {"properties": {"b": {"type": "number"}, "a": {"type": "string"}}, "type": "object"}

        result1 = compiler.compile(schema1, "test", "1.0.0")
        result2 = compiler.compile(schema2, "test", "1.0.0")

        assert result1.ir.schema_hash == result2.ir.schema_hash


# ==============================================================================
# Test Class 7: Compiler Tests - Property Flattening
# ==============================================================================

class TestSchemaCompilerPropertyFlattening:
    """Tests for property flattening."""

    def test_flatten_simple_properties(self, compiler, simple_schema):
        """Test flattening simple properties."""
        result = compiler.compile(simple_schema, "test", "1.0.0")

        assert "/name" in result.ir.properties
        assert "/age" in result.ir.properties
        assert "/active" in result.ir.properties

    def test_flatten_nested_properties(self, compiler, complex_schema):
        """Test flattening nested object properties."""
        result = compiler.compile(complex_schema, "test", "1.0.0")

        assert "/organization" in result.ir.properties
        assert "/organization/name" in result.ir.properties
        assert "/organization/facilities" in result.ir.properties

    def test_flatten_array_items(self, compiler, complex_schema):
        """Test flattening array item properties."""
        result = compiler.compile(complex_schema, "test", "1.0.0")

        # Array items should be flattened
        assert "/organization/facilities/items" in result.ir.properties
        assert "/organization/facilities/items/id" in result.ir.properties

    def test_property_types_preserved(self, compiler, simple_schema):
        """Test property types are preserved."""
        result = compiler.compile(simple_schema, "test", "1.0.0")

        assert result.ir.properties["/name"].type == "string"
        assert result.ir.properties["/age"].type == "integer"
        assert result.ir.properties["/active"].type == "boolean"

    def test_required_properties_marked(self, compiler, simple_schema):
        """Test required properties are marked."""
        result = compiler.compile(simple_schema, "test", "1.0.0")

        assert result.ir.properties["/name"].required is True
        assert result.ir.properties["/age"].required is False

    def test_required_paths_collected(self, compiler, simple_schema):
        """Test required paths are collected."""
        result = compiler.compile(simple_schema, "test", "1.0.0")

        assert "/name" in result.ir.required_paths
        assert "/age" not in result.ir.required_paths


# ==============================================================================
# Test Class 8: Compiler Tests - Constraint Compilation
# ==============================================================================

class TestSchemaCompilerConstraints:
    """Tests for constraint compilation."""

    def test_numeric_constraints_compiled(self, compiler, simple_schema):
        """Test numeric constraints are compiled."""
        result = compiler.compile(simple_schema, "test", "1.0.0")

        age_constraint = result.ir.get_numeric_constraint("/age")
        assert age_constraint is not None
        assert age_constraint.minimum == 0
        assert age_constraint.maximum == 150

    def test_string_constraints_compiled(self, compiler, simple_schema):
        """Test string constraints are compiled."""
        result = compiler.compile(simple_schema, "test", "1.0.0")

        name_constraint = result.ir.get_string_constraint("/name")
        assert name_constraint is not None
        assert name_constraint.min_length == 1
        assert name_constraint.max_length == 100

    def test_array_constraints_compiled(self, compiler, complex_schema):
        """Test array constraints are compiled."""
        result = compiler.compile(complex_schema, "test", "1.0.0")

        array_constraint = result.ir.get_array_constraint("/organization/facilities")
        assert array_constraint is not None
        assert array_constraint.min_items == 1
        assert array_constraint.max_items == 100

    def test_nested_constraints_compiled(self, compiler, complex_schema):
        """Test nested property constraints are compiled."""
        result = compiler.compile(complex_schema, "test", "1.0.0")

        energy_constraint = result.ir.get_numeric_constraint(
            "/organization/facilities/items/energy_consumption"
        )
        assert energy_constraint is not None
        assert energy_constraint.minimum == 0


# ==============================================================================
# Test Class 9: Compiler Tests - Pattern Compilation
# ==============================================================================

class TestSchemaCompilerPatterns:
    """Tests for pattern compilation."""

    def test_pattern_compiled(self, compiler, complex_schema):
        """Test patterns are compiled."""
        result = compiler.compile(complex_schema, "test", "1.0.0")

        pattern = result.ir.get_pattern("/organization/facilities/items/id")
        assert pattern is not None
        assert pattern.pattern == "^FAC-[0-9]{4}$"

    def test_safe_pattern_marked_safe(self, compiler):
        """Test safe patterns are marked as safe."""
        schema = {
            "type": "object",
            "properties": {
                "email": {"type": "string", "pattern": "^[a-z]+@[a-z]+\\.[a-z]+$"},
            },
        }

        result = compiler.compile(schema, "test", "1.0.0")

        pattern = result.ir.get_pattern("/email")
        assert pattern.is_safe is True

    def test_unsafe_pattern_marked_unsafe(self, compiler):
        """Test unsafe patterns are marked as unsafe."""
        schema = {
            "type": "object",
            "properties": {
                "bad": {"type": "string", "pattern": "(a+)+"},
            },
        }

        result = compiler.compile(schema, "test", "1.0.0")

        pattern = result.ir.get_pattern("/bad")
        assert pattern.is_safe is False
        assert pattern.vulnerability_type == "nested_quantifier"

    def test_invalid_pattern_handled(self, compiler):
        """Test invalid regex patterns are handled gracefully."""
        schema = {
            "type": "object",
            "properties": {
                "invalid": {"type": "string", "pattern": "[unclosed"},
            },
        }

        result = compiler.compile(schema, "test", "1.0.0")

        pattern = result.ir.get_pattern("/invalid")
        assert pattern.is_safe is False
        assert pattern.vulnerability_type == "invalid_pattern"


# ==============================================================================
# Test Class 10: Compiler Tests - Unit Specifications
# ==============================================================================

class TestSchemaCompilerUnitSpecs:
    """Tests for unit specification extraction."""

    def test_unit_specs_extracted(self, compiler, complex_schema):
        """Test unit specifications are extracted."""
        result = compiler.compile(complex_schema, "test", "1.0.0")

        energy_unit = result.ir.get_unit_spec(
            "/organization/facilities/items/energy_consumption"
        )
        assert energy_unit is not None
        assert energy_unit.dimension == "energy"
        assert energy_unit.canonical == "kWh"

    def test_multiple_unit_specs(self, compiler, complex_schema):
        """Test multiple unit specifications are extracted."""
        result = compiler.compile(complex_schema, "test", "1.0.0")

        assert len(result.ir.unit_specs) >= 2

    def test_incomplete_unit_spec_warning(self, compiler):
        """Test incomplete unit spec produces warning."""
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "$unit": {"dimension": "energy"},  # Missing canonical
                },
            },
        }

        result = compiler.compile(schema, "test", "1.0.0")

        assert len(result.warnings) > 0
        assert len(result.ir.unit_specs) == 0


# ==============================================================================
# Test Class 11: Compiler Tests - Rule Bindings
# ==============================================================================

class TestSchemaCompilerRuleBindings:
    """Tests for rule binding extraction."""

    def test_document_level_rules_extracted(self, compiler, complex_schema):
        """Test document-level rules are extracted."""
        result = compiler.compile(complex_schema, "test", "1.0.0")

        assert len(result.ir.rule_bindings) >= 1
        rule = result.ir.rule_bindings[0]
        assert rule.rule_id == "facility_emissions_check"

    def test_property_level_rules_extracted(self, compiler):
        """Test property-level rules are extracted."""
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "$rules": [
                        {
                            "rule_id": "value_range",
                            "severity": "error",
                            "check": {"between": ["$value", 0, 100]},
                            "message": "Value must be 0-100",
                        }
                    ],
                },
            },
        }

        result = compiler.compile(schema, "test", "1.0.0")

        assert len(result.ir.rule_bindings) == 1
        assert result.ir.rule_bindings[0].rule_id == "value_range"


# ==============================================================================
# Test Class 12: Compiler Tests - Deprecations
# ==============================================================================

class TestSchemaCompilerDeprecations:
    """Tests for deprecation extraction."""

    def test_deprecated_fields_extracted(self, compiler, schema_with_deprecations):
        """Test deprecated fields are extracted."""
        result = compiler.compile(schema_with_deprecations, "test", "1.0.0")

        assert "/legacy_field" in result.ir.deprecated_fields
        deprecation = result.ir.get_deprecation_info("/legacy_field")
        assert deprecation["since_version"] == "2.0.0"
        assert deprecation["replacement"] == "/modern_field"

    def test_renamed_fields_extracted(self, compiler, schema_with_deprecations):
        """Test renamed fields are extracted."""
        result = compiler.compile(schema_with_deprecations, "test", "1.0.0")

        assert "/old_field" in result.ir.renamed_fields
        assert result.ir.get_renamed_to("/old_field") == "/new_field"

    def test_is_deprecated_method(self, compiler, schema_with_deprecations):
        """Test is_deprecated method."""
        result = compiler.compile(schema_with_deprecations, "test", "1.0.0")

        assert result.ir.is_deprecated("/legacy_field") is True
        assert result.ir.is_deprecated("/modern_field") is False


# ==============================================================================
# Test Class 13: Compiler Tests - Enums
# ==============================================================================

class TestSchemaCompilerEnums:
    """Tests for enum extraction."""

    def test_enums_extracted(self, compiler, complex_schema):
        """Test enum values are extracted."""
        result = compiler.compile(complex_schema, "test", "1.0.0")

        enum_values = result.ir.get_enum("/status")
        assert enum_values is not None
        assert "active" in enum_values
        assert "inactive" in enum_values
        assert "pending" in enum_values


# ==============================================================================
# Test Class 14: Compiler Tests - CompilationResult
# ==============================================================================

class TestCompilationResult:
    """Tests for CompilationResult model."""

    def test_success_property(self, compiler, simple_schema):
        """Test success property."""
        result = compiler.compile(simple_schema, "test", "1.0.0")

        assert result.success is True

    def test_success_false_on_error(self):
        """Test success is False when errors exist."""
        result = CompilationResult(
            ir=None,
            errors=["Some error"],
        )

        assert result.success is False

    def test_add_warning(self):
        """Test add_warning method."""
        result = CompilationResult()
        result.add_warning("Warning 1")
        result.add_warning("Warning 2")

        assert len(result.warnings) == 2

    def test_add_error(self):
        """Test add_error method."""
        result = CompilationResult()
        result.add_error("Error 1")

        assert len(result.errors) == 1


# ==============================================================================
# Test Class 15: Determinism Tests
# ==============================================================================

class TestSchemaCompilerDeterminism:
    """Tests for deterministic compilation."""

    def test_same_input_same_ir(self, compiler, simple_schema):
        """Test same input produces same IR."""
        result1 = compiler.compile(simple_schema, "test", "1.0.0")
        result2 = compiler.compile(simple_schema, "test", "1.0.0")

        # Compare key IR attributes
        assert result1.ir.schema_hash == result2.ir.schema_hash
        assert len(result1.ir.properties) == len(result2.ir.properties)
        assert result1.ir.required_paths == result2.ir.required_paths

    def test_property_order_consistent(self, compiler, simple_schema):
        """Test property order is consistent."""
        results = []
        for _ in range(5):
            result = compiler.compile(simple_schema, "test", "1.0.0")
            results.append(list(result.ir.properties.keys()))

        # All runs should have same property order
        for i in range(1, len(results)):
            assert results[i] == results[0]

    def test_constraint_compilation_deterministic(self, compiler, simple_schema):
        """Test constraint compilation is deterministic."""
        results = []
        for _ in range(3):
            result = compiler.compile(simple_schema, "test", "1.0.0")
            results.append(len(result.ir.numeric_constraints))

        assert len(set(results)) == 1


# ==============================================================================
# Test Class 16: Edge Cases
# ==============================================================================

class TestSchemaCompilerEdgeCases:
    """Edge case tests."""

    def test_empty_schema(self, compiler):
        """Test compiling empty schema."""
        result = compiler.compile({}, "test", "1.0.0")

        assert result.success is True
        assert len(result.ir.properties) == 0

    def test_schema_with_definitions(self, compiler):
        """Test schema with definitions/$defs."""
        schema = {
            "type": "object",
            "properties": {
                "item": {"$ref": "#/$defs/Item"},
            },
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                },
            },
        }

        result = compiler.compile(schema, "test", "1.0.0")

        assert result.success is True
        # Definitions should be flattened
        assert "/$defs/Item/name" in result.ir.properties

    def test_union_types(self, compiler):
        """Test schema with union types."""
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "null"]},
            },
        }

        result = compiler.compile(schema, "test", "1.0.0")

        assert result.ir.properties["/value"].type == "string|null"

    def test_deeply_nested_schema(self, compiler):
        """Test deeply nested schema."""
        # Create 10 levels of nesting
        schema = {"type": "object"}
        current = schema
        for i in range(10):
            current["properties"] = {
                f"level{i}": {"type": "object"},
            }
            current = current["properties"][f"level{i}"]
        current["properties"] = {"value": {"type": "string"}}

        result = compiler.compile(schema, "test", "1.0.0")

        assert result.success is True


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
