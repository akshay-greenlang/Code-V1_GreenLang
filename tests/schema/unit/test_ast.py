# -*- coding: utf-8 -*-
"""
Unit Tests for Schema AST (GL-FOUND-X-002 Task 1.2)
===================================================

Comprehensive unit tests for the Schema Abstract Syntax Tree implementation.

Tests cover:
    - All AST node types (TypeNode, ObjectTypeNode, ArrayTypeNode, etc.)
    - GreenLang extensions (UnitSpec, DeprecationInfo, RuleBinding)
    - Helper functions (create_node_id, parse_type_node, build_ast)
    - AST validation (validate_ast)
    - Immutability constraints (frozen models)
    - Serialization and hashing

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

import json
import pytest
from typing import Any, Dict

from greenlang.schema.compiler.ast import (
    # Constants
    JSON_SCHEMA_DRAFT_2020_12,
    JSON_SCHEMA_TYPES,
    # Enums
    RuleSeverity,
    # Extension Models
    UnitSpec,
    DeprecationInfo,
    RuleBinding,
    GreenLangExtensions,
    # AST Nodes
    SchemaNode,
    TypeNode,
    ObjectTypeNode,
    ArrayTypeNode,
    StringTypeNode,
    NumericTypeNode,
    BooleanTypeNode,
    NullTypeNode,
    RefNode,
    CompositionNode,
    EnumTypeNode,
    SchemaDocument,
    # Helper Functions
    create_node_id,
    create_unique_node_id,
    parse_type_node,
    build_ast,
    validate_ast,
)
from pydantic import ValidationError


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_string_node() -> StringTypeNode:
    """Create a simple string type node for testing."""
    return StringTypeNode(
        node_id="/properties/name",
        type="string",
        min_length=1,
        max_length=100,
    )


@pytest.fixture
def simple_numeric_node() -> NumericTypeNode:
    """Create a simple numeric type node for testing."""
    return NumericTypeNode(
        node_id="/properties/quantity",
        type="integer",
        minimum=0,
        maximum=1000,
    )


@pytest.fixture
def simple_object_node(simple_string_node, simple_numeric_node) -> ObjectTypeNode:
    """Create a simple object type node for testing."""
    return ObjectTypeNode(
        node_id="/",
        type="object",
        properties={
            "name": simple_string_node,
            "quantity": simple_numeric_node,
        },
        required=["name"],
    )


@pytest.fixture
def sample_schema_dict() -> Dict[str, Any]:
    """Create a sample schema dictionary for testing."""
    return {
        "type": "object",
        "title": "Activity Data",
        "description": "Schema for activity data",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "energy": {
                "type": "number",
                "minimum": 0,
                "$unit": {
                    "dimension": "energy",
                    "canonical": "kWh",
                    "allowed": ["kWh", "MWh"],
                },
            },
            "status": {"type": "string", "enum": ["active", "inactive"]},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "energy"],
        "$defs": {
            "Address": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            }
        },
    }


# =============================================================================
# Test UnitSpec
# =============================================================================


class TestUnitSpec:
    """Tests for UnitSpec model."""

    def test_create_valid_unit_spec(self):
        """Test creating a valid UnitSpec."""
        unit = UnitSpec(
            dimension="energy",
            canonical="kWh",
            allowed=["kWh", "MWh", "GJ"],
        )
        assert unit.dimension == "energy"
        assert unit.canonical == "kWh"
        assert unit.allowed == ["kWh", "MWh", "GJ"]

    def test_unit_spec_immutable(self):
        """Test that UnitSpec is immutable (frozen)."""
        unit = UnitSpec(dimension="energy", canonical="kWh")
        with pytest.raises(ValidationError):
            unit.dimension = "mass"

    def test_is_unit_allowed(self):
        """Test is_unit_allowed method."""
        unit = UnitSpec(
            dimension="energy",
            canonical="kWh",
            allowed=["kWh", "MWh"],
        )
        assert unit.is_unit_allowed("kWh") is True
        assert unit.is_unit_allowed("MWh") is True
        assert unit.is_unit_allowed("GJ") is False

    def test_is_unit_allowed_empty_list(self):
        """Test is_unit_allowed with no restrictions."""
        unit = UnitSpec(dimension="energy", canonical="kWh", allowed=[])
        assert unit.is_unit_allowed("anything") is True

    def test_contains_canonical(self):
        """Test contains_canonical method."""
        unit = UnitSpec(
            dimension="energy",
            canonical="kWh",
            allowed=["kWh", "MWh"],
        )
        assert unit.contains_canonical() is True

        unit2 = UnitSpec(dimension="energy", canonical="kWh", allowed=["MWh"])
        assert unit2.contains_canonical() is False

    def test_invalid_dimension_format(self):
        """Test that invalid dimension format is rejected."""
        with pytest.raises(ValidationError):
            UnitSpec(dimension="Energy", canonical="kWh")  # Must be lowercase

    def test_empty_canonical_rejected(self):
        """Test that empty canonical is rejected."""
        with pytest.raises(ValidationError):
            UnitSpec(dimension="energy", canonical="  ")


# =============================================================================
# Test DeprecationInfo
# =============================================================================


class TestDeprecationInfo:
    """Tests for DeprecationInfo model."""

    def test_create_valid_deprecation(self):
        """Test creating a valid DeprecationInfo."""
        dep = DeprecationInfo(
            since_version="2.0.0",
            message="Use energy_consumption instead",
            replacement="/energy_consumption",
            removal_version="3.0.0",
        )
        assert dep.since_version == "2.0.0"
        assert dep.message == "Use energy_consumption instead"
        assert dep.replacement == "/energy_consumption"
        assert dep.removal_version == "3.0.0"

    def test_deprecation_without_optional_fields(self):
        """Test creating DeprecationInfo without optional fields."""
        dep = DeprecationInfo(
            since_version="1.0.0",
            message="Deprecated field",
        )
        assert dep.replacement is None
        assert dep.removal_version is None

    def test_invalid_version_format(self):
        """Test that invalid version format is rejected."""
        with pytest.raises(ValidationError):
            DeprecationInfo(since_version="v2.0", message="Test")

    def test_is_removal_imminent(self):
        """Test is_removal_imminent method."""
        dep = DeprecationInfo(
            since_version="1.0.0",
            message="Test",
            removal_version="2.0.0",
        )
        assert dep.is_removal_imminent("1.0.0") is True
        assert dep.is_removal_imminent("0.5.0") is True
        assert dep.is_removal_imminent("3.0.0") is True  # Already past

    def test_is_removal_imminent_no_removal(self):
        """Test is_removal_imminent with no removal version."""
        dep = DeprecationInfo(since_version="1.0.0", message="Test")
        assert dep.is_removal_imminent("1.0.0") is False


# =============================================================================
# Test RuleBinding
# =============================================================================


class TestRuleBinding:
    """Tests for RuleBinding model."""

    def test_create_valid_rule(self):
        """Test creating a valid RuleBinding."""
        rule = RuleBinding(
            rule_id="scope_sum_check",
            severity=RuleSeverity.ERROR,
            check={"$eq": ["$scope1 + $scope2", "$total"]},
            message="Scope 1 + Scope 2 must equal total",
        )
        assert rule.rule_id == "scope_sum_check"
        assert rule.severity == RuleSeverity.ERROR
        assert rule.is_blocking() is True

    def test_rule_with_condition(self):
        """Test creating a rule with when condition."""
        rule = RuleBinding(
            rule_id="conditional_check",
            severity=RuleSeverity.WARNING,
            when={"$exists": "$optional_field"},
            check={"$gt": ["$optional_field", 0]},
            message="Optional field must be positive if present",
        )
        assert rule.has_condition() is True
        assert rule.is_blocking() is False

    def test_invalid_rule_id_format(self):
        """Test that invalid rule_id format is rejected."""
        with pytest.raises(ValidationError):
            RuleBinding(
                rule_id="123-invalid",  # Must start with letter
                severity=RuleSeverity.ERROR,
                check={},
                message="Test",
            )

    def test_rule_severity_levels(self):
        """Test all severity levels."""
        assert RuleSeverity.ERROR.is_blocking() is True
        assert RuleSeverity.WARNING.is_blocking() is False
        assert RuleSeverity.INFO.is_blocking() is False


# =============================================================================
# Test GreenLangExtensions
# =============================================================================


class TestGreenLangExtensions:
    """Tests for GreenLangExtensions model."""

    def test_create_empty_extensions(self):
        """Test creating empty extensions."""
        ext = GreenLangExtensions()
        assert ext.has_unit() is False
        assert ext.has_rules() is False
        assert ext.has_aliases() is False
        assert ext.is_deprecated() is False

    def test_create_extensions_with_unit(self):
        """Test creating extensions with unit."""
        unit = UnitSpec(dimension="energy", canonical="kWh")
        ext = GreenLangExtensions(unit=unit)
        assert ext.has_unit() is True
        assert ext.get_effective_dimension() == "energy"

    def test_create_extensions_with_aliases(self):
        """Test creating extensions with aliases."""
        ext = GreenLangExtensions(
            aliases={"energyConsumption": "energy_consumption"}
        )
        assert ext.has_aliases() is True
        assert ext.aliases["energyConsumption"] == "energy_consumption"

    def test_invalid_alias_to_self(self):
        """Test that alias to self is rejected."""
        with pytest.raises(ValidationError):
            GreenLangExtensions(aliases={"name": "name"})

    def test_merge_extensions(self):
        """Test merging two extensions objects."""
        ext1 = GreenLangExtensions(
            unit=UnitSpec(dimension="energy", canonical="kWh"),
            aliases={"e": "energy"},
        )
        ext2 = GreenLangExtensions(
            dimension="mass",  # Should not override ext1's unit.dimension
            aliases={"m": "mass"},
        )
        merged = ext1.merge_with(ext2)
        assert merged.get_effective_dimension() == "energy"  # ext1 takes precedence
        assert "e" in merged.aliases
        assert "m" in merged.aliases


# =============================================================================
# Test TypeNode Base Class
# =============================================================================


class TestTypeNode:
    """Tests for TypeNode base class."""

    def test_create_basic_type_node(self):
        """Test creating a basic TypeNode."""
        node = TypeNode(node_id="/test", type="string")
        assert node.node_id == "/test"
        assert node.type == "string"

    def test_type_node_with_union_type(self):
        """Test TypeNode with union type (array)."""
        node = TypeNode(node_id="/test", type=["string", "null"])
        assert node.is_type("string") is True
        assert node.is_type("null") is True
        assert node.is_nullable() is True

    def test_is_type_method(self):
        """Test is_type method."""
        node = TypeNode(node_id="/test", type="string")
        assert node.is_type("string") is True
        assert node.is_type("number") is False

    def test_invalid_type_rejected(self):
        """Test that invalid type is rejected."""
        with pytest.raises(ValidationError):
            TypeNode(node_id="/test", type="invalid_type")

    def test_type_node_with_default(self):
        """Test TypeNode with default value."""
        node = TypeNode(node_id="/test", type="string", default="hello")
        assert node.has_default() is True
        assert node.default == "hello"

    def test_type_node_with_const(self):
        """Test TypeNode with const value."""
        node = TypeNode(node_id="/test", type="string", const="fixed")
        assert node.has_const() is True
        assert node.const == "fixed"

    def test_get_extensions(self):
        """Test get_extensions method."""
        node = TypeNode(node_id="/test")
        ext = node.get_extensions()
        assert isinstance(ext, GreenLangExtensions)


# =============================================================================
# Test StringTypeNode
# =============================================================================


class TestStringTypeNode:
    """Tests for StringTypeNode."""

    def test_create_string_node(self):
        """Test creating a StringTypeNode."""
        node = StringTypeNode(
            node_id="/test",
            type="string",
            min_length=1,
            max_length=100,
            pattern="^[a-z]+$",
        )
        assert node.min_length == 1
        assert node.max_length == 100
        assert node.has_pattern() is True

    def test_string_with_format(self):
        """Test StringTypeNode with format."""
        node = StringTypeNode(
            node_id="/test",
            type="string",
            format="email",
        )
        assert node.has_format() is True
        assert node.format == "email"

    def test_invalid_min_max_length(self):
        """Test that min_length > max_length is rejected."""
        with pytest.raises(ValidationError):
            StringTypeNode(
                node_id="/test",
                type="string",
                min_length=100,
                max_length=10,
            )

    def test_invalid_regex_pattern(self):
        """Test that invalid regex pattern is rejected."""
        with pytest.raises(ValidationError):
            StringTypeNode(
                node_id="/test",
                type="string",
                pattern="[invalid",  # Unclosed bracket
            )


# =============================================================================
# Test NumericTypeNode
# =============================================================================


class TestNumericTypeNode:
    """Tests for NumericTypeNode."""

    def test_create_numeric_node(self):
        """Test creating a NumericTypeNode."""
        node = NumericTypeNode(
            node_id="/test",
            type="number",
            minimum=0,
            maximum=100,
        )
        assert node.minimum == 0
        assert node.maximum == 100

    def test_integer_type(self):
        """Test integer type."""
        node = NumericTypeNode(node_id="/test", type="integer")
        assert node.is_integer() is True

    def test_exclusive_bounds(self):
        """Test exclusive minimum/maximum."""
        node = NumericTypeNode(
            node_id="/test",
            type="number",
            exclusive_minimum=0,
            exclusive_maximum=100,
        )
        assert node.get_effective_minimum() == 0
        assert node.get_effective_maximum() == 100

    def test_has_range(self):
        """Test has_range method."""
        node1 = NumericTypeNode(node_id="/test", type="number")
        assert node1.has_range() is False

        node2 = NumericTypeNode(node_id="/test", type="number", minimum=0)
        assert node2.has_range() is True

    def test_invalid_min_max(self):
        """Test that minimum > maximum is rejected."""
        with pytest.raises(ValidationError):
            NumericTypeNode(
                node_id="/test",
                type="number",
                minimum=100,
                maximum=0,
            )

    def test_multiple_of(self):
        """Test multipleOf constraint."""
        node = NumericTypeNode(
            node_id="/test",
            type="number",
            multiple_of=0.5,
        )
        assert node.multiple_of == 0.5

    def test_multiple_of_must_be_positive(self):
        """Test that multipleOf must be positive."""
        with pytest.raises(ValidationError):
            NumericTypeNode(
                node_id="/test",
                type="number",
                multiple_of=0,
            )


# =============================================================================
# Test ObjectTypeNode
# =============================================================================


class TestObjectTypeNode:
    """Tests for ObjectTypeNode."""

    def test_create_object_node(self, simple_string_node):
        """Test creating an ObjectTypeNode."""
        node = ObjectTypeNode(
            node_id="/",
            type="object",
            properties={"name": simple_string_node},
            required=["name"],
        )
        assert node.has_property("name") is True
        assert node.is_required("name") is True
        assert node.allows_additional_properties() is True

    def test_additional_properties_false(self):
        """Test additionalProperties: false."""
        node = ObjectTypeNode(
            node_id="/",
            type="object",
            additional_properties=False,
        )
        assert node.allows_additional_properties() is False

    def test_additional_properties_schema(self, simple_string_node):
        """Test additionalProperties with schema."""
        node = ObjectTypeNode(
            node_id="/",
            type="object",
            additional_properties=simple_string_node,
        )
        assert node.allows_additional_properties() is True

    def test_property_names_constraint(self):
        """Test propertyNames constraint."""
        name_schema = StringTypeNode(
            node_id="/propertyNames",
            type="string",
            pattern="^[a-z]+$",
        )
        node = ObjectTypeNode(
            node_id="/",
            type="object",
            property_names=name_schema,
        )
        assert node.property_names is not None

    def test_invalid_min_max_properties(self):
        """Test that minProperties > maxProperties is rejected."""
        with pytest.raises(ValidationError):
            ObjectTypeNode(
                node_id="/",
                type="object",
                min_properties=10,
                max_properties=5,
            )


# =============================================================================
# Test ArrayTypeNode
# =============================================================================


class TestArrayTypeNode:
    """Tests for ArrayTypeNode."""

    def test_create_array_node(self, simple_string_node):
        """Test creating an ArrayTypeNode."""
        node = ArrayTypeNode(
            node_id="/test",
            type="array",
            items=simple_string_node,
            min_items=1,
            max_items=10,
        )
        assert node.has_item_schema() is True
        assert node.min_items == 1
        assert node.max_items == 10

    def test_tuple_validation(self, simple_string_node, simple_numeric_node):
        """Test tuple validation with prefixItems."""
        node = ArrayTypeNode(
            node_id="/test",
            type="array",
            prefix_items=[simple_string_node, simple_numeric_node],
        )
        assert node.is_tuple() is True

    def test_unique_items(self):
        """Test uniqueItems constraint."""
        node = ArrayTypeNode(
            node_id="/test",
            type="array",
            unique_items=True,
        )
        assert node.unique_items is True

    def test_contains_constraint(self, simple_string_node):
        """Test contains constraint."""
        node = ArrayTypeNode(
            node_id="/test",
            type="array",
            contains=simple_string_node,
            min_contains=1,
            max_contains=5,
        )
        assert node.has_contains() is True
        assert node.min_contains == 1

    def test_invalid_min_max_items(self):
        """Test that minItems > maxItems is rejected."""
        with pytest.raises(ValidationError):
            ArrayTypeNode(
                node_id="/test",
                type="array",
                min_items=10,
                max_items=5,
            )

    def test_min_contains_without_contains(self):
        """Test that minContains without contains is rejected."""
        with pytest.raises(ValidationError):
            ArrayTypeNode(
                node_id="/test",
                type="array",
                min_contains=1,
            )


# =============================================================================
# Test RefNode
# =============================================================================


class TestRefNode:
    """Tests for RefNode."""

    def test_create_local_ref(self):
        """Test creating a local reference."""
        node = RefNode(node_id="/test", ref="#/$defs/Address")
        assert node.is_local_ref() is True
        assert node.is_external_ref() is False
        assert node.get_local_path() == "/$defs/Address"

    def test_create_external_ref(self):
        """Test creating an external reference."""
        node = RefNode(
            node_id="/test",
            ref="gl://schemas/common/address@1.0.0",
        )
        assert node.is_local_ref() is False
        assert node.is_external_ref() is True
        assert node.is_greenlang_ref() is True

    def test_ref_not_resolved_initially(self):
        """Test that ref is not resolved initially."""
        node = RefNode(node_id="/test", ref="#/$defs/Address")
        assert node.is_resolved() is False

    def test_empty_ref_rejected(self):
        """Test that empty ref is rejected."""
        with pytest.raises(ValidationError):
            RefNode(node_id="/test", ref="")


# =============================================================================
# Test CompositionNode
# =============================================================================


class TestCompositionNode:
    """Tests for CompositionNode."""

    def test_all_of(self, simple_string_node, simple_numeric_node):
        """Test allOf composition."""
        node = CompositionNode(
            node_id="/test",
            all_of=[simple_string_node, simple_numeric_node],
        )
        assert node.has_all_of() is True
        assert node.get_composition_type() == "allOf"

    def test_any_of(self, simple_string_node, simple_numeric_node):
        """Test anyOf composition."""
        node = CompositionNode(
            node_id="/test",
            any_of=[simple_string_node, simple_numeric_node],
        )
        assert node.has_any_of() is True
        assert node.get_composition_type() == "anyOf"

    def test_one_of(self, simple_string_node, simple_numeric_node):
        """Test oneOf composition."""
        node = CompositionNode(
            node_id="/test",
            one_of=[simple_string_node, simple_numeric_node],
        )
        assert node.has_one_of() is True
        assert node.get_composition_type() == "oneOf"

    def test_not(self, simple_string_node):
        """Test not composition."""
        node = CompositionNode(
            node_id="/test",
            not_=simple_string_node,
        )
        assert node.has_not() is True
        assert node.get_composition_type() == "not"

    def test_conditional(self, simple_string_node, simple_numeric_node):
        """Test if/then/else composition."""
        node = CompositionNode(
            node_id="/test",
            if_=simple_string_node,
            then_=simple_numeric_node,
        )
        assert node.has_conditional() is True
        assert node.get_composition_type() == "conditional"


# =============================================================================
# Test EnumTypeNode
# =============================================================================


class TestEnumTypeNode:
    """Tests for EnumTypeNode."""

    def test_create_enum_node(self):
        """Test creating an EnumTypeNode."""
        node = EnumTypeNode(
            node_id="/test",
            type="string",
            enum=["active", "inactive", "pending"],
        )
        assert node.get_value_count() == 3
        assert node.contains_value("active") is True
        assert node.contains_value("unknown") is False

    def test_empty_enum_rejected(self):
        """Test that empty enum is rejected."""
        with pytest.raises(ValidationError):
            EnumTypeNode(node_id="/test", enum=[])


# =============================================================================
# Test SchemaDocument
# =============================================================================


class TestSchemaDocument:
    """Tests for SchemaDocument."""

    def test_create_schema_document(self, simple_object_node):
        """Test creating a SchemaDocument."""
        doc = SchemaDocument(
            node_id="root",
            schema_id="test/activity",
            version="1.0.0",
            title="Activity Data",
            root=simple_object_node,
        )
        assert doc.schema_id == "test/activity"
        assert doc.version == "1.0.0"
        assert doc.dialect == JSON_SCHEMA_DRAFT_2020_12

    def test_schema_uri(self, simple_object_node):
        """Test to_uri method."""
        doc = SchemaDocument(
            node_id="root",
            schema_id="emissions/activity",
            version="1.3.0",
            root=simple_object_node,
        )
        assert doc.to_uri() == "gl://schemas/emissions/activity@1.3.0"

    def test_definitions(self, simple_object_node, simple_string_node):
        """Test schema with definitions."""
        doc = SchemaDocument(
            node_id="root",
            schema_id="test/schema",
            version="1.0.0",
            root=simple_object_node,
            definitions={"Address": simple_string_node},
        )
        assert doc.has_definition("Address") is True
        assert doc.get_definition("Address") == simple_string_node
        assert "Address" in doc.get_definition_names()

    def test_invalid_schema_id(self, simple_object_node):
        """Test that invalid schema_id is rejected."""
        with pytest.raises(ValidationError):
            SchemaDocument(
                node_id="root",
                schema_id="InvalidId",  # Must be lowercase
                version="1.0.0",
                root=simple_object_node,
            )

    def test_invalid_version(self, simple_object_node):
        """Test that invalid version is rejected."""
        with pytest.raises(ValidationError):
            SchemaDocument(
                node_id="root",
                schema_id="test/schema",
                version="v1.0",  # Must be semver
                root=simple_object_node,
            )

    def test_compute_hash(self, simple_object_node):
        """Test compute_hash method."""
        doc = SchemaDocument(
            node_id="root",
            schema_id="test/schema",
            version="1.0.0",
            root=simple_object_node,
        )
        hash1 = doc.compute_hash()
        hash2 = doc.compute_hash()
        assert hash1 == hash2  # Deterministic
        assert len(hash1) == 64  # SHA-256 hex length


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_node_id(self):
        """Test create_node_id function."""
        assert create_node_id("/properties/name") == "/properties/name"
        assert create_node_id("/items", 0) == "/items"
        assert create_node_id("/items", 1) == "/items[1]"

    def test_create_unique_node_id(self):
        """Test create_unique_node_id function."""
        id1 = create_unique_node_id("ref")
        id2 = create_unique_node_id("ref")
        assert id1.startswith("ref_")
        assert id1 != id2  # Should be unique


class TestParseTypeNode:
    """Tests for parse_type_node function."""

    def test_parse_string_type(self):
        """Test parsing string type."""
        data = {"type": "string", "minLength": 1, "maxLength": 100}
        node = parse_type_node(data, "/test")
        assert isinstance(node, StringTypeNode)
        assert node.min_length == 1

    def test_parse_numeric_type(self):
        """Test parsing numeric types."""
        data = {"type": "number", "minimum": 0, "maximum": 100}
        node = parse_type_node(data, "/test")
        assert isinstance(node, NumericTypeNode)
        assert node.minimum == 0

        data = {"type": "integer", "minimum": 0}
        node = parse_type_node(data, "/test")
        assert isinstance(node, NumericTypeNode)
        assert node.is_integer() is True

    def test_parse_object_type(self):
        """Test parsing object type."""
        data = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        node = parse_type_node(data, "/test")
        assert isinstance(node, ObjectTypeNode)
        assert node.is_required("name") is True

    def test_parse_array_type(self):
        """Test parsing array type."""
        data = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }
        node = parse_type_node(data, "/test")
        assert isinstance(node, ArrayTypeNode)
        assert node.has_item_schema() is True

    def test_parse_ref(self):
        """Test parsing $ref."""
        data = {"$ref": "#/$defs/Address"}
        node = parse_type_node(data, "/test")
        assert isinstance(node, RefNode)
        assert node.ref == "#/$defs/Address"

    def test_parse_composition(self):
        """Test parsing composition schemas."""
        data = {
            "anyOf": [{"type": "string"}, {"type": "number"}]
        }
        node = parse_type_node(data, "/test")
        assert isinstance(node, CompositionNode)
        assert node.has_any_of() is True

    def test_parse_enum(self):
        """Test parsing enum."""
        data = {"type": "string", "enum": ["a", "b", "c"]}
        node = parse_type_node(data, "/test")
        assert isinstance(node, EnumTypeNode)
        assert node.get_value_count() == 3

    def test_parse_with_gl_extensions(self):
        """Test parsing with GreenLang extensions."""
        data = {
            "type": "number",
            "$unit": {
                "dimension": "energy",
                "canonical": "kWh",
                "allowed": ["kWh", "MWh"],
            },
        }
        node = parse_type_node(data, "/test")
        assert node.gl_extensions is not None
        assert node.gl_extensions.unit.dimension == "energy"


class TestBuildAst:
    """Tests for build_ast function."""

    def test_build_simple_ast(self, sample_schema_dict):
        """Test building AST from schema dict."""
        doc = build_ast(sample_schema_dict, "test/activity", "1.0.0")
        assert doc.schema_id == "test/activity"
        assert doc.version == "1.0.0"
        assert doc.title == "Activity Data"
        assert isinstance(doc.root, ObjectTypeNode)

    def test_build_ast_with_definitions(self, sample_schema_dict):
        """Test that definitions are parsed."""
        doc = build_ast(sample_schema_dict, "test/activity", "1.0.0")
        assert doc.has_definition("Address") is True

    def test_build_ast_properties(self, sample_schema_dict):
        """Test that properties are correctly parsed."""
        doc = build_ast(sample_schema_dict, "test/activity", "1.0.0")
        root = doc.root
        assert isinstance(root, ObjectTypeNode)
        assert "name" in root.properties
        assert "energy" in root.properties
        assert "status" in root.properties


class TestValidateAst:
    """Tests for validate_ast function."""

    def test_validate_valid_ast(self, sample_schema_dict):
        """Test validating a valid AST."""
        doc = build_ast(sample_schema_dict, "test/activity", "1.0.0")
        errors = validate_ast(doc)
        assert len(errors) == 0

    def test_validate_detects_invalid_pattern(self):
        """Test that invalid patterns in patternProperties are detected."""
        root = ObjectTypeNode(
            node_id="/",
            type="object",
            pattern_properties={
                "[invalid": StringTypeNode(node_id="/patternProperties/[invalid", type="string")
            },
        )
        doc = SchemaDocument(
            node_id="root",
            schema_id="test/schema",
            version="1.0.0",
            root=root,
        )
        errors = validate_ast(doc)
        assert len(errors) > 0
        assert "Invalid regex pattern" in errors[0]


# =============================================================================
# Test Immutability
# =============================================================================


class TestImmutability:
    """Tests for model immutability (frozen=True)."""

    def test_schema_node_immutable(self):
        """Test that SchemaNode is immutable."""
        node = TypeNode(node_id="/test", type="string")
        with pytest.raises(ValidationError):
            node.node_id = "/changed"

    def test_unit_spec_immutable(self):
        """Test that UnitSpec is immutable."""
        unit = UnitSpec(dimension="energy", canonical="kWh")
        with pytest.raises(ValidationError):
            unit.dimension = "mass"

    def test_greenlang_extensions_immutable(self):
        """Test that GreenLangExtensions is immutable."""
        ext = GreenLangExtensions()
        with pytest.raises(ValidationError):
            ext.dimension = "energy"


# =============================================================================
# Test Serialization
# =============================================================================


class TestSerialization:
    """Tests for model serialization."""

    def test_to_dict(self, simple_string_node):
        """Test to_dict method."""
        d = simple_string_node.to_dict()
        assert isinstance(d, dict)
        assert d["node_id"] == "/properties/name"
        assert d["type"] == "string"

    def test_json_serializable(self, simple_object_node):
        """Test that models are JSON serializable."""
        doc = SchemaDocument(
            node_id="root",
            schema_id="test/schema",
            version="1.0.0",
            root=simple_object_node,
        )
        json_str = json.dumps(doc.to_dict())
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["schema_id"] == "test/schema"
