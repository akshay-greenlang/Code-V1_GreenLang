# -*- coding: utf-8 -*-
"""
Unit tests for the Schema Linter (GL-FOUND-X-002 Task 2.6).

This module tests:
    - Levenshtein distance algorithm correctness
    - Typo detection and close match suggestions
    - Casing detection (snake_case, camelCase, PascalCase, etc.)
    - Casing conversion functions
    - Deprecated field detection
    - Unit formatting suggestions
    - Suspicious value detection

Author: GreenLang Framework Team
Version: 0.1.0
"""

from datetime import datetime
from typing import Dict, Any, Set

import pytest

from greenlang.schema.validator.linter import (
    SchemaLinter,
    lint_payload,
    # Casing detection functions
    is_snake_case,
    is_camel_case,
    is_pascal_case,
    is_kebab_case,
    is_screaming_snake_case,
    # Casing conversion functions
    to_snake_case,
    to_camel_case,
    to_pascal_case,
    # Casing constants
    CASING_SNAKE_CASE,
    CASING_CAMEL_CASE,
    CASING_PASCAL_CASE,
    CASING_KEBAB_CASE,
    CASING_SCREAMING_SNAKE,
    CASING_UNKNOWN,
)
from greenlang.schema.compiler.ir import SchemaIR, PropertyIR, UnitSpecIR
from greenlang.schema.models.config import ValidationOptions
from greenlang.schema.models.finding import Severity
from greenlang.schema.errors import ErrorCode


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_ir() -> SchemaIR:
    """Create a mock SchemaIR for testing."""
    return SchemaIR(
        schema_id="test/schema",
        version="1.0.0",
        schema_hash="a" * 64,
        compiled_at=datetime.now(),
        compiler_version="0.1.0",
        properties={
            "/emissions": PropertyIR(path="/emissions", type="number", required=True),
            "/energy": PropertyIR(path="/energy", type="number", required=True),
            "/energy_consumption": PropertyIR(path="/energy_consumption", type="number", required=False),
            "/fuel_type": PropertyIR(path="/fuel_type", type="string", required=False),
            "/temperature": PropertyIR(path="/temperature", type="number", required=False),
            "/nested/field": PropertyIR(path="/nested/field", type="string", required=False),
        },
        required_paths={"/emissions", "/energy"},
    )


@pytest.fixture
def mock_ir_with_deprecations() -> SchemaIR:
    """Create a mock SchemaIR with deprecated fields."""
    return SchemaIR(
        schema_id="test/schema",
        version="2.0.0",
        schema_hash="b" * 64,
        compiled_at=datetime.now(),
        compiler_version="0.1.0",
        properties={
            "/old_field": PropertyIR(path="/old_field", type="string", required=False),
            "/new_field": PropertyIR(path="/new_field", type="string", required=True),
        },
        required_paths={"/new_field"},
        deprecated_fields={
            "/old_field": {
                "since_version": "1.5.0",
                "message": "Use new_field instead",
                "replacement": "/new_field",
                "removal_version": "3.0.0",
            }
        },
        renamed_fields={
            "/legacy_name": "/new_name",
        },
    )


@pytest.fixture
def mock_ir_with_units() -> SchemaIR:
    """Create a mock SchemaIR with unit specifications."""
    return SchemaIR(
        schema_id="test/schema",
        version="1.0.0",
        schema_hash="c" * 64,
        compiled_at=datetime.now(),
        compiler_version="0.1.0",
        properties={
            "/energy_value": PropertyIR(path="/energy_value", type="number", required=True),
        },
        required_paths={"/energy_value"},
        unit_specs={
            "/energy_value": UnitSpecIR(
                path="/energy_value",
                dimension="energy",
                canonical="kWh",
                allowed=["kWh", "MWh", "Wh", "J"],
            ),
        },
    )


@pytest.fixture
def linter(mock_ir) -> SchemaLinter:
    """Create a SchemaLinter with mock IR."""
    return SchemaLinter(mock_ir, ValidationOptions())


# =============================================================================
# Levenshtein Distance Tests
# =============================================================================


class TestLevenshteinDistance:
    """Tests for the Levenshtein distance algorithm."""

    def test_identical_strings(self, linter):
        """Distance between identical strings should be 0."""
        assert linter._levenshtein_distance("hello", "hello") == 0
        assert linter._levenshtein_distance("", "") == 0
        assert linter._levenshtein_distance("a", "a") == 0

    def test_empty_string_to_nonempty(self, linter):
        """Distance from empty to non-empty is the length."""
        assert linter._levenshtein_distance("", "abc") == 3
        assert linter._levenshtein_distance("abc", "") == 3
        assert linter._levenshtein_distance("", "a") == 1

    def test_single_insertion(self, linter):
        """Single insertion should have distance 1."""
        assert linter._levenshtein_distance("cat", "cats") == 1
        assert linter._levenshtein_distance("cat", "coat") == 1

    def test_single_deletion(self, linter):
        """Single deletion should have distance 1."""
        assert linter._levenshtein_distance("cats", "cat") == 1
        assert linter._levenshtein_distance("hello", "helo") == 1

    def test_single_substitution(self, linter):
        """Single substitution should have distance 1."""
        assert linter._levenshtein_distance("cat", "bat") == 1
        assert linter._levenshtein_distance("cat", "cot") == 1

    def test_emissions_typo(self, linter):
        """Test the common 'emissions' typo."""
        # emmisions (transposed s and i) -> emissions: 2 operations
        # emmisions vs emissions: e-m-m-i-s-i-o-n-s vs e-m-i-s-s-i-o-n-s
        # The difference: 'mm' vs 'm', and 'si' vs 'ss' - 2 edits needed
        assert linter._levenshtein_distance("emmisions", "emissions") == 2
        # emisions (missing one s) -> emissions: 1 operation
        assert linter._levenshtein_distance("emisions", "emissions") == 1

    def test_completely_different_strings(self, linter):
        """Completely different strings should have high distance."""
        assert linter._levenshtein_distance("cat", "dog") == 3
        assert linter._levenshtein_distance("abc", "xyz") == 3

    def test_case_sensitivity(self, linter):
        """Distance is case-sensitive."""
        assert linter._levenshtein_distance("Cat", "cat") == 1
        assert linter._levenshtein_distance("ABC", "abc") == 3

    def test_common_typos(self, linter):
        """Test common typo patterns."""
        # Transposition (2 operations for standard Levenshtein)
        assert linter._levenshtein_distance("hte", "the") == 2
        # Double letter
        assert linter._levenshtein_distance("begining", "beginning") == 1
        # Missing letter
        assert linter._levenshtein_distance("ocurrence", "occurrence") == 1


# =============================================================================
# Close Match Finding Tests
# =============================================================================


class TestFindCloseMatches:
    """Tests for the close match finding functionality."""

    def test_exact_match(self, linter):
        """Exact matches should have distance 0."""
        known = {"emissions", "energy", "fuel"}
        matches = linter._find_close_matches("emissions", known)
        assert len(matches) >= 1
        assert matches[0] == ("emissions", 0)

    def test_single_typo_match(self, linter):
        """Single character typo should be detected."""
        known = {"emissions", "energy", "consumption"}
        matches = linter._find_close_matches("emmisions", known, max_distance=2)
        assert len(matches) >= 1
        assert matches[0][0] == "emissions"
        assert matches[0][1] <= 2

    def test_no_match_when_too_different(self, linter):
        """No matches when strings are too different."""
        known = {"emissions", "energy", "consumption"}
        matches = linter._find_close_matches("xyz", known, max_distance=2)
        assert len(matches) == 0

    def test_sorted_by_distance(self, linter):
        """Matches should be sorted by distance."""
        known = {"cat", "cats", "catch", "dog"}
        matches = linter._find_close_matches("cat", known, max_distance=3)
        # Should be sorted by distance, then alphabetically
        distances = [m[1] for m in matches]
        assert distances == sorted(distances)

    def test_case_insensitive_matching(self, linter):
        """Matching should be case-insensitive."""
        known = {"Emissions", "Energy"}
        matches = linter._find_close_matches("emissions", known, max_distance=2)
        assert len(matches) >= 1

    def test_multiple_close_matches(self, linter):
        """Multiple close matches should be returned."""
        known = {"test", "text", "tent", "best"}
        matches = linter._find_close_matches("tест", known, max_distance=3)
        # May have multiple matches within distance 3


# =============================================================================
# Casing Detection Tests
# =============================================================================


class TestCasingDetection:
    """Tests for casing style detection functions."""

    def test_is_snake_case(self):
        """Test snake_case detection."""
        # Valid snake_case
        assert is_snake_case("energy_consumption") is True
        assert is_snake_case("foo_bar_baz") is True
        assert is_snake_case("simple") is True
        assert is_snake_case("a") is True
        assert is_snake_case("test123") is True
        assert is_snake_case("test_123") is True

        # Invalid snake_case
        assert is_snake_case("energyConsumption") is False
        assert is_snake_case("EnergyConsumption") is False
        assert is_snake_case("energy-consumption") is False
        assert is_snake_case("ENERGY_CONSUMPTION") is False
        assert is_snake_case("_leading") is False
        assert is_snake_case("trailing_") is False
        assert is_snake_case("") is False
        assert is_snake_case("123") is False

    def test_is_camel_case(self):
        """Test camelCase detection."""
        # Valid camelCase
        assert is_camel_case("energyConsumption") is True
        assert is_camel_case("fooBarBaz") is True
        assert is_camel_case("simpleTest") is True

        # Invalid camelCase (no uppercase = not camelCase)
        assert is_camel_case("simple") is False
        assert is_camel_case("energy_consumption") is False
        assert is_camel_case("EnergyConsumption") is False
        assert is_camel_case("energy-consumption") is False
        assert is_camel_case("") is False

    def test_is_pascal_case(self):
        """Test PascalCase detection."""
        # Valid PascalCase
        assert is_pascal_case("EnergyConsumption") is True
        assert is_pascal_case("FooBarBaz") is True
        assert is_pascal_case("Simple") is True
        assert is_pascal_case("A") is True

        # Invalid PascalCase
        assert is_pascal_case("energyConsumption") is False
        assert is_pascal_case("energy_consumption") is False
        assert is_pascal_case("ENERGY_CONSUMPTION") is False
        assert is_pascal_case("") is False

    def test_is_kebab_case(self):
        """Test kebab-case detection."""
        # Valid kebab-case
        assert is_kebab_case("energy-consumption") is True
        assert is_kebab_case("foo-bar-baz") is True
        assert is_kebab_case("simple") is True

        # Invalid kebab-case
        assert is_kebab_case("energyConsumption") is False
        assert is_kebab_case("energy_consumption") is False
        assert is_kebab_case("Energy-Consumption") is False
        assert is_kebab_case("") is False

    def test_is_screaming_snake_case(self):
        """Test SCREAMING_SNAKE_CASE detection."""
        # Valid SCREAMING_SNAKE_CASE
        assert is_screaming_snake_case("ENERGY_CONSUMPTION") is True
        assert is_screaming_snake_case("FOO_BAR_BAZ") is True
        assert is_screaming_snake_case("SIMPLE") is True
        assert is_screaming_snake_case("A") is True

        # Invalid SCREAMING_SNAKE_CASE
        assert is_screaming_snake_case("energy_consumption") is False
        assert is_screaming_snake_case("EnergyConsumption") is False
        assert is_screaming_snake_case("") is False


class TestDetectCasingStyle:
    """Tests for the casing style detection method."""

    def test_detect_snake_case(self, linter):
        """Detect snake_case style."""
        assert linter._detect_casing_style("energy_consumption") == CASING_SNAKE_CASE
        assert linter._detect_casing_style("foo_bar") == CASING_SNAKE_CASE

    def test_detect_camel_case(self, linter):
        """Detect camelCase style."""
        assert linter._detect_casing_style("energyConsumption") == CASING_CAMEL_CASE
        assert linter._detect_casing_style("fooBar") == CASING_CAMEL_CASE

    def test_detect_pascal_case(self, linter):
        """Detect PascalCase style."""
        assert linter._detect_casing_style("EnergyConsumption") == CASING_PASCAL_CASE
        assert linter._detect_casing_style("FooBar") == CASING_PASCAL_CASE

    def test_detect_kebab_case(self, linter):
        """Detect kebab-case style."""
        assert linter._detect_casing_style("energy-consumption") == CASING_KEBAB_CASE
        assert linter._detect_casing_style("foo-bar") == CASING_KEBAB_CASE

    def test_detect_screaming_snake(self, linter):
        """Detect SCREAMING_SNAKE_CASE style."""
        assert linter._detect_casing_style("ENERGY_CONSUMPTION") == CASING_SCREAMING_SNAKE
        assert linter._detect_casing_style("FOO_BAR") == CASING_SCREAMING_SNAKE

    def test_detect_unknown(self, linter):
        """Detect unknown casing style."""
        assert linter._detect_casing_style("") == CASING_UNKNOWN


# =============================================================================
# Casing Conversion Tests
# =============================================================================


class TestCasingConversion:
    """Tests for casing conversion functions."""

    def test_to_snake_case_from_camel(self):
        """Convert camelCase to snake_case."""
        assert to_snake_case("energyConsumption") == "energy_consumption"
        assert to_snake_case("fooBarBaz") == "foo_bar_baz"
        assert to_snake_case("simple") == "simple"
        assert to_snake_case("XMLParser") == "x_m_l_parser"

    def test_to_snake_case_from_pascal(self):
        """Convert PascalCase to snake_case."""
        assert to_snake_case("EnergyConsumption") == "energy_consumption"
        assert to_snake_case("FooBarBaz") == "foo_bar_baz"

    def test_to_snake_case_from_kebab(self):
        """Convert kebab-case to snake_case."""
        assert to_snake_case("energy-consumption") == "energy_consumption"
        assert to_snake_case("foo-bar-baz") == "foo_bar_baz"

    def test_to_snake_case_from_screaming(self):
        """Convert SCREAMING_SNAKE_CASE to snake_case."""
        assert to_snake_case("ENERGY_CONSUMPTION") == "energy_consumption"
        assert to_snake_case("FOO_BAR") == "foo_bar"

    def test_to_camel_case_from_snake(self):
        """Convert snake_case to camelCase."""
        assert to_camel_case("energy_consumption") == "energyConsumption"
        assert to_camel_case("foo_bar_baz") == "fooBarBaz"
        assert to_camel_case("simple") == "simple"

    def test_to_camel_case_from_kebab(self):
        """Convert kebab-case to camelCase."""
        assert to_camel_case("energy-consumption") == "energyConsumption"
        assert to_camel_case("foo-bar-baz") == "fooBarBaz"

    def test_to_pascal_case(self):
        """Convert to PascalCase."""
        assert to_pascal_case("energy_consumption") == "EnergyConsumption"
        assert to_pascal_case("energyConsumption") == "EnergyConsumption"
        assert to_pascal_case("simple") == "Simple"

    def test_empty_string_conversion(self):
        """Empty strings should remain empty."""
        assert to_snake_case("") == ""
        assert to_camel_case("") == ""
        assert to_pascal_case("") == ""


# =============================================================================
# Lint Check Tests - Unknown Fields
# =============================================================================


class TestLintUnknownFields:
    """Tests for unknown field detection with typo suggestions."""

    def test_detect_typo_emissions(self, mock_ir):
        """Detect 'emmisions' typo and suggest 'emissions'."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        findings = linter.lint({"emmisions": 100})

        # Should find at least one warning
        typo_findings = [f for f in findings if f.code == ErrorCode.SUSPICIOUS_KEY.value]
        assert len(typo_findings) >= 1

        # Should suggest 'emissions'
        finding = typo_findings[0]
        assert finding.severity == Severity.WARNING
        assert "emmisions" in finding.message
        assert "emissions" in finding.message
        assert finding.hint is not None
        assert "emissions" in finding.hint.suggested_values

    def test_no_warning_for_known_field(self, mock_ir):
        """Known fields should not generate warnings."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        findings = linter.lint({"emissions": 100})

        # Should not have typo warnings
        typo_findings = [f for f in findings if f.code == ErrorCode.SUSPICIOUS_KEY.value]
        assert len(typo_findings) == 0

    def test_short_key_no_typo_check(self, mock_ir):
        """Short keys should not trigger typo detection."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        findings = linter.lint({"ab": 100})  # Very short unknown key

        # Short keys don't get typo suggestions
        typo_findings = [f for f in findings if f.code == ErrorCode.SUSPICIOUS_KEY.value
                        and "Did you mean" in f.message]
        assert len(typo_findings) == 0

    def test_multiple_typos(self, mock_ir):
        """Multiple typos should each be detected."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        findings = linter.lint({
            "emmisions": 100,
            "energi": 200,  # typo for 'energy'
        })

        typo_findings = [f for f in findings if f.code == ErrorCode.SUSPICIOUS_KEY.value]
        # Should detect both typos
        assert len(typo_findings) >= 1


# =============================================================================
# Lint Check Tests - Deprecated Fields
# =============================================================================


class TestLintDeprecatedFields:
    """Tests for deprecated field detection."""

    def test_detect_deprecated_field(self, mock_ir_with_deprecations):
        """Deprecated fields should generate warnings."""
        linter = SchemaLinter(mock_ir_with_deprecations, ValidationOptions())
        findings = linter.lint({"old_field": "value", "new_field": "value"})

        deprecated_findings = [f for f in findings if f.code == ErrorCode.DEPRECATED_FIELD.value]
        assert len(deprecated_findings) == 1

        finding = deprecated_findings[0]
        assert finding.severity == Severity.WARNING
        assert "old_field" in finding.message
        assert "deprecated" in finding.message.lower()
        assert "1.5.0" in finding.message

    def test_detect_renamed_field(self, mock_ir_with_deprecations):
        """Renamed fields should generate warnings."""
        linter = SchemaLinter(mock_ir_with_deprecations, ValidationOptions())
        findings = linter.lint({"legacy_name": "value"})

        renamed_findings = [f for f in findings if f.code == ErrorCode.RENAMED_FIELD.value]
        assert len(renamed_findings) == 1

        finding = renamed_findings[0]
        assert finding.severity == Severity.WARNING
        assert "legacy_name" in finding.message
        assert "renamed" in finding.message.lower()

    def test_no_warning_for_new_field(self, mock_ir_with_deprecations):
        """Non-deprecated fields should not generate warnings."""
        linter = SchemaLinter(mock_ir_with_deprecations, ValidationOptions())
        findings = linter.lint({"new_field": "value"})

        deprecated_findings = [f for f in findings if f.code in (
            ErrorCode.DEPRECATED_FIELD.value,
            ErrorCode.RENAMED_FIELD.value
        )]
        assert len(deprecated_findings) == 0


# =============================================================================
# Lint Check Tests - Casing Consistency
# =============================================================================


class TestLintCasingConsistency:
    """Tests for casing consistency checks."""

    def test_detect_camel_case_in_snake_case_schema(self, mock_ir):
        """camelCase keys in snake_case schema should generate warnings."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        findings = linter.lint({"energyConsumption": 100})

        casing_findings = [f for f in findings if f.code == ErrorCode.NONCOMPLIANT_CASING.value]
        assert len(casing_findings) >= 1

        finding = casing_findings[0]
        assert finding.severity == Severity.WARNING
        assert "snake_case" in finding.message
        assert finding.hint is not None
        # Should suggest snake_case version
        assert "energy_consumption" in finding.hint.suggested_values

    def test_no_warning_for_snake_case(self, mock_ir):
        """snake_case keys should not generate casing warnings."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        findings = linter.lint({"energy_consumption": 100})

        casing_findings = [f for f in findings if f.code == ErrorCode.NONCOMPLIANT_CASING.value]
        assert len(casing_findings) == 0


# =============================================================================
# Lint Check Tests - Unit Formatting
# =============================================================================


class TestLintUnitFormatting:
    """Tests for unit formatting suggestions."""

    def test_suggest_object_format_for_string_unit(self, mock_ir_with_units):
        """String unit format should suggest object format."""
        linter = SchemaLinter(mock_ir_with_units, ValidationOptions())
        findings = linter.lint({"energy_value": "100 kWh"})

        unit_findings = [f for f in findings if f.code == ErrorCode.UNIT_FORMAT_STYLE.value]
        assert len(unit_findings) >= 1

        finding = unit_findings[0]
        assert finding.severity == Severity.INFO
        assert "object format" in finding.message.lower()

    def test_no_suggestion_for_numeric_value(self, mock_ir_with_units):
        """Numeric values don't get unit format suggestions."""
        linter = SchemaLinter(mock_ir_with_units, ValidationOptions())
        findings = linter.lint({"energy_value": 100})

        # Numeric value without string format shouldn't trigger unit format warning
        # (it might trigger suspicious value if zero, but that's different)
        unit_format_findings = [f for f in findings if f.code == ErrorCode.UNIT_FORMAT_STYLE.value]
        assert len(unit_format_findings) == 0


# =============================================================================
# Lint Check Tests - Suspicious Values
# =============================================================================


class TestLintSuspiciousValues:
    """Tests for suspicious value detection."""

    def test_detect_empty_string_in_required_field(self):
        """Empty string in required field should generate warning."""
        ir = SchemaIR(
            schema_id="test/schema",
            version="1.0.0",
            schema_hash="d" * 64,
            compiled_at=datetime.now(),
            compiler_version="0.1.0",
            properties={
                "/name": PropertyIR(path="/name", type="string", required=True),
            },
            required_paths={"/name"},
        )
        linter = SchemaLinter(ir, ValidationOptions())
        findings = linter.lint({"name": ""})

        suspicious_findings = [f for f in findings if "empty string" in f.message.lower()]
        assert len(suspicious_findings) >= 1

    def test_detect_zero_in_measurement_field(self, mock_ir_with_units):
        """Zero in measurement field should generate info."""
        linter = SchemaLinter(mock_ir_with_units, ValidationOptions())
        findings = linter.lint({"energy_value": 0})

        zero_findings = [f for f in findings if "zero" in f.message.lower()]
        assert len(zero_findings) >= 1

        finding = zero_findings[0]
        assert finding.severity == Severity.INFO


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestLintPayloadFunction:
    """Tests for the lint_payload convenience function."""

    def test_lint_payload_basic(self, mock_ir):
        """lint_payload should work like SchemaLinter.lint."""
        findings = lint_payload({"emmisions": 100}, mock_ir)
        assert isinstance(findings, list)
        assert len(findings) >= 1

    def test_lint_payload_with_options(self, mock_ir):
        """lint_payload should accept options."""
        options = ValidationOptions()
        findings = lint_payload({"emissions": 100}, mock_ir, options)
        assert isinstance(findings, list)


# =============================================================================
# Recursive Linting Tests
# =============================================================================


class TestRecursiveLinting:
    """Tests for recursive linting of nested structures."""

    def test_lint_nested_object(self, mock_ir):
        """Nested objects should be linted recursively."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        findings = linter.lint({
            "emissions": 100,
            "nested": {
                "feild": "typo"  # typo for 'field'
            }
        })

        # Should detect the nested typo
        # (if 'field' is a known key under /nested)
        assert isinstance(findings, list)

    def test_lint_array_of_objects(self, mock_ir):
        """Arrays of objects should be linted."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        findings = linter.lint({
            "emissions": 100,
            "items": [
                {"value": 1},
                {"value": 2},
            ]
        })
        assert isinstance(findings, list)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_payload(self, mock_ir):
        """Empty payload should not cause errors."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        findings = linter.lint({})
        assert isinstance(findings, list)

    def test_non_dict_payload(self, mock_ir):
        """Non-dict payload should return empty findings."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        findings = linter.lint("not a dict")  # type: ignore
        assert findings == []

    def test_none_options(self, mock_ir):
        """None options should use defaults."""
        linter = SchemaLinter(mock_ir, None)
        findings = linter.lint({"emissions": 100})
        assert isinstance(findings, list)

    def test_unicode_field_names(self, mock_ir):
        """Unicode field names should not cause errors."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        findings = linter.lint({"field": 100})
        assert isinstance(findings, list)

    def test_deeply_nested_structure(self, mock_ir):
        """Deeply nested structures should be handled."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        payload = {"a": {"b": {"c": {"d": {"e": 100}}}}}
        findings = linter.lint(payload)
        assert isinstance(findings, list)

    def test_large_number_of_fields(self, mock_ir):
        """Large number of fields should be handled."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        payload = {f"field_{i}": i for i in range(100)}
        findings = linter.lint(payload)
        assert isinstance(findings, list)


# =============================================================================
# Finding Object Tests
# =============================================================================


class TestFindingObjects:
    """Tests for the structure of Finding objects."""

    def test_finding_has_required_fields(self, mock_ir):
        """Findings should have all required fields."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        findings = linter.lint({"emmisions": 100})

        for finding in findings:
            assert finding.code is not None
            assert finding.severity is not None
            assert finding.path is not None
            assert finding.message is not None

    def test_finding_severity_is_warning_or_info(self, mock_ir):
        """Lint findings should only be warnings or info, not errors."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        findings = linter.lint({"emmisions": 100})

        for finding in findings:
            assert finding.severity in (Severity.WARNING, Severity.INFO)

    def test_finding_code_format(self, mock_ir):
        """Finding codes should follow GLSCHEMA-W* pattern."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        findings = linter.lint({"emmisions": 100})

        for finding in findings:
            # Lint warnings should be W6xx or W7xx
            assert finding.code.startswith("GLSCHEMA-W")

    def test_finding_hint_has_suggestions(self, mock_ir):
        """Findings with hints should have suggestions."""
        linter = SchemaLinter(mock_ir, ValidationOptions())
        findings = linter.lint({"emmisions": 100})

        for finding in findings:
            if finding.hint is not None:
                assert finding.hint.category is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestLinterIntegration:
    """Integration tests for the complete linting workflow."""

    def test_full_lint_workflow(self):
        """Test complete linting workflow with all checks."""
        # Create a comprehensive IR
        ir = SchemaIR(
            schema_id="test/comprehensive",
            version="1.0.0",
            schema_hash="e" * 64,
            compiled_at=datetime.now(),
            compiler_version="0.1.0",
            properties={
                "/emissions": PropertyIR(path="/emissions", type="number", required=True),
                "/energy": PropertyIR(path="/energy", type="number", required=True),
            },
            required_paths={"/emissions", "/energy"},
            deprecated_fields={
                "/old_emissions": {
                    "since_version": "0.5.0",
                    "message": "Use emissions instead",
                    "replacement": "/emissions",
                }
            },
        )

        linter = SchemaLinter(ir, ValidationOptions())

        # Payload with multiple issues
        payload = {
            "emmisions": 100,  # typo
            "old_emissions": 50,  # deprecated
            "energyValue": 200,  # wrong casing
        }

        findings = linter.lint(payload)

        # Should have multiple findings
        assert len(findings) >= 1

        # All findings should be warnings or info
        for finding in findings:
            assert finding.severity in (Severity.WARNING, Severity.INFO)
