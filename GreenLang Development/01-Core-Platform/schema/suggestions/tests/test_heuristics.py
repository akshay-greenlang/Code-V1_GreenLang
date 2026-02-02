# -*- coding: utf-8 -*-
"""
Unit tests for Fix Heuristics (GL-FOUND-X-002 Task 4.3).

This module tests the FixHeuristics class and its heuristic methods.

Author: GreenLang Framework Team
Version: 0.1.0
"""

import pytest
from datetime import datetime
from typing import Any, Dict

from greenlang.schema.suggestions.heuristics import (
    FixHeuristics,
    MAX_TYPO_DISTANCE,
    MAX_ENUM_DISTANCE,
    RANGE_TYPO_THRESHOLD,
    BOOLEAN_TRUE_VALUES,
    BOOLEAN_FALSE_VALUES,
)
from greenlang.schema.compiler.ir import SchemaIR, PropertyIR, NumericConstraintIR, UnitSpecIR
from greenlang.schema.models.finding import Finding, Severity
from greenlang.schema.models.patch import PatchSafety


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_schema_ir() -> SchemaIR:
    """Create a mock SchemaIR for testing."""
    return SchemaIR(
        schema_id="test-schema",
        version="1.0.0",
        schema_hash="a" * 64,
        compiled_at=datetime.now(),
        properties={},
        required_paths=set(),
        enums={},
        numeric_constraints={},
        unit_specs={},
        renamed_fields={},
        deprecated_fields={},
    )


@pytest.fixture
def schema_ir_with_properties() -> SchemaIR:
    """Create a SchemaIR with properties for testing."""
    return SchemaIR(
        schema_id="test-schema",
        version="1.0.0",
        schema_hash="a" * 64,
        compiled_at=datetime.now(),
        properties={
            "/name": PropertyIR(
                path="/name",
                type="string",
                required=True,
                has_default=False,
            ),
            "/count": PropertyIR(
                path="/count",
                type="integer",
                required=False,
                has_default=True,
                default_value=0,
            ),
            "/enabled": PropertyIR(
                path="/enabled",
                type="boolean",
                required=False,
                has_default=True,
                default_value=False,
            ),
            "/energy": PropertyIR(
                path="/energy",
                type="object",
                required=False,
                has_default=False,
            ),
            "/category": PropertyIR(
                path="/category",
                type="string",
                required=True,
                has_default=False,
            ),
        },
        required_paths={"/name", "/category"},
        enums={
            "/category": ["scope1", "scope2", "scope3"],
        },
        numeric_constraints={
            "/count": NumericConstraintIR(
                path="/count",
                minimum=0,
                maximum=100,
            ),
        },
        unit_specs={
            "/energy": UnitSpecIR(
                path="/energy",
                dimension="energy",
                canonical="kWh",
                allowed=["kWh", "MWh", "Wh", "J"],
            ),
        },
        renamed_fields={
            "old_name": "name",
        },
        deprecated_fields={},
    )


@pytest.fixture
def heuristics(schema_ir_with_properties: SchemaIR) -> FixHeuristics:
    """Create FixHeuristics instance for testing."""
    return FixHeuristics(schema_ir_with_properties)


# =============================================================================
# TEST: Levenshtein Distance
# =============================================================================


class TestLevenshteinDistance:
    """Tests for Levenshtein edit distance calculation."""

    def test_identical_strings(self, heuristics: FixHeuristics) -> None:
        """Identical strings have distance 0."""
        assert heuristics._levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self, heuristics: FixHeuristics) -> None:
        """Empty string distance equals length of other string."""
        assert heuristics._levenshtein_distance("", "hello") == 5
        assert heuristics._levenshtein_distance("hello", "") == 5

    def test_single_substitution(self, heuristics: FixHeuristics) -> None:
        """Single character substitution has distance 1."""
        assert heuristics._levenshtein_distance("cat", "bat") == 1

    def test_single_insertion(self, heuristics: FixHeuristics) -> None:
        """Single character insertion has distance 1."""
        assert heuristics._levenshtein_distance("cat", "cats") == 1

    def test_single_deletion(self, heuristics: FixHeuristics) -> None:
        """Single character deletion has distance 1."""
        assert heuristics._levenshtein_distance("cats", "cat") == 1

    def test_complex_edit(self, heuristics: FixHeuristics) -> None:
        """Complex edits compute correctly."""
        # "kitten" -> "sitting" requires 3 operations
        assert heuristics._levenshtein_distance("kitten", "sitting") == 3


# =============================================================================
# TEST: Type Coercion
# =============================================================================


class TestTypeCoercion:
    """Tests for safe type coercion."""

    def test_string_to_integer(self, heuristics: FixHeuristics) -> None:
        """String '42' can be coerced to integer 42."""
        can_coerce, value = heuristics._can_coerce("42", "integer")
        assert can_coerce is True
        assert value == 42

    def test_string_to_integer_invalid(self, heuristics: FixHeuristics) -> None:
        """String 'abc' cannot be coerced to integer."""
        can_coerce, value = heuristics._can_coerce("abc", "integer")
        assert can_coerce is False

    def test_string_to_number(self, heuristics: FixHeuristics) -> None:
        """String '3.14' can be coerced to float 3.14."""
        can_coerce, value = heuristics._can_coerce("3.14", "number")
        assert can_coerce is True
        assert value == 3.14

    def test_string_to_boolean_true(self, heuristics: FixHeuristics) -> None:
        """String 'true' can be coerced to True."""
        can_coerce, value = heuristics._can_coerce("true", "boolean")
        assert can_coerce is True
        assert value is True

    def test_string_to_boolean_false(self, heuristics: FixHeuristics) -> None:
        """String 'false' can be coerced to False."""
        can_coerce, value = heuristics._can_coerce("false", "boolean")
        assert can_coerce is True
        assert value is False

    def test_string_yes_to_boolean(self, heuristics: FixHeuristics) -> None:
        """String 'yes' can be coerced to True."""
        can_coerce, value = heuristics._can_coerce("yes", "boolean")
        assert can_coerce is True
        assert value is True

    def test_non_string_cannot_coerce(self, heuristics: FixHeuristics) -> None:
        """Non-string values cannot be coerced."""
        can_coerce, value = heuristics._can_coerce(42, "integer")
        assert can_coerce is False


# =============================================================================
# TEST: Close Match Finding
# =============================================================================


class TestCloseMatches:
    """Tests for finding close matches via edit distance."""

    def test_exact_match(self, heuristics: FixHeuristics) -> None:
        """Exact match has distance 0."""
        matches = heuristics._find_close_matches(
            "name",
            {"name", "count", "enabled"},
            max_distance=2
        )
        assert len(matches) == 1
        assert matches[0] == ("name", 0)

    def test_typo_match(self, heuristics: FixHeuristics) -> None:
        """Typo 'nme' matches 'name' with distance 1."""
        matches = heuristics._find_close_matches(
            "nme",
            {"name", "count", "enabled"},
            max_distance=2
        )
        assert len(matches) == 1
        assert matches[0] == ("name", 1)

    def test_no_match(self, heuristics: FixHeuristics) -> None:
        """No match when distance exceeds threshold."""
        matches = heuristics._find_close_matches(
            "xyz",
            {"name", "count", "enabled"},
            max_distance=2
        )
        assert len(matches) == 0

    def test_multiple_matches_sorted(self, heuristics: FixHeuristics) -> None:
        """Multiple matches are sorted by distance."""
        matches = heuristics._find_close_matches(
            "countt",
            {"count", "county", "mount"},
            max_distance=2
        )
        # "count" has distance 1, others have distance 2
        assert len(matches) >= 1
        assert matches[0][0] == "count"


# =============================================================================
# TEST: Fix Suggestions
# =============================================================================


class TestTypeMismatchSuggestion:
    """Tests for type mismatch fix suggestions."""

    def test_suggest_string_to_integer_coercion(
        self,
        heuristics: FixHeuristics
    ) -> None:
        """Suggests coercion from string to integer."""
        finding = Finding(
            code="GLSCHEMA-E102",
            path="/count",
            severity=Severity.ERROR,
            message="Expected integer, got string",
            expected={"type": "integer"},
            actual="42",
        )
        payload = {"count": "42"}

        suggestion = heuristics.suggest_for_type_mismatch(finding, payload)

        assert suggestion is not None
        assert suggestion.safety == PatchSafety.SAFE
        assert suggestion.confidence >= 0.9
        assert len(suggestion.patch) >= 1


class TestUnknownFieldSuggestion:
    """Tests for unknown field fix suggestions."""

    def test_suggest_typo_correction(
        self,
        heuristics: FixHeuristics
    ) -> None:
        """Suggests typo correction for unknown field."""
        finding = Finding(
            code="GLSCHEMA-E101",
            path="/nme",
            severity=Severity.ERROR,
            message="Unknown field 'nme'",
        )
        payload = {"nme": "test"}

        suggestion = heuristics.suggest_for_unknown_field(finding, payload)

        assert suggestion is not None
        assert suggestion.safety == PatchSafety.NEEDS_REVIEW
        assert "typo" in suggestion.rationale.lower()

    def test_suggest_schema_rename(
        self,
        heuristics: FixHeuristics
    ) -> None:
        """Suggests rename for field with renamed_from declaration."""
        finding = Finding(
            code="GLSCHEMA-E101",
            path="/old_name",
            severity=Severity.ERROR,
            message="Unknown field 'old_name'",
        )
        payload = {"old_name": "test"}

        suggestion = heuristics.suggest_for_unknown_field(finding, payload)

        assert suggestion is not None
        assert suggestion.safety == PatchSafety.SAFE
        assert suggestion.confidence == 1.0


class TestEnumViolationSuggestion:
    """Tests for enum violation fix suggestions."""

    def test_suggest_close_enum_value(
        self,
        heuristics: FixHeuristics
    ) -> None:
        """Suggests closest enum value for typo."""
        finding = Finding(
            code="GLSCHEMA-E202",
            path="/category",
            severity=Severity.ERROR,
            message="Invalid enum value",
            expected={"enum": ["scope1", "scope2", "scope3"]},
            actual="scop1",  # Typo
        )
        payload = {"category": "scop1"}

        suggestion = heuristics.suggest_for_enum_violation(finding, payload)

        assert suggestion is not None
        assert suggestion.safety == PatchSafety.NEEDS_REVIEW
        assert "scope1" in suggestion.rationale


# =============================================================================
# TEST: Dispatch
# =============================================================================


class TestSuggestionDispatch:
    """Tests for main suggestion dispatch method."""

    def test_dispatch_to_type_mismatch(
        self,
        heuristics: FixHeuristics
    ) -> None:
        """Dispatches E102 to type mismatch handler."""
        finding = Finding(
            code="GLSCHEMA-E102",
            path="/count",
            severity=Severity.ERROR,
            message="Type mismatch",
            expected={"type": "integer"},
            actual="42",
        )
        payload = {"count": "42"}

        suggestion = heuristics.suggest_for_finding(finding, payload)
        assert suggestion is not None

    def test_dispatch_unknown_code_returns_none(
        self,
        heuristics: FixHeuristics
    ) -> None:
        """Unknown error code returns None."""
        finding = Finding(
            code="GLSCHEMA-E999",
            path="/unknown",
            severity=Severity.ERROR,
            message="Unknown error",
        )
        payload = {}

        suggestion = heuristics.suggest_for_finding(finding, payload)
        assert suggestion is None


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    "TestLevenshteinDistance",
    "TestTypeCoercion",
    "TestCloseMatches",
    "TestTypeMismatchSuggestion",
    "TestUnknownFieldSuggestion",
    "TestEnumViolationSuggestion",
    "TestSuggestionDispatch",
]
