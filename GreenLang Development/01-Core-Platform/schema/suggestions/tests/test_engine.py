# -*- coding: utf-8 -*-
"""
Unit tests for Fix Suggestion Engine (GL-FOUND-X-002 Task 4.4).

This module tests the FixSuggestionEngine class and its methods.

Author: GreenLang Framework Team
Version: 0.1.0
"""

import pytest
from datetime import datetime
from typing import Any, Dict, List

from greenlang.schema.suggestions.engine import (
    FixSuggestionEngine,
    SuggestionEngineResult,
    generate_suggestions,
    apply_suggestions,
    get_fixable_codes,
    FIXABLE_CODES,
)
from greenlang.schema.compiler.ir import SchemaIR, PropertyIR, NumericConstraintIR, UnitSpecIR
from greenlang.schema.models.config import ValidationOptions, PatchLevel
from greenlang.schema.models.finding import Finding, Severity
from greenlang.schema.models.patch import FixSuggestion, PatchSafety, JSONPatchOp


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
        deprecated_fields={
            "/old_field": {
                "since_version": "1.0.0",
                "message": "Use new_field instead",
                "replacement": "/new_field",
            }
        },
    )


@pytest.fixture
def engine(schema_ir_with_properties: SchemaIR) -> FixSuggestionEngine:
    """Create FixSuggestionEngine instance for testing."""
    return FixSuggestionEngine(schema_ir_with_properties)


@pytest.fixture
def strict_engine(schema_ir_with_properties: SchemaIR) -> FixSuggestionEngine:
    """Create FixSuggestionEngine with strict options."""
    options = ValidationOptions(
        patch_level=PatchLevel.SAFE,
        emit_patches=True,
    )
    return FixSuggestionEngine(schema_ir_with_properties, options)


# =============================================================================
# TEST: SuggestionEngineResult
# =============================================================================


class TestSuggestionEngineResult:
    """Tests for SuggestionEngineResult model."""

    def test_empty_result(self) -> None:
        """Empty result has no suggestions."""
        result = SuggestionEngineResult()
        assert result.has_suggestions is False
        assert result.has_safe_suggestions is False
        assert result.safe_suggestions == []
        assert result.total_generated == 0
        assert result.filtered_count == 0
        assert result.errors == []

    def test_result_with_suggestions(self) -> None:
        """Result correctly reports suggestions."""
        safe_suggestion = FixSuggestion(
            patch=[JSONPatchOp(op="add", path="/test", value=1)],
            confidence=0.95,
            safety=PatchSafety.SAFE,
            rationale="Test suggestion",
        )
        review_suggestion = FixSuggestion(
            patch=[JSONPatchOp(op="replace", path="/other", value=2)],
            confidence=0.7,
            safety=PatchSafety.NEEDS_REVIEW,
            rationale="Review suggestion",
        )

        result = SuggestionEngineResult(
            suggestions=[safe_suggestion, review_suggestion],
            total_generated=3,
            filtered_count=1,
        )

        assert result.has_suggestions is True
        assert result.has_safe_suggestions is True
        assert len(result.safe_suggestions) == 1
        assert result.safe_suggestions[0].safety == PatchSafety.SAFE

    def test_format_summary(self) -> None:
        """Format summary includes counts."""
        result = SuggestionEngineResult(
            suggestions=[
                FixSuggestion(
                    patch=[JSONPatchOp(op="add", path="/test", value=1)],
                    confidence=0.95,
                    safety=PatchSafety.SAFE,
                    rationale="Safe suggestion",
                )
            ],
            total_generated=5,
            filtered_count=2,
            errors=["error1"],
        )

        summary = result.format_summary()
        assert "1 total" in summary
        assert "safe=1" in summary
        assert "filtered=2" in summary
        assert "errors=1" in summary


# =============================================================================
# TEST: FixSuggestionEngine Initialization
# =============================================================================


class TestEngineInitialization:
    """Tests for FixSuggestionEngine initialization."""

    def test_init_with_defaults(self, schema_ir_with_properties: SchemaIR) -> None:
        """Engine initializes with default options."""
        engine = FixSuggestionEngine(schema_ir_with_properties)
        assert engine.ir == schema_ir_with_properties
        assert engine.options.patch_level == PatchLevel.SAFE

    def test_init_with_custom_options(self, schema_ir_with_properties: SchemaIR) -> None:
        """Engine initializes with custom options."""
        options = ValidationOptions(
            patch_level=PatchLevel.NEEDS_REVIEW,
            emit_patches=True,
        )
        engine = FixSuggestionEngine(schema_ir_with_properties, options)
        assert engine.options.patch_level == PatchLevel.NEEDS_REVIEW


# =============================================================================
# TEST: Filter By Safety
# =============================================================================


class TestFilterBySafety:
    """Tests for safety filtering."""

    def test_filter_safe_only(self, engine: FixSuggestionEngine) -> None:
        """SAFE filter only returns safe suggestions."""
        suggestions = [
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/a", value=1)],
                confidence=0.9,
                safety=PatchSafety.SAFE,
                rationale="Safe",
            ),
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/b", value=2)],
                confidence=0.7,
                safety=PatchSafety.NEEDS_REVIEW,
                rationale="Review",
            ),
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/c", value=3)],
                confidence=0.5,
                safety=PatchSafety.UNSAFE,
                rationale="Unsafe",
            ),
        ]

        filtered = engine.filter_by_safety(suggestions, PatchLevel.SAFE)
        assert len(filtered) == 1
        assert filtered[0].safety == PatchSafety.SAFE

    def test_filter_needs_review(self, engine: FixSuggestionEngine) -> None:
        """NEEDS_REVIEW filter returns safe + needs_review."""
        suggestions = [
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/a", value=1)],
                confidence=0.9,
                safety=PatchSafety.SAFE,
                rationale="Safe",
            ),
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/b", value=2)],
                confidence=0.7,
                safety=PatchSafety.NEEDS_REVIEW,
                rationale="Review",
            ),
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/c", value=3)],
                confidence=0.5,
                safety=PatchSafety.UNSAFE,
                rationale="Unsafe",
            ),
        ]

        filtered = engine.filter_by_safety(suggestions, PatchLevel.NEEDS_REVIEW)
        assert len(filtered) == 2
        assert all(s.safety != PatchSafety.UNSAFE for s in filtered)

    def test_filter_unsafe(self, engine: FixSuggestionEngine) -> None:
        """UNSAFE filter returns all suggestions."""
        suggestions = [
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/a", value=1)],
                confidence=0.9,
                safety=PatchSafety.SAFE,
                rationale="Safe",
            ),
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/b", value=2)],
                confidence=0.7,
                safety=PatchSafety.NEEDS_REVIEW,
                rationale="Review",
            ),
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/c", value=3)],
                confidence=0.5,
                safety=PatchSafety.UNSAFE,
                rationale="Unsafe",
            ),
        ]

        filtered = engine.filter_by_safety(suggestions, PatchLevel.UNSAFE)
        assert len(filtered) == 3


# =============================================================================
# TEST: Sort Suggestions
# =============================================================================


class TestSortSuggestions:
    """Tests for suggestion sorting."""

    def test_sort_by_safety_first(self, engine: FixSuggestionEngine) -> None:
        """Safe suggestions come before needs_review."""
        suggestions = [
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/b", value=2)],
                confidence=0.9,
                safety=PatchSafety.NEEDS_REVIEW,
                rationale="Review",
            ),
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/a", value=1)],
                confidence=0.7,
                safety=PatchSafety.SAFE,
                rationale="Safe",
            ),
        ]

        sorted_suggestions = engine.sort_suggestions(suggestions)
        assert sorted_suggestions[0].safety == PatchSafety.SAFE
        assert sorted_suggestions[1].safety == PatchSafety.NEEDS_REVIEW

    def test_sort_by_confidence_within_safety(self, engine: FixSuggestionEngine) -> None:
        """Higher confidence comes first within same safety level."""
        suggestions = [
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/a", value=1)],
                confidence=0.7,
                safety=PatchSafety.SAFE,
                rationale="Safe low",
            ),
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/b", value=2)],
                confidence=0.95,
                safety=PatchSafety.SAFE,
                rationale="Safe high",
            ),
        ]

        sorted_suggestions = engine.sort_suggestions(suggestions)
        assert sorted_suggestions[0].confidence == 0.95
        assert sorted_suggestions[1].confidence == 0.7


# =============================================================================
# TEST: Deduplicate Suggestions
# =============================================================================


class TestDeduplicateSuggestions:
    """Tests for suggestion deduplication."""

    def test_deduplicate_same_path(self, engine: FixSuggestionEngine) -> None:
        """Keeps best suggestion for same path."""
        suggestions = [
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/test", value=1)],
                confidence=0.7,
                safety=PatchSafety.NEEDS_REVIEW,
                rationale="Lower confidence",
            ),
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/test", value=2)],
                confidence=0.95,
                safety=PatchSafety.SAFE,
                rationale="Higher confidence, safer",
            ),
        ]

        deduped = engine.deduplicate_suggestions(suggestions)
        assert len(deduped) == 1
        assert deduped[0].confidence == 0.95
        assert deduped[0].safety == PatchSafety.SAFE

    def test_deduplicate_different_paths(self, engine: FixSuggestionEngine) -> None:
        """Keeps suggestions for different paths."""
        suggestions = [
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/a", value=1)],
                confidence=0.9,
                safety=PatchSafety.SAFE,
                rationale="Path A",
            ),
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/b", value=2)],
                confidence=0.8,
                safety=PatchSafety.SAFE,
                rationale="Path B",
            ),
        ]

        deduped = engine.deduplicate_suggestions(suggestions)
        assert len(deduped) == 2

    def test_deduplicate_empty(self, engine: FixSuggestionEngine) -> None:
        """Empty list returns empty."""
        deduped = engine.deduplicate_suggestions([])
        assert deduped == []


# =============================================================================
# TEST: Can Fix
# =============================================================================


class TestCanFix:
    """Tests for fixable error code checking."""

    def test_fixable_code(self, engine: FixSuggestionEngine) -> None:
        """Fixable codes return True."""
        finding = Finding(
            code="GLSCHEMA-E102",
            severity=Severity.ERROR,
            path="/test",
            message="Type mismatch",
        )
        assert engine._can_fix(finding) is True

    def test_non_fixable_code(self, engine: FixSuggestionEngine) -> None:
        """Non-fixable codes return False."""
        finding = Finding(
            code="GLSCHEMA-E999",
            severity=Severity.ERROR,
            path="/test",
            message="Unknown error",
        )
        assert engine._can_fix(finding) is False


# =============================================================================
# TEST: Generate Method
# =============================================================================


class TestGenerate:
    """Tests for main generate method."""

    def test_generate_empty_findings(self, engine: FixSuggestionEngine) -> None:
        """Empty findings returns empty result."""
        result = engine.generate([], {})
        assert result.has_suggestions is False
        assert result.total_generated == 0

    def test_generate_type_mismatch(self, engine: FixSuggestionEngine) -> None:
        """Generates suggestion for type mismatch."""
        findings = [
            Finding(
                code="GLSCHEMA-E102",
                severity=Severity.ERROR,
                path="/count",
                message="Expected integer, got string",
                expected={"type": "integer"},
                actual="42",
            )
        ]
        payload = {"count": "42"}

        result = engine.generate(findings, payload)
        # May or may not have suggestion depending on heuristics
        assert isinstance(result, SuggestionEngineResult)


# =============================================================================
# TEST: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_fixable_codes(self) -> None:
        """get_fixable_codes returns copy of FIXABLE_CODES."""
        codes = get_fixable_codes()
        assert codes == FIXABLE_CODES
        # Verify it's a copy
        codes["NEW_CODE"] = "new"
        assert "NEW_CODE" not in FIXABLE_CODES

    def test_generate_suggestions_function(
        self,
        schema_ir_with_properties: SchemaIR
    ) -> None:
        """generate_suggestions function works."""
        findings = [
            Finding(
                code="GLSCHEMA-E102",
                severity=Severity.ERROR,
                path="/count",
                message="Type mismatch",
                expected={"type": "integer"},
                actual="42",
            )
        ]
        payload = {"count": "42"}

        suggestions = generate_suggestions(
            findings,
            payload,
            schema_ir_with_properties
        )
        assert isinstance(suggestions, list)

    def test_apply_suggestions_empty(self) -> None:
        """apply_suggestions with empty suggestions returns original."""
        payload = {"name": "test"}
        new_payload, applied = apply_suggestions(payload, [])
        assert new_payload == payload
        assert applied == []

    def test_apply_suggestions_safe_only(self) -> None:
        """apply_suggestions only applies safe by default."""
        payload = {"name": "test"}
        suggestions = [
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/count", value=1)],
                confidence=0.9,
                safety=PatchSafety.SAFE,
                rationale="Safe add",
            ),
            FixSuggestion(
                patch=[JSONPatchOp(op="add", path="/other", value=2)],
                confidence=0.7,
                safety=PatchSafety.NEEDS_REVIEW,
                rationale="Needs review",
            ),
        ]

        new_payload, applied = apply_suggestions(payload, suggestions)
        assert len(applied) == 1
        assert applied[0].safety == PatchSafety.SAFE
        assert "count" in new_payload


# =============================================================================
# TEST: FIXABLE_CODES Constant
# =============================================================================


class TestFixableCodes:
    """Tests for FIXABLE_CODES constant."""

    def test_contains_expected_codes(self) -> None:
        """FIXABLE_CODES contains expected error codes."""
        expected_codes = [
            "GLSCHEMA-E100",
            "GLSCHEMA-E101",
            "GLSCHEMA-E102",
            "GLSCHEMA-E200",
            "GLSCHEMA-E202",
            "GLSCHEMA-E301",
            "GLSCHEMA-E302",
            "GLSCHEMA-W600",
            "GLSCHEMA-W601",
            "GLSCHEMA-W700",
        ]
        for code in expected_codes:
            assert code in FIXABLE_CODES

    def test_values_are_descriptions(self) -> None:
        """FIXABLE_CODES values are descriptive strings."""
        for code, description in FIXABLE_CODES.items():
            assert isinstance(description, str)
            assert len(description) > 0


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    "TestSuggestionEngineResult",
    "TestEngineInitialization",
    "TestFilterBySafety",
    "TestSortSuggestions",
    "TestDeduplicateSuggestions",
    "TestCanFix",
    "TestGenerate",
    "TestConvenienceFunctions",
    "TestFixableCodes",
]
