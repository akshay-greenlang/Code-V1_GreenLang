# -*- coding: utf-8 -*-
"""
Unit tests for the Patch Safety Classifier (Task 4.2).

This module provides comprehensive tests for the PatchSafetyClassifier,
covering all safety levels and operation types.

Test Coverage:
    - PatchSafety enum
    - PatchOp enum
    - JSONPatchOperation model
    - PatchContext model
    - SafetyClassification model
    - PatchSafetyClassifier.classify()
    - _classify_add()
    - _classify_replace()
    - _classify_remove()
    - _classify_move()
    - _classify_copy()
    - _is_exact_coercion()
    - _is_schema_defined_rename()
    - Convenience functions (is_safe_patch, classify_patches, filter_safe_patches)

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 4.2 Tests
"""

import pytest
from datetime import datetime
from typing import Any, Dict, List

from greenlang.schema.suggestions.safety import (
    PatchSafety,
    PatchOp,
    JSONPatchOperation,
    PatchContext,
    SafetyClassification,
    PatchSafetyClassifier,
    is_safe_patch,
    classify_patches,
    filter_safe_patches,
    SAFE_OPERATIONS,
    NEEDS_REVIEW_OPERATIONS,
    UNSAFE_OPERATIONS,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    LARGE_UNIT_FACTOR_THRESHOLD,
)
from greenlang.schema.models.finding import Finding, Severity
from greenlang.schema.compiler.ir import SchemaIR, PropertyIR


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_schema_ir() -> SchemaIR:
    """Create a sample SchemaIR for testing."""
    return SchemaIR(
        schema_id="test/sample",
        version="1.0.0",
        schema_hash="a" * 64,
        compiled_at=datetime.now(),
        compiler_version="0.1.0",
        properties={
            "/name": PropertyIR(path="/name", type="string", required=True),
            "/age": PropertyIR(path="/age", type="integer", required=False, has_default=True, default_value=0),
            "/email": PropertyIR(path="/email", type="string", required=False),
            "/energy": PropertyIR(path="/energy", type="number", required=True),
        },
        required_paths={"/name", "/energy"},
        renamed_fields={"old_name": "name", "legacy_email": "email"},
        deprecated_fields={"/old_field": {"since_version": "1.0.0", "message": "Use new_field instead"}},
    )


@pytest.fixture
def sample_finding() -> Finding:
    """Create a sample Finding for testing."""
    return Finding(
        code="GLSCHEMA-E102",
        severity=Severity.ERROR,
        path="/energy/value",
        message="Type mismatch: expected number, got string",
    )


@pytest.fixture
def sample_add_patch() -> JSONPatchOperation:
    """Create a sample add patch for testing."""
    return JSONPatchOperation(
        op=PatchOp.ADD,
        path="/age",
        value=25,
    )


@pytest.fixture
def sample_replace_patch() -> JSONPatchOperation:
    """Create a sample replace patch for testing."""
    return JSONPatchOperation(
        op=PatchOp.REPLACE,
        path="/energy/value",
        value=100,
    )


@pytest.fixture
def sample_remove_patch() -> JSONPatchOperation:
    """Create a sample remove patch for testing."""
    return JSONPatchOperation(
        op=PatchOp.REMOVE,
        path="/unknown_field",
    )


@pytest.fixture
def sample_move_patch() -> JSONPatchOperation:
    """Create a sample move patch for testing."""
    return JSONPatchOperation(
        op=PatchOp.MOVE,
        path="/name",
        from_="/old_name",
    )


# =============================================================================
# PATCH SAFETY ENUM TESTS
# =============================================================================


class TestPatchSafety:
    """Tests for PatchSafety enum."""

    def test_safe_value(self):
        """Test SAFE enum value."""
        assert PatchSafety.SAFE.value == "safe"

    def test_needs_review_value(self):
        """Test NEEDS_REVIEW enum value."""
        assert PatchSafety.NEEDS_REVIEW.value == "needs_review"

    def test_unsafe_value(self):
        """Test UNSAFE enum value."""
        assert PatchSafety.UNSAFE.value == "unsafe"

    def test_allows_auto_apply_safe(self):
        """Test that SAFE allows auto-apply."""
        assert PatchSafety.SAFE.allows_auto_apply() is True

    def test_allows_auto_apply_needs_review(self):
        """Test that NEEDS_REVIEW does not allow auto-apply."""
        assert PatchSafety.NEEDS_REVIEW.allows_auto_apply() is False

    def test_allows_auto_apply_unsafe(self):
        """Test that UNSAFE does not allow auto-apply."""
        assert PatchSafety.UNSAFE.allows_auto_apply() is False

    def test_requires_review_safe(self):
        """Test that SAFE does not require review."""
        assert PatchSafety.SAFE.requires_review() is False

    def test_requires_review_needs_review(self):
        """Test that NEEDS_REVIEW requires review."""
        assert PatchSafety.NEEDS_REVIEW.requires_review() is True

    def test_requires_review_unsafe(self):
        """Test that UNSAFE requires review."""
        assert PatchSafety.UNSAFE.requires_review() is True

    def test_numeric_level_ordering(self):
        """Test that numeric levels are correctly ordered."""
        assert PatchSafety.SAFE.numeric_level() < PatchSafety.NEEDS_REVIEW.numeric_level()
        assert PatchSafety.NEEDS_REVIEW.numeric_level() < PatchSafety.UNSAFE.numeric_level()

    def test_from_string_valid(self):
        """Test from_string with valid values."""
        assert PatchSafety.from_string("safe") == PatchSafety.SAFE
        assert PatchSafety.from_string("SAFE") == PatchSafety.SAFE
        assert PatchSafety.from_string("needs_review") == PatchSafety.NEEDS_REVIEW
        assert PatchSafety.from_string("unsafe") == PatchSafety.UNSAFE

    def test_from_string_invalid(self):
        """Test from_string with invalid value."""
        with pytest.raises(ValueError, match="Invalid safety level"):
            PatchSafety.from_string("invalid")


# =============================================================================
# PATCH OP ENUM TESTS
# =============================================================================


class TestPatchOp:
    """Tests for PatchOp enum."""

    def test_all_operations_defined(self):
        """Test that all RFC 6902 operations are defined."""
        ops = {PatchOp.ADD, PatchOp.REMOVE, PatchOp.REPLACE, PatchOp.MOVE, PatchOp.COPY, PatchOp.TEST}
        assert len(ops) == 6

    def test_operation_values(self):
        """Test operation string values."""
        assert PatchOp.ADD.value == "add"
        assert PatchOp.REMOVE.value == "remove"
        assert PatchOp.REPLACE.value == "replace"
        assert PatchOp.MOVE.value == "move"
        assert PatchOp.COPY.value == "copy"
        assert PatchOp.TEST.value == "test"


# =============================================================================
# JSON PATCH OPERATION MODEL TESTS
# =============================================================================


class TestJSONPatchOperation:
    """Tests for JSONPatchOperation model."""

    def test_create_add_operation(self):
        """Test creating an add operation."""
        patch = JSONPatchOperation(
            op=PatchOp.ADD,
            path="/field",
            value="test",
        )
        assert patch.op == PatchOp.ADD
        assert patch.path == "/field"
        assert patch.value == "test"
        assert patch.from_ is None

    def test_create_move_operation(self):
        """Test creating a move operation."""
        patch = JSONPatchOperation(
            op=PatchOp.MOVE,
            path="/new_field",
            from_="/old_field",
        )
        assert patch.op == PatchOp.MOVE
        assert patch.path == "/new_field"
        assert patch.from_ == "/old_field"

    def test_invalid_path_no_slash(self):
        """Test that path without leading slash fails."""
        with pytest.raises(ValueError, match="Invalid JSON Pointer path"):
            JSONPatchOperation(
                op=PatchOp.ADD,
                path="field",  # Missing leading slash
                value="test",
            )

    def test_empty_path_is_valid(self):
        """Test that empty path (root) is valid."""
        patch = JSONPatchOperation(
            op=PatchOp.ADD,
            path="",
            value={"key": "value"},
        )
        assert patch.path == ""

    def test_is_additive(self):
        """Test is_additive method."""
        add_patch = JSONPatchOperation(op=PatchOp.ADD, path="/field", value="test")
        copy_patch = JSONPatchOperation(op=PatchOp.COPY, path="/dest", from_="/src")
        remove_patch = JSONPatchOperation(op=PatchOp.REMOVE, path="/field")

        assert add_patch.is_additive() is True
        assert copy_patch.is_additive() is True
        assert remove_patch.is_additive() is False

    def test_is_destructive(self):
        """Test is_destructive method."""
        remove_patch = JSONPatchOperation(op=PatchOp.REMOVE, path="/field")
        add_patch = JSONPatchOperation(op=PatchOp.ADD, path="/field", value="test")

        assert remove_patch.is_destructive() is True
        assert add_patch.is_destructive() is False

    def test_is_modification(self):
        """Test is_modification method."""
        replace_patch = JSONPatchOperation(op=PatchOp.REPLACE, path="/field", value="new")
        move_patch = JSONPatchOperation(op=PatchOp.MOVE, path="/new", from_="/old")
        add_patch = JSONPatchOperation(op=PatchOp.ADD, path="/field", value="test")

        assert replace_patch.is_modification() is True
        assert move_patch.is_modification() is True
        assert add_patch.is_modification() is False

    def test_to_dict_add(self):
        """Test to_dict for add operation."""
        patch = JSONPatchOperation(op=PatchOp.ADD, path="/field", value=42)
        result = patch.to_dict()
        assert result == {"op": "add", "path": "/field", "value": 42}

    def test_to_dict_move(self):
        """Test to_dict for move operation."""
        patch = JSONPatchOperation(op=PatchOp.MOVE, path="/new", from_="/old")
        result = patch.to_dict()
        assert result == {"op": "move", "path": "/new", "from": "/old"}


# =============================================================================
# PATCH CONTEXT MODEL TESTS
# =============================================================================


class TestPatchContext:
    """Tests for PatchContext model."""

    def test_create_context(self, sample_finding):
        """Test creating a PatchContext."""
        context = PatchContext(
            finding=sample_finding,
            original_value="100",
            suggested_value=100,
            derivation="exact_type_coercion",
            is_type_coercion=True,
        )
        assert context.finding == sample_finding
        assert context.original_value == "100"
        assert context.suggested_value == 100
        assert context.derivation == "exact_type_coercion"
        assert context.is_type_coercion is True

    def test_default_values(self, sample_finding):
        """Test default values in PatchContext."""
        context = PatchContext(
            finding=sample_finding,
            suggested_value=100,
            derivation="test",
        )
        assert context.original_value is None
        assert context.schema_has_default is False
        assert context.is_required_field is False
        assert context.is_alias_resolution is False
        assert context.is_unit_conversion is False
        assert context.is_type_coercion is False
        assert context.unit_conversion_factor is None
        assert context.edit_distance is None


# =============================================================================
# SAFETY CLASSIFICATION MODEL TESTS
# =============================================================================


class TestSafetyClassification:
    """Tests for SafetyClassification model."""

    def test_create_classification(self):
        """Test creating a SafetyClassification."""
        classification = SafetyClassification(
            safety=PatchSafety.SAFE,
            confidence=0.95,
            rationale="Exact type coercion",
            risks=[],
            requires_human_review=False,
        )
        assert classification.safety == PatchSafety.SAFE
        assert classification.confidence == 0.95
        assert classification.rationale == "Exact type coercion"
        assert classification.risks == []
        assert classification.requires_human_review is False

    def test_is_auto_applicable_safe_high_confidence(self):
        """Test is_auto_applicable for safe, high confidence."""
        classification = SafetyClassification(
            safety=PatchSafety.SAFE,
            confidence=0.95,
            rationale="Test",
            requires_human_review=False,
        )
        assert classification.is_auto_applicable() is True

    def test_is_auto_applicable_safe_low_confidence(self):
        """Test is_auto_applicable for safe, low confidence."""
        classification = SafetyClassification(
            safety=PatchSafety.SAFE,
            confidence=0.5,
            rationale="Test",
            requires_human_review=False,
        )
        assert classification.is_auto_applicable() is False

    def test_is_auto_applicable_needs_review(self):
        """Test is_auto_applicable for needs_review."""
        classification = SafetyClassification(
            safety=PatchSafety.NEEDS_REVIEW,
            confidence=0.95,
            rationale="Test",
            requires_human_review=True,
        )
        assert classification.is_auto_applicable() is False

    def test_format_summary(self):
        """Test format_summary method."""
        classification = SafetyClassification(
            safety=PatchSafety.SAFE,
            confidence=0.95,
            rationale="Test rationale",
            risks=["Risk 1", "Risk 2"],
        )
        summary = classification.format_summary()
        assert "[SAFE]" in summary
        assert "95%" in summary
        assert "Test rationale" in summary
        assert "Risk 1" in summary


# =============================================================================
# PATCH SAFETY CLASSIFIER TESTS
# =============================================================================


class TestPatchSafetyClassifier:
    """Tests for PatchSafetyClassifier class."""

    def test_init(self, sample_schema_ir):
        """Test classifier initialization."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        assert classifier.ir == sample_schema_ir

    # -------------------------------------------------------------------------
    # _classify_add tests
    # -------------------------------------------------------------------------

    def test_classify_add_alias_resolution_safe(self, sample_schema_ir, sample_finding):
        """Test that add from alias resolution is safe."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.ADD, path="/name", value="test")
        context = PatchContext(
            finding=sample_finding,
            suggested_value="test",
            derivation="alias_resolution",
            is_alias_resolution=True,
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.SAFE
        assert result.operation_category == "add_from_alias"

    def test_classify_add_optional_with_default_safe(self, sample_schema_ir, sample_finding):
        """Test that adding optional field with default is safe."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.ADD, path="/age", value=0)
        context = PatchContext(
            finding=sample_finding,
            suggested_value=0,
            derivation="schema_default",
            schema_has_default=True,
            is_required_field=False,
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.SAFE
        assert result.operation_category == "add_default_optional"

    def test_classify_add_required_with_default_needs_review(self, sample_schema_ir, sample_finding):
        """Test that adding required field with default needs review."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.ADD, path="/name", value="default_name")
        context = PatchContext(
            finding=sample_finding,
            suggested_value="default_name",
            derivation="schema_default",
            schema_has_default=True,
            is_required_field=True,
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.NEEDS_REVIEW
        assert result.requires_human_review is True

    def test_classify_add_inferred_value_needs_review(self, sample_schema_ir, sample_finding):
        """Test that adding inferred value needs review."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.ADD, path="/email", value="test@example.com")
        context = PatchContext(
            finding=sample_finding,
            suggested_value="test@example.com",
            derivation="inferred_from_context",
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.NEEDS_REVIEW
        assert result.operation_category == "add_inferred_value"

    def test_classify_add_required_no_default_unsafe(self, sample_schema_ir, sample_finding):
        """Test that adding required field without default is unsafe."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.ADD, path="/energy", value=0)
        context = PatchContext(
            finding=sample_finding,
            suggested_value=0,
            derivation="guessed",
            is_required_field=True,
            schema_has_default=False,
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.UNSAFE
        assert result.operation_category == "add_required_no_default"

    # -------------------------------------------------------------------------
    # _classify_replace tests
    # -------------------------------------------------------------------------

    def test_classify_replace_exact_coercion_safe(self, sample_schema_ir, sample_finding):
        """Test that exact type coercion is safe."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.REPLACE, path="/energy/value", value=100)
        context = PatchContext(
            finding=sample_finding,
            original_value="100",
            suggested_value=100,
            derivation="type_coercion",
            is_type_coercion=True,
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.SAFE
        assert result.operation_category == "replace_exact_coercion"

    def test_classify_replace_lossy_coercion_needs_review(self, sample_schema_ir, sample_finding):
        """Test that lossy type coercion needs review."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.REPLACE, path="/energy/value", value=100)
        context = PatchContext(
            finding=sample_finding,
            original_value="100.5",  # Will lose .5
            suggested_value=100,
            derivation="type_coercion",
            is_type_coercion=True,
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.NEEDS_REVIEW
        assert result.operation_category == "replace_lossy_coercion"

    def test_classify_replace_unit_conversion_safe(self, sample_schema_ir, sample_finding):
        """Test that small unit conversion is safe."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.REPLACE, path="/energy/value", value=1.0)
        context = PatchContext(
            finding=sample_finding,
            original_value=1000,  # Wh
            suggested_value=1.0,  # kWh
            derivation="unit_conversion",
            is_unit_conversion=True,
            unit_conversion_factor=0.001,  # Wh to kWh
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.SAFE
        assert result.operation_category == "replace_unit_same_dimension"

    def test_classify_replace_large_unit_factor_needs_review(self, sample_schema_ir, sample_finding):
        """Test that large unit conversion factor needs review."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.REPLACE, path="/energy/value", value=1000000)
        context = PatchContext(
            finding=sample_finding,
            original_value=1,
            suggested_value=1000000,
            derivation="unit_conversion",
            is_unit_conversion=True,
            unit_conversion_factor=1000000,  # Very large factor
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.NEEDS_REVIEW
        assert result.operation_category == "replace_large_unit_factor"

    def test_classify_replace_speculative_unsafe(self, sample_schema_ir, sample_finding):
        """Test that speculative replacement is unsafe."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.REPLACE, path="/energy/value", value=100)
        context = PatchContext(
            finding=sample_finding,
            original_value=50,
            suggested_value=100,
            derivation="speculative_guess",
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.UNSAFE
        assert result.operation_category == "replace_speculative"

    # -------------------------------------------------------------------------
    # _classify_remove tests
    # -------------------------------------------------------------------------

    def test_classify_remove_empty_optional_safe(self, sample_schema_ir, sample_finding):
        """Test that removing empty optional field is safe."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.REMOVE, path="/email")
        context = PatchContext(
            finding=sample_finding,
            original_value=None,
            suggested_value=None,
            derivation="remove_empty",
            is_required_field=False,
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.SAFE
        assert result.operation_category == "remove_empty_field"

    def test_classify_remove_deprecated_needs_review(self, sample_schema_ir, sample_finding):
        """Test that removing deprecated field needs review."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.REMOVE, path="/old_field")
        context = PatchContext(
            finding=sample_finding,
            original_value="old_value",
            suggested_value=None,
            derivation="remove_deprecated",
            is_required_field=False,
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.NEEDS_REVIEW
        assert result.operation_category == "remove_deprecated_field"

    def test_classify_remove_unknown_with_data_unsafe(self, sample_schema_ir, sample_finding):
        """Test that removing unknown field with data is unsafe."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.REMOVE, path="/unknown_field")
        context = PatchContext(
            finding=sample_finding,
            original_value="important_data",
            suggested_value=None,
            derivation="remove_unknown_field",
            is_required_field=False,
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.UNSAFE
        assert result.operation_category == "remove_data_loss"

    def test_classify_remove_data_unsafe(self, sample_schema_ir, sample_finding):
        """Test that removing any field with data is unsafe."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.REMOVE, path="/name")
        context = PatchContext(
            finding=sample_finding,
            original_value="John",
            suggested_value=None,
            derivation="arbitrary_removal",
            is_required_field=True,
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.UNSAFE
        assert result.operation_category == "remove_data_loss"

    # -------------------------------------------------------------------------
    # _classify_move tests
    # -------------------------------------------------------------------------

    def test_classify_move_schema_defined_rename_safe(self, sample_schema_ir, sample_finding):
        """Test that schema-defined rename is safe."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.MOVE, path="/name", from_="/old_name")
        context = PatchContext(
            finding=sample_finding,
            original_value="John",
            suggested_value="John",
            derivation="schema_rename",
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.SAFE
        assert result.operation_category == "move_declared_rename"

    def test_classify_move_typo_correction_needs_review(self, sample_schema_ir, sample_finding):
        """Test that typo correction needs review."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.MOVE, path="/name", from_="/nane")  # typo
        context = PatchContext(
            finding=sample_finding,
            original_value="John",
            suggested_value="John",
            derivation="typo_correction",
            edit_distance=1,
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.NEEDS_REVIEW
        assert result.operation_category == "move_typo_correction"

    def test_classify_move_alias_resolution_safe(self, sample_schema_ir, sample_finding):
        """Test that alias resolution move is safe."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.MOVE, path="/email", from_="/legacy_email")
        context = PatchContext(
            finding=sample_finding,
            original_value="test@example.com",
            suggested_value="test@example.com",
            derivation="alias_resolution",
            is_alias_resolution=True,
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.SAFE

    def test_classify_move_arbitrary_rename_unsafe(self, sample_schema_ir, sample_finding):
        """Test that arbitrary rename is unsafe."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.MOVE, path="/new_field", from_="/some_field")
        context = PatchContext(
            finding=sample_finding,
            original_value="data",
            suggested_value="data",
            derivation="arbitrary_rename",
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.UNSAFE
        assert result.operation_category == "move_arbitrary_rename"

    def test_classify_move_missing_from_unsafe(self, sample_schema_ir, sample_finding):
        """Test that move without from is unsafe."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        # Note: We need to bypass the validation for this test
        patch = JSONPatchOperation(op=PatchOp.MOVE, path="/new_field")
        patch.from_ = None  # Manually set to None
        context = PatchContext(
            finding=sample_finding,
            original_value="data",
            suggested_value="data",
            derivation="move",
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.UNSAFE

    # -------------------------------------------------------------------------
    # _classify_copy tests
    # -------------------------------------------------------------------------

    def test_classify_copy_safe(self, sample_schema_ir, sample_finding):
        """Test that copy to new location is safe."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.COPY, path="/new_field", from_="/name")
        context = PatchContext(
            finding=sample_finding,
            original_value="John",
            suggested_value="John",
            derivation="copy",
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.SAFE

    def test_classify_copy_to_existing_needs_review(self, sample_schema_ir, sample_finding):
        """Test that copy to existing location needs review."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.COPY, path="/name", from_="/email")  # /name exists
        context = PatchContext(
            finding=sample_finding,
            original_value="test@example.com",
            suggested_value="test@example.com",
            derivation="copy",
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.NEEDS_REVIEW

    # -------------------------------------------------------------------------
    # _classify test operation
    # -------------------------------------------------------------------------

    def test_classify_test_always_safe(self, sample_schema_ir, sample_finding):
        """Test that test operations are always safe."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.TEST, path="/name", value="expected")
        context = PatchContext(
            finding=sample_finding,
            original_value="expected",
            suggested_value="expected",
            derivation="test_precondition",
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.SAFE
        assert result.confidence == 1.0

    # -------------------------------------------------------------------------
    # _is_exact_coercion tests
    # -------------------------------------------------------------------------

    def test_is_exact_coercion_string_to_int(self, sample_schema_ir):
        """Test exact coercion from string to int."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        assert classifier._is_exact_coercion("42", 42) is True
        assert classifier._is_exact_coercion("42", 43) is False

    def test_is_exact_coercion_string_to_float(self, sample_schema_ir):
        """Test exact coercion from string to float."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        assert classifier._is_exact_coercion("3.14", 3.14) is True
        assert classifier._is_exact_coercion("3.14", 3.15) is False

    def test_is_exact_coercion_string_to_bool(self, sample_schema_ir):
        """Test exact coercion from string to bool."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        assert classifier._is_exact_coercion("true", True) is True
        assert classifier._is_exact_coercion("false", False) is True
        assert classifier._is_exact_coercion("TRUE", True) is True
        assert classifier._is_exact_coercion("yes", True) is False  # Not standard

    def test_is_exact_coercion_none_values(self, sample_schema_ir):
        """Test that None values return False."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        assert classifier._is_exact_coercion(None, 42) is False
        assert classifier._is_exact_coercion("42", None) is False

    def test_is_exact_coercion_same_type(self, sample_schema_ir):
        """Test same type comparison."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        assert classifier._is_exact_coercion(42, 42) is True
        assert classifier._is_exact_coercion(42, 43) is False
        assert classifier._is_exact_coercion("test", "test") is True

    # -------------------------------------------------------------------------
    # _is_schema_defined_rename tests
    # -------------------------------------------------------------------------

    def test_is_schema_defined_rename_true(self, sample_schema_ir, sample_finding):
        """Test detecting schema-defined rename."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        context = PatchContext(
            finding=sample_finding,
            suggested_value="value",
            derivation="test",
        )

        assert classifier._is_schema_defined_rename("/old_name", "/name", context) is True

    def test_is_schema_defined_rename_false(self, sample_schema_ir, sample_finding):
        """Test non-schema-defined rename."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        context = PatchContext(
            finding=sample_finding,
            suggested_value="value",
            derivation="test",
        )

        assert classifier._is_schema_defined_rename("/random", "/name", context) is False


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_is_safe_patch(self, sample_schema_ir, sample_finding):
        """Test is_safe_patch function."""
        patch = JSONPatchOperation(op=PatchOp.ADD, path="/age", value=0)
        context = PatchContext(
            finding=sample_finding,
            suggested_value=0,
            derivation="schema_default",
            schema_has_default=True,
            is_required_field=False,
        )

        assert is_safe_patch(patch, context, sample_schema_ir) is True

    def test_is_safe_patch_false(self, sample_schema_ir, sample_finding):
        """Test is_safe_patch returns False for unsafe patch."""
        patch = JSONPatchOperation(op=PatchOp.REMOVE, path="/name")
        context = PatchContext(
            finding=sample_finding,
            original_value="John",
            suggested_value=None,
            derivation="arbitrary",
        )

        assert is_safe_patch(patch, context, sample_schema_ir) is False

    def test_classify_patches(self, sample_schema_ir, sample_finding):
        """Test classify_patches function."""
        patches = [
            JSONPatchOperation(op=PatchOp.ADD, path="/age", value=0),
            JSONPatchOperation(op=PatchOp.REMOVE, path="/name"),
        ]
        contexts = [
            PatchContext(
                finding=sample_finding,
                suggested_value=0,
                derivation="schema_default",
                schema_has_default=True,
                is_required_field=False,
            ),
            PatchContext(
                finding=sample_finding,
                original_value="John",
                suggested_value=None,
                derivation="arbitrary",
            ),
        ]

        results = classify_patches(patches, contexts, sample_schema_ir)

        assert len(results) == 2
        assert results[0].safety == PatchSafety.SAFE
        assert results[1].safety == PatchSafety.UNSAFE

    def test_classify_patches_mismatched_lengths(self, sample_schema_ir, sample_finding):
        """Test classify_patches raises error for mismatched lengths."""
        patches = [JSONPatchOperation(op=PatchOp.ADD, path="/age", value=0)]
        contexts = []

        with pytest.raises(ValueError, match="must have the same length"):
            classify_patches(patches, contexts, sample_schema_ir)

    def test_filter_safe_patches(self, sample_schema_ir, sample_finding):
        """Test filter_safe_patches function."""
        patches = [
            JSONPatchOperation(op=PatchOp.ADD, path="/age", value=0),
            JSONPatchOperation(op=PatchOp.REMOVE, path="/name"),
        ]
        contexts = [
            PatchContext(
                finding=sample_finding,
                suggested_value=0,
                derivation="schema_default",
                schema_has_default=True,
                is_required_field=False,
            ),
            PatchContext(
                finding=sample_finding,
                original_value="John",
                suggested_value=None,
                derivation="arbitrary",
            ),
        ]

        results = filter_safe_patches(patches, contexts, sample_schema_ir)

        assert len(results) == 1
        assert results[0][0].op == PatchOp.ADD


# =============================================================================
# CONSTANTS TESTS
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_safe_operations_defined(self):
        """Test that SAFE_OPERATIONS has expected keys."""
        expected_keys = {
            "add_default_optional",
            "add_from_alias",
            "replace_exact_coercion",
            "replace_unit_same_dimension",
            "move_declared_rename",
            "remove_empty_field",
        }
        assert set(SAFE_OPERATIONS.keys()) == expected_keys

    def test_needs_review_operations_defined(self):
        """Test that NEEDS_REVIEW_OPERATIONS has expected keys."""
        expected_keys = {
            "replace_lossy_coercion",
            "replace_large_unit_factor",
            "move_typo_correction",
            "add_inferred_value",
            "remove_deprecated_field",
        }
        assert set(NEEDS_REVIEW_OPERATIONS.keys()) == expected_keys

    def test_unsafe_operations_defined(self):
        """Test that UNSAFE_OPERATIONS has expected keys."""
        expected_keys = {
            "add_required_no_default",
            "remove_unknown_field",
            "replace_speculative",
            "remove_data_loss",
            "move_arbitrary_rename",
        }
        assert set(UNSAFE_OPERATIONS.keys()) == expected_keys

    def test_confidence_thresholds(self):
        """Test confidence threshold values."""
        assert HIGH_CONFIDENCE_THRESHOLD == 0.9
        assert MEDIUM_CONFIDENCE_THRESHOLD == 0.7
        assert LOW_CONFIDENCE_THRESHOLD == 0.5

    def test_unit_factor_threshold(self):
        """Test unit factor threshold value."""
        assert LARGE_UNIT_FACTOR_THRESHOLD == 1000.0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_path(self, sample_schema_ir, sample_finding):
        """Test classification with empty path (root)."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.REPLACE, path="", value={"key": "value"})
        context = PatchContext(
            finding=sample_finding,
            original_value={},
            suggested_value={"key": "value"},
            derivation="replace_root",
        )

        result = classifier.classify(patch, context)
        # Should classify as needs_review for general replace
        assert result.safety in (PatchSafety.NEEDS_REVIEW, PatchSafety.UNSAFE)

    def test_deep_nested_path(self, sample_schema_ir, sample_finding):
        """Test classification with deeply nested path."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(
            op=PatchOp.ADD,
            path="/level1/level2/level3/level4/field",
            value="test",
        )
        context = PatchContext(
            finding=sample_finding,
            suggested_value="test",
            derivation="deep_add",
        )

        result = classifier.classify(patch, context)
        assert result is not None  # Should not error

    def test_large_edit_distance_reduces_confidence(self, sample_schema_ir, sample_finding):
        """Test that large edit distance reduces confidence."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.MOVE, path="/name", from_="/aaaa")  # large distance
        context = PatchContext(
            finding=sample_finding,
            original_value="John",
            suggested_value="John",
            derivation="typo_correction",
            edit_distance=3,  # Large edit distance
        )

        result = classifier.classify(patch, context)
        # Large edit distance should not be treated as typo
        assert result.safety == PatchSafety.UNSAFE

    def test_null_value_handling(self, sample_schema_ir, sample_finding):
        """Test handling of null values in context."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        patch = JSONPatchOperation(op=PatchOp.ADD, path="/field", value=None)
        context = PatchContext(
            finding=sample_finding,
            original_value=None,
            suggested_value=None,
            derivation="add_null",
            schema_has_default=True,
            is_required_field=False,
        )

        result = classifier.classify(patch, context)
        assert result.safety == PatchSafety.SAFE

    def test_float_precision_coercion(self, sample_schema_ir):
        """Test float precision in exact coercion check."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        # Should be considered exact despite float representation
        assert classifier._is_exact_coercion("0.1", 0.1) is True
        # Large floats
        assert classifier._is_exact_coercion("1000000.0", 1000000.0) is True

    def test_scientific_notation_coercion(self, sample_schema_ir):
        """Test scientific notation in coercion check."""
        classifier = PatchSafetyClassifier(sample_schema_ir)
        assert classifier._is_exact_coercion("1e10", 1e10) is True
        assert classifier._is_exact_coercion("1.5e-5", 1.5e-5) is True
