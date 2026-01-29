# -*- coding: utf-8 -*-
"""
Property-Based Tests: Patch Monotonicity

Tests the property that applying safe patches reduces (or maintains) the
number of validation errors. Safe patches should never make things worse.

Monotonicity Property:
    errors(apply_patches(payload, safe_patches)) <= errors(payload)

This property is essential for:
    - Automated fix suggestions: Safe patches are safe to auto-apply
    - Progressive correction: Each patch moves toward valid state
    - Trust in the system: Patches don't introduce new problems

Uses Hypothesis to generate random payloads with errors and verify that
applying suggested patches reduces the error count.

GL-FOUND-X-002: Schema Compiler & Validator - Property Tests
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pytest
from hypothesis import given, settings, assume, HealthCheck, Phase
from hypothesis import strategies as st

# Import components under test
from greenlang.schema.models.patch import JSONPatchOp, PatchSafety, FixSuggestion
from greenlang.schema.models.finding import Finding, Severity, FindingHint
from greenlang.schema.models.config import ValidationOptions, ValidationProfile
from greenlang.schema.compiler.ir import (
    SchemaIR, PropertyIR, NumericConstraintIR, StringConstraintIR,
    ArrayConstraintIR, CompiledPattern
)


# =============================================================================
# PATCH APPLICATION UTILITIES
# =============================================================================

def apply_json_patch(payload: Dict[str, Any], patch_op: JSONPatchOp) -> Dict[str, Any]:
    """
    Apply a single JSON Patch operation to a payload.

    Implements RFC 6902 JSON Patch operations:
    - add: Add a value at the target path
    - remove: Remove the value at the target path
    - replace: Replace the value at the target path
    - move: Move a value from one path to another
    - copy: Copy a value from one path to another
    - test: Verify a value exists at the path

    Args:
        payload: The payload to modify
        patch_op: The patch operation to apply

    Returns:
        Modified payload (deep copy)

    Raises:
        ValueError: If operation fails
    """
    result = copy.deepcopy(payload)

    path_parts = patch_op.path.strip('/').split('/') if patch_op.path else []

    if patch_op.op == "add":
        _set_at_path(result, path_parts, patch_op.value)
    elif patch_op.op == "remove":
        _remove_at_path(result, path_parts)
    elif patch_op.op == "replace":
        _set_at_path(result, path_parts, patch_op.value)
    elif patch_op.op == "move":
        from_parts = patch_op.from_.strip('/').split('/') if patch_op.from_ else []
        value = _get_at_path(result, from_parts)
        _remove_at_path(result, from_parts)
        _set_at_path(result, path_parts, value)
    elif patch_op.op == "copy":
        from_parts = patch_op.from_.strip('/').split('/') if patch_op.from_ else []
        value = _get_at_path(result, from_parts)
        _set_at_path(result, path_parts, copy.deepcopy(value))
    elif patch_op.op == "test":
        actual = _get_at_path(result, path_parts)
        if actual != patch_op.value:
            raise ValueError(f"Test failed: expected {patch_op.value}, got {actual}")

    return result


def _get_at_path(obj: Any, path_parts: List[str]) -> Any:
    """Get value at JSON Pointer path."""
    current = obj
    for part in path_parts:
        if not part:
            continue
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise ValueError(f"Cannot traverse path at {part}")
    return current


def _set_at_path(obj: Dict[str, Any], path_parts: List[str], value: Any) -> None:
    """Set value at JSON Pointer path."""
    if not path_parts or (len(path_parts) == 1 and not path_parts[0]):
        return

    current = obj
    for part in path_parts[:-1]:
        if not part:
            continue
        if isinstance(current, dict):
            if part not in current:
                current[part] = {}
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]

    final_key = path_parts[-1]
    if isinstance(current, dict):
        current[final_key] = value
    elif isinstance(current, list):
        idx = int(final_key)
        if idx == len(current):
            current.append(value)
        else:
            current[idx] = value


def _remove_at_path(obj: Dict[str, Any], path_parts: List[str]) -> None:
    """Remove value at JSON Pointer path."""
    if not path_parts:
        return

    current = obj
    for part in path_parts[:-1]:
        if not part:
            continue
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]

    final_key = path_parts[-1]
    if isinstance(current, dict) and final_key in current:
        del current[final_key]
    elif isinstance(current, list):
        del current[int(final_key)]


def apply_fix_suggestion(
    payload: Dict[str, Any],
    suggestion: FixSuggestion
) -> Dict[str, Any]:
    """
    Apply a complete fix suggestion to a payload.

    First applies preconditions (test operations), then applies
    the actual patch operations.

    Args:
        payload: The payload to fix
        suggestion: The fix suggestion to apply

    Returns:
        Modified payload

    Raises:
        ValueError: If preconditions fail or patch fails
    """
    result = copy.deepcopy(payload)

    # Apply preconditions first
    for op in suggestion.preconditions:
        result = apply_json_patch(result, op)

    # Apply patch operations
    for op in suggestion.patch:
        result = apply_json_patch(result, op)

    return result


# =============================================================================
# MOCK VALIDATION FUNCTION
# =============================================================================

def count_validation_errors(
    payload: Dict[str, Any],
    schema_ir: SchemaIR
) -> int:
    """
    Count validation errors in a payload against a schema IR.

    This is a simplified validation that checks:
    - Required fields present
    - Type constraints
    - Numeric constraints
    - String constraints

    Args:
        payload: The payload to validate
        schema_ir: The schema IR to validate against

    Returns:
        Number of validation errors
    """
    errors = 0

    # Check required fields
    for path in schema_ir.required_paths:
        key = path.strip('/').split('/')[0] if path else None
        if key and key not in payload:
            errors += 1

    # Check type constraints
    for path, prop_ir in schema_ir.properties.items():
        key = path.strip('/').split('/')[0] if path else None
        if not key or key not in payload:
            continue

        value = payload[key]

        if prop_ir.type:
            if not _check_type(value, prop_ir.type):
                errors += 1

    # Check numeric constraints
    for path, constraint in schema_ir.numeric_constraints.items():
        key = path.strip('/').split('/')[0] if path else None
        if not key or key not in payload:
            continue

        value = payload[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if constraint.minimum is not None and value < constraint.minimum:
                errors += 1
            if constraint.maximum is not None and value > constraint.maximum:
                errors += 1
            if constraint.exclusive_minimum is not None and value <= constraint.exclusive_minimum:
                errors += 1
            if constraint.exclusive_maximum is not None and value >= constraint.exclusive_maximum:
                errors += 1

    # Check string constraints
    for path, constraint in schema_ir.string_constraints.items():
        key = path.strip('/').split('/')[0] if path else None
        if not key or key not in payload:
            continue

        value = payload[key]
        if isinstance(value, str):
            if constraint.min_length is not None and len(value) < constraint.min_length:
                errors += 1
            if constraint.max_length is not None and len(value) > constraint.max_length:
                errors += 1

    # Check enum constraints
    for path, enum_values in schema_ir.enums.items():
        key = path.strip('/').split('/')[0] if path else None
        if not key or key not in payload:
            continue

        value = payload[key]
        if value not in enum_values:
            errors += 1

    return errors


def _check_type(value: Any, expected_type: str) -> bool:
    """Check if value matches expected JSON Schema type."""
    if expected_type == "string":
        return isinstance(value, str)
    elif expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    elif expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    elif expected_type == "boolean":
        return isinstance(value, bool)
    elif expected_type == "null":
        return value is None
    elif expected_type == "object":
        return isinstance(value, dict)
    elif expected_type == "array":
        return isinstance(value, list)
    return True


# =============================================================================
# HYPOTHESIS STRATEGIES
# =============================================================================

# Strategy for patch operations
patch_operations = st.sampled_from(["add", "remove", "replace"])

# Strategy for JSON paths
json_paths = st.one_of(
    st.just("/name"),
    st.just("/value"),
    st.just("/count"),
    st.just("/status"),
    st.just("/items"),
    st.text(
        alphabet='abcdefghijklmnopqrstuvwxyz_',
        min_size=1,
        max_size=20
    ).map(lambda x: f"/{x}")
)

# Strategy for patch values
patch_values = st.one_of(
    st.integers(min_value=0, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1000),
    st.text(min_size=1, max_size=50),
    st.booleans(),
)

# Strategy for safe patches
safe_patch_ops = st.builds(
    JSONPatchOp,
    op=st.sampled_from(["add", "replace"]),
    path=json_paths,
    value=patch_values,
)

# Strategy for fix suggestions
safe_fix_suggestions = st.builds(
    FixSuggestion,
    patch=st.lists(safe_patch_ops, min_size=1, max_size=3),
    preconditions=st.just([]),
    confidence=st.floats(min_value=0.9, max_value=1.0),
    safety=st.just(PatchSafety.SAFE),
    rationale=st.just("Automatically generated safe fix"),
)

# Strategy for payloads with missing required fields
payloads_with_errors = st.fixed_dictionaries({
    # Missing 'name' which is required
    'value': st.integers(min_value=-100, max_value=100),  # May violate min constraint
    'status': st.sampled_from(['invalid_status', 'pending', 'completed']),  # May violate enum
})


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def schema_ir_with_constraints() -> SchemaIR:
    """Create a schema IR with various constraints for testing patches."""
    return SchemaIR(
        schema_id="test/with_constraints",
        version="1.0.0",
        schema_hash="c" * 64,
        compiled_at=datetime.now(),
        compiler_version="0.1.0",
        properties={
            "/name": PropertyIR(path="/name", type="string", required=True),
            "/value": PropertyIR(path="/value", type="number", required=False),
            "/count": PropertyIR(path="/count", type="integer", required=False),
            "/status": PropertyIR(path="/status", type="string", required=False),
        },
        required_paths={"/name"},
        numeric_constraints={
            "/value": NumericConstraintIR(
                path="/value",
                minimum=0,
                maximum=1000,
            ),
            "/count": NumericConstraintIR(
                path="/count",
                minimum=1,
                maximum=100,
            ),
        },
        string_constraints={
            "/name": StringConstraintIR(
                path="/name",
                min_length=1,
                max_length=100,
            ),
        },
        enums={
            "/status": ["pending", "active", "completed", "cancelled"],
        },
    )


# =============================================================================
# MONOTONICITY TESTS
# =============================================================================

@pytest.mark.property
class TestPatchMonotonicity:
    """
    Property-based tests for patch monotonicity.

    The key property: applying safe patches reduces or maintains error count.
    """

    def test_adding_missing_required_field_reduces_errors(
        self,
        schema_ir_with_constraints,
    ):
        """
        Test that adding a missing required field reduces error count.

        This is the most common type of safe patch - providing missing data.
        """
        # Payload missing required 'name' field
        payload = {"value": 50}

        errors_before = count_validation_errors(payload, schema_ir_with_constraints)
        assert errors_before > 0, "Expected at least one error for missing required field"

        # Apply patch to add required field
        patch = JSONPatchOp(
            op="add",
            path="/name",
            value="test_name"
        )
        patched_payload = apply_json_patch(payload, patch)

        errors_after = count_validation_errors(patched_payload, schema_ir_with_constraints)

        assert errors_after < errors_before, (
            f"Expected errors to decrease after adding required field.\n"
            f"Before: {errors_before}, After: {errors_after}\n"
            f"Payload: {payload} -> {patched_payload}"
        )

    def test_fixing_enum_value_reduces_errors(
        self,
        schema_ir_with_constraints,
    ):
        """
        Test that replacing invalid enum value with valid one reduces errors.
        """
        # Payload with invalid enum value
        payload = {"name": "test", "status": "invalid_status"}

        errors_before = count_validation_errors(payload, schema_ir_with_constraints)
        assert errors_before > 0, "Expected error for invalid enum value"

        # Apply patch to fix enum value
        patch = JSONPatchOp(
            op="replace",
            path="/status",
            value="active"
        )
        patched_payload = apply_json_patch(payload, patch)

        errors_after = count_validation_errors(patched_payload, schema_ir_with_constraints)

        assert errors_after < errors_before, (
            f"Expected errors to decrease after fixing enum value.\n"
            f"Before: {errors_before}, After: {errors_after}"
        )

    def test_fixing_numeric_constraint_reduces_errors(
        self,
        schema_ir_with_constraints,
    ):
        """
        Test that fixing out-of-range numeric value reduces errors.
        """
        # Payload with value below minimum
        payload = {"name": "test", "value": -50}

        errors_before = count_validation_errors(payload, schema_ir_with_constraints)
        assert errors_before > 0, "Expected error for value below minimum"

        # Apply patch to fix numeric value
        patch = JSONPatchOp(
            op="replace",
            path="/value",
            value=50
        )
        patched_payload = apply_json_patch(payload, patch)

        errors_after = count_validation_errors(patched_payload, schema_ir_with_constraints)

        assert errors_after < errors_before, (
            f"Expected errors to decrease after fixing numeric value.\n"
            f"Before: {errors_before}, After: {errors_after}"
        )

    @given(
        original_value=st.integers(min_value=-1000, max_value=-1),  # Below minimum
        fixed_value=st.integers(min_value=0, max_value=1000),  # Within range
    )
    @settings(max_examples=100, deadline=None)
    def test_monotonic_numeric_fix_property(
        self,
        original_value: int,
        fixed_value: int,
        schema_ir_with_constraints,
    ):
        """
        Property test: fixing numeric constraint violations is monotonic.

        For any out-of-range value replaced with in-range value,
        error count should decrease.
        """
        payload = {"name": "test", "value": original_value}

        errors_before = count_validation_errors(payload, schema_ir_with_constraints)

        patch = JSONPatchOp(
            op="replace",
            path="/value",
            value=fixed_value
        )
        patched_payload = apply_json_patch(payload, patch)

        errors_after = count_validation_errors(patched_payload, schema_ir_with_constraints)

        assert errors_after <= errors_before, (
            f"Error count should not increase after fixing numeric value.\n"
            f"Original: {original_value}, Fixed: {fixed_value}\n"
            f"Errors before: {errors_before}, after: {errors_after}"
        )

    @given(
        name=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('L', 'N'),
        ))
    )
    @settings(max_examples=100, deadline=None)
    def test_adding_required_field_property(
        self,
        name: str,
        schema_ir_with_constraints,
    ):
        """
        Property test: adding missing required field reduces errors.
        """
        # Payload missing required field
        payload = {"value": 50}

        errors_before = count_validation_errors(payload, schema_ir_with_constraints)

        patch = JSONPatchOp(
            op="add",
            path="/name",
            value=name
        )
        patched_payload = apply_json_patch(payload, patch)

        errors_after = count_validation_errors(patched_payload, schema_ir_with_constraints)

        assert errors_after <= errors_before, (
            f"Adding required field should not increase errors.\n"
            f"Errors before: {errors_before}, after: {errors_after}"
        )


# =============================================================================
# FIX SUGGESTION TESTS
# =============================================================================

@pytest.mark.property
class TestFixSuggestionMonotonicity:
    """
    Test that applying FixSuggestion objects maintains monotonicity.
    """

    def test_safe_fix_suggestion_reduces_errors(
        self,
        schema_ir_with_constraints,
    ):
        """
        Test that safe fix suggestions reduce error count.
        """
        # Payload with multiple errors
        payload = {"value": -50, "status": "invalid"}

        errors_before = count_validation_errors(payload, schema_ir_with_constraints)

        # Create fix suggestion
        suggestion = FixSuggestion(
            patch=[
                JSONPatchOp(op="add", path="/name", value="fixed_name"),
                JSONPatchOp(op="replace", path="/value", value=50),
                JSONPatchOp(op="replace", path="/status", value="active"),
            ],
            preconditions=[],
            confidence=0.95,
            safety=PatchSafety.SAFE,
            rationale="Fix missing required field and constraint violations",
        )

        patched_payload = apply_fix_suggestion(payload, suggestion)
        errors_after = count_validation_errors(patched_payload, schema_ir_with_constraints)

        assert errors_after <= errors_before, (
            f"Safe fix suggestion should not increase errors.\n"
            f"Errors before: {errors_before}, after: {errors_after}"
        )

    def test_high_confidence_fix_is_effective(
        self,
        schema_ir_with_constraints,
    ):
        """
        Test that high-confidence fixes actually reduce errors.
        """
        payload = {}  # Missing all required fields

        errors_before = count_validation_errors(payload, schema_ir_with_constraints)
        assert errors_before > 0

        suggestion = FixSuggestion(
            patch=[
                JSONPatchOp(op="add", path="/name", value="test_name"),
            ],
            preconditions=[],
            confidence=0.99,
            safety=PatchSafety.SAFE,
            rationale="Add missing required field 'name'",
        )

        patched_payload = apply_fix_suggestion(payload, suggestion)
        errors_after = count_validation_errors(patched_payload, schema_ir_with_constraints)

        assert errors_after < errors_before, (
            "High-confidence fix should reduce errors"
        )

    @given(
        enum_value=st.sampled_from(["pending", "active", "completed", "cancelled"])
    )
    @settings(max_examples=50, deadline=None)
    def test_enum_fix_suggestion_property(
        self,
        enum_value: str,
        schema_ir_with_constraints,
    ):
        """
        Property test: fixing invalid enum with valid value is monotonic.
        """
        payload = {"name": "test", "status": "definitely_invalid_value"}

        errors_before = count_validation_errors(payload, schema_ir_with_constraints)

        suggestion = FixSuggestion(
            patch=[
                JSONPatchOp(op="replace", path="/status", value=enum_value),
            ],
            preconditions=[],
            confidence=0.95,
            safety=PatchSafety.SAFE,
            rationale=f"Replace invalid enum with '{enum_value}'",
        )

        patched_payload = apply_fix_suggestion(payload, suggestion)
        errors_after = count_validation_errors(patched_payload, schema_ir_with_constraints)

        assert errors_after <= errors_before


# =============================================================================
# CUMULATIVE PATCH TESTS
# =============================================================================

@pytest.mark.property
class TestCumulativePatchMonotonicity:
    """
    Test that applying multiple patches cumulatively maintains monotonicity.
    """

    def test_multiple_patches_monotonic(
        self,
        schema_ir_with_constraints,
    ):
        """
        Test that applying multiple patches one by one is monotonically
        non-increasing in error count.
        """
        # Start with maximum errors
        payload = {"value": -1000}  # Missing required, below minimum

        patches = [
            JSONPatchOp(op="add", path="/name", value="test"),
            JSONPatchOp(op="replace", path="/value", value=500),
        ]

        current_payload = copy.deepcopy(payload)
        previous_errors = count_validation_errors(current_payload, schema_ir_with_constraints)

        for i, patch in enumerate(patches):
            current_payload = apply_json_patch(current_payload, patch)
            current_errors = count_validation_errors(current_payload, schema_ir_with_constraints)

            assert current_errors <= previous_errors, (
                f"Error count increased after patch {i+1}.\n"
                f"Before: {previous_errors}, After: {current_errors}\n"
                f"Patch: {patch}"
            )

            previous_errors = current_errors

    @given(
        name=st.text(min_size=1, max_size=30, alphabet='abcdefghijklmnopqrstuvwxyz'),
        value=st.integers(min_value=0, max_value=1000),
        count=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100, deadline=None)
    def test_cumulative_fix_property(
        self,
        name: str,
        value: int,
        count: int,
        schema_ir_with_constraints,
    ):
        """
        Property test: cumulative fixes are monotonically non-increasing.
        """
        # Start with all violations
        payload = {
            "value": -1000,  # Below minimum
            "count": 1000,   # Above maximum
        }

        # Apply fixes one by one
        fixes = [
            JSONPatchOp(op="add", path="/name", value=name),
            JSONPatchOp(op="replace", path="/value", value=value),
            JSONPatchOp(op="replace", path="/count", value=count),
        ]

        current_payload = copy.deepcopy(payload)
        previous_errors = count_validation_errors(current_payload, schema_ir_with_constraints)

        for fix in fixes:
            current_payload = apply_json_patch(current_payload, fix)
            current_errors = count_validation_errors(current_payload, schema_ir_with_constraints)
            assert current_errors <= previous_errors
            previous_errors = current_errors


# =============================================================================
# SAFETY LEVEL TESTS
# =============================================================================

@pytest.mark.property
class TestPatchSafetyLevels:
    """
    Test that safety level classification correlates with monotonicity.
    """

    def test_safe_patch_never_increases_errors(
        self,
        schema_ir_with_constraints,
    ):
        """
        Test that patches classified as SAFE never increase error count.
        """
        payload = {"value": 50}  # Missing required name

        # This is a safe patch: adding missing required field
        safe_patch = JSONPatchOp(
            op="add",
            path="/name",
            value="safe_name"
        )

        errors_before = count_validation_errors(payload, schema_ir_with_constraints)
        patched = apply_json_patch(payload, safe_patch)
        errors_after = count_validation_errors(patched, schema_ir_with_constraints)

        # SAFE patches must not increase errors
        assert errors_after <= errors_before

    def test_remove_operation_may_increase_errors(
        self,
        schema_ir_with_constraints,
    ):
        """
        Test that remove operations on required fields increase errors.

        This is why 'remove' on required fields should be UNSAFE.
        """
        payload = {"name": "test", "value": 50}

        errors_before = count_validation_errors(payload, schema_ir_with_constraints)
        assert errors_before == 0, "Payload should be valid before removal"

        # Remove required field - this is UNSAFE
        remove_patch = JSONPatchOp(
            op="remove",
            path="/name"
        )

        patched = apply_json_patch(payload, remove_patch)
        errors_after = count_validation_errors(patched, schema_ir_with_constraints)

        # This should increase errors
        assert errors_after > errors_before, (
            "Removing required field should increase errors"
        )


# =============================================================================
# PRECONDITION TESTS
# =============================================================================

@pytest.mark.property
class TestPatchPreconditions:
    """
    Test that preconditions protect against invalid patch application.
    """

    def test_precondition_prevents_invalid_patch(self):
        """
        Test that precondition failure prevents patch application.
        """
        payload = {"name": "original_name", "value": 100}

        # Suggestion with precondition that should fail
        suggestion = FixSuggestion(
            patch=[
                JSONPatchOp(op="replace", path="/value", value=200),
            ],
            preconditions=[
                JSONPatchOp(op="test", path="/name", value="wrong_name"),
            ],
            confidence=0.95,
            safety=PatchSafety.NEEDS_REVIEW,
            rationale="Only apply if name matches expected value",
        )

        with pytest.raises(ValueError, match="Test failed"):
            apply_fix_suggestion(payload, suggestion)

    def test_precondition_success_allows_patch(self):
        """
        Test that matching precondition allows patch application.
        """
        payload = {"name": "expected_name", "value": 100}

        suggestion = FixSuggestion(
            patch=[
                JSONPatchOp(op="replace", path="/value", value=200),
            ],
            preconditions=[
                JSONPatchOp(op="test", path="/name", value="expected_name"),
            ],
            confidence=0.95,
            safety=PatchSafety.SAFE,
            rationale="Apply value update after verifying name",
        )

        patched = apply_fix_suggestion(payload, suggestion)
        assert patched["value"] == 200
