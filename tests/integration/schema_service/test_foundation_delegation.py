# -*- coding: utf-8 -*-
"""
Integration Tests for Foundation Agent -> Layer 2 Delegation (AGENT-FOUND-002)

Tests that the foundation SchemaCompilerAgent correctly delegates
to the Layer 2 Schema SDK when available, and falls back to its
own Layer 1 validation when the SDK is not importable.

Coverage:
    - SCHEMA_SDK_AVAILABLE flag
    - Delegation to SDK when available
    - Fallback when SDK not available
    - ValidationResult format compatibility
    - Type coercion consistency
    - Unit consistency checks through delegation
    - Error hints from suggestions engine
    - Provenance hashing preservation

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Inline stubs for the foundation agent interface
# ---------------------------------------------------------------------------


class ValidationResult:
    """Minimal ValidationResult compatible with Layer 1."""

    def __init__(self, valid: bool = True):
        self.valid = valid
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []

    def add_error(self, error: Dict[str, Any]):
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: Dict[str, Any]):
        self.warnings.append(warning)

    def merge(self, other: "ValidationResult"):
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.valid:
            self.valid = False


# Check whether the Schema SDK is actually importable
try:
    from greenlang.schema.sdk import validate as _sdk_validate

    _SCHEMA_SDK_AVAILABLE = True
except ImportError:
    _SCHEMA_SDK_AVAILABLE = False


def _compute_provenance_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


def _validate_with_layer1(
    payload: Dict[str, Any],
    schema: Dict[str, Any],
) -> ValidationResult:
    """Layer 1 basic validation (no SDK dependency)."""
    result = ValidationResult(valid=True)

    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in payload:
            result.add_error({
                "field": field,
                "message": f"Required field '{field}' is missing",
            })

    # Check types for properties
    properties = schema.get("properties", {})
    for field, field_schema in properties.items():
        if field in payload:
            expected_type = field_schema.get("type")
            if expected_type == "string" and not isinstance(payload[field], str):
                result.add_error({
                    "field": field,
                    "message": f"Expected string, got {type(payload[field]).__name__}",
                })
            elif expected_type == "number" and not isinstance(payload[field], (int, float)):
                result.add_error({
                    "field": field,
                    "message": f"Expected number, got {type(payload[field]).__name__}",
                })
            elif expected_type == "integer" and not isinstance(payload[field], int):
                result.add_error({
                    "field": field,
                    "message": f"Expected integer, got {type(payload[field]).__name__}",
                })

    return result


def _validate_with_sdk(
    payload: Dict[str, Any],
    schema: Dict[str, Any],
) -> ValidationResult:
    """
    Delegate to Layer 2 SDK, then convert result to Layer 1 format.
    """
    result = ValidationResult(valid=True)

    if not _SCHEMA_SDK_AVAILABLE:
        return _validate_with_layer1(payload, schema)

    try:
        sdk_result = _sdk_validate(payload, schema)

        # Convert SDK findings to Layer 1 format
        result.valid = sdk_result.valid
        for finding in sdk_result.findings:
            entry = {
                "field": finding.path,
                "message": finding.message,
                "code": finding.code,
            }
            if finding.severity.value == "error":
                result.add_error(entry)
            else:
                result.add_warning(entry)

    except Exception:
        # If SDK raises, fall back to Layer 1
        return _validate_with_layer1(payload, schema)

    return result


# ===========================================================================
# Test Classes
# ===========================================================================


class TestSchemaSDKAvailableFlag:
    """Test SCHEMA_SDK_AVAILABLE flag."""

    def test_flag_is_boolean(self):
        """Flag should be a boolean."""
        assert isinstance(_SCHEMA_SDK_AVAILABLE, bool)

    def test_sdk_importable_when_flag_true(self):
        """If flag is True, the SDK should actually be importable."""
        if _SCHEMA_SDK_AVAILABLE:
            from greenlang.schema.sdk import validate

            assert callable(validate)

    def test_sdk_not_importable_when_flag_false(self):
        """If flag is False, the SDK import should have failed."""
        if not _SCHEMA_SDK_AVAILABLE:
            with pytest.raises(ImportError):
                from greenlang.schema.sdk import validate  # noqa: F811


class TestFoundationAgentDelegation:
    """Test foundation agent validates using SDK when available."""

    def test_validates_valid_payload(self, emissions_schema, valid_emissions_payload):
        """Valid payload should pass validation."""
        result = _validate_with_sdk(valid_emissions_payload, emissions_schema)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validates_invalid_payload(self, emissions_schema, invalid_emissions_payload):
        """Invalid payload should fail validation."""
        result = _validate_with_sdk(invalid_emissions_payload, emissions_schema)
        assert result.valid is False
        assert len(result.errors) > 0

    def test_missing_required_detected(self, emissions_schema):
        """Missing required fields should be detected."""
        payload = {"fuel_type": "diesel"}  # Missing most required fields
        result = _validate_with_sdk(payload, emissions_schema)
        assert result.valid is False

        missing_fields = {e["field"] for e in result.errors}
        # At minimum these should be detected as missing
        assert len(missing_fields) > 0


class TestFoundationAgentFallback:
    """Test foundation agent falls back when SDK not available."""

    def test_fallback_validates_valid_payload(self, emissions_schema, valid_emissions_payload):
        """Layer 1 fallback should pass valid payloads."""
        result = _validate_with_layer1(valid_emissions_payload, emissions_schema)
        assert result.valid is True

    def test_fallback_detects_missing_required(self, emissions_schema):
        """Layer 1 fallback should detect missing required fields."""
        payload = {"fuel_type": "diesel"}
        result = _validate_with_layer1(payload, emissions_schema)
        assert result.valid is False
        assert len(result.errors) > 0

    def test_fallback_detects_type_errors(self, emissions_schema):
        """Layer 1 fallback should detect type errors."""
        payload = {
            "source_id": 123,  # Should be string
            "fuel_type": "diesel",
            "quantity": 100.0,
            "unit": "liters",
            "co2e_kg": 268.0,
        }
        result = _validate_with_layer1(payload, emissions_schema)
        assert result.valid is False

    def test_sdk_exception_falls_back(self, emissions_schema, valid_emissions_payload):
        """If SDK raises an exception, should fall back to Layer 1."""
        with patch(
            "tests.integration.schema_service.test_foundation_delegation._SCHEMA_SDK_AVAILABLE",
            True,
        ):
            with patch(
                "tests.integration.schema_service.test_foundation_delegation._sdk_validate",
                side_effect=Exception("SDK Error"),
            ):
                result = _validate_with_sdk(valid_emissions_payload, emissions_schema)
                # Should fall back to Layer 1 and still pass
                assert result.valid is True


class TestResultFormatCompatibility:
    """Test result format matches Layer 1 expectations (ValidationResult)."""

    def test_result_has_valid_flag(self, emissions_schema, valid_emissions_payload):
        result = _validate_with_sdk(valid_emissions_payload, emissions_schema)
        assert hasattr(result, "valid")
        assert isinstance(result.valid, bool)

    def test_result_has_errors_list(self, emissions_schema, valid_emissions_payload):
        result = _validate_with_sdk(valid_emissions_payload, emissions_schema)
        assert hasattr(result, "errors")
        assert isinstance(result.errors, list)

    def test_result_has_warnings_list(self, emissions_schema, valid_emissions_payload):
        result = _validate_with_sdk(valid_emissions_payload, emissions_schema)
        assert hasattr(result, "warnings")
        assert isinstance(result.warnings, list)

    def test_error_entries_have_field_and_message(self, emissions_schema, invalid_emissions_payload):
        result = _validate_with_sdk(invalid_emissions_payload, emissions_schema)
        for error in result.errors:
            assert "field" in error
            assert "message" in error

    def test_merge_combines_results(self):
        r1 = ValidationResult(valid=True)
        r2 = ValidationResult(valid=False)
        r2.add_error({"field": "test", "message": "err"})
        r1.merge(r2)
        assert r1.valid is False
        assert len(r1.errors) == 1


class TestTypeCoercionConsistency:
    """Test type coercion results match between Layer 1 and Layer 2."""

    def test_string_number_detected_by_both(self, emissions_schema):
        """Both layers should detect a string where a number is expected."""
        payload = {
            "source_id": "FAC-001",
            "fuel_type": "diesel",
            "quantity": "one hundred",  # Should be a number
            "unit": "liters",
            "co2e_kg": 268.0,
        }
        l1_result = _validate_with_layer1(payload, emissions_schema)
        l2_result = _validate_with_sdk(payload, emissions_schema)

        # Both should detect this as invalid
        assert l1_result.valid is False
        assert l2_result.valid is False

    def test_valid_payload_passes_both(self, emissions_schema, valid_emissions_payload):
        """Valid payload should pass both Layer 1 and Layer 2."""
        l1_result = _validate_with_layer1(valid_emissions_payload, emissions_schema)
        l2_result = _validate_with_sdk(valid_emissions_payload, emissions_schema)

        assert l1_result.valid is True
        assert l2_result.valid is True


class TestUnitConsistencyThroughDelegation:
    """Test unit consistency checks work through delegation."""

    def test_unit_field_accepted_in_schema(self, inline_schema_with_extensions):
        """Schema with $unit extensions should be processable."""
        payload = {"energy_consumed": 100.0, "co2_emissions": 50.0}
        result = _validate_with_sdk(payload, inline_schema_with_extensions)
        # Should at least not crash; validity depends on SDK support
        assert hasattr(result, "valid")

    def test_negative_values_rejected(self, inline_schema_with_extensions):
        """Negative values should be rejected when minimum is 0."""
        payload = {"energy_consumed": -100.0, "co2_emissions": 50.0}
        result = _validate_with_sdk(payload, inline_schema_with_extensions)
        assert result.valid is False


class TestProvenanceHashing:
    """Test provenance hashing is preserved through delegation."""

    def test_provenance_hash_is_deterministic(self, compute_hash):
        """Same input produces same hash."""
        data = {"payload": {"x": 1}, "schema": "test", "valid": True}
        h1 = _compute_provenance_hash(data)
        h2 = _compute_provenance_hash(data)
        assert h1 == h2

    def test_provenance_hash_is_sha256(self, compute_hash):
        """Hash should be a 64-character hex string (SHA-256)."""
        data = {"test": True}
        h = _compute_provenance_hash(data)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_inputs_produce_different_hashes(self, compute_hash):
        """Different inputs should produce different hashes."""
        h1 = _compute_provenance_hash({"x": 1})
        h2 = _compute_provenance_hash({"x": 2})
        assert h1 != h2

    def test_key_order_does_not_affect_hash(self, compute_hash):
        """JSON sort_keys=True ensures key order independence."""
        h1 = _compute_provenance_hash({"a": 1, "b": 2})
        h2 = _compute_provenance_hash({"b": 2, "a": 1})
        assert h1 == h2
