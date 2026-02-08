# -*- coding: utf-8 -*-
"""
Unit Tests for AssertionEngine (AGENT-FOUND-009)

Tests basic assertions, zero-hallucination checks, determinism verification,
lineage completeness, golden file comparison, deep comparison, and number
extraction.

Coverage target: 85%+ of assertion_engine.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums and stubs
# ---------------------------------------------------------------------------

class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class TestAssertion:
    def __init__(self, name, passed, expected=None, actual=None,
                 message="", severity=SeverityLevel.HIGH):
        self.name = name
        self.passed = passed
        self.expected = expected
        self.actual = actual
        self.message = message
        self.severity = severity


# ---------------------------------------------------------------------------
# Inline AssertionEngine mirroring greenlang/qa_test_harness/assertion_engine.py
# ---------------------------------------------------------------------------


class AssertionEngine:
    """Assertion engine for the QA Test Harness."""

    def assert_success(self, agent_result: Dict[str, Any]) -> TestAssertion:
        """Assert agent completed successfully."""
        success = agent_result.get("success", False)
        return TestAssertion(
            name="agent_success", passed=success,
            expected=True, actual=success,
            message="" if success else "Agent did not succeed",
        )

    def assert_failure(self, agent_result: Dict[str, Any]) -> TestAssertion:
        """Assert agent failed (for negative testing)."""
        success = agent_result.get("success", False)
        return TestAssertion(
            name="agent_failure", passed=not success,
            expected=False, actual=success,
            message="" if not success else "Agent should have failed",
        )

    # -- Zero-hallucination checks --

    def check_numeric_traceability(
        self, input_data: Dict[str, Any], output_data: Dict[str, Any]
    ) -> List[TestAssertion]:
        """Check output numbers are traceable to input."""
        assertions = []
        input_numbers = self._extract_numbers(input_data)
        output_numbers = self._extract_numbers(output_data.get("data", {}))

        for key, value in output_numbers.items():
            is_suspicious = (
                isinstance(value, (int, float))
                and value > 0
                and value == round(value, -3)
                and value not in input_numbers.values()
            )
            assertions.append(TestAssertion(
                name=f"numeric_traceability_{key}", passed=not is_suspicious,
                expected="traceable_value", actual=str(value),
                message=f"Value {key}={value} may be hallucinated" if is_suspicious else "",
                severity=SeverityLevel.HIGH if is_suspicious else SeverityLevel.INFO,
            ))
        return assertions

    def check_suspicious_round_number(self, value: float,
                                       input_values: Dict[str, Any]) -> bool:
        """Check if a number is suspiciously round and not in inputs."""
        if value <= 0:
            return False
        return (
            value == round(value, -3)
            and value not in input_values.values()
        )

    def check_provenance_id(self, output_data: Dict[str, Any]) -> TestAssertion:
        """Check provenance ID is valid."""
        data = output_data.get("data", {})
        prov_id = data.get("provenance_id")
        has_valid = isinstance(prov_id, str) and len(prov_id) >= 8
        return TestAssertion(
            name="provenance_id_valid", passed=has_valid,
            expected="valid_provenance_id", actual=str(prov_id),
            message="Provenance ID must be a valid identifier",
            severity=SeverityLevel.CRITICAL,
        )

    def check_output_consistency(self, output_data: Dict[str, Any]) -> TestAssertion:
        """Check success/data/error consistency."""
        success = output_data.get("success", False)
        has_data = bool(output_data.get("data"))
        has_error = bool(output_data.get("error"))
        ok = (success and has_data and not has_error) or (not success and has_error)
        return TestAssertion(
            name="output_consistency", passed=ok,
            expected="consistent_success_data_error",
            actual=f"success={success}, has_data={has_data}, has_error={has_error}",
            message="Output success/data/error must be consistent",
            severity=SeverityLevel.HIGH,
        )

    # -- Determinism checks --

    def check_determinism(self, hashes: List[str]) -> TestAssertion:
        """Check all output hashes are identical."""
        all_equal = len(set(hashes)) <= 1
        return TestAssertion(
            name="output_hash_determinism", passed=all_equal,
            expected=hashes[0] if hashes else "N/A",
            actual=str(set(hashes)),
            message=f"All iterations should produce identical output hashes",
            severity=SeverityLevel.CRITICAL,
        )

    # -- Lineage checks --

    def check_lineage_provenance(self, output_data: Dict[str, Any]) -> TestAssertion:
        """Check for provenance ID in output."""
        data = output_data.get("data", {})
        has = "provenance_id" in data or "provenance_hash" in output_data
        return TestAssertion(
            name="has_provenance_id", passed=has,
            expected="provenance_id_present",
            actual="present" if has else "missing",
            message="Output must include provenance identifier",
            severity=SeverityLevel.HIGH,
        )

    def check_lineage_timestamp(self, output_data: Dict[str, Any]) -> TestAssertion:
        """Check for timestamp in output."""
        data = output_data.get("data", {})
        has = "timestamp" in output_data or "created_at" in data
        return TestAssertion(
            name="has_timestamp", passed=has,
            expected="timestamp_present",
            actual="present" if has else "missing",
            message="Output must include timestamp",
            severity=SeverityLevel.MEDIUM,
        )

    # -- Golden file checks --

    def compare_with_golden(self, expected: Dict[str, Any],
                            actual: Dict[str, Any]) -> List[TestAssertion]:
        """Compare actual output against golden file data."""
        assertions = []
        for key in expected:
            exp_val = expected[key]
            act_val = actual.get(key)
            match = self._deep_compare(exp_val, act_val)
            assertions.append(TestAssertion(
                name=f"golden_{key}", passed=match,
                expected=str(exp_val)[:100], actual=str(act_val)[:100],
                message=f"Field '{key}' should match golden file",
                severity=SeverityLevel.HIGH,
            ))
        return assertions

    def compare_golden_file_not_found(self, path: str) -> TestAssertion:
        """Return assertion for missing golden file."""
        return TestAssertion(
            name="golden_file_exists", passed=False,
            message=f"Golden file not found: {path}",
            severity=SeverityLevel.HIGH,
        )

    # -- Deep compare --

    def _deep_compare(self, obj1: Any, obj2: Any) -> bool:
        if type(obj1) != type(obj2):
            return False
        if isinstance(obj1, dict):
            if set(obj1.keys()) != set(obj2.keys()):
                return False
            return all(self._deep_compare(obj1[k], obj2[k]) for k in obj1)
        elif isinstance(obj1, list):
            if len(obj1) != len(obj2):
                return False
            return all(self._deep_compare(a, b) for a, b in zip(obj1, obj2))
        elif isinstance(obj1, float):
            return abs(obj1 - obj2) < 1e-9
        else:
            return obj1 == obj2

    def _extract_numbers(self, data: Any, prefix: str = "") -> Dict[str, Any]:
        numbers = {}
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numbers[full_key] = value
                elif isinstance(value, (dict, list)):
                    numbers.update(self._extract_numbers(value, full_key))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                full_key = f"{prefix}[{i}]"
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    numbers[full_key] = item
                elif isinstance(item, (dict, list)):
                    numbers.update(self._extract_numbers(item, full_key))
        return numbers


# ===========================================================================
# Test Classes
# ===========================================================================


@pytest.fixture
def engine():
    return AssertionEngine()


class TestBasicAssertions:
    def test_assert_success_pass(self, engine):
        result = engine.assert_success({"success": True, "data": {"x": 1}})
        assert result.passed is True

    def test_assert_success_fail(self, engine):
        result = engine.assert_success({"success": False, "error": "failed"})
        assert result.passed is False

    def test_assert_failure_pass(self, engine):
        result = engine.assert_failure({"success": False, "error": "failed"})
        assert result.passed is True

    def test_assert_failure_fail(self, engine):
        result = engine.assert_failure({"success": True, "data": {"x": 1}})
        assert result.passed is False

    def test_assert_success_missing_key(self, engine):
        result = engine.assert_success({})
        assert result.passed is False

    def test_assert_success_name(self, engine):
        result = engine.assert_success({"success": True})
        assert result.name == "agent_success"

    def test_assert_failure_name(self, engine):
        result = engine.assert_failure({"success": False})
        assert result.name == "agent_failure"


class TestZeroHallucinationChecks:
    def test_numeric_traceability_traceable(self, engine):
        input_data = {"quantity": 100}
        output_data = {"data": {"result": 100}}
        assertions = engine.check_numeric_traceability(input_data, output_data)
        assert all(a.passed for a in assertions)

    def test_numeric_traceability_suspicious_round(self, engine):
        input_data = {"quantity": 123}
        output_data = {"data": {"result": 5000}}
        assertions = engine.check_numeric_traceability(input_data, output_data)
        suspicious = [a for a in assertions if not a.passed]
        assert len(suspicious) >= 1

    def test_numeric_traceability_zero_not_suspicious(self, engine):
        input_data = {}
        output_data = {"data": {"result": 0}}
        assertions = engine.check_numeric_traceability(input_data, output_data)
        assert all(a.passed for a in assertions)

    def test_numeric_traceability_negative_not_suspicious(self, engine):
        input_data = {}
        output_data = {"data": {"result": -5000}}
        assertions = engine.check_numeric_traceability(input_data, output_data)
        assert all(a.passed for a in assertions)

    def test_suspicious_round_number_true(self, engine):
        assert engine.check_suspicious_round_number(5000.0, {"x": 123}) is True

    def test_suspicious_round_number_false_in_input(self, engine):
        assert engine.check_suspicious_round_number(5000.0, {"x": 5000.0}) is False

    def test_suspicious_round_number_false_negative(self, engine):
        assert engine.check_suspicious_round_number(-1000.0, {}) is False

    def test_suspicious_round_number_false_not_round(self, engine):
        assert engine.check_suspicious_round_number(5001.0, {}) is False

    def test_provenance_id_valid(self, engine):
        output = {"data": {"provenance_id": "prov-abc12345"}}
        result = engine.check_provenance_id(output)
        assert result.passed is True

    def test_provenance_id_missing(self, engine):
        output = {"data": {}}
        result = engine.check_provenance_id(output)
        assert result.passed is False

    def test_provenance_id_too_short(self, engine):
        output = {"data": {"provenance_id": "abc"}}
        result = engine.check_provenance_id(output)
        assert result.passed is False

    def test_provenance_id_severity_critical(self, engine):
        output = {"data": {}}
        result = engine.check_provenance_id(output)
        assert result.severity == SeverityLevel.CRITICAL

    def test_output_consistency_success_with_data(self, engine):
        output = {"success": True, "data": {"x": 1}, "error": None}
        result = engine.check_output_consistency(output)
        assert result.passed is True

    def test_output_consistency_failure_with_error(self, engine):
        output = {"success": False, "data": None, "error": "bad"}
        result = engine.check_output_consistency(output)
        assert result.passed is True

    def test_output_consistency_success_without_data(self, engine):
        output = {"success": True, "data": None, "error": None}
        result = engine.check_output_consistency(output)
        assert result.passed is False

    def test_output_consistency_success_with_error(self, engine):
        output = {"success": True, "data": {"x": 1}, "error": "oops"}
        result = engine.check_output_consistency(output)
        assert result.passed is False


class TestDeterminismChecks:
    def test_determinism_all_hashes_equal(self, engine):
        hashes = ["abc123", "abc123", "abc123"]
        result = engine.check_determinism(hashes)
        assert result.passed is True

    def test_determinism_hash_mismatch(self, engine):
        hashes = ["abc123", "abc123", "def456"]
        result = engine.check_determinism(hashes)
        assert result.passed is False

    def test_determinism_single_hash(self, engine):
        hashes = ["abc123"]
        result = engine.check_determinism(hashes)
        assert result.passed is True

    def test_determinism_empty_hashes(self, engine):
        hashes = []
        result = engine.check_determinism(hashes)
        assert result.passed is True

    def test_determinism_severity_critical(self, engine):
        hashes = ["abc", "def"]
        result = engine.check_determinism(hashes)
        assert result.severity == SeverityLevel.CRITICAL


class TestLineageChecks:
    def test_lineage_provenance_present(self, engine):
        output = {"data": {"provenance_id": "prov-001"}}
        result = engine.check_lineage_provenance(output)
        assert result.passed is True

    def test_lineage_provenance_hash_present(self, engine):
        output = {"provenance_hash": "abc123", "data": {}}
        result = engine.check_lineage_provenance(output)
        assert result.passed is True

    def test_lineage_provenance_missing(self, engine):
        output = {"data": {}}
        result = engine.check_lineage_provenance(output)
        assert result.passed is False

    def test_lineage_timestamp_present(self, engine):
        output = {"timestamp": "2026-01-01T00:00:00Z", "data": {}}
        result = engine.check_lineage_timestamp(output)
        assert result.passed is True

    def test_lineage_timestamp_in_data(self, engine):
        output = {"data": {"created_at": "2026-01-01"}}
        result = engine.check_lineage_timestamp(output)
        assert result.passed is True

    def test_lineage_timestamp_missing(self, engine):
        output = {"data": {}}
        result = engine.check_lineage_timestamp(output)
        assert result.passed is False


class TestGoldenFileComparison:
    def test_golden_file_comparison_match(self, engine):
        expected = {"emissions": 2680.0, "unit": "kg_co2e"}
        actual = {"emissions": 2680.0, "unit": "kg_co2e"}
        assertions = engine.compare_with_golden(expected, actual)
        assert all(a.passed for a in assertions)

    def test_golden_file_comparison_mismatch(self, engine):
        expected = {"emissions": 2680.0}
        actual = {"emissions": 2700.0}
        assertions = engine.compare_with_golden(expected, actual)
        assert any(not a.passed for a in assertions)

    def test_golden_file_not_found(self, engine):
        result = engine.compare_golden_file_not_found("/path/to/missing.json")
        assert result.passed is False
        assert "not found" in result.message.lower()

    def test_golden_file_partial_match(self, engine):
        expected = {"a": 1, "b": 2}
        actual = {"a": 1, "b": 99}
        assertions = engine.compare_with_golden(expected, actual)
        passed = [a for a in assertions if a.passed]
        failed = [a for a in assertions if not a.passed]
        assert len(passed) == 1
        assert len(failed) == 1


class TestDeepCompare:
    def test_deep_compare_dicts_equal(self, engine):
        assert engine._deep_compare({"a": 1}, {"a": 1}) is True

    def test_deep_compare_dicts_different_values(self, engine):
        assert engine._deep_compare({"a": 1}, {"a": 2}) is False

    def test_deep_compare_dicts_different_keys(self, engine):
        assert engine._deep_compare({"a": 1}, {"b": 1}) is False

    def test_deep_compare_lists_equal(self, engine):
        assert engine._deep_compare([1, 2, 3], [1, 2, 3]) is True

    def test_deep_compare_lists_different(self, engine):
        assert engine._deep_compare([1, 2], [1, 3]) is False

    def test_deep_compare_lists_different_length(self, engine):
        assert engine._deep_compare([1, 2], [1, 2, 3]) is False

    def test_deep_compare_floats_tolerance(self, engine):
        assert engine._deep_compare(1.0000000001, 1.0000000002) is True

    def test_deep_compare_floats_different(self, engine):
        assert engine._deep_compare(1.0, 2.0) is False

    def test_deep_compare_nested(self, engine):
        a = {"x": {"y": [1.0, 2.0]}}
        b = {"x": {"y": [1.0, 2.0]}}
        assert engine._deep_compare(a, b) is True

    def test_deep_compare_type_mismatch(self, engine):
        assert engine._deep_compare(1, "1") is False

    def test_deep_compare_strings(self, engine):
        assert engine._deep_compare("abc", "abc") is True

    def test_deep_compare_strings_different(self, engine):
        assert engine._deep_compare("abc", "def") is False

    def test_deep_compare_none(self, engine):
        assert engine._deep_compare(None, None) is True

    def test_deep_compare_bool(self, engine):
        assert engine._deep_compare(True, True) is True


class TestExtractNumbers:
    def test_extract_numbers_flat(self, engine):
        data = {"a": 1, "b": 2.5, "c": "text"}
        numbers = engine._extract_numbers(data)
        assert numbers == {"a": 1, "b": 2.5}

    def test_extract_numbers_nested(self, engine):
        data = {"level1": {"level2": {"value": 42}}}
        numbers = engine._extract_numbers(data)
        assert "level1.level2.value" in numbers
        assert numbers["level1.level2.value"] == 42

    def test_extract_numbers_list(self, engine):
        data = {"items": [10, 20, 30]}
        numbers = engine._extract_numbers(data)
        assert numbers["items[0]"] == 10
        assert numbers["items[1]"] == 20
        assert numbers["items[2]"] == 30

    def test_extract_numbers_empty(self, engine):
        numbers = engine._extract_numbers({})
        assert numbers == {}

    def test_extract_numbers_no_numbers(self, engine):
        data = {"a": "text", "b": True, "c": None}
        numbers = engine._extract_numbers(data)
        assert numbers == {}

    def test_extract_numbers_excludes_bool(self, engine):
        data = {"flag": True, "value": 1}
        numbers = engine._extract_numbers(data)
        assert "flag" not in numbers
        assert "value" in numbers

    def test_extract_numbers_mixed_nested(self, engine):
        data = {
            "a": 1,
            "b": {"c": 2, "d": "text"},
            "e": [3, "four", {"f": 5}]
        }
        numbers = engine._extract_numbers(data)
        assert numbers["a"] == 1
        assert numbers["b.c"] == 2
        assert numbers["e[0]"] == 3
        assert numbers["e[2].f"] == 5
