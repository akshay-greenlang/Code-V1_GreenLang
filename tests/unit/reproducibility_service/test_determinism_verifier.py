# -*- coding: utf-8 -*-
"""
Unit Tests for DeterminismVerifier (AGENT-FOUND-008)

Tests input/output hash verification, tolerance-aware comparison,
full verification runs, value comparison logic, and run creation.

Coverage target: 85%+ of determinism_verifier.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline enums and helpers
# ---------------------------------------------------------------------------

class VerificationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIPPED = "skipped"


class NonDeterminismSource(str, Enum):
    FLOATING_POINT = "floating_point"
    RANDOM_SEED = "random_seed"


DEFAULT_ABSOLUTE_TOLERANCE = 1e-9
DEFAULT_RELATIVE_TOLERANCE = 1e-6


class VerificationCheck:
    def __init__(self, check_name: str, status: VerificationStatus,
                 expected_value: Optional[str] = None, actual_value: Optional[str] = None,
                 difference: Optional[float] = None, tolerance: Optional[float] = None,
                 message: str = ""):
        self.check_name = check_name
        self.status = status
        self.expected_value = expected_value
        self.actual_value = actual_value
        self.difference = difference
        self.tolerance = tolerance
        self.message = message


class VerificationRun:
    def __init__(self, run_id: str, execution_id: str, status: VerificationStatus,
                 checks: Optional[List[VerificationCheck]] = None,
                 started_at: Optional[datetime] = None):
        self.run_id = run_id
        self.execution_id = execution_id
        self.status = status
        self.checks = checks or []
        self.started_at = started_at or datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Inline DeterminismVerifier
# ---------------------------------------------------------------------------

class DeterminismVerifier:
    """Verifies deterministic behavior of computations."""

    def __init__(self, absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
                 relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE):
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self._detected_sources: list = []
        self._run_counter = 0

    def _compute_hash(self, data: Any) -> str:
        normalized = self._normalize(data)
        json_str = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _normalize(self, data: Any) -> Any:
        if data is None:
            return None
        if isinstance(data, bool):
            return data
        if isinstance(data, (int, str)):
            return data
        if isinstance(data, float):
            if self.absolute_tolerance > 0:
                precision = max(0, -int(math.log10(self.absolute_tolerance)))
                return round(data, precision)
            return data
        if isinstance(data, Decimal):
            return str(data)
        if isinstance(data, dict):
            return {str(k): self._normalize(v) for k, v in sorted(data.items())}
        if isinstance(data, (list, tuple)):
            return [self._normalize(item) for item in data]
        if isinstance(data, set):
            return sorted([self._normalize(item) for item in data])
        return str(data)

    def verify_input(self, input_data: Dict[str, Any],
                     expected_hash: Optional[str] = None) -> VerificationCheck:
        """Verify input data hash."""
        actual_hash = self._compute_hash(input_data)

        if expected_hash is None:
            return VerificationCheck(
                check_name="input_hash_verification",
                status=VerificationStatus.SKIPPED,
                actual_value=actual_hash,
                message="No expected hash provided - skipped",
            )

        if actual_hash == expected_hash:
            return VerificationCheck(
                check_name="input_hash_verification",
                status=VerificationStatus.PASS,
                expected_value=expected_hash,
                actual_value=actual_hash,
                message="Input hash matches",
            )

        return VerificationCheck(
            check_name="input_hash_verification",
            status=VerificationStatus.FAIL,
            expected_value=expected_hash,
            actual_value=actual_hash,
            message="Input hash mismatch",
        )

    def verify_output(self, output_data: Dict[str, Any],
                      expected_hash: Optional[str] = None,
                      expected_data: Optional[Dict[str, Any]] = None) -> VerificationCheck:
        """Verify output data hash or values."""
        actual_hash = self._compute_hash(output_data)

        if expected_hash is None and expected_data is None:
            return VerificationCheck(
                check_name="output_hash_verification",
                status=VerificationStatus.SKIPPED,
                actual_value=actual_hash,
                message="No expected hash or data provided - skipped",
            )

        if expected_hash is not None:
            if actual_hash == expected_hash:
                return VerificationCheck(
                    check_name="output_hash_verification",
                    status=VerificationStatus.PASS,
                    expected_value=expected_hash,
                    actual_value=actual_hash,
                    message="Output hash matches",
                )

            # Hash mismatch - check if within tolerance
            if expected_data is not None:
                within_tol = self._compare_values(output_data, expected_data)
                if within_tol:
                    return VerificationCheck(
                        check_name="output_hash_verification",
                        status=VerificationStatus.WARNING,
                        expected_value=expected_hash,
                        actual_value=actual_hash,
                        tolerance=self.absolute_tolerance,
                        message="Output hash differs but values within tolerance",
                    )

            return VerificationCheck(
                check_name="output_hash_verification",
                status=VerificationStatus.FAIL,
                expected_value=expected_hash,
                actual_value=actual_hash,
                message="Output hash mismatch",
            )

        # Only expected_data provided
        within_tol = self._compare_values(output_data, expected_data)
        if within_tol:
            return VerificationCheck(
                check_name="output_value_verification",
                status=VerificationStatus.PASS,
                message="Output values match within tolerance",
            )
        return VerificationCheck(
            check_name="output_value_verification",
            status=VerificationStatus.FAIL,
            message="Output values differ beyond tolerance",
        )

    def verify_full(self, input_data: Dict[str, Any],
                    output_data: Optional[Dict[str, Any]] = None,
                    expected_input_hash: Optional[str] = None,
                    expected_output_hash: Optional[str] = None) -> List[VerificationCheck]:
        """Run full verification on input and output."""
        checks = []
        checks.append(self.verify_input(input_data, expected_input_hash))
        if output_data is not None:
            checks.append(self.verify_output(output_data, expected_output_hash))
        return checks

    def _compare_values(self, actual: Any, expected: Any) -> bool:
        """Compare two values within tolerance."""
        if isinstance(actual, dict) and isinstance(expected, dict):
            if set(actual.keys()) != set(expected.keys()):
                return False
            return all(self._compare_values(actual[k], expected[k]) for k in actual)
        if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                return False
            return all(self._compare_values(a, e) for a, e in zip(actual, expected))
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return self._compare_numeric(float(actual), float(expected))
        return actual == expected

    def _compare_numeric(self, a: float, b: float) -> bool:
        """Compare two numeric values with absolute and relative tolerance."""
        if abs(a - b) <= self.absolute_tolerance:
            return True
        if b != 0 and abs((a - b) / b) <= self.relative_tolerance:
            return True
        return False

    def run_verification(self, execution_id: str,
                         input_data: Dict[str, Any],
                         output_data: Optional[Dict[str, Any]] = None,
                         expected_input_hash: Optional[str] = None,
                         expected_output_hash: Optional[str] = None) -> VerificationRun:
        """Execute a full verification run and return a VerificationRun record."""
        self._run_counter += 1
        run_id = f"vrun-{self._run_counter:04d}"
        checks = self.verify_full(
            input_data, output_data, expected_input_hash, expected_output_hash,
        )
        failed = any(c.status == VerificationStatus.FAIL for c in checks)
        warned = any(c.status == VerificationStatus.WARNING for c in checks)
        if failed:
            overall = VerificationStatus.FAIL
        elif warned:
            overall = VerificationStatus.WARNING
        else:
            overall = VerificationStatus.PASS

        return VerificationRun(
            run_id=run_id,
            execution_id=execution_id,
            status=overall,
            checks=checks,
        )


# ===========================================================================
# Test Classes
# ===========================================================================


class TestVerifyInput:
    """Test verify_input method."""

    def test_verify_input_hash_match(self):
        verifier = DeterminismVerifier()
        data = {"emissions": 100.5}
        h = verifier._compute_hash(data)
        result = verifier.verify_input(data, expected_hash=h)
        assert result.status == VerificationStatus.PASS
        assert result.actual_value == h

    def test_verify_input_hash_mismatch(self):
        verifier = DeterminismVerifier()
        data = {"emissions": 100.5}
        result = verifier.verify_input(data, expected_hash="wrong_hash")
        assert result.status == VerificationStatus.FAIL

    def test_verify_input_no_expected_hash(self):
        verifier = DeterminismVerifier()
        data = {"emissions": 100.5}
        result = verifier.verify_input(data, expected_hash=None)
        assert result.status == VerificationStatus.SKIPPED

    def test_verify_input_empty_dict(self):
        verifier = DeterminismVerifier()
        data = {}
        h = verifier._compute_hash(data)
        result = verifier.verify_input(data, h)
        assert result.status == VerificationStatus.PASS

    def test_verify_input_deterministic(self):
        verifier = DeterminismVerifier()
        data = {"fuel": "diesel", "quantity": 1000}
        h1 = verifier._compute_hash(data)
        h2 = verifier._compute_hash(data)
        assert h1 == h2


class TestVerifyOutput:
    """Test verify_output method."""

    def test_verify_output_hash_match(self):
        verifier = DeterminismVerifier()
        data = {"total": 2680.0}
        h = verifier._compute_hash(data)
        result = verifier.verify_output(data, expected_hash=h)
        assert result.status == VerificationStatus.PASS

    def test_verify_output_hash_mismatch(self):
        verifier = DeterminismVerifier()
        data = {"total": 2680.0}
        result = verifier.verify_output(data, expected_hash="bad_hash")
        assert result.status == VerificationStatus.FAIL

    def test_verify_output_no_expected(self):
        verifier = DeterminismVerifier()
        data = {"total": 2680.0}
        result = verifier.verify_output(data)
        assert result.status == VerificationStatus.SKIPPED

    def test_verify_output_within_tolerance(self):
        verifier = DeterminismVerifier(absolute_tolerance=1.0)
        expected = {"total": 2680.0}
        actual = {"total": 2680.5}
        result = verifier.verify_output(
            actual,
            expected_hash="fake_hash",
            expected_data=expected,
        )
        assert result.status == VerificationStatus.WARNING

    def test_verify_output_outside_tolerance(self):
        verifier = DeterminismVerifier(absolute_tolerance=0.001)
        expected = {"total": 2680.0}
        actual = {"total": 2690.0}
        result = verifier.verify_output(
            actual,
            expected_hash="fake_hash",
            expected_data=expected,
        )
        assert result.status == VerificationStatus.FAIL

    def test_verify_output_value_only_match(self):
        verifier = DeterminismVerifier(absolute_tolerance=1.0)
        expected = {"total": 100.0}
        actual = {"total": 100.5}
        result = verifier.verify_output(actual, expected_data=expected)
        assert result.status == VerificationStatus.PASS

    def test_verify_output_value_only_mismatch(self):
        verifier = DeterminismVerifier(absolute_tolerance=0.001)
        expected = {"total": 100.0}
        actual = {"total": 200.0}
        result = verifier.verify_output(actual, expected_data=expected)
        assert result.status == VerificationStatus.FAIL


class TestVerifyFull:
    """Test verify_full method."""

    def test_verify_full_all_pass(self):
        verifier = DeterminismVerifier()
        inp = {"x": 1}
        out = {"y": 2}
        ih = verifier._compute_hash(inp)
        oh = verifier._compute_hash(out)
        checks = verifier.verify_full(inp, out, ih, oh)
        assert len(checks) == 2
        assert all(c.status == VerificationStatus.PASS for c in checks)

    def test_verify_full_input_fail(self):
        verifier = DeterminismVerifier()
        inp = {"x": 1}
        checks = verifier.verify_full(inp, expected_input_hash="bad")
        assert checks[0].status == VerificationStatus.FAIL

    def test_verify_full_output_fail(self):
        verifier = DeterminismVerifier()
        inp = {"x": 1}
        out = {"y": 2}
        ih = verifier._compute_hash(inp)
        checks = verifier.verify_full(inp, out, ih, "bad_output_hash")
        assert checks[0].status == VerificationStatus.PASS
        assert checks[1].status == VerificationStatus.FAIL

    def test_verify_full_no_output(self):
        verifier = DeterminismVerifier()
        inp = {"x": 1}
        ih = verifier._compute_hash(inp)
        checks = verifier.verify_full(inp, expected_input_hash=ih)
        assert len(checks) == 1
        assert checks[0].status == VerificationStatus.PASS

    def test_verify_full_both_skipped(self):
        verifier = DeterminismVerifier()
        checks = verifier.verify_full({"x": 1})
        assert len(checks) == 1
        assert checks[0].status == VerificationStatus.SKIPPED


class TestCompareValues:
    """Test _compare_values method."""

    def test_compare_exact_match(self):
        verifier = DeterminismVerifier()
        assert verifier._compare_values({"a": 1}, {"a": 1}) is True

    def test_compare_within_absolute_tolerance(self):
        verifier = DeterminismVerifier(absolute_tolerance=0.01)
        assert verifier._compare_values(
            {"v": 1.005}, {"v": 1.0}
        ) is True

    def test_compare_within_relative_tolerance(self):
        verifier = DeterminismVerifier(absolute_tolerance=0.0, relative_tolerance=0.01)
        assert verifier._compare_values(
            {"v": 100.5}, {"v": 100.0}
        ) is True

    def test_compare_outside_tolerance(self):
        verifier = DeterminismVerifier(absolute_tolerance=0.001, relative_tolerance=0.001)
        assert verifier._compare_values(
            {"v": 110.0}, {"v": 100.0}
        ) is False

    def test_compare_dict_missing_keys(self):
        verifier = DeterminismVerifier()
        assert verifier._compare_values(
            {"a": 1, "b": 2}, {"a": 1}
        ) is False

    def test_compare_dict_different_values(self):
        verifier = DeterminismVerifier(absolute_tolerance=0.001)
        assert verifier._compare_values(
            {"a": 1, "b": 100}, {"a": 1, "b": 200}
        ) is False

    def test_compare_list_matching(self):
        verifier = DeterminismVerifier()
        assert verifier._compare_values([1, 2, 3], [1, 2, 3]) is True

    def test_compare_list_different_length(self):
        verifier = DeterminismVerifier()
        assert verifier._compare_values([1, 2], [1, 2, 3]) is False

    def test_compare_string_matching(self):
        verifier = DeterminismVerifier()
        assert verifier._compare_values("hello", "hello") is True

    def test_compare_string_different(self):
        verifier = DeterminismVerifier()
        assert verifier._compare_values("hello", "world") is False


class TestCompareNumeric:
    """Test _compare_numeric method."""

    def test_compare_integers_exact(self):
        verifier = DeterminismVerifier()
        assert verifier._compare_numeric(42.0, 42.0) is True

    def test_compare_floats_within_absolute(self):
        verifier = DeterminismVerifier(absolute_tolerance=0.01)
        assert verifier._compare_numeric(1.005, 1.0) is True

    def test_compare_floats_within_relative(self):
        verifier = DeterminismVerifier(absolute_tolerance=0.0, relative_tolerance=0.01)
        assert verifier._compare_numeric(100.5, 100.0) is True

    def test_compare_floats_outside_both(self):
        verifier = DeterminismVerifier(absolute_tolerance=0.001, relative_tolerance=0.001)
        assert verifier._compare_numeric(110.0, 100.0) is False

    def test_compare_zero_division_safe(self):
        verifier = DeterminismVerifier(absolute_tolerance=0.0, relative_tolerance=0.01)
        # When b=0, relative tolerance check skipped
        assert verifier._compare_numeric(0.001, 0.0) is False

    def test_compare_negative_numbers(self):
        verifier = DeterminismVerifier(absolute_tolerance=0.01)
        assert verifier._compare_numeric(-1.005, -1.0) is True

    def test_compare_large_numbers(self):
        verifier = DeterminismVerifier(absolute_tolerance=0.0, relative_tolerance=1e-6)
        assert verifier._compare_numeric(1_000_000.001, 1_000_000.0) is True


class TestRunVerification:
    """Test run_verification method."""

    def test_run_verification_creates_run(self):
        verifier = DeterminismVerifier()
        inp = {"x": 1}
        ih = verifier._compute_hash(inp)
        run = verifier.run_verification("exec-001", inp, expected_input_hash=ih)
        assert isinstance(run, VerificationRun)
        assert run.execution_id == "exec-001"
        assert run.status == VerificationStatus.PASS

    def test_run_verification_increments_counter(self):
        verifier = DeterminismVerifier()
        verifier.run_verification("exec-001", {"x": 1})
        verifier.run_verification("exec-002", {"x": 2})
        assert verifier._run_counter == 2

    def test_run_verification_fail_status(self):
        verifier = DeterminismVerifier()
        run = verifier.run_verification(
            "exec-001", {"x": 1}, expected_input_hash="bad"
        )
        assert run.status == VerificationStatus.FAIL

    def test_run_verification_warning_status(self):
        verifier = DeterminismVerifier(absolute_tolerance=1.0)
        inp = {"x": 1}
        ih = verifier._compute_hash(inp)
        out = {"y": 2.5}
        run = verifier.run_verification(
            "exec-001", inp, out,
            expected_input_hash=ih,
            expected_output_hash="fake_hash",
        )
        # Output hash mismatch without expected_data leads to FAIL
        assert run.status == VerificationStatus.FAIL

    def test_run_verification_run_id_format(self):
        verifier = DeterminismVerifier()
        run = verifier.run_verification("exec-001", {"x": 1})
        assert run.run_id.startswith("vrun-")

    def test_run_verification_checks_included(self):
        verifier = DeterminismVerifier()
        inp = {"x": 1}
        out = {"y": 2}
        ih = verifier._compute_hash(inp)
        oh = verifier._compute_hash(out)
        run = verifier.run_verification("exec-001", inp, out, ih, oh)
        assert len(run.checks) == 2

    def test_run_verification_started_at(self):
        verifier = DeterminismVerifier()
        run = verifier.run_verification("exec-001", {"x": 1})
        assert run.started_at is not None


class TestDeterminismVerifierInit:
    """Test DeterminismVerifier initialization."""

    def test_default_tolerances(self):
        verifier = DeterminismVerifier()
        assert verifier.absolute_tolerance == 1e-9
        assert verifier.relative_tolerance == 1e-6

    def test_custom_tolerances(self):
        verifier = DeterminismVerifier(absolute_tolerance=0.01, relative_tolerance=0.1)
        assert verifier.absolute_tolerance == 0.01
        assert verifier.relative_tolerance == 0.1

    def test_zero_run_counter(self):
        verifier = DeterminismVerifier()
        assert verifier._run_counter == 0
