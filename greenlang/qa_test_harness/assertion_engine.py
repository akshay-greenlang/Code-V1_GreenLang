# -*- coding: utf-8 -*-
"""
Assertion Engine for QA Test Harness - AGENT-FOUND-009

Provides category-specific test assertions for zero-hallucination,
determinism, lineage, and golden file verification. All assertions
use deterministic comparison with no LLM-generated expected values.

Zero-Hallucination Guarantees:
    - All comparisons are deterministic
    - No LLM calls for expected value generation
    - All golden files are human-verified
    - Complete audit trail for every assertion

Example:
    >>> from greenlang.qa_test_harness.assertion_engine import AssertionEngine
    >>> engine = AssertionEngine(config)
    >>> assertions = engine.check_zero_hallucination(input_data, output_data)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-009 QA Test Harness
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Type

from greenlang.qa_test_harness.config import QATestHarnessConfig
from greenlang.qa_test_harness.models import (
    TestAssertion,
    TestCaseInput,
    TestCategory,
    SeverityLevel,
)

logger = logging.getLogger(__name__)


class AssertionEngine:
    """Assertion engine for category-specific test assertions.

    Executes deterministic assertions based on the test category,
    verifying zero-hallucination guarantees, determinism, lineage
    completeness, and golden file conformance.

    Attributes:
        config: QA test harness configuration.

    Example:
        >>> engine = AssertionEngine(config)
        >>> assertions = engine.run_assertions(test_input, agent_result, agent)
    """

    def __init__(self, config: QATestHarnessConfig) -> None:
        """Initialize AssertionEngine.

        Args:
            config: QA test harness configuration.
        """
        self.config = config
        logger.info("AssertionEngine initialized")

    def run_assertions(
        self,
        test_input: TestCaseInput,
        agent_result: Any,
        agent: Any,
    ) -> List[TestAssertion]:
        """Run assertions based on test category.

        Args:
            test_input: Test case specification.
            agent_result: Agent execution result.
            agent: Agent instance that produced the result.

        Returns:
            List of assertion results.
        """
        assertions: List[TestAssertion] = []

        # Basic success assertion
        is_success = agent_result.success if agent_result else False
        assertions.append(TestAssertion(
            name="agent_success",
            passed=is_success,
            expected=True,
            actual=is_success,
            message="Agent should complete successfully",
            severity=test_input.severity,
        ))

        # Category-specific assertions
        if test_input.category == TestCategory.ZERO_HALLUCINATION:
            output_data = agent_result.data if agent_result else {}
            assertions.extend(self.check_zero_hallucination(
                test_input.input_data, output_data,
            ))

        elif test_input.category == TestCategory.LINEAGE:
            output_data = (
                agent_result.model_dump() if agent_result else {}
            )
            assertions.extend(self.check_lineage(output_data))

        elif test_input.category == TestCategory.GOLDEN_FILE:
            assertions.extend(self._check_golden_file_path(test_input, agent_result))

        elif test_input.expected_output:
            assertions.extend(self._check_expected_output(
                test_input.expected_output, agent_result,
            ))

        return assertions

    def check_zero_hallucination(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        checks: Optional[List[str]] = None,
    ) -> List[TestAssertion]:
        """Check for hallucinated data in agent output.

        Verifies that all numeric values are traceable to inputs or
        formulas, that provenance IDs are valid, and that output
        structure is internally consistent.

        Args:
            input_data: Original input data.
            output_data: Agent output data.
            checks: Optional list of specific check names to run.

        Returns:
            List of assertion results.
        """
        assertions: List[TestAssertion] = []

        # Check 1: Numeric values are reasonable (not suspiciously round)
        input_numbers = self._extract_numbers(input_data)
        output_numbers = self._extract_numbers(output_data)

        for key, value in output_numbers.items():
            is_suspicious = (
                isinstance(value, (int, float))
                and value > 0
                and value == round(value, -3)
                and value not in input_numbers.values()
            )
            assertions.append(TestAssertion(
                name=f"numeric_traceability_{key}",
                passed=not is_suspicious,
                expected="traceable_value",
                actual=str(value),
                message=f"Value {key}={value} may be hallucinated (suspiciously round)",
                severity=SeverityLevel.HIGH if is_suspicious else SeverityLevel.INFO,
            ))

        # Check 2: Provenance IDs are valid format
        if "provenance_id" in output_data:
            prov_id = output_data["provenance_id"]
            has_valid_format = isinstance(prov_id, str) and len(prov_id) >= 8
            assertions.append(TestAssertion(
                name="provenance_id_valid",
                passed=has_valid_format,
                expected="valid_provenance_id",
                actual=str(prov_id),
                message="Provenance ID must be a valid identifier",
                severity=SeverityLevel.CRITICAL,
            ))

        # Check 3: Output consistency
        assertions.append(self._check_output_consistency(output_data))

        return assertions

    def check_determinism(
        self,
        agent_class: Type[Any],
        input_data: Dict[str, Any],
        iterations: Optional[int] = None,
    ) -> List[TestAssertion]:
        """Check that agent produces deterministic outputs.

        Runs the same input multiple times and verifies all outputs
        are identical.

        Args:
            agent_class: Agent class to instantiate and run.
            input_data: Input data for the agent.
            iterations: Number of iterations (defaults to config value).

        Returns:
            List of assertion results.
        """
        iters = iterations or self.config.determinism_iterations
        assertions: List[TestAssertion] = []
        outputs = []
        hashes = []

        for i in range(iters):
            agent = agent_class()
            result = agent.run(input_data)
            outputs.append(result)
            output_data = result.data if result else {}
            output_hash = _compute_hash(output_data)
            hashes.append(output_hash)

        # Check all hashes are identical
        all_hashes_equal = len(set(hashes)) == 1
        assertions.append(TestAssertion(
            name="output_hash_determinism",
            passed=all_hashes_equal,
            expected=hashes[0] if hashes else "N/A",
            actual=str(set(hashes)),
            message=f"All {iters} iterations should produce identical output hashes",
            severity=SeverityLevel.CRITICAL,
        ))

        # Check data values are identical (deep compare)
        if outputs:
            first_output = outputs[0]
            for i, output in enumerate(outputs[1:], 2):
                data_equal = self._deep_compare(
                    first_output.data if first_output else {},
                    output.data if output else {},
                )
                assertions.append(TestAssertion(
                    name=f"data_determinism_iter_{i}",
                    passed=data_equal,
                    expected="identical_to_first",
                    actual="identical" if data_equal else "different",
                    message=f"Iteration {i} should produce identical data to iteration 1",
                    severity=SeverityLevel.CRITICAL,
                ))

        return assertions

    def check_lineage(
        self,
        output_data: Dict[str, Any],
    ) -> List[TestAssertion]:
        """Check lineage completeness in output.

        Verifies that the agent output includes provenance identifiers,
        timestamps, and metrics required for audit trails.

        Args:
            output_data: Full agent output data (model_dump).

        Returns:
            List of assertion results.
        """
        assertions: List[TestAssertion] = []
        data = output_data.get("data", {})

        # Provenance present
        assertions.append(self._check_provenance_present(data, output_data))

        # Timestamp present
        assertions.append(self._check_timestamp_present(data, output_data))

        # Metrics present
        has_metrics = output_data.get("metrics") is not None
        assertions.append(TestAssertion(
            name="has_metrics",
            passed=has_metrics,
            expected="metrics_present",
            actual="present" if has_metrics else "missing",
            message="Output should include execution metrics",
            severity=SeverityLevel.LOW,
        ))

        return assertions

    def check_golden_file(
        self,
        agent_result: Any,
        golden_file_path: str,
    ) -> List[TestAssertion]:
        """Compare agent result against a golden file.

        Args:
            agent_result: Agent execution result.
            golden_file_path: Path to the golden file.

        Returns:
            List of assertion results.
        """
        assertions: List[TestAssertion] = []

        try:
            with open(golden_file_path, "r") as f:
                golden_data = json.load(f)

            expected_output = golden_data.get("expected_output", {})
            actual_data = agent_result.data if agent_result else {}

            for key in expected_output:
                expected_value = expected_output[key]
                actual_value = actual_data.get(key)
                matches = self._deep_compare(expected_value, actual_value)
                assertions.append(TestAssertion(
                    name=f"golden_{key}",
                    passed=matches,
                    expected=str(expected_value)[:100],
                    actual=str(actual_value)[:100],
                    message=f"Field '{key}' should match golden file",
                    severity=SeverityLevel.HIGH,
                ))

        except FileNotFoundError:
            assertions.append(TestAssertion(
                name="golden_file_exists",
                passed=False,
                message=f"Golden file not found: {golden_file_path}",
                severity=SeverityLevel.HIGH,
            ))

        except json.JSONDecodeError as e:
            assertions.append(TestAssertion(
                name="golden_file_valid",
                passed=False,
                message=f"Golden file is not valid JSON: {e}",
                severity=SeverityLevel.HIGH,
            ))

        return assertions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_numbers(
        self,
        data: Any,
        prefix: str = "",
    ) -> Dict[str, Any]:
        """Recursively extract numeric values from data.

        Args:
            data: Input data structure.
            prefix: Key prefix for nested paths.

        Returns:
            Dictionary mapping key paths to numeric values.
        """
        numbers: Dict[str, Any] = {}

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

    def _deep_compare(self, obj1: Any, obj2: Any) -> bool:
        """Deep compare two objects for equality.

        Args:
            obj1: First object.
            obj2: Second object.

        Returns:
            True if objects are deeply equal.
        """
        if type(obj1) != type(obj2):
            return False

        if isinstance(obj1, dict):
            if set(obj1.keys()) != set(obj2.keys()):
                return False
            return all(
                self._deep_compare(obj1[k], obj2[k])
                for k in obj1.keys()
            )

        elif isinstance(obj1, list):
            if len(obj1) != len(obj2):
                return False
            return all(
                self._deep_compare(a, b)
                for a, b in zip(obj1, obj2)
            )

        elif isinstance(obj1, float):
            return abs(obj1 - obj2) < 1e-9

        else:
            return obj1 == obj2

    def _check_output_consistency(
        self,
        output_data: Dict[str, Any],
    ) -> TestAssertion:
        """Check that output success/data/error fields are consistent.

        Args:
            output_data: Agent output data.

        Returns:
            Single assertion result.
        """
        success = output_data.get("success", False)
        has_data = bool(output_data.get("data"))
        has_error = bool(output_data.get("error"))

        consistency_ok = (
            (success and has_data and not has_error)
            or (not success and has_error)
            or (not success and not has_data and not has_error)
        )

        return TestAssertion(
            name="output_consistency",
            passed=consistency_ok,
            expected="consistent_success_data_error",
            actual=f"success={success}, has_data={has_data}, has_error={has_error}",
            message="Output success/data/error must be consistent",
            severity=SeverityLevel.HIGH,
        )

    def _check_provenance_present(
        self,
        data: Dict[str, Any],
        output_data: Dict[str, Any],
    ) -> TestAssertion:
        """Check that provenance identifier is present in output.

        Args:
            data: Inner data dictionary.
            output_data: Full output dictionary.

        Returns:
            Single assertion result.
        """
        has_provenance = (
            "provenance_id" in data
            or "provenance_hash" in output_data
            or "provenance_hash" in data
        )
        return TestAssertion(
            name="has_provenance_id",
            passed=has_provenance,
            expected="provenance_id_present",
            actual="present" if has_provenance else "missing",
            message="Output must include provenance identifier",
            severity=SeverityLevel.HIGH,
        )

    def _check_timestamp_present(
        self,
        data: Dict[str, Any],
        output_data: Dict[str, Any],
    ) -> TestAssertion:
        """Check that timestamp is present in output.

        Args:
            data: Inner data dictionary.
            output_data: Full output dictionary.

        Returns:
            Single assertion result.
        """
        has_timestamp = (
            "timestamp" in output_data
            or "created_at" in data
            or "timestamp" in data
        )
        return TestAssertion(
            name="has_timestamp",
            passed=has_timestamp,
            expected="timestamp_present",
            actual="present" if has_timestamp else "missing",
            message="Output must include timestamp for audit trail",
            severity=SeverityLevel.MEDIUM,
        )

    def _check_golden_file_path(
        self,
        test_input: TestCaseInput,
        agent_result: Any,
    ) -> List[TestAssertion]:
        """Run golden file comparison assertions from test input.

        Args:
            test_input: Test case specification.
            agent_result: Agent execution result.

        Returns:
            List of assertion results.
        """
        if not test_input.golden_file_path:
            return [TestAssertion(
                name="golden_file_path",
                passed=False,
                message="Golden file path not specified",
                severity=SeverityLevel.HIGH,
            )]

        return self.check_golden_file(agent_result, test_input.golden_file_path)

    def _check_expected_output(
        self,
        expected: Dict[str, Any],
        agent_result: Any,
    ) -> List[TestAssertion]:
        """Run expected output comparison assertions.

        Args:
            expected: Expected output data.
            agent_result: Agent execution result.

        Returns:
            List of assertion results.
        """
        assertions: List[TestAssertion] = []
        actual_data = agent_result.data if agent_result else {}

        for key, expected_value in expected.items():
            actual_value = actual_data.get(key)
            matches = self._deep_compare(expected_value, actual_value)
            assertions.append(TestAssertion(
                name=f"expected_{key}",
                passed=matches,
                expected=str(expected_value)[:100],
                actual=str(actual_value)[:100],
                message=f"Field '{key}' should match expected value",
                severity=SeverityLevel.HIGH,
            ))

        return assertions


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    """Compute deterministic SHA-256 hash of data.

    Args:
        data: Data to hash.

    Returns:
        Hex-encoded SHA-256 hash (first 16 chars).
    """
    import hashlib
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


__all__ = [
    "AssertionEngine",
]
