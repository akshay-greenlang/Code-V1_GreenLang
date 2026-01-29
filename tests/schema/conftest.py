"""
Pytest configuration and shared fixtures for GL-FOUND-X-002 tests.

This module provides:
    1. Fixtures for loading golden test schemas
    2. Fixtures for loading test payloads
    3. Fixtures for creating ValidationOptions with different profiles
    4. Fixtures for temporary directories
    5. Helper functions for comparing validation reports
    6. Custom markers: @pytest.mark.golden, @pytest.mark.property, @pytest.mark.security, @pytest.mark.slow

Example:
    >>> def test_valid_payload(load_golden_schema, load_golden_payload):
    ...     schema = load_golden_schema("basic/simple_object.yaml")
    ...     payload = load_golden_payload("valid/simple_valid.yaml")
    ...     # ... validate ...
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest
import yaml


# =============================================================================
# Path Constants
# =============================================================================

TESTS_SCHEMA_DIR = Path(__file__).parent
GOLDEN_DIR = TESTS_SCHEMA_DIR / "golden"
GOLDEN_SCHEMAS_DIR = GOLDEN_DIR / "schemas"
GOLDEN_PAYLOADS_DIR = GOLDEN_DIR / "payloads"
GOLDEN_EXPECTED_DIR = GOLDEN_DIR / "expected"


# =============================================================================
# Custom Markers Registration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "golden: marks tests that use golden test data"
    )
    config.addinivalue_line(
        "markers", "property: marks property-based tests (using Hypothesis)"
    )
    config.addinivalue_line(
        "markers", "security: marks security-focused tests (ReDoS, YAML bombs, etc.)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests that take more than 1 second"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


# =============================================================================
# Golden Test Data Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def golden_dir() -> Path:
    """Get the golden test data directory path."""
    return GOLDEN_DIR


@pytest.fixture(scope="session")
def golden_schemas_dir() -> Path:
    """Get the golden schemas directory path."""
    return GOLDEN_SCHEMAS_DIR


@pytest.fixture(scope="session")
def golden_payloads_dir() -> Path:
    """Get the golden payloads directory path."""
    return GOLDEN_PAYLOADS_DIR


@pytest.fixture(scope="session")
def golden_expected_dir() -> Path:
    """Get the golden expected outputs directory path."""
    return GOLDEN_EXPECTED_DIR


@pytest.fixture
def load_golden_schema(golden_schemas_dir):
    """
    Fixture factory for loading golden test schemas.

    Returns a function that loads a schema from the golden/schemas directory.

    Example:
        >>> def test_something(load_golden_schema):
        ...     schema = load_golden_schema("basic/simple_object.yaml")
    """
    def _load(schema_path: str) -> Dict[str, Any]:
        """Load a schema from the golden/schemas directory."""
        full_path = golden_schemas_dir / schema_path
        if not full_path.exists():
            raise FileNotFoundError(f"Golden schema not found: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            if schema_path.endswith(".yaml") or schema_path.endswith(".yml"):
                return yaml.safe_load(f)
            else:
                return json.load(f)

    return _load


@pytest.fixture
def load_golden_payload(golden_payloads_dir):
    """
    Fixture factory for loading golden test payloads.

    Returns a function that loads a payload from the golden/payloads directory.

    Example:
        >>> def test_something(load_golden_payload):
        ...     valid_payload = load_golden_payload("valid/simple_valid.yaml")
        ...     invalid_payload = load_golden_payload("invalid/missing_required.yaml")
    """
    def _load(payload_path: str) -> Dict[str, Any]:
        """Load a payload from the golden/payloads directory."""
        full_path = golden_payloads_dir / payload_path
        if not full_path.exists():
            raise FileNotFoundError(f"Golden payload not found: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            if payload_path.endswith(".yaml") or payload_path.endswith(".yml"):
                return yaml.safe_load(f)
            else:
                return json.load(f)

    return _load


@pytest.fixture
def load_expected_report(golden_expected_dir):
    """
    Fixture factory for loading expected validation reports.

    Returns a function that loads an expected output from the golden/expected directory.

    Example:
        >>> def test_something(load_expected_report):
        ...     expected = load_expected_report("missing_required_report.json")
    """
    def _load(report_path: str) -> Dict[str, Any]:
        """Load an expected report from the golden/expected directory."""
        full_path = golden_expected_dir / report_path
        if not full_path.exists():
            raise FileNotFoundError(f"Expected report not found: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return _load


@pytest.fixture
def all_golden_schemas(golden_schemas_dir) -> List[Path]:
    """Get all golden schema files."""
    schemas = []
    for pattern in ["**/*.yaml", "**/*.yml", "**/*.json"]:
        schemas.extend(golden_schemas_dir.glob(pattern))
    return sorted(schemas)


@pytest.fixture
def all_valid_payloads(golden_payloads_dir) -> List[Path]:
    """Get all valid payload files."""
    payloads_dir = golden_payloads_dir / "valid"
    if not payloads_dir.exists():
        return []
    payloads = []
    for pattern in ["**/*.yaml", "**/*.yml", "**/*.json"]:
        payloads.extend(payloads_dir.glob(pattern))
    return sorted(payloads)


@pytest.fixture
def all_invalid_payloads(golden_payloads_dir) -> List[Path]:
    """Get all invalid payload files."""
    payloads_dir = golden_payloads_dir / "invalid"
    if not payloads_dir.exists():
        return []
    payloads = []
    for pattern in ["**/*.yaml", "**/*.yml", "**/*.json"]:
        payloads.extend(payloads_dir.glob(pattern))
    return sorted(payloads)


# =============================================================================
# Validation Options Fixtures
# =============================================================================

@pytest.fixture
def strict_validation_options() -> Dict[str, Any]:
    """
    Get validation options for strict profile.

    Strict mode:
        - Unknown fields are errors
        - No coercion allowed
        - Units must be canonical
    """
    return {
        "profile": "strict",
        "normalize": True,
        "emit_patches": True,
        "patch_level": "safe",
        "max_errors": 100,
        "fail_fast": False,
        "unit_system": "SI",
        "unknown_field_policy": "error",
        "coercion_policy": "off",
    }


@pytest.fixture
def standard_validation_options() -> Dict[str, Any]:
    """
    Get validation options for standard profile (default).

    Standard mode:
        - Unknown fields are warnings
        - Safe coercion allowed
        - Non-canonical units trigger warnings
    """
    return {
        "profile": "standard",
        "normalize": True,
        "emit_patches": True,
        "patch_level": "safe",
        "max_errors": 100,
        "fail_fast": False,
        "unit_system": "SI",
        "unknown_field_policy": "warn",
        "coercion_policy": "safe",
    }


@pytest.fixture
def permissive_validation_options() -> Dict[str, Any]:
    """
    Get validation options for permissive profile.

    Permissive mode:
        - Unknown fields are ignored
        - Aggressive coercion allowed
        - Non-canonical units accepted silently
    """
    return {
        "profile": "permissive",
        "normalize": True,
        "emit_patches": True,
        "patch_level": "needs_review",
        "max_errors": 1000,
        "fail_fast": False,
        "unit_system": "SI",
        "unknown_field_policy": "ignore",
        "coercion_policy": "aggressive",
    }


@pytest.fixture
def fail_fast_options() -> Dict[str, Any]:
    """Get validation options that fail on first error."""
    return {
        "profile": "standard",
        "normalize": False,
        "emit_patches": False,
        "max_errors": 1,
        "fail_fast": True,
        "unit_system": "SI",
        "unknown_field_policy": "warn",
        "coercion_policy": "safe",
    }


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_schema_dir(tmp_path) -> Path:
    """
    Create a temporary directory for schema files.

    The directory is automatically cleaned up after the test.
    """
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    return schema_dir


@pytest.fixture
def temp_payload_dir(tmp_path) -> Path:
    """
    Create a temporary directory for payload files.

    The directory is automatically cleaned up after the test.
    """
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    return payload_dir


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """
    Create a temporary directory for output files.

    The directory is automatically cleaned up after the test.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# =============================================================================
# Helper Functions and Classes
# =============================================================================

class ValidationReportComparator:
    """
    Helper class for comparing validation reports.

    Provides methods to compare reports with various levels of strictness,
    ignoring non-deterministic fields like timestamps and processing times.
    """

    # Fields to ignore during comparison (non-deterministic)
    IGNORE_FIELDS = {
        "timings_ms",
        "processing_time_ms",
        "timestamp",
        "created_at",
        "schema_hash",  # May differ based on compilation
    }

    @classmethod
    def compare_reports(
        cls,
        actual: Dict[str, Any],
        expected: Dict[str, Any],
        ignore_order: bool = True,
        ignore_fields: Optional[set] = None,
    ) -> bool:
        """
        Compare two validation reports.

        Args:
            actual: The actual validation report.
            expected: The expected validation report.
            ignore_order: If True, ignore order of findings list.
            ignore_fields: Additional fields to ignore during comparison.

        Returns:
            True if reports match, False otherwise.
        """
        fields_to_ignore = cls.IGNORE_FIELDS.copy()
        if ignore_fields:
            fields_to_ignore.update(ignore_fields)

        actual_clean = cls._remove_fields(actual, fields_to_ignore)
        expected_clean = cls._remove_fields(expected, fields_to_ignore)

        if ignore_order:
            actual_clean = cls._sort_findings(actual_clean)
            expected_clean = cls._sort_findings(expected_clean)

        return actual_clean == expected_clean

    @classmethod
    def assert_reports_equal(
        cls,
        actual: Dict[str, Any],
        expected: Dict[str, Any],
        ignore_order: bool = True,
        ignore_fields: Optional[set] = None,
        message: str = "",
    ) -> None:
        """
        Assert that two validation reports are equal.

        Args:
            actual: The actual validation report.
            expected: The expected validation report.
            ignore_order: If True, ignore order of findings list.
            ignore_fields: Additional fields to ignore.
            message: Optional message for assertion failure.
        """
        if not cls.compare_reports(actual, expected, ignore_order, ignore_fields):
            diff = cls._get_diff(actual, expected)
            msg = message or "Validation reports do not match"
            raise AssertionError(f"{msg}\n\nDifferences:\n{diff}")

    @classmethod
    def assert_finding_exists(
        cls,
        report: Dict[str, Any],
        code: str,
        path: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> None:
        """
        Assert that a specific finding exists in the report.

        Args:
            report: The validation report.
            code: The expected error code.
            path: Optional expected JSON path.
            severity: Optional expected severity.
        """
        findings = report.get("findings", [])
        for finding in findings:
            if finding.get("code") == code:
                if path is not None and finding.get("path") != path:
                    continue
                if severity is not None and finding.get("severity") != severity:
                    continue
                return  # Found matching finding

        raise AssertionError(
            f"Finding with code={code}, path={path}, severity={severity} "
            f"not found in report. Findings: {findings}"
        )

    @classmethod
    def assert_no_findings(cls, report: Dict[str, Any]) -> None:
        """Assert that the report has no findings."""
        findings = report.get("findings", [])
        if findings:
            raise AssertionError(
                f"Expected no findings, but found {len(findings)}: {findings}"
            )

    @classmethod
    def assert_valid(cls, report: Dict[str, Any]) -> None:
        """Assert that the validation result is valid."""
        if not report.get("valid", False):
            findings = report.get("findings", [])
            raise AssertionError(
                f"Expected valid=True, but got valid=False. "
                f"Findings: {findings}"
            )

    @classmethod
    def assert_invalid(cls, report: Dict[str, Any]) -> None:
        """Assert that the validation result is invalid."""
        if report.get("valid", True):
            raise AssertionError(
                f"Expected valid=False, but got valid=True"
            )

    @staticmethod
    def _remove_fields(obj: Any, fields: set) -> Any:
        """Recursively remove specified fields from a dict."""
        if isinstance(obj, dict):
            return {
                k: ValidationReportComparator._remove_fields(v, fields)
                for k, v in obj.items()
                if k not in fields
            }
        elif isinstance(obj, list):
            return [
                ValidationReportComparator._remove_fields(item, fields)
                for item in obj
            ]
        return obj

    @staticmethod
    def _sort_findings(obj: Any) -> Any:
        """Sort findings list by code and path for deterministic comparison."""
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k == "findings" and isinstance(v, list):
                    result[k] = sorted(
                        ValidationReportComparator._sort_findings(v),
                        key=lambda f: (f.get("code", ""), f.get("path", ""))
                    )
                else:
                    result[k] = ValidationReportComparator._sort_findings(v)
            return result
        elif isinstance(obj, list):
            return [
                ValidationReportComparator._sort_findings(item) for item in obj
            ]
        return obj

    @staticmethod
    def _get_diff(actual: Any, expected: Any) -> str:
        """Get a human-readable diff between two objects."""
        import difflib
        import json

        actual_str = json.dumps(actual, indent=2, sort_keys=True)
        expected_str = json.dumps(expected, indent=2, sort_keys=True)

        diff = difflib.unified_diff(
            expected_str.splitlines(),
            actual_str.splitlines(),
            fromfile="expected",
            tofile="actual",
            lineterm=""
        )
        return "\n".join(diff)


@pytest.fixture
def report_comparator() -> ValidationReportComparator:
    """Get the validation report comparator instance."""
    return ValidationReportComparator()


@pytest.fixture
def compare_reports(report_comparator) -> callable:
    """Fixture that returns the compare_reports method."""
    return report_comparator.compare_reports


@pytest.fixture
def assert_reports_equal(report_comparator) -> callable:
    """Fixture that returns the assert_reports_equal method."""
    return report_comparator.assert_reports_equal


@pytest.fixture
def assert_finding_exists(report_comparator) -> callable:
    """Fixture that returns the assert_finding_exists method."""
    return report_comparator.assert_finding_exists


@pytest.fixture
def assert_valid(report_comparator) -> callable:
    """Fixture that returns the assert_valid method."""
    return report_comparator.assert_valid


@pytest.fixture
def assert_invalid(report_comparator) -> callable:
    """Fixture that returns the assert_invalid method."""
    return report_comparator.assert_invalid


# =============================================================================
# Schema and Payload Builders
# =============================================================================

@pytest.fixture
def build_simple_schema():
    """
    Factory fixture for building simple test schemas.

    Example:
        >>> def test_something(build_simple_schema):
        ...     schema = build_simple_schema(
        ...         properties={"name": {"type": "string"}},
        ...         required=["name"]
        ...     )
    """
    def _build(
        properties: Optional[Dict[str, Any]] = None,
        required: Optional[List[str]] = None,
        additional_properties: bool = True,
        title: str = "Test Schema",
        description: str = "A test schema",
    ) -> Dict[str, Any]:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "urn:greenlang:test:schema",
            "title": title,
            "description": description,
            "type": "object",
            "properties": properties or {},
            "additionalProperties": additional_properties,
        }
        if required:
            schema["required"] = required
        return schema

    return _build


@pytest.fixture
def build_payload():
    """
    Factory fixture for building test payloads.

    Example:
        >>> def test_something(build_payload):
        ...     payload = build_payload(name="test", value=42)
    """
    def _build(**kwargs) -> Dict[str, Any]:
        return dict(kwargs)

    return _build


# =============================================================================
# Error Code Utilities
# =============================================================================

@pytest.fixture
def error_code_pattern():
    """Get the regex pattern for valid error codes."""
    import re
    return re.compile(r"^GLSCHEMA-[EW]\d{3}$")


@pytest.fixture
def is_valid_error_code(error_code_pattern):
    """
    Fixture that returns a function to validate error code format.

    Example:
        >>> def test_something(is_valid_error_code):
        ...     assert is_valid_error_code("GLSCHEMA-E100")
        ...     assert not is_valid_error_code("INVALID")
    """
    def _validate(code: str) -> bool:
        return bool(error_code_pattern.match(code))

    return _validate


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture
def env_vars(monkeypatch):
    """
    Fixture for setting environment variables during test.

    Returns a function that sets environment variables.
    Variables are automatically unset after the test.

    Example:
        >>> def test_something(env_vars):
        ...     env_vars(GL_SCHEMA_MAX_PAYLOAD_BYTES="2097152")
    """
    def _set(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, str(value))

    return _set


@pytest.fixture
def reset_limits(monkeypatch):
    """
    Fixture for resetting schema limits to test values.

    Sets smaller limits for faster testing.
    """
    monkeypatch.setenv("GL_SCHEMA_MAX_PAYLOAD_BYTES", "10240")  # 10KB
    monkeypatch.setenv("GL_SCHEMA_MAX_OBJECT_DEPTH", "10")
    monkeypatch.setenv("GL_SCHEMA_MAX_ARRAY_ITEMS", "100")
    monkeypatch.setenv("GL_SCHEMA_MAX_TOTAL_NODES", "1000")
    monkeypatch.setenv("GL_SCHEMA_MAX_REF_EXPANSIONS", "50")
    monkeypatch.setenv("GL_SCHEMA_MAX_FINDINGS", "20")
