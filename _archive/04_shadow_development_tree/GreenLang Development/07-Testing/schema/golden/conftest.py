# -*- coding: utf-8 -*-
"""
Pytest configuration for GL-FOUND-X-002 Golden Tests.

This module provides fixtures and configuration specific to golden tests,
building on the parent conftest.py fixtures.

Key Fixtures:
    - golden_test_registry: Mock schema registry for golden tests
    - schema_validator: Configured SchemaValidator for testing
    - validation_options: Pre-configured validation options
    - error_codes: Complete list of expected error codes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pytest
import yaml


# =============================================================================
# PATH CONSTANTS
# =============================================================================

GOLDEN_DIR = Path(__file__).parent
SCHEMAS_DIR = GOLDEN_DIR / "schemas"
PAYLOADS_DIR = GOLDEN_DIR / "payloads"
EXPECTED_DIR = GOLDEN_DIR / "expected"


# =============================================================================
# ERROR CODE DEFINITIONS
# =============================================================================

# Complete list of GL-FOUND-X-002 error codes
STRUCTURAL_ERROR_CODES: Set[str] = {
    "GLSCHEMA-E100",  # MISSING_REQUIRED
    "GLSCHEMA-E101",  # UNKNOWN_FIELD
    "GLSCHEMA-E102",  # TYPE_MISMATCH
    "GLSCHEMA-E103",  # INVALID_NULL
    "GLSCHEMA-E104",  # CONTAINER_TYPE_MISMATCH
    "GLSCHEMA-E105",  # PROPERTY_COUNT_VIOLATION
    "GLSCHEMA-E106",  # REQUIRED_PROPERTIES_MISSING
    "GLSCHEMA-E107",  # DUPLICATE_KEY
}

CONSTRAINT_ERROR_CODES: Set[str] = {
    "GLSCHEMA-E200",  # RANGE_VIOLATION
    "GLSCHEMA-E201",  # PATTERN_MISMATCH
    "GLSCHEMA-E202",  # ENUM_VIOLATION
    "GLSCHEMA-E203",  # LENGTH_VIOLATION
    "GLSCHEMA-E204",  # UNIQUE_VIOLATION
    "GLSCHEMA-E205",  # MULTIPLE_OF_VIOLATION
    "GLSCHEMA-E206",  # FORMAT_VIOLATION
    "GLSCHEMA-E207",  # CONST_VIOLATION
    "GLSCHEMA-E208",  # CONTAINS_VIOLATION
    "GLSCHEMA-E209",  # PROPERTY_NAME_VIOLATION
}

UNIT_ERROR_CODES: Set[str] = {
    "GLSCHEMA-E300",  # UNIT_MISSING
    "GLSCHEMA-E301",  # UNIT_INCOMPATIBLE
    "GLSCHEMA-E302",  # UNIT_NONCANONICAL (warning)
    "GLSCHEMA-E303",  # UNIT_UNKNOWN
}

RULE_ERROR_CODES: Set[str] = {
    "GLSCHEMA-E400",  # RULE_VIOLATION
    "GLSCHEMA-E401",  # CONDITIONAL_REQUIRED
    "GLSCHEMA-E402",  # CONSISTENCY_ERROR
    "GLSCHEMA-E403",  # DEPENDENCY_VIOLATION
    "GLSCHEMA-E404",  # CROSS_FIELD_VIOLATION
    "GLSCHEMA-E405",  # ONE_OF_VIOLATION
    "GLSCHEMA-E406",  # ANY_OF_VIOLATION
    "GLSCHEMA-E407",  # ALL_OF_VIOLATION
}

ALL_ERROR_CODES: Set[str] = (
    STRUCTURAL_ERROR_CODES
    | CONSTRAINT_ERROR_CODES
    | UNIT_ERROR_CODES
    | RULE_ERROR_CODES
)


# =============================================================================
# FIXTURES: ERROR CODES
# =============================================================================

@pytest.fixture
def structural_error_codes() -> Set[str]:
    """Get all structural error codes (E1xx)."""
    return STRUCTURAL_ERROR_CODES.copy()


@pytest.fixture
def constraint_error_codes() -> Set[str]:
    """Get all constraint error codes (E2xx)."""
    return CONSTRAINT_ERROR_CODES.copy()


@pytest.fixture
def unit_error_codes() -> Set[str]:
    """Get all unit error codes (E3xx)."""
    return UNIT_ERROR_CODES.copy()


@pytest.fixture
def rule_error_codes() -> Set[str]:
    """Get all rule error codes (E4xx)."""
    return RULE_ERROR_CODES.copy()


@pytest.fixture
def all_error_codes() -> Set[str]:
    """Get all GL-FOUND-X-002 error codes."""
    return ALL_ERROR_CODES.copy()


# =============================================================================
# FIXTURES: FILE LOADING
# =============================================================================

@pytest.fixture
def load_schema():
    """
    Factory fixture for loading golden test schemas.

    Returns:
        Function that loads a schema from schemas/ directory
    """
    def _load(schema_path: str) -> Dict[str, Any]:
        full_path = SCHEMAS_DIR / schema_path
        if not full_path.exists():
            raise FileNotFoundError(f"Schema not found: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            if full_path.suffix in (".yaml", ".yml"):
                return yaml.safe_load(f)
            return json.load(f)

    return _load


@pytest.fixture
def load_valid_payload():
    """
    Factory fixture for loading valid test payloads.

    Returns:
        Function that loads a payload from payloads/valid/ directory
    """
    def _load(payload_name: str) -> Dict[str, Any]:
        full_path = PAYLOADS_DIR / "valid" / payload_name
        if not full_path.exists():
            # Try with .yaml extension
            full_path = PAYLOADS_DIR / "valid" / f"{payload_name}.yaml"

        if not full_path.exists():
            raise FileNotFoundError(f"Valid payload not found: {payload_name}")

        with open(full_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    return _load


@pytest.fixture
def load_invalid_payload():
    """
    Factory fixture for loading invalid test payloads.

    Returns:
        Function that loads a payload from payloads/invalid/ directory
    """
    def _load(payload_name: str) -> Dict[str, Any]:
        full_path = PAYLOADS_DIR / "invalid" / payload_name
        if not full_path.exists():
            full_path = PAYLOADS_DIR / "invalid" / f"{payload_name}.yaml"

        if not full_path.exists():
            raise FileNotFoundError(f"Invalid payload not found: {payload_name}")

        with open(full_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    return _load


@pytest.fixture
def load_expected():
    """
    Factory fixture for loading expected validation reports.

    Returns:
        Function that loads expected output from expected/ directory
    """
    def _load(expected_name: str) -> Dict[str, Any]:
        full_path = EXPECTED_DIR / expected_name
        if not full_path.exists():
            full_path = EXPECTED_DIR / f"{expected_name}.json"

        if not full_path.exists():
            raise FileNotFoundError(f"Expected report not found: {expected_name}")

        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return _load


# =============================================================================
# FIXTURES: PAYLOAD DISCOVERY
# =============================================================================

@pytest.fixture
def all_schemas() -> List[Path]:
    """Get all schema files in the golden test directory."""
    schemas = []
    for pattern in ["**/*.yaml", "**/*.yml", "**/*.json"]:
        schemas.extend(SCHEMAS_DIR.glob(pattern))
    return [s for s in schemas if s.name != "__init__.py"]


@pytest.fixture
def valid_payloads() -> List[Path]:
    """Get all valid payload files."""
    valid_dir = PAYLOADS_DIR / "valid"
    payloads = []
    for pattern in ["*.yaml", "*.yml", "*.json"]:
        payloads.extend(valid_dir.glob(pattern))
    return [p for p in payloads if not p.name.startswith("_")]


@pytest.fixture
def invalid_payloads() -> List[Path]:
    """Get all invalid payload files."""
    invalid_dir = PAYLOADS_DIR / "invalid"
    payloads = []
    for pattern in ["*.yaml", "*.yml", "*.json"]:
        payloads.extend(invalid_dir.glob(pattern))
    return [p for p in payloads if not p.name.startswith("_")]


@pytest.fixture
def expected_reports() -> List[Path]:
    """Get all expected report files."""
    reports = []
    for pattern in ["*.json"]:
        reports.extend(EXPECTED_DIR.glob(pattern))
    return reports


# =============================================================================
# FIXTURES: VALIDATION OPTIONS
# =============================================================================

@pytest.fixture
def strict_options() -> Dict[str, Any]:
    """Strict validation options for golden tests."""
    return {
        "profile": "strict",
        "normalize": True,
        "emit_patches": True,
        "max_errors": 100,
        "fail_fast": False,
        "unknown_field_policy": "error",
        "coercion_policy": "off",
    }


@pytest.fixture
def standard_options() -> Dict[str, Any]:
    """Standard validation options for golden tests."""
    return {
        "profile": "standard",
        "normalize": True,
        "emit_patches": True,
        "max_errors": 100,
        "fail_fast": False,
        "unknown_field_policy": "warn",
        "coercion_policy": "safe",
    }


@pytest.fixture
def permissive_options() -> Dict[str, Any]:
    """Permissive validation options for golden tests."""
    return {
        "profile": "permissive",
        "normalize": True,
        "emit_patches": True,
        "max_errors": 1000,
        "fail_fast": False,
        "unknown_field_policy": "ignore",
        "coercion_policy": "aggressive",
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract test metadata from a payload.

    Args:
        payload: Payload dict that may contain _test_metadata

    Returns:
        Metadata dict (empty if not present)
    """
    return payload.pop("_test_metadata", {})


def get_expected_errors(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get expected errors from test metadata.

    Args:
        metadata: Test metadata dict

    Returns:
        List of expected error specifications
    """
    return metadata.get("expected_errors", [])


def schema_name_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
    """
    Get schema name from test metadata.

    Args:
        metadata: Test metadata dict

    Returns:
        Schema path string or None
    """
    return metadata.get("schema")


@pytest.fixture
def extract_test_metadata():
    """Fixture that provides the extract_metadata function."""
    return extract_metadata


@pytest.fixture
def get_expected_error_codes():
    """Fixture that provides get_expected_errors function."""
    return get_expected_errors
