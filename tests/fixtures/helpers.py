# -*- coding: utf-8 -*-
"""
Shared test assertion helpers for GreenLang tests.

These are plain functions (not pytest fixtures) that can be imported
in ``conftest.py`` files **or** directly in test modules.  Each helper
raises ``AssertionError`` with a descriptive message on failure so that
pytest displays a clear diff.

Usage::

    from tests.fixtures.helpers import (
        assert_valid_provenance_hash,
        assert_decimal_close,
        assert_valid_agent_response,
    )

    def test_calculation(result):
        assert_valid_provenance_hash(result["provenance_hash"])
        assert_decimal_close(result["total"], Decimal("42.123456"))
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from tests.fixtures.constants import (
    DEFAULT_DECIMAL_PLACES,
    DEFAULT_FLOAT_ABS_TOL,
    DEFAULT_FLOAT_REL_TOL,
    EMISSIONS_DECIMAL_PLACES,
)


# =============================================================================
# Provenance / Hash Assertions
# =============================================================================

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def assert_valid_provenance_hash(hash_value: str) -> None:
    """
    Assert that *hash_value* is a valid SHA-256 hex digest.

    A valid hash is exactly 64 lowercase hexadecimal characters.

    Args:
        hash_value: The string to validate.

    Raises:
        AssertionError: If the value is not a valid SHA-256 hex string.
    """
    assert isinstance(hash_value, str), (
        f"Provenance hash must be str, got {type(hash_value).__name__}"
    )
    assert _SHA256_PATTERN.match(hash_value), (
        f"Invalid SHA-256 hash: expected 64 hex chars, got "
        f"{len(hash_value)} chars: {hash_value!r}"
    )


def assert_valid_md5_hash(hash_value: str) -> None:
    """
    Assert that *hash_value* is a valid MD5 hex digest (32 hex chars).

    Args:
        hash_value: The string to validate.

    Raises:
        AssertionError: If the value is not a valid MD5 hex string.
    """
    assert isinstance(hash_value, str), (
        f"MD5 hash must be str, got {type(hash_value).__name__}"
    )
    assert re.match(r"^[0-9a-f]{32}$", hash_value), (
        f"Invalid MD5 hash: expected 32 hex chars, got "
        f"{len(hash_value)} chars: {hash_value!r}"
    )


# =============================================================================
# Numeric Assertions
# =============================================================================


def assert_decimal_close(
    actual: Union[Decimal, float, str],
    expected: Union[Decimal, float, str],
    places: int = DEFAULT_DECIMAL_PLACES,
) -> None:
    """
    Assert two ``Decimal`` values are equal to *places* decimal places.

    Both *actual* and *expected* are coerced to ``Decimal`` if needed.

    Args:
        actual: Computed value.
        expected: Expected value.
        places: Number of decimal places for comparison.

    Raises:
        AssertionError: If values differ beyond the specified precision.
    """
    try:
        a = Decimal(str(actual))
        e = Decimal(str(expected))
    except (InvalidOperation, ValueError) as exc:
        raise AssertionError(
            f"Cannot convert to Decimal: actual={actual!r}, expected={expected!r}"
        ) from exc

    tolerance = Decimal(10) ** -places
    diff = abs(a - e)
    assert diff <= tolerance, (
        f"Decimal values differ beyond {places} places: "
        f"actual={a}, expected={e}, diff={diff}, tolerance={tolerance}"
    )


def assert_float_close(
    actual: float,
    expected: float,
    rel_tol: float = DEFAULT_FLOAT_REL_TOL,
    abs_tol: float = DEFAULT_FLOAT_ABS_TOL,
) -> None:
    """
    Assert two floats are close using both relative and absolute tolerance.

    Mirrors :func:`math.isclose` semantics but raises ``AssertionError``
    with a diagnostic message.

    Args:
        actual: Computed float value.
        expected: Expected float value.
        rel_tol: Maximum allowed relative difference.
        abs_tol: Maximum allowed absolute difference.

    Raises:
        AssertionError: If the values are not close.
    """
    import math

    assert math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol), (
        f"Floats not close: actual={actual}, expected={expected}, "
        f"rel_tol={rel_tol}, abs_tol={abs_tol}, diff={abs(actual - expected)}"
    )


def assert_positive(value: Union[int, float, Decimal], label: str = "value") -> None:
    """Assert that *value* is strictly positive (> 0)."""
    assert value > 0, f"Expected positive {label}, got {value}"


def assert_non_negative(value: Union[int, float, Decimal], label: str = "value") -> None:
    """Assert that *value* is non-negative (>= 0)."""
    assert value >= 0, f"Expected non-negative {label}, got {value}"


# =============================================================================
# Agent Response Contract Assertions
# =============================================================================

_REQUIRED_AGENT_RESPONSE_KEYS: Set[str] = {
    "status",
    "result",
    "provenance_hash",
}

_VALID_AGENT_STATUSES: Set[str] = {"success", "error", "partial", "pending"}


def assert_valid_agent_response(
    response: Dict[str, Any],
    agent_name: str = "",
    required_keys: Optional[Set[str]] = None,
) -> None:
    """
    Assert an agent response follows the GreenLang response contract.

    Required top-level keys: ``status``, ``result``, ``provenance_hash``.
    Valid statuses: ``success``, ``error``, ``partial``, ``pending``.

    Args:
        response: The agent response dict.
        agent_name: Optional agent name for clearer error messages.
        required_keys: Override the default required keys set.

    Raises:
        AssertionError: If the response violates the contract.
    """
    prefix = f"[{agent_name}] " if agent_name else ""
    assert isinstance(response, dict), (
        f"{prefix}Agent response must be dict, got {type(response).__name__}"
    )

    keys = required_keys or _REQUIRED_AGENT_RESPONSE_KEYS
    missing = keys - response.keys()
    assert not missing, (
        f"{prefix}Agent response missing required keys: {missing}. "
        f"Got keys: {set(response.keys())}"
    )

    status = response["status"]
    assert status in _VALID_AGENT_STATUSES, (
        f"{prefix}Invalid agent status: {status!r}. "
        f"Must be one of {_VALID_AGENT_STATUSES}"
    )

    if "provenance_hash" in response and response["provenance_hash"]:
        assert_valid_provenance_hash(response["provenance_hash"])

    if "processing_time_ms" in response:
        assert response["processing_time_ms"] >= 0, (
            f"{prefix}processing_time_ms must be >= 0, "
            f"got {response['processing_time_ms']}"
        )


# =============================================================================
# Emissions Result Assertions
# =============================================================================

_REQUIRED_EMISSIONS_KEYS: Set[str] = {
    "total_emissions",
    "unit",
}


def assert_emissions_result(
    result: Dict[str, Any],
    expected_scope: Optional[str] = None,
    expected_unit: str = "tCO2e",
) -> None:
    """
    Assert an emissions calculation result has the required fields
    and valid values.

    Args:
        result: The emissions result dict.
        expected_scope: If provided, assert ``scope`` matches
            (e.g. ``"scope_1"``, ``"scope_2"``, ``"scope_3"``).
        expected_unit: Expected unit string (default ``"tCO2e"``).

    Raises:
        AssertionError: If the result is invalid.
    """
    assert isinstance(result, dict), (
        f"Emissions result must be dict, got {type(result).__name__}"
    )

    missing = _REQUIRED_EMISSIONS_KEYS - result.keys()
    assert not missing, (
        f"Emissions result missing keys: {missing}. Got: {set(result.keys())}"
    )

    total = result["total_emissions"]
    assert isinstance(total, (int, float, Decimal)), (
        f"total_emissions must be numeric, got {type(total).__name__}"
    )
    assert total >= 0, f"total_emissions must be >= 0, got {total}"

    assert result["unit"] == expected_unit, (
        f"Expected unit={expected_unit!r}, got {result['unit']!r}"
    )

    if expected_scope is not None:
        assert "scope" in result, "Emissions result missing 'scope' key"
        assert result["scope"] == expected_scope, (
            f"Expected scope={expected_scope!r}, got {result['scope']!r}"
        )

    # If breakdown is present, validate it sums correctly
    if "breakdown" in result and isinstance(result["breakdown"], dict):
        breakdown_total = sum(
            v for v in result["breakdown"].values()
            if isinstance(v, (int, float, Decimal))
        )
        assert_decimal_close(
            breakdown_total,
            total,
            places=EMISSIONS_DECIMAL_PLACES,
        )


# =============================================================================
# Timestamp Assertions
# =============================================================================

_ISO8601_PATTERNS = [
    # Full datetime with timezone
    re.compile(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})$"
    ),
    # Full datetime without timezone
    re.compile(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?$"
    ),
    # Date only
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
]


def assert_valid_iso_timestamp(timestamp_str: str) -> None:
    """
    Assert that *timestamp_str* is a valid ISO 8601 timestamp.

    Accepts date-only (``YYYY-MM-DD``), datetime with or without
    timezone, and fractional seconds.

    Args:
        timestamp_str: The timestamp string to validate.

    Raises:
        AssertionError: If the string is not valid ISO 8601.
    """
    assert isinstance(timestamp_str, str), (
        f"Timestamp must be str, got {type(timestamp_str).__name__}"
    )

    matched = any(p.match(timestamp_str) for p in _ISO8601_PATTERNS)
    assert matched, (
        f"Not a valid ISO 8601 timestamp: {timestamp_str!r}"
    )

    # Verify it actually parses (catches impossible dates like 2024-02-30)
    try:
        # Strip trailing Z for fromisoformat compatibility
        parseable = timestamp_str.replace("Z", "+00:00")
        datetime.fromisoformat(parseable)
    except ValueError as exc:
        raise AssertionError(
            f"ISO 8601 string does not parse as a real date: "
            f"{timestamp_str!r} ({exc})"
        ) from exc


# =============================================================================
# PII Detection Assertions
# =============================================================================

# Pre-compiled patterns for common PII types
_PII_PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone_us": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "phone_intl": re.compile(r"\+\d{1,3}[-.\s]?\d{4,14}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    "ip_address": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
}

# Known safe values that should not trigger PII detection
_PII_SAFE_PATTERNS: Set[str] = {
    "0.0.0.0",
    "127.0.0.1",
    "localhost",
}


def assert_no_pii(
    text: str,
    fields: Optional[Sequence[str]] = None,
    allow_list: Optional[Set[str]] = None,
) -> None:
    """
    Assert that *text* does not contain common PII patterns.

    Args:
        text: The text to scan.
        fields: Specific PII types to check. If ``None``, checks all
            known types: ``email``, ``phone_us``, ``phone_intl``,
            ``ssn``, ``credit_card``, ``ip_address``.
        allow_list: Set of strings that are known-safe and should not
            trigger a PII match (e.g. ``{"127.0.0.1"}``).

    Raises:
        AssertionError: If any PII pattern is found.
    """
    assert isinstance(text, str), (
        f"text must be str, got {type(text).__name__}"
    )

    check_fields = fields or list(_PII_PATTERNS.keys())
    safe = (allow_list or set()) | _PII_SAFE_PATTERNS

    for field_name in check_fields:
        pattern = _PII_PATTERNS.get(field_name)
        if pattern is None:
            continue

        matches = pattern.findall(text)
        # Filter out known-safe values
        real_matches = [m for m in matches if m not in safe]

        assert not real_matches, (
            f"PII detected ({field_name}): found {real_matches} in text. "
            f"If these are expected test values, add them to allow_list."
        )


# =============================================================================
# Data Structure Assertions
# =============================================================================


def assert_dict_has_keys(
    data: Dict[str, Any],
    required_keys: Set[str],
    label: str = "dict",
) -> None:
    """
    Assert that *data* contains all *required_keys*.

    Args:
        data: The dictionary to check.
        required_keys: Set of keys that must be present.
        label: Label for error messages.

    Raises:
        AssertionError: If any key is missing.
    """
    assert isinstance(data, dict), (
        f"{label} must be dict, got {type(data).__name__}"
    )
    missing = required_keys - data.keys()
    assert not missing, (
        f"{label} missing required keys: {missing}. Got: {set(data.keys())}"
    )


def assert_list_of_dicts(
    data: Any,
    min_length: int = 0,
    required_keys: Optional[Set[str]] = None,
    label: str = "list",
) -> None:
    """
    Assert that *data* is a list of dicts, optionally with minimum
    length and required keys per element.

    Args:
        data: The value to validate.
        min_length: Minimum number of elements.
        required_keys: Keys each dict must contain.
        label: Label for error messages.

    Raises:
        AssertionError: On any validation failure.
    """
    assert isinstance(data, list), (
        f"{label} must be list, got {type(data).__name__}"
    )
    assert len(data) >= min_length, (
        f"{label} must have >= {min_length} elements, got {len(data)}"
    )
    for i, item in enumerate(data):
        assert isinstance(item, dict), (
            f"{label}[{i}] must be dict, got {type(item).__name__}"
        )
        if required_keys:
            missing = required_keys - item.keys()
            assert not missing, (
                f"{label}[{i}] missing keys: {missing}"
            )


# =============================================================================
# Audit Trail Assertions
# =============================================================================


def assert_valid_audit_entry(entry: Dict[str, Any]) -> None:
    """
    Assert that *entry* is a valid GreenLang audit trail entry.

    Required keys: ``timestamp``, ``action``, ``actor``, ``provenance_hash``.

    Args:
        entry: The audit entry dict.

    Raises:
        AssertionError: If the entry is invalid.
    """
    required = {"timestamp", "action", "actor", "provenance_hash"}
    assert_dict_has_keys(entry, required, label="audit_entry")
    assert_valid_iso_timestamp(entry["timestamp"])
    assert_valid_provenance_hash(entry["provenance_hash"])
    assert isinstance(entry["action"], str) and entry["action"], (
        "audit_entry.action must be a non-empty string"
    )
    assert isinstance(entry["actor"], str) and entry["actor"], (
        "audit_entry.actor must be a non-empty string"
    )
