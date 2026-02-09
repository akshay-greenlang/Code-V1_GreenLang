# -*- coding: utf-8 -*-
"""
Validity Checker Engine - AGENT-DATA-010: Data Quality Profiler (GL-DATA-X-013)

Format validation and type conformance checking across datasets. Supports
20+ format validators (email, phone, URL, IP, UUID, dates, currency,
latitude/longitude, zip codes, credit card with Luhn check), custom regex
rules, range checks, domain checks, and cross-field constraint validation.

Zero-Hallucination Guarantees:
    - All format checks use deterministic regex matching
    - Luhn algorithm for credit card validation is deterministic arithmetic
    - No ML/LLM calls in the validation path
    - SHA-256 provenance on every validation mutation
    - Thread-safe in-memory storage

Supported Format Validators (20+):
    email, phone, url, ipv4, ipv6, uuid, date_iso, date_us, date_eu,
    datetime_iso, currency, percentage, zip_code_us, zip_code_uk,
    country_code_iso2, country_code_iso3, latitude, longitude,
    hex_color, credit_card

Example:
    >>> from greenlang.data_quality_profiler.validity_checker import ValidityChecker
    >>> checker = ValidityChecker()
    >>> data = [
    ...     {"email": "alice@example.com", "age": 30},
    ...     {"email": "invalid-email", "age": -5},
    ... ]
    >>> rules = [{"column": "email", "format_type": "email"}]
    >>> result = checker.validate(data, rules=rules)
    >>> print(result["validity_score"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-010 Data Quality Profiler (GL-DATA-X-013)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

__all__ = [
    "ValidityChecker",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "VLD") -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        String of the form ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash for a validity operation.

    Args:
        operation: Name of the operation.
        data_repr: Serialised representation of the data involved.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Format Regex Patterns
# ---------------------------------------------------------------------------

_RE_EMAIL = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)
_RE_PHONE = re.compile(
    r"^\+?[1-9]\d{0,2}[\s\-.]?\(?\d{1,4}\)?[\s\-.]?\d{1,4}[\s\-.]?\d{1,9}$"
)
_RE_URL = re.compile(
    r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE
)
_RE_IPV4 = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)
_RE_IPV6 = re.compile(
    r"^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$"
)
_RE_IPV6_COMPRESSED = re.compile(
    r"^(?:[0-9a-fA-F]{1,4}:)*:(?::[0-9a-fA-F]{1,4})*$"
)
_RE_UUID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_RE_DATE_ISO = re.compile(
    r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$"
)
_RE_DATE_US = re.compile(
    r"^(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/\d{4}$"
)
_RE_DATE_EU = re.compile(
    r"^(?:0[1-9]|[12]\d|3[01])/(?:0[1-9]|1[0-2])/\d{4}$"
)
_RE_DATETIME_ISO = re.compile(
    r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+\-]\d{2}:?\d{2})?$"
)
_RE_CURRENCY = re.compile(
    r"^[\$\u20ac\u00a3\u00a5]\s?[\d,]+\.?\d*$"
)
_RE_PERCENTAGE = re.compile(
    r"^-?\d+\.?\d*\s*%$"
)
_RE_ZIP_US = re.compile(
    r"^\d{5}(-\d{4})?$"
)
_RE_ZIP_UK = re.compile(
    r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$", re.IGNORECASE
)
_RE_COUNTRY_ISO2 = re.compile(
    r"^[A-Z]{2}$"
)
_RE_COUNTRY_ISO3 = re.compile(
    r"^[A-Z]{3}$"
)
_RE_LATITUDE = re.compile(
    r"^-?(?:90(?:\.0+)?|[1-8]?\d(?:\.\d+)?)$"
)
_RE_LONGITUDE = re.compile(
    r"^-?(?:180(?:\.0+)?|1[0-7]\d(?:\.\d+)?|\d{1,2}(?:\.\d+)?)$"
)
_RE_HEX_COLOR = re.compile(
    r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$"
)
_RE_CREDIT_CARD = re.compile(
    r"^\d{13,19}$"
)

# Mapping format_type -> regex
_FORMAT_REGEX_MAP: Dict[str, re.Pattern[str]] = {
    "email": _RE_EMAIL,
    "phone": _RE_PHONE,
    "url": _RE_URL,
    "ipv4": _RE_IPV4,
    "ipv6": _RE_IPV6,
    "uuid": _RE_UUID,
    "date_iso": _RE_DATE_ISO,
    "date_us": _RE_DATE_US,
    "date_eu": _RE_DATE_EU,
    "datetime_iso": _RE_DATETIME_ISO,
    "currency": _RE_CURRENCY,
    "percentage": _RE_PERCENTAGE,
    "zip_code_us": _RE_ZIP_US,
    "zip_code_uk": _RE_ZIP_UK,
    "country_code_iso2": _RE_COUNTRY_ISO2,
    "country_code_iso3": _RE_COUNTRY_ISO3,
    "latitude": _RE_LATITUDE,
    "longitude": _RE_LONGITUDE,
    "hex_color": _RE_HEX_COLOR,
}

# Severity levels
SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"
SEVERITY_INFO = "info"

# Supported operators
OPERATOR_EQUALS = "EQUALS"
OPERATOR_NOT_EQUALS = "NOT_EQUALS"
OPERATOR_GREATER_THAN = "GREATER_THAN"
OPERATOR_LESS_THAN = "LESS_THAN"
OPERATOR_BETWEEN = "BETWEEN"
OPERATOR_MATCHES = "MATCHES"
OPERATOR_CONTAINS = "CONTAINS"
OPERATOR_IN_SET = "IN_SET"

ALL_OPERATORS = frozenset({
    OPERATOR_EQUALS, OPERATOR_NOT_EQUALS, OPERATOR_GREATER_THAN,
    OPERATOR_LESS_THAN, OPERATOR_BETWEEN, OPERATOR_MATCHES,
    OPERATOR_CONTAINS, OPERATOR_IN_SET,
})


# ---------------------------------------------------------------------------
# Luhn Algorithm (credit card)
# ---------------------------------------------------------------------------


def _luhn_check(number_str: str) -> bool:
    """Validate a number string using the Luhn algorithm.

    Args:
        number_str: Numeric string to validate.

    Returns:
        True if the number passes the Luhn check.
    """
    digits = [int(d) for d in number_str if d.isdigit()]
    if len(digits) < 2:
        return False

    # Reverse digits
    digits = digits[::-1]
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d

    return total % 10 == 0


# ---------------------------------------------------------------------------
# Type Conformance Helpers
# ---------------------------------------------------------------------------

_TYPE_CHECKERS: Dict[str, Callable[[Any], bool]] = {}


def _is_integer(value: Any) -> bool:
    """Check if a value is or can be parsed as an integer."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    try:
        int(str(value).strip())
        return True
    except (ValueError, TypeError):
        return False


def _is_float(value: Any) -> bool:
    """Check if a value is or can be parsed as a float."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    try:
        float(str(value).strip())
        return True
    except (ValueError, TypeError):
        return False


def _is_boolean(value: Any) -> bool:
    """Check if a value is or represents a boolean."""
    if isinstance(value, bool):
        return True
    s = str(value).strip().lower()
    return s in {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n", "on", "off"}


def _is_string(value: Any) -> bool:
    """Check if a value is a non-empty string."""
    return isinstance(value, str) and len(value.strip()) > 0


_TYPE_CHECKERS = {
    "integer": _is_integer,
    "float": _is_float,
    "boolean": _is_boolean,
    "string": _is_string,
    "email": lambda v: bool(_RE_EMAIL.match(str(v).strip())),
    "url": lambda v: bool(_RE_URL.match(str(v).strip())),
    "uuid": lambda v: bool(_RE_UUID.match(str(v).strip())),
    "ipv4": lambda v: bool(_RE_IPV4.match(str(v).strip())),
    "date": lambda v: bool(_RE_DATE_ISO.match(str(v).strip())),
    "datetime": lambda v: bool(_RE_DATETIME_ISO.match(str(v).strip())),
    "phone": lambda v: bool(_RE_PHONE.match(str(v).strip())) and len(str(v).strip()) >= 7,
}


# ---------------------------------------------------------------------------
# ValidityChecker Engine
# ---------------------------------------------------------------------------


class ValidityChecker:
    """Format validation and type conformance engine.

    Validates dataset values against 20+ format types, range constraints,
    domain lists, regex patterns, and cross-field constraints. Computes
    per-column and overall validity scores with severity-classified issues.

    Thread-safe: all mutations to internal storage are protected by
    a threading lock. SHA-256 provenance hashes on every validation.

    Attributes:
        _config: Configuration dictionary.
        _lock: Threading lock for thread-safe storage access.
        _validations: In-memory storage of completed validations.
        _stats: Aggregate validation statistics.

    Example:
        >>> checker = ValidityChecker()
        >>> data = [{"email": "a@b.com"}, {"email": "bad"}]
        >>> rules = [{"column": "email", "format_type": "email"}]
        >>> result = checker.validate(data, rules=rules)
        >>> assert 0.0 <= result["validity_score"] <= 1.0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ValidityChecker.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``strict_mode``: bool, fail on first error (default False)
                - ``max_issues``: int, max issues to report (default 1000)
                - ``custom_formats``: dict of name -> regex pattern string
        """
        self._config = config or {}
        self._strict_mode: bool = self._config.get("strict_mode", False)
        self._max_issues: int = self._config.get("max_issues", 1000)
        self._custom_formats: Dict[str, re.Pattern[str]] = {}
        for name, pattern_str in self._config.get("custom_formats", {}).items():
            self._custom_formats[name] = re.compile(pattern_str)
        self._lock = threading.Lock()
        self._validations: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, Any] = {
            "validations_completed": 0,
            "total_rows_validated": 0,
            "total_violations": 0,
            "total_validation_time_ms": 0.0,
        }
        logger.info(
            "ValidityChecker initialized: strict=%s, max_issues=%d, custom_formats=%d",
            self._strict_mode, self._max_issues, len(self._custom_formats),
        )

    # ------------------------------------------------------------------
    # Public API - Full Dataset Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Validate dataset against format and type rules.

        Args:
            data: List of row dictionaries to validate.
            columns: Optional subset of columns to validate.
            rules: Optional list of validation rule dicts. Each rule has:
                - column: str (column name)
                - format_type: str (email, phone, etc.) [optional]
                - expected_type: str (integer, float, etc.) [optional]
                - min_val / max_val: numeric range [optional]
                - pattern: regex string [optional]
                - allowed_values: list [optional]
                - cross_field: dict with constraint [optional]

        Returns:
            Validity assessment dict with: validation_id, validity_score,
            column_validity, violations, issues, provenance_hash.

        Raises:
            ValueError: If data is empty.
        """
        start = time.monotonic()
        if not data:
            raise ValueError("Cannot validate empty dataset")

        validation_id = _generate_id("VLD")
        all_keys = columns if columns else list(data[0].keys())
        rule_list = rules or []

        # Per-column validity
        column_results: Dict[str, Dict[str, Any]] = {}
        all_violations: List[Dict[str, Any]] = []

        # Validate each rule
        for rule in rule_list:
            col = rule.get("column", "")
            if col and col not in all_keys:
                continue

            col_values = [row.get(col) for row in data]
            col_result = self.validate_column(
                values=col_values,
                column_name=col,
                expected_type=rule.get("expected_type"),
                format_pattern=rule.get("format_type"),
                min_val=rule.get("min_val"),
                max_val=rule.get("max_val"),
                regex_pattern=rule.get("pattern"),
                allowed_values=rule.get("allowed_values"),
            )
            column_results[col] = col_result

            # Collect violations
            for violation in col_result.get("violations", []):
                if len(all_violations) < self._max_issues:
                    all_violations.append(violation)

        # Cross-field constraints
        cross_violations = self._evaluate_cross_field_rules(data, rule_list)
        for cv in cross_violations:
            if len(all_violations) < self._max_issues:
                all_violations.append(cv)

        # Overall validity score
        validity_score = self.compute_validity_score(data, rule_list, all_keys)

        # Generate issues
        issues = self.generate_validity_issues(
            column_results, all_violations, validity_score
        )

        # Provenance
        provenance_data = json.dumps({
            "validation_id": validation_id,
            "row_count": len(data),
            "rule_count": len(rule_list),
            "validity_score": validity_score,
            "violation_count": len(all_violations),
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("validate", provenance_data)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        result: Dict[str, Any] = {
            "validation_id": validation_id,
            "validity_score": round(validity_score, 4),
            "row_count": len(data),
            "column_count": len(all_keys),
            "rule_count": len(rule_list),
            "violation_count": len(all_violations),
            "column_validity": column_results,
            "violations": all_violations,
            "cross_field_violations": cross_violations,
            "issues": issues,
            "issue_count": len(issues),
            "provenance_hash": provenance_hash,
            "validation_time_ms": round(elapsed_ms, 2),
            "created_at": _utcnow().isoformat(),
        }

        with self._lock:
            self._validations[validation_id] = result
            self._stats["validations_completed"] += 1
            self._stats["total_rows_validated"] += len(data)
            self._stats["total_violations"] += len(all_violations)
            self._stats["total_validation_time_ms"] += elapsed_ms

        logger.info(
            "Validity check: id=%s, score=%.4f, violations=%d, time=%.1fms",
            validation_id, validity_score, len(all_violations), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Per-Column Validation
    # ------------------------------------------------------------------

    def validate_column(
        self,
        values: List[Any],
        column_name: str,
        expected_type: Optional[str] = None,
        format_pattern: Optional[str] = None,
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None,
        regex_pattern: Optional[str] = None,
        allowed_values: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Validate a single column against specified rules.

        Args:
            values: List of values for this column.
            column_name: Name of the column.
            expected_type: Expected data type for conformance check.
            format_pattern: Format type name (email, phone, etc.).
            min_val: Minimum acceptable value (for range check).
            max_val: Maximum acceptable value (for range check).
            regex_pattern: Custom regex pattern string.
            allowed_values: List of allowed values (domain check).

        Returns:
            Dict with: column_name, total_count, valid_count, invalid_count,
            validity_rate, violations, conformance_ratio.
        """
        total = len(values)
        violations: List[Dict[str, Any]] = []
        valid_count = 0

        for idx, value in enumerate(values):
            if value is None or (isinstance(value, str) and value.strip() == ""):
                # Null/empty values are not validity violations
                # (completeness is handled separately)
                valid_count += 1
                continue

            is_valid = True
            reasons: List[str] = []

            # Type conformance check
            if expected_type:
                if not self.check_type_conformance_single(value, expected_type):
                    is_valid = False
                    reasons.append(f"type_mismatch: expected {expected_type}")

            # Format check
            if format_pattern:
                if not self.check_format(value, format_pattern):
                    is_valid = False
                    reasons.append(f"format_invalid: expected {format_pattern}")

            # Range check
            if min_val is not None or max_val is not None:
                if not self.check_range(value, min_val, max_val):
                    is_valid = False
                    reasons.append(
                        f"out_of_range: [{min_val}, {max_val}]"
                    )

            # Regex check
            if regex_pattern:
                if not self.check_regex(str(value), regex_pattern):
                    is_valid = False
                    reasons.append(f"regex_mismatch: {regex_pattern}")

            # Domain check
            if allowed_values is not None:
                if not self.check_domain(value, allowed_values):
                    is_valid = False
                    reasons.append("not_in_allowed_values")

            if is_valid:
                valid_count += 1
            else:
                violations.append({
                    "row_index": idx,
                    "column": column_name,
                    "value": str(value)[:200],
                    "reasons": reasons,
                })

        invalid_count = total - valid_count
        validity_rate = valid_count / total if total > 0 else 1.0

        # Conformance ratio (if expected_type provided)
        conformance_ratio = 1.0
        if expected_type:
            conformance_ratio = self.check_type_conformance(values, expected_type)

        provenance_data = json.dumps({
            "column_name": column_name,
            "total": total,
            "valid_count": valid_count,
            "format_pattern": format_pattern,
            "expected_type": expected_type,
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("validate_column", provenance_data)

        return {
            "column_name": column_name,
            "total_count": total,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "validity_rate": round(validity_rate, 4),
            "conformance_ratio": round(conformance_ratio, 4),
            "violations": violations[:self._max_issues],
            "violation_count": len(violations),
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # Type Conformance
    # ------------------------------------------------------------------

    def check_type_conformance(
        self,
        values: List[Any],
        expected_type: str,
    ) -> float:
        """Check what fraction of values conform to the expected type.

        Args:
            values: List of values to check.
            expected_type: Expected type name (integer, float, etc.).

        Returns:
            Conformance ratio (0.0 to 1.0).
        """
        if not values:
            return 1.0

        checker = _TYPE_CHECKERS.get(expected_type)
        if not checker:
            logger.warning("Unknown type '%s' for conformance check", expected_type)
            return 1.0

        non_null = [v for v in values if v is not None]
        if not non_null:
            return 1.0

        conforms = sum(1 for v in non_null if checker(v))
        return conforms / len(non_null)

    def check_type_conformance_single(
        self,
        value: Any,
        expected_type: str,
    ) -> bool:
        """Check if a single value conforms to the expected type.

        Args:
            value: Value to check.
            expected_type: Expected type name.

        Returns:
            True if value conforms.
        """
        checker = _TYPE_CHECKERS.get(expected_type)
        if not checker:
            return True
        return checker(value)

    # ------------------------------------------------------------------
    # Format Checking
    # ------------------------------------------------------------------

    def check_format(self, value: Any, format_type: str) -> bool:
        """Check if a value matches the specified format type.

        Supports 20+ built-in formats plus custom formats registered
        via configuration.

        Args:
            value: Value to validate.
            format_type: Format type name (email, phone, url, etc.).

        Returns:
            True if the value matches the format.
        """
        if value is None:
            return False

        s = str(value).strip()
        if not s:
            return False

        # Special handling for credit card (Luhn check)
        if format_type == "credit_card":
            if not _RE_CREDIT_CARD.match(s):
                return False
            return _luhn_check(s)

        # Check built-in formats
        regex = _FORMAT_REGEX_MAP.get(format_type)
        if regex:
            return bool(regex.match(s))

        # Check custom formats
        custom_regex = self._custom_formats.get(format_type)
        if custom_regex:
            return bool(custom_regex.match(s))

        # IPv6 compressed format
        if format_type == "ipv6":
            return bool(_RE_IPV6.match(s) or _RE_IPV6_COMPRESSED.match(s))

        logger.warning("Unknown format type: %s", format_type)
        return True

    # ------------------------------------------------------------------
    # Range Checking
    # ------------------------------------------------------------------

    def check_range(
        self,
        value: Any,
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None,
    ) -> bool:
        """Check if a numeric value falls within the specified range.

        Args:
            value: Value to check (will attempt numeric conversion).
            min_val: Minimum acceptable value (inclusive).
            max_val: Maximum acceptable value (inclusive).

        Returns:
            True if the value is within range.
        """
        try:
            num = float(str(value))
        except (ValueError, TypeError):
            return False

        if math.isnan(num) or math.isinf(num):
            return False

        if min_val is not None and num < min_val:
            return False
        if max_val is not None and num > max_val:
            return False
        return True

    # ------------------------------------------------------------------
    # Regex Checking
    # ------------------------------------------------------------------

    def check_regex(self, value: str, pattern: str) -> bool:
        """Check if a string value matches a regex pattern.

        Args:
            value: String to check.
            pattern: Regular expression pattern.

        Returns:
            True if the pattern matches.
        """
        try:
            return bool(re.match(pattern, value))
        except re.error as e:
            logger.warning("Invalid regex pattern '%s': %s", pattern, e)
            return False

    # ------------------------------------------------------------------
    # Domain Checking
    # ------------------------------------------------------------------

    def check_domain(self, value: Any, allowed_values: List[Any]) -> bool:
        """Check if a value is in the allowed domain set.

        Args:
            value: Value to check.
            allowed_values: List of acceptable values.

        Returns:
            True if the value is in the allowed set.
        """
        return value in allowed_values

    # ------------------------------------------------------------------
    # Cross-Field Validation
    # ------------------------------------------------------------------

    def check_cross_field(
        self,
        record: Dict[str, Any],
        constraint: Dict[str, Any],
    ) -> bool:
        """Check a cross-field constraint on a single record.

        Constraint format:
            {
                "field_a": "start_date",
                "operator": "LESS_THAN",
                "field_b": "end_date"
            }

        Supported operators: EQUALS, NOT_EQUALS, GREATER_THAN, LESS_THAN.

        Args:
            record: Row dictionary.
            constraint: Constraint definition dict.

        Returns:
            True if the constraint is satisfied.
        """
        field_a = constraint.get("field_a", "")
        field_b = constraint.get("field_b", "")
        operator = constraint.get("operator", OPERATOR_EQUALS)

        val_a = record.get(field_a)
        val_b = record.get(field_b)

        if val_a is None or val_b is None:
            # Cannot evaluate constraint with missing values
            return True

        try:
            if operator == OPERATOR_EQUALS:
                return val_a == val_b
            elif operator == OPERATOR_NOT_EQUALS:
                return val_a != val_b
            elif operator == OPERATOR_LESS_THAN:
                return val_a < val_b
            elif operator == OPERATOR_GREATER_THAN:
                return val_a > val_b
            else:
                logger.warning("Unsupported cross-field operator: %s", operator)
                return True
        except TypeError:
            # Values not comparable
            return False

    def _evaluate_cross_field_rules(
        self,
        data: List[Dict[str, Any]],
        rules: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Evaluate all cross-field constraints across the dataset.

        Args:
            data: Row dictionaries.
            rules: Rule list (extracts cross_field constraints).

        Returns:
            List of cross-field violation dicts.
        """
        violations: List[Dict[str, Any]] = []

        for rule in rules:
            constraint = rule.get("cross_field")
            if not constraint:
                continue

            for idx, record in enumerate(data):
                if not self.check_cross_field(record, constraint):
                    violations.append({
                        "row_index": idx,
                        "type": "cross_field_violation",
                        "constraint": constraint,
                        "field_a_value": str(record.get(constraint.get("field_a", "")))[:200],
                        "field_b_value": str(record.get(constraint.get("field_b", "")))[:200],
                    })
                    if len(violations) >= self._max_issues:
                        return violations

        return violations

    # ------------------------------------------------------------------
    # Validity Score
    # ------------------------------------------------------------------

    def compute_validity_score(
        self,
        data: List[Dict[str, Any]],
        rules: Optional[List[Dict[str, Any]]] = None,
        columns: Optional[List[str]] = None,
    ) -> float:
        """Compute overall validity score for a dataset.

        Score is the fraction of rule checks that pass across all rows.
        If no rules are provided, returns 1.0 (all valid by default).

        Args:
            data: List of row dictionaries.
            rules: Optional rule list.
            columns: Optional column subset.

        Returns:
            Float between 0.0 and 1.0.
        """
        if not data or not rules:
            return 1.0

        total_checks = 0
        passed_checks = 0

        for rule in rules:
            col = rule.get("column", "")
            format_type = rule.get("format_type")
            expected_type = rule.get("expected_type")
            min_val = rule.get("min_val")
            max_val = rule.get("max_val")
            regex_pattern = rule.get("pattern")
            allowed_values = rule.get("allowed_values")

            for row in data:
                value = row.get(col)
                if value is None or (isinstance(value, str) and not value.strip()):
                    # Skip null values for validity scoring
                    continue

                total_checks += 1
                is_valid = True

                if expected_type and not self.check_type_conformance_single(value, expected_type):
                    is_valid = False
                if format_type and not self.check_format(value, format_type):
                    is_valid = False
                if (min_val is not None or max_val is not None) and not self.check_range(value, min_val, max_val):
                    is_valid = False
                if regex_pattern and not self.check_regex(str(value), regex_pattern):
                    is_valid = False
                if allowed_values is not None and not self.check_domain(value, allowed_values):
                    is_valid = False

                if is_valid:
                    passed_checks += 1

        return passed_checks / total_checks if total_checks > 0 else 1.0

    # ------------------------------------------------------------------
    # Issue Generation
    # ------------------------------------------------------------------

    def generate_validity_issues(
        self,
        column_results: Dict[str, Dict[str, Any]],
        violations: List[Dict[str, Any]],
        validity_score: float,
    ) -> List[Dict[str, Any]]:
        """Generate validity quality issues from validation results.

        Args:
            column_results: Per-column validation results.
            violations: List of violation dicts.
            validity_score: Overall validity score.

        Returns:
            List of issue dicts with: issue_id, type, severity, message, details.
        """
        issues: List[Dict[str, Any]] = []

        # Per-column issues
        for col_name, col_result in column_results.items():
            invalid_count = col_result.get("invalid_count", 0)
            total = col_result.get("total_count", 0)
            if invalid_count > 0:
                rate = invalid_count / total if total > 0 else 0.0
                severity = self._classify_severity(rate)
                issues.append({
                    "issue_id": _generate_id("ISS"),
                    "type": "validity_violation",
                    "severity": severity,
                    "column": col_name,
                    "message": (
                        f"Column '{col_name}' has {invalid_count}/{total} "
                        f"({rate:.1%}) invalid values"
                    ),
                    "details": {
                        "invalid_count": invalid_count,
                        "total_count": total,
                        "invalid_rate": round(rate, 4),
                        "validity_rate": col_result.get("validity_rate", 0.0),
                    },
                    "created_at": _utcnow().isoformat(),
                })

        # Dataset-level issue
        if validity_score < 0.9:
            severity = self._classify_severity(1.0 - validity_score)
            issues.append({
                "issue_id": _generate_id("ISS"),
                "type": "low_validity",
                "severity": severity,
                "column": "__dataset__",
                "message": (
                    f"Dataset validity score is {validity_score:.1%} "
                    f"(below 90% threshold)"
                ),
                "details": {
                    "validity_score": round(validity_score, 4),
                    "total_violations": len(violations),
                },
                "created_at": _utcnow().isoformat(),
            })

        return issues

    def _classify_severity(self, error_rate: float) -> str:
        """Classify severity based on error rate.

        Args:
            error_rate: Float between 0.0 and 1.0.

        Returns:
            Severity string.
        """
        if error_rate >= 0.5:
            return SEVERITY_CRITICAL
        if error_rate >= 0.3:
            return SEVERITY_HIGH
        if error_rate >= 0.1:
            return SEVERITY_MEDIUM
        if error_rate > 0.0:
            return SEVERITY_LOW
        return SEVERITY_INFO

    # ------------------------------------------------------------------
    # Storage and Retrieval
    # ------------------------------------------------------------------

    def get_validation(self, validation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored validation by ID.

        Args:
            validation_id: The validation identifier.

        Returns:
            Validation dict or None if not found.
        """
        with self._lock:
            return self._validations.get(validation_id)

    def list_validations(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List stored validations with pagination.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of validation dicts sorted by creation time descending.
        """
        with self._lock:
            all_validations = sorted(
                self._validations.values(),
                key=lambda v: v.get("created_at", ""),
                reverse=True,
            )
            return all_validations[offset:offset + limit]

    def delete_validation(self, validation_id: str) -> bool:
        """Delete a stored validation.

        Args:
            validation_id: The validation identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if validation_id in self._validations:
                del self._validations[validation_id]
                logger.info("Validation deleted: %s", validation_id)
                return True
            return False

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate validation statistics.

        Returns:
            Dictionary with counters and totals for all validity
            checks performed by this engine instance.
        """
        with self._lock:
            completed = self._stats["validations_completed"]
            avg_time = (
                self._stats["total_validation_time_ms"] / completed
                if completed > 0 else 0.0
            )
            avg_violations = (
                self._stats["total_violations"] / completed
                if completed > 0 else 0.0
            )
            return {
                "validations_completed": completed,
                "total_rows_validated": self._stats["total_rows_validated"],
                "total_violations": self._stats["total_violations"],
                "avg_violations_per_validation": round(avg_violations, 2),
                "total_validation_time_ms": round(
                    self._stats["total_validation_time_ms"], 2
                ),
                "avg_validation_time_ms": round(avg_time, 2),
                "stored_validations": len(self._validations),
                "timestamp": _utcnow().isoformat(),
            }
