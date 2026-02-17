# -*- coding: utf-8 -*-
"""
Comparison Engine - AGENT-DATA-015

Field-by-field tolerance-aware comparison of matched records across data
sources.  Supports numeric, string, date, boolean, categorical, currency,
and unit-value comparisons with configurable tolerance rules and synonym
mappings.  Produces structured FieldComparison results with discrepancy
severity classification and provenance-tracked audit trails.

Engine 3 of 7 in the Cross-Source Reconciliation Agent SDK.

Zero-Hallucination: All calculations use deterministic Python arithmetic
(math, statistics, datetime, hashlib). No LLM calls for numeric
computations or comparison logic. No external numerical libraries required.

Example:
    >>> from greenlang.cross_source_reconciliation.comparison_engine import ComparisonEngine
    >>> engine = ComparisonEngine()
    >>> result = engine.compare_numeric(100.0, 102.0, tolerance_abs=5.0, tolerance_pct=5.0)
    >>> assert result.status == "within_tolerance"

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["ComparisonEngine"]


# ---------------------------------------------------------------------------
# Unit Conversion Tables
# ---------------------------------------------------------------------------
# Each dimension maps (from_unit, to_canonical_unit) -> multiplier.
# Canonical units: kg, kWh, liters, km, m2, kgCO2e.

UNIT_CONVERSIONS: Dict[str, Dict[str, float]] = {
    # Mass -> canonical unit: kg
    "mass": {
        "kg": 1.0,
        "g": 0.001,
        "mg": 1e-6,
        "tonnes": 1000.0,
        "metric_tons": 1000.0,
        "t": 1000.0,
        "lbs": 0.453592,
        "lb": 0.453592,
        "pounds": 0.453592,
        "oz": 0.0283495,
        "ounces": 0.0283495,
        "short_tons": 907.185,
        "long_tons": 1016.05,
    },
    # Energy -> canonical unit: kWh
    "energy": {
        "kwh": 1.0,
        "kWh": 1.0,
        "mwh": 1000.0,
        "MWh": 1000.0,
        "gwh": 1_000_000.0,
        "GWh": 1_000_000.0,
        "wh": 0.001,
        "Wh": 0.001,
        "j": 2.77778e-7,
        "J": 2.77778e-7,
        "kj": 2.77778e-4,
        "kJ": 2.77778e-4,
        "mj": 0.277778,
        "MJ": 0.277778,
        "gj": 277.778,
        "GJ": 277.778,
        "btu": 0.000293071,
        "BTU": 0.000293071,
        "therms": 29.3071,
        "therm": 29.3071,
        "cal": 1.16222e-6,
        "kcal": 1.16222e-3,
    },
    # Volume -> canonical unit: liters
    "volume": {
        "liters": 1.0,
        "litres": 1.0,
        "l": 1.0,
        "L": 1.0,
        "ml": 0.001,
        "mL": 0.001,
        "m3": 1000.0,
        "cubic_meters": 1000.0,
        "gallons": 3.78541,
        "gal": 3.78541,
        "us_gallons": 3.78541,
        "imperial_gallons": 4.54609,
        "barrels": 158.987,
        "bbl": 158.987,
        "ft3": 28.3168,
        "cubic_feet": 28.3168,
    },
    # Distance -> canonical unit: km
    "distance": {
        "km": 1.0,
        "m": 0.001,
        "cm": 1e-5,
        "mm": 1e-6,
        "miles": 1.60934,
        "mi": 1.60934,
        "yards": 0.0009144,
        "yd": 0.0009144,
        "feet": 0.0003048,
        "ft": 0.0003048,
        "nautical_miles": 1.852,
        "nm": 1.852,
    },
    # Area -> canonical unit: m2
    "area": {
        "m2": 1.0,
        "sq_m": 1.0,
        "km2": 1_000_000.0,
        "sq_km": 1_000_000.0,
        "hectares": 10_000.0,
        "ha": 10_000.0,
        "acres": 4046.86,
        "sq_ft": 0.092903,
        "sq_mi": 2_589_988.0,
        "sq_yd": 0.836127,
    },
    # Emissions -> canonical unit: kgCO2e
    "emissions": {
        "kgCO2e": 1.0,
        "kgco2e": 1.0,
        "kg_co2e": 1.0,
        "tCO2e": 1000.0,
        "tco2e": 1000.0,
        "t_co2e": 1000.0,
        "tonnes_co2e": 1000.0,
        "gCO2e": 0.001,
        "gco2e": 0.001,
        "g_co2e": 0.001,
        "mtCO2e": 1_000_000_000.0,
        "MtCO2e": 1_000_000_000.0,
    },
}

# Reverse lookup: unit_name (lowercase) -> (dimension, multiplier)
_UNIT_LOOKUP: Dict[str, Tuple[str, float]] = {}
for _dim, _units in UNIT_CONVERSIONS.items():
    for _unit_name, _multiplier in _units.items():
        _UNIT_LOOKUP[_unit_name.lower()] = (_dim, _multiplier)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FieldType(str, Enum):
    """Supported field types for comparison routing."""

    NUMERIC = "numeric"
    STRING = "string"
    DATE = "date"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    CURRENCY = "currency"
    UNIT_VALUE = "unit_value"


class ComparisonResult(str, Enum):
    """Outcome of a single field comparison."""

    MATCH = "match"
    WITHIN_TOLERANCE = "within_tolerance"
    MISMATCH = "mismatch"
    MISSING_LEFT = "missing_left"
    MISSING_RIGHT = "missing_right"
    MISSING_BOTH = "missing_both"
    INCOMPARABLE = "incomparable"


class DiscrepancySeverity(str, Enum):
    """Severity classification of a discrepancy."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    NONE = "none"


# ---------------------------------------------------------------------------
# Data Models (self-contained until models.py is built)
# ---------------------------------------------------------------------------


@dataclass
class ToleranceRule:
    """Tolerance rule for a specific field comparison.

    Attributes:
        field_name: Name of the field this rule applies to.
        tolerance_abs: Absolute tolerance for numeric comparison.
        tolerance_pct: Percentage tolerance for numeric comparison.
        case_sensitive: Whether string comparison is case-sensitive.
        strip_whitespace: Whether to strip whitespace before string comparison.
        max_days_diff: Maximum allowed day difference for date comparison.
        synonyms: Synonym mapping for categorical fields.
        rounding_digits: Number of decimal places for rounding before compare.
        date_format_a: Date format string for source A.
        date_format_b: Date format string for source B.
        currency_a: Currency code for source A.
        currency_b: Currency code for source B.
        unit_a: Unit label for source A.
        unit_b: Unit label for source B.
    """

    field_name: str = ""
    tolerance_abs: Optional[float] = None
    tolerance_pct: Optional[float] = None
    case_sensitive: bool = True
    strip_whitespace: bool = True
    max_days_diff: int = 0
    synonyms: Dict[str, List[str]] = field(default_factory=dict)
    rounding_digits: Optional[int] = None
    date_format_a: Optional[str] = None
    date_format_b: Optional[str] = None
    currency_a: str = "USD"
    currency_b: str = "USD"
    unit_a: str = ""
    unit_b: str = ""


@dataclass
class FieldComparison:
    """Result of comparing a single field across two records.

    Attributes:
        field_name: Name of the field compared.
        field_type: Type of comparison performed.
        value_a: Value from source A (as original type).
        value_b: Value from source B (as original type).
        normalized_a: Normalised value from source A after conversion.
        normalized_b: Normalised value from source B after conversion.
        status: Comparison outcome.
        absolute_diff: Absolute numeric difference (None for non-numeric).
        relative_diff_pct: Relative difference as percentage (None for non-numeric).
        severity: Discrepancy severity classification.
        message: Human-readable comparison summary.
        provenance_hash: SHA-256 hash for audit trail.
    """

    field_name: str = ""
    field_type: str = ""
    value_a: Any = None
    value_b: Any = None
    normalized_a: Any = None
    normalized_b: Any = None
    status: str = ComparisonResult.MATCH.value
    absolute_diff: Optional[float] = None
    relative_diff_pct: Optional[float] = None
    severity: str = DiscrepancySeverity.NONE.value
    message: str = ""
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the comparison result.
        """
        return asdict(self)


@dataclass
class ComparisonSummary:
    """Aggregated summary of multiple field comparisons.

    Attributes:
        total_fields: Total number of fields compared.
        matches: Number of exact matches.
        within_tolerance: Number of fields within tolerance.
        mismatches: Number of mismatches.
        missing: Number of missing fields (left, right, or both).
        incomparable: Number of incomparable fields.
        match_rate: Ratio of (matches + within_tolerance) / total_fields.
        severity_counts: Count by severity level.
        provenance_hash: SHA-256 hash for audit trail.
    """

    total_fields: int = 0
    matches: int = 0
    within_tolerance: int = 0
    mismatches: int = 0
    missing: int = 0
    incomparable: int = 0
    match_rate: float = 0.0
    severity_counts: Dict[str, int] = field(default_factory=dict)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the summary.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# Metrics helpers (safe when prometheus_client is absent)
# ---------------------------------------------------------------------------

try:
    from greenlang.cross_source_reconciliation.metrics import (
        inc_comparisons as _inc_comparisons_raw,
        observe_duration as _observe_duration_raw,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _inc_comparisons_raw = None  # type: ignore[assignment]
    _observe_duration_raw = None  # type: ignore[assignment]


def _inc_comparisons(result: str, count: int = 1) -> None:
    """Safely increment comparison counter.

    Args:
        result: Comparison result category (match, mismatch,
            within_tolerance, missing_left, missing_right, etc.).
        count: Number of comparisons to record.
    """
    if _METRICS_AVAILABLE and _inc_comparisons_raw is not None:
        try:
            _inc_comparisons_raw(result, count)
        except Exception:
            pass


def _observe_duration(duration: float) -> None:
    """Safely observe processing duration.

    Args:
        duration: Duration in seconds.
    """
    if _METRICS_AVAILABLE and _observe_duration_raw is not None:
        try:
            _observe_duration_raw(duration)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Provenance helper (safe when provenance module is absent)
# ---------------------------------------------------------------------------

try:
    from greenlang.cross_source_reconciliation.provenance import (
        ProvenanceTracker,
    )
    _PROVENANCE_MODULE_AVAILABLE = True
except ImportError:
    _PROVENANCE_MODULE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_TRUTHY_STRINGS = frozenset({
    "true", "yes", "1", "y", "on", "t",
})
_FALSY_STRINGS = frozenset({
    "false", "no", "0", "n", "off", "f",
})

_DEFAULT_DATE_FORMATS: Tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m-%d-%Y",
    "%m/%d/%Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%Y%m%d",
)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _is_missing(value: Any) -> bool:
    """Determine whether a value represents a missing data point.

    Treats None, empty string, float('nan'), and float('inf') as missing.

    Args:
        value: Value to check.

    Returns:
        True if the value is considered missing.
    """
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    return False


def _to_float(value: Any) -> Optional[float]:
    """Safely convert a value to float.

    Args:
        value: Value to convert.

    Returns:
        Float value or None if conversion fails.
    """
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            cleaned = value.strip().replace(",", "")
            return float(cleaned)
        except (ValueError, TypeError):
            return None
    return None


def _to_bool(value: Any) -> Optional[bool]:
    """Convert a value to boolean, handling truthy/falsy strings.

    Args:
        value: Value to convert.

    Returns:
        Boolean value or None if not interpretable.
    """
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUTHY_STRINGS:
            return True
        if normalized in _FALSY_STRINGS:
            return False
    return None


def _build_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, numeric, or other).

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ===========================================================================
# ComparisonEngine
# ===========================================================================


class ComparisonEngine:
    """Field-by-field tolerance-aware comparison engine.

    Compares matched records from different data sources using
    type-appropriate comparison methods. Supports numeric tolerance
    (absolute and percentage), string normalization, date parsing
    with configurable formats, boolean coercion, categorical synonym
    matching, currency conversion, and unit-of-measure conversion
    for sustainability metrics.

    All arithmetic is deterministic Python (zero-hallucination).
    Every comparison produces a SHA-256 provenance hash for audit
    trail tracking.

    Attributes:
        _provenance: SHA-256 provenance tracker for audit trails.
        _comparison_count: Running count of comparisons performed.

    Example:
        >>> engine = ComparisonEngine()
        >>> fc = engine.compare_numeric(100.0, 102.0, tolerance_abs=5.0, tolerance_pct=5.0)
        >>> assert fc.status == "within_tolerance"
        >>> fc = engine.compare_string("hello", "HELLO", case_sensitive=False)
        >>> assert fc.status == "match"
    """

    def __init__(self) -> None:
        """Initialize ComparisonEngine with provenance tracker."""
        if _PROVENANCE_MODULE_AVAILABLE:
            self._provenance = ProvenanceTracker()
        else:
            self._provenance = None  # type: ignore[assignment]
        self._comparison_count: int = 0
        logger.info("ComparisonEngine initialized (cross-source reconciliation)")

    # ------------------------------------------------------------------
    # 1. compare_records
    # ------------------------------------------------------------------

    def compare_records(
        self,
        record_a: Dict[str, Any],
        record_b: Dict[str, Any],
        fields: List[str],
        tolerance_rules: Optional[Dict[str, ToleranceRule]] = None,
        field_types: Optional[Dict[str, FieldType]] = None,
    ) -> List[FieldComparison]:
        """Compare each field of two records using type-appropriate methods.

        Iterates through the specified field list, selects the comparison
        method based on the field type, applies tolerance rules, and
        returns a list of FieldComparison results.

        Args:
            record_a: First record (source A) as a dictionary.
            record_b: Second record (source B) as a dictionary.
            fields: Ordered list of field names to compare.
            tolerance_rules: Optional per-field tolerance rules keyed by
                field name.  Falls back to default tolerances when absent.
            field_types: Optional per-field type mapping.  Falls back to
                FieldType.STRING when absent.

        Returns:
            List of FieldComparison results, one per field.

        Example:
            >>> engine = ComparisonEngine()
            >>> comparisons = engine.compare_records(
            ...     {"amount": 100, "name": "Acme"},
            ...     {"amount": 102, "name": "acme"},
            ...     fields=["amount", "name"],
            ...     field_types={"amount": FieldType.NUMERIC, "name": FieldType.STRING},
            ...     tolerance_rules={"amount": ToleranceRule(tolerance_pct=5.0)},
            ... )
            >>> assert len(comparisons) == 2
        """
        start_time = time.monotonic()
        tolerance_rules = tolerance_rules or {}
        field_types = field_types or {}
        comparisons: List[FieldComparison] = []

        for field_name in fields:
            value_a = record_a.get(field_name)
            value_b = record_b.get(field_name)
            ft = field_types.get(field_name, FieldType.STRING)
            rule = tolerance_rules.get(field_name, ToleranceRule())

            comparison = self._compare_field(
                field_name=field_name,
                value_a=value_a,
                value_b=value_b,
                field_type=ft,
                rule=rule,
            )
            comparisons.append(comparison)

        elapsed = time.monotonic() - start_time
        _inc_comparisons("record", len(fields))
        _observe_duration(elapsed)

        logger.debug(
            "compare_records completed: %d fields in %.3fms",
            len(fields),
            elapsed * 1000,
        )
        return comparisons

    # ------------------------------------------------------------------
    # 2. compare_numeric
    # ------------------------------------------------------------------

    def compare_numeric(
        self,
        value_a: Any,
        value_b: Any,
        tolerance_abs: Optional[float] = None,
        tolerance_pct: Optional[float] = None,
        rounding_digits: Optional[int] = None,
        field_name: str = "",
    ) -> FieldComparison:
        """Compare two numeric values with tolerance thresholds.

        Computes absolute and relative differences and checks them
        against the provided tolerance thresholds.  Both tolerances
        must be satisfied for the result to be ``within_tolerance``.

        Args:
            value_a: Numeric value from source A.
            value_b: Numeric value from source B.
            tolerance_abs: Maximum absolute difference allowed.
            tolerance_pct: Maximum relative difference percentage allowed.
            rounding_digits: Optional rounding precision before compare.
            field_name: Name of the field being compared.

        Returns:
            FieldComparison with status, absolute_diff, and relative_diff_pct.

        Example:
            >>> engine = ComparisonEngine()
            >>> fc = engine.compare_numeric(100.0, 105.0, tolerance_pct=10.0)
            >>> assert fc.status == "within_tolerance"
        """
        start_time = time.monotonic()
        self._comparison_count += 1

        # Handle missing values
        a_missing = _is_missing(value_a)
        b_missing = _is_missing(value_b)

        if a_missing and b_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.NUMERIC.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_BOTH,
                message="Both values are missing",
            )
            self._record_metric_and_provenance(
                "compare_numeric", start_time, result,
            )
            return result

        if a_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.NUMERIC.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_LEFT,
                message="Source A value is missing",
            )
            self._record_metric_and_provenance(
                "compare_numeric", start_time, result,
            )
            return result

        if b_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.NUMERIC.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_RIGHT,
                message="Source B value is missing",
            )
            self._record_metric_and_provenance(
                "compare_numeric", start_time, result,
            )
            return result

        # Convert to float
        float_a = _to_float(value_a)
        float_b = _to_float(value_b)

        if float_a is None or float_b is None:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.NUMERIC.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.INCOMPARABLE,
                message="One or both values could not be parsed as numeric",
            )
            self._record_metric_and_provenance(
                "compare_numeric", start_time, result,
            )
            return result

        # Apply rounding
        if rounding_digits is not None:
            float_a = round(float_a, rounding_digits)
            float_b = round(float_b, rounding_digits)

        # Compute differences
        abs_diff = abs(float_a - float_b)
        denominator = max(abs(float_a), abs(float_b), 1e-10)
        rel_diff_pct = (abs_diff / denominator) * 100.0

        # Determine status
        status = self._evaluate_numeric_tolerance(
            abs_diff, rel_diff_pct, tolerance_abs, tolerance_pct,
        )

        message = self._format_numeric_message(
            float_a, float_b, abs_diff, rel_diff_pct, status,
        )

        result = self._make_comparison(
            field_name=field_name,
            field_type=FieldType.NUMERIC.value,
            value_a=value_a,
            value_b=value_b,
            normalized_a=float_a,
            normalized_b=float_b,
            status=status,
            absolute_diff=abs_diff,
            relative_diff_pct=rel_diff_pct,
            message=message,
        )
        self._record_metric_and_provenance(
            "compare_numeric", start_time, result,
        )
        return result

    # ------------------------------------------------------------------
    # 3. compare_string
    # ------------------------------------------------------------------

    def compare_string(
        self,
        value_a: Any,
        value_b: Any,
        case_sensitive: bool = True,
        strip_whitespace: bool = True,
        field_name: str = "",
    ) -> FieldComparison:
        """Compare two string values with optional normalization.

        Args:
            value_a: String value from source A.
            value_b: String value from source B.
            case_sensitive: If False, compare in lowercase.
            strip_whitespace: If True, strip leading/trailing whitespace.
            field_name: Name of the field being compared.

        Returns:
            FieldComparison with status match or mismatch.

        Example:
            >>> engine = ComparisonEngine()
            >>> fc = engine.compare_string("Hello World", "hello world", case_sensitive=False)
            >>> assert fc.status == "match"
        """
        start_time = time.monotonic()
        self._comparison_count += 1

        # Handle missing
        a_missing = _is_missing(value_a)
        b_missing = _is_missing(value_b)

        if a_missing and b_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.STRING.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_BOTH,
                message="Both values are missing",
            )
            self._record_metric_and_provenance(
                "compare_string", start_time, result,
            )
            return result

        if a_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.STRING.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_LEFT,
                message="Source A value is missing",
            )
            self._record_metric_and_provenance(
                "compare_string", start_time, result,
            )
            return result

        if b_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.STRING.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_RIGHT,
                message="Source B value is missing",
            )
            self._record_metric_and_provenance(
                "compare_string", start_time, result,
            )
            return result

        # Convert to strings
        str_a = str(value_a)
        str_b = str(value_b)

        # Normalize
        norm_a = self._normalize_string(str_a, case_sensitive, strip_whitespace)
        norm_b = self._normalize_string(str_b, case_sensitive, strip_whitespace)

        if norm_a == norm_b:
            status = ComparisonResult.MATCH
            message = "Exact match"
            if str_a != str_b:
                message = "Match after normalization"
        else:
            status = ComparisonResult.MISMATCH
            message = f"String mismatch: '{str_a}' != '{str_b}'"

        result = self._make_comparison(
            field_name=field_name,
            field_type=FieldType.STRING.value,
            value_a=value_a,
            value_b=value_b,
            normalized_a=norm_a,
            normalized_b=norm_b,
            status=status,
            message=message,
        )
        self._record_metric_and_provenance(
            "compare_string", start_time, result,
        )
        return result

    # ------------------------------------------------------------------
    # 4. compare_date
    # ------------------------------------------------------------------

    def compare_date(
        self,
        value_a: Any,
        value_b: Any,
        max_days_diff: int = 0,
        date_format_a: Optional[str] = None,
        date_format_b: Optional[str] = None,
        field_name: str = "",
    ) -> FieldComparison:
        """Compare two date values with configurable day tolerance.

        Parses dates using specified or default format strings, computes
        the day difference, and checks against the tolerance.

        Args:
            value_a: Date value from source A (str, date, or datetime).
            value_b: Date value from source B (str, date, or datetime).
            max_days_diff: Maximum allowed day difference.  Zero means
                exact match required.
            date_format_a: Optional date format for parsing source A.
            date_format_b: Optional date format for parsing source B.
            field_name: Name of the field being compared.

        Returns:
            FieldComparison with status and absolute_diff in days.

        Example:
            >>> engine = ComparisonEngine()
            >>> fc = engine.compare_date("2025-01-15", "2025-01-17", max_days_diff=3)
            >>> assert fc.status == "within_tolerance"
        """
        start_time = time.monotonic()
        self._comparison_count += 1

        # Handle missing
        a_missing = _is_missing(value_a)
        b_missing = _is_missing(value_b)

        if a_missing and b_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.DATE.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_BOTH,
                message="Both date values are missing",
            )
            self._record_metric_and_provenance(
                "compare_date", start_time, result,
            )
            return result

        if a_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.DATE.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_LEFT,
                message="Source A date is missing",
            )
            self._record_metric_and_provenance(
                "compare_date", start_time, result,
            )
            return result

        if b_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.DATE.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_RIGHT,
                message="Source B date is missing",
            )
            self._record_metric_and_provenance(
                "compare_date", start_time, result,
            )
            return result

        # Parse dates
        formats_a = (date_format_a,) if date_format_a else _DEFAULT_DATE_FORMATS
        formats_b = (date_format_b,) if date_format_b else _DEFAULT_DATE_FORMATS

        parsed_a = self._parse_date(value_a, formats_a)
        parsed_b = self._parse_date(value_b, formats_b)

        if parsed_a is None or parsed_b is None:
            unparseable = []
            if parsed_a is None:
                unparseable.append("A")
            if parsed_b is None:
                unparseable.append("B")
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.DATE.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.INCOMPARABLE,
                message=(
                    f"Could not parse date from source(s): "
                    f"{', '.join(unparseable)}"
                ),
            )
            self._record_metric_and_provenance(
                "compare_date", start_time, result,
            )
            return result

        # Compute day difference
        day_diff = abs((parsed_a - parsed_b).days)

        if day_diff == 0:
            status = ComparisonResult.MATCH
            message = "Exact date match"
        elif day_diff <= max_days_diff:
            status = ComparisonResult.WITHIN_TOLERANCE
            message = (
                f"Date difference of {day_diff} day(s) within "
                f"tolerance of {max_days_diff} day(s)"
            )
        else:
            status = ComparisonResult.MISMATCH
            message = (
                f"Date difference of {day_diff} day(s) exceeds "
                f"tolerance of {max_days_diff} day(s)"
            )

        result = self._make_comparison(
            field_name=field_name,
            field_type=FieldType.DATE.value,
            value_a=value_a,
            value_b=value_b,
            normalized_a=parsed_a.isoformat(),
            normalized_b=parsed_b.isoformat(),
            status=status,
            absolute_diff=float(day_diff),
            message=message,
        )
        self._record_metric_and_provenance(
            "compare_date", start_time, result,
        )
        return result

    # ------------------------------------------------------------------
    # 5. compare_boolean
    # ------------------------------------------------------------------

    def compare_boolean(
        self,
        value_a: Any,
        value_b: Any,
        field_name: str = "",
    ) -> FieldComparison:
        """Compare two boolean values with truthy/falsy string handling.

        Handles native booleans, numeric (0/1), and string representations
        (true/false, yes/no, 1/0, on/off).

        Args:
            value_a: Boolean-like value from source A.
            value_b: Boolean-like value from source B.
            field_name: Name of the field being compared.

        Returns:
            FieldComparison with status match or mismatch.

        Example:
            >>> engine = ComparisonEngine()
            >>> fc = engine.compare_boolean("yes", True)
            >>> assert fc.status == "match"
        """
        start_time = time.monotonic()
        self._comparison_count += 1

        # Handle missing
        a_missing = _is_missing(value_a)
        b_missing = _is_missing(value_b)

        if a_missing and b_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.BOOLEAN.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_BOTH,
                message="Both boolean values are missing",
            )
            self._record_metric_and_provenance(
                "compare_boolean", start_time, result,
            )
            return result

        if a_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.BOOLEAN.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_LEFT,
                message="Source A boolean is missing",
            )
            self._record_metric_and_provenance(
                "compare_boolean", start_time, result,
            )
            return result

        if b_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.BOOLEAN.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_RIGHT,
                message="Source B boolean is missing",
            )
            self._record_metric_and_provenance(
                "compare_boolean", start_time, result,
            )
            return result

        # Convert to boolean
        bool_a = _to_bool(value_a)
        bool_b = _to_bool(value_b)

        if bool_a is None or bool_b is None:
            unparseable = []
            if bool_a is None:
                unparseable.append(f"A='{value_a}'")
            if bool_b is None:
                unparseable.append(f"B='{value_b}'")
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.BOOLEAN.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.INCOMPARABLE,
                message=(
                    f"Cannot interpret as boolean: "
                    f"{', '.join(unparseable)}"
                ),
            )
            self._record_metric_and_provenance(
                "compare_boolean", start_time, result,
            )
            return result

        if bool_a == bool_b:
            status = ComparisonResult.MATCH
            message = f"Boolean match: both {bool_a}"
        else:
            status = ComparisonResult.MISMATCH
            message = f"Boolean mismatch: A={bool_a}, B={bool_b}"

        result = self._make_comparison(
            field_name=field_name,
            field_type=FieldType.BOOLEAN.value,
            value_a=value_a,
            value_b=value_b,
            normalized_a=bool_a,
            normalized_b=bool_b,
            status=status,
            message=message,
        )
        self._record_metric_and_provenance(
            "compare_boolean", start_time, result,
        )
        return result

    # ------------------------------------------------------------------
    # 6. compare_categorical
    # ------------------------------------------------------------------

    def compare_categorical(
        self,
        value_a: Any,
        value_b: Any,
        synonyms: Optional[Dict[str, List[str]]] = None,
        field_name: str = "",
    ) -> FieldComparison:
        """Compare two categorical values using direct equality and synonyms.

        Checks direct string equality first, then consults the synonym
        mapping to determine if the values are semantically equivalent
        (e.g., "electricity" = "electric" = "power").

        Args:
            value_a: Categorical value from source A.
            value_b: Categorical value from source B.
            synonyms: Mapping of canonical term to list of synonyms.
                Example: {"electricity": ["electric", "power", "elec"]}.
            field_name: Name of the field being compared.

        Returns:
            FieldComparison with status match (including synonym match)
            or mismatch.

        Example:
            >>> engine = ComparisonEngine()
            >>> syns = {"electricity": ["electric", "power"]}
            >>> fc = engine.compare_categorical("electricity", "power", synonyms=syns)
            >>> assert fc.status == "match"
        """
        start_time = time.monotonic()
        self._comparison_count += 1
        synonyms = synonyms or {}

        # Handle missing
        a_missing = _is_missing(value_a)
        b_missing = _is_missing(value_b)

        if a_missing and b_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.CATEGORICAL.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_BOTH,
                message="Both categorical values are missing",
            )
            self._record_metric_and_provenance(
                "compare_categorical", start_time, result,
            )
            return result

        if a_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.CATEGORICAL.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_LEFT,
                message="Source A categorical value is missing",
            )
            self._record_metric_and_provenance(
                "compare_categorical", start_time, result,
            )
            return result

        if b_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.CATEGORICAL.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_RIGHT,
                message="Source B categorical value is missing",
            )
            self._record_metric_and_provenance(
                "compare_categorical", start_time, result,
            )
            return result

        str_a = str(value_a).strip().lower()
        str_b = str(value_b).strip().lower()

        # Direct equality check
        if str_a == str_b:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.CATEGORICAL.value,
                value_a=value_a,
                value_b=value_b,
                normalized_a=str_a,
                normalized_b=str_b,
                status=ComparisonResult.MATCH,
                message="Exact categorical match",
            )
            self._record_metric_and_provenance(
                "compare_categorical", start_time, result,
            )
            return result

        # Check synonym mapping
        if self._are_synonymous(str_a, str_b, synonyms):
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.CATEGORICAL.value,
                value_a=value_a,
                value_b=value_b,
                normalized_a=str_a,
                normalized_b=str_b,
                status=ComparisonResult.MATCH,
                message=(
                    f"Synonym match: '{value_a}' and '{value_b}' "
                    f"are equivalent"
                ),
            )
            self._record_metric_and_provenance(
                "compare_categorical", start_time, result,
            )
            return result

        result = self._make_comparison(
            field_name=field_name,
            field_type=FieldType.CATEGORICAL.value,
            value_a=value_a,
            value_b=value_b,
            normalized_a=str_a,
            normalized_b=str_b,
            status=ComparisonResult.MISMATCH,
            message=f"Categorical mismatch: '{value_a}' != '{value_b}'",
        )
        self._record_metric_and_provenance(
            "compare_categorical", start_time, result,
        )
        return result

    # ------------------------------------------------------------------
    # 7. compare_currency
    # ------------------------------------------------------------------

    def compare_currency(
        self,
        value_a: Any,
        value_b: Any,
        currency_a: str = "USD",
        currency_b: str = "USD",
        exchange_rates: Optional[Dict[str, float]] = None,
        tolerance_pct: Optional[float] = None,
        field_name: str = "",
    ) -> FieldComparison:
        """Compare two monetary values with currency conversion.

        Converts both values to a common currency (USD by default)
        using the provided exchange rates, then performs numeric
        comparison with tolerance.

        Args:
            value_a: Monetary value from source A.
            value_b: Monetary value from source B.
            currency_a: Currency code for source A (ISO 4217).
            currency_b: Currency code for source B (ISO 4217).
            exchange_rates: Mapping of currency code to USD rate.
                Example: {"EUR": 1.10, "GBP": 1.27, "USD": 1.0}.
            tolerance_pct: Percentage tolerance for the comparison.
            field_name: Name of the field being compared.

        Returns:
            FieldComparison with converted values and tolerance check.

        Example:
            >>> engine = ComparisonEngine()
            >>> rates = {"EUR": 1.10, "USD": 1.0}
            >>> fc = engine.compare_currency(100, 110, "EUR", "USD", rates, tolerance_pct=1.0)
            >>> assert fc.status == "match"
        """
        start_time = time.monotonic()
        self._comparison_count += 1
        exchange_rates = exchange_rates or {"USD": 1.0}

        # Handle missing
        a_missing = _is_missing(value_a)
        b_missing = _is_missing(value_b)

        if a_missing and b_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.CURRENCY.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_BOTH,
                message="Both currency values are missing",
            )
            self._record_metric_and_provenance(
                "compare_currency", start_time, result,
            )
            return result

        if a_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.CURRENCY.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_LEFT,
                message="Source A currency value is missing",
            )
            self._record_metric_and_provenance(
                "compare_currency", start_time, result,
            )
            return result

        if b_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.CURRENCY.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_RIGHT,
                message="Source B currency value is missing",
            )
            self._record_metric_and_provenance(
                "compare_currency", start_time, result,
            )
            return result

        # Convert to float
        float_a = _to_float(value_a)
        float_b = _to_float(value_b)

        if float_a is None or float_b is None:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.CURRENCY.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.INCOMPARABLE,
                message="One or both currency values cannot be parsed as numeric",
            )
            self._record_metric_and_provenance(
                "compare_currency", start_time, result,
            )
            return result

        # Convert to common currency (USD)
        converted_a = self._convert_currency(
            float_a, currency_a.upper(), "USD", exchange_rates,
        )
        converted_b = self._convert_currency(
            float_b, currency_b.upper(), "USD", exchange_rates,
        )

        if converted_a is None or converted_b is None:
            missing_rates = []
            if converted_a is None:
                missing_rates.append(currency_a)
            if converted_b is None:
                missing_rates.append(currency_b)
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.CURRENCY.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.INCOMPARABLE,
                message=(
                    f"Missing exchange rate(s) for: "
                    f"{', '.join(missing_rates)}"
                ),
            )
            self._record_metric_and_provenance(
                "compare_currency", start_time, result,
            )
            return result

        # Numeric comparison on converted values
        abs_diff = abs(converted_a - converted_b)
        denominator = max(abs(converted_a), abs(converted_b), 1e-10)
        rel_diff_pct = (abs_diff / denominator) * 100.0

        status = self._evaluate_numeric_tolerance(
            abs_diff, rel_diff_pct, None, tolerance_pct,
        )

        message = (
            f"Currency comparison: {value_a} {currency_a} "
            f"({converted_a:.2f} USD) vs {value_b} {currency_b} "
            f"({converted_b:.2f} USD), "
            f"diff={abs_diff:.2f} USD ({rel_diff_pct:.2f}%)"
        )

        result = self._make_comparison(
            field_name=field_name,
            field_type=FieldType.CURRENCY.value,
            value_a=value_a,
            value_b=value_b,
            normalized_a=converted_a,
            normalized_b=converted_b,
            status=status,
            absolute_diff=abs_diff,
            relative_diff_pct=rel_diff_pct,
            message=message,
        )
        self._record_metric_and_provenance(
            "compare_currency", start_time, result,
        )
        return result

    # ------------------------------------------------------------------
    # 8. compare_unit_value
    # ------------------------------------------------------------------

    def compare_unit_value(
        self,
        value_a: Any,
        value_b: Any,
        unit_a: str = "",
        unit_b: str = "",
        unit_conversions: Optional[Dict[str, Dict[str, float]]] = None,
        tolerance_pct: Optional[float] = None,
        tolerance_abs: Optional[float] = None,
        field_name: str = "",
    ) -> FieldComparison:
        """Compare two values with different units of measure.

        Converts both values to a canonical unit within the same
        dimension (e.g., both to kg, both to kWh), then performs
        numeric comparison with tolerance.

        Built-in conversions cover common sustainability units:
        - Mass: kg, tonnes, lbs, g
        - Energy: kWh, MWh, GJ, BTU, therms
        - Volume: liters, m3, gallons
        - Distance: km, miles, m
        - Area: m2, hectares, acres
        - Emissions: tCO2e, kgCO2e, gCO2e

        Args:
            value_a: Numeric value from source A.
            value_b: Numeric value from source B.
            unit_a: Unit of measure for source A.
            unit_b: Unit of measure for source B.
            unit_conversions: Optional custom conversion tables to
                supplement or override built-in tables.  Format matches
                UNIT_CONVERSIONS module constant.
            tolerance_pct: Percentage tolerance for the comparison.
            tolerance_abs: Absolute tolerance after unit conversion.
            field_name: Name of the field being compared.

        Returns:
            FieldComparison with converted values and tolerance check.

        Example:
            >>> engine = ComparisonEngine()
            >>> fc = engine.compare_unit_value(1.0, 1000.0, "tonnes", "kg")
            >>> assert fc.status == "match"
        """
        start_time = time.monotonic()
        self._comparison_count += 1

        # Handle missing
        a_missing = _is_missing(value_a)
        b_missing = _is_missing(value_b)

        if a_missing and b_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.UNIT_VALUE.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_BOTH,
                message="Both unit values are missing",
            )
            self._record_metric_and_provenance(
                "compare_unit_value", start_time, result,
            )
            return result

        if a_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.UNIT_VALUE.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_LEFT,
                message="Source A unit value is missing",
            )
            self._record_metric_and_provenance(
                "compare_unit_value", start_time, result,
            )
            return result

        if b_missing:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.UNIT_VALUE.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.MISSING_RIGHT,
                message="Source B unit value is missing",
            )
            self._record_metric_and_provenance(
                "compare_unit_value", start_time, result,
            )
            return result

        # Convert to float
        float_a = _to_float(value_a)
        float_b = _to_float(value_b)

        if float_a is None or float_b is None:
            result = self._make_comparison(
                field_name=field_name,
                field_type=FieldType.UNIT_VALUE.value,
                value_a=value_a,
                value_b=value_b,
                status=ComparisonResult.INCOMPARABLE,
                message="One or both values cannot be parsed as numeric",
            )
            self._record_metric_and_provenance(
                "compare_unit_value", start_time, result,
            )
            return result

        # Convert units
        converted_a = self._convert_unit(
            float_a, unit_a, unit_b, unit_conversions,
        )
        converted_b = float_b

        # If direct conversion from unit_a to unit_b failed, try
        # converting both to canonical unit within the same dimension.
        if converted_a is None:
            all_conv = dict(UNIT_CONVERSIONS)
            if unit_conversions:
                for dim, units in unit_conversions.items():
                    if dim in all_conv:
                        merged = dict(all_conv[dim])
                        merged.update(units)
                        all_conv[dim] = merged
                    else:
                        all_conv[dim] = dict(units)

            info_a = self._find_unit_info(
                unit_a.lower().strip(), all_conv,
            ) if unit_a else None
            info_b = self._find_unit_info(
                unit_b.lower().strip(), all_conv,
            ) if unit_b else None

            # Both units must be known AND in the same dimension
            if (
                info_a is not None
                and info_b is not None
                and info_a[0] == info_b[0]
            ):
                converted_a = float_a * info_a[1]
                converted_b = float_b * info_b[1]
            else:
                result = self._make_comparison(
                    field_name=field_name,
                    field_type=FieldType.UNIT_VALUE.value,
                    value_a=value_a,
                    value_b=value_b,
                    status=ComparisonResult.INCOMPARABLE,
                    message=(
                        f"Cannot convert between units: "
                        f"'{unit_a}' and '{unit_b}'"
                    ),
                )
                self._record_metric_and_provenance(
                    "compare_unit_value", start_time, result,
                )
                return result

        # Numeric comparison on converted values
        abs_diff = abs(converted_a - converted_b)
        denominator = max(abs(converted_a), abs(converted_b), 1e-10)
        rel_diff_pct = (abs_diff / denominator) * 100.0

        status = self._evaluate_numeric_tolerance(
            abs_diff, rel_diff_pct, tolerance_abs, tolerance_pct,
        )

        message = (
            f"Unit comparison: {value_a} {unit_a} "
            f"({converted_a:.6g} canonical) vs {value_b} {unit_b} "
            f"({converted_b:.6g} canonical), "
            f"diff={abs_diff:.6g} ({rel_diff_pct:.2f}%)"
        )

        result = self._make_comparison(
            field_name=field_name,
            field_type=FieldType.UNIT_VALUE.value,
            value_a=value_a,
            value_b=value_b,
            normalized_a=converted_a,
            normalized_b=converted_b,
            status=status,
            absolute_diff=abs_diff,
            relative_diff_pct=rel_diff_pct,
            message=message,
        )
        self._record_metric_and_provenance(
            "compare_unit_value", start_time, result,
        )
        return result

    # ------------------------------------------------------------------
    # 9. compare_batch
    # ------------------------------------------------------------------

    def compare_batch(
        self,
        matched_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        fields: List[str],
        tolerance_rules: Optional[Dict[str, ToleranceRule]] = None,
        field_types: Optional[Dict[str, FieldType]] = None,
    ) -> List[List[FieldComparison]]:
        """Batch-compare all matched record pairs.

        Iterates through the list of (record_a, record_b) tuples and
        compares each pair across the specified fields.

        Args:
            matched_pairs: List of (record_a, record_b) tuples.
            fields: Ordered list of field names to compare.
            tolerance_rules: Optional per-field tolerance rules.
            field_types: Optional per-field type mapping.

        Returns:
            List of comparison lists, one per matched pair.

        Example:
            >>> engine = ComparisonEngine()
            >>> pairs = [
            ...     ({"amount": 100}, {"amount": 102}),
            ...     ({"amount": 200}, {"amount": 200}),
            ... ]
            >>> results = engine.compare_batch(
            ...     pairs, ["amount"],
            ...     field_types={"amount": FieldType.NUMERIC},
            ...     tolerance_rules={"amount": ToleranceRule(tolerance_pct=5.0)},
            ... )
            >>> assert len(results) == 2
        """
        start_time = time.monotonic()
        all_comparisons: List[List[FieldComparison]] = []

        for idx, (rec_a, rec_b) in enumerate(matched_pairs):
            comparisons = self.compare_records(
                record_a=rec_a,
                record_b=rec_b,
                fields=fields,
                tolerance_rules=tolerance_rules,
                field_types=field_types,
            )
            all_comparisons.append(comparisons)

        elapsed = time.monotonic() - start_time
        _inc_comparisons("batch", len(matched_pairs))
        _observe_duration(elapsed)

        logger.info(
            "compare_batch completed: %d pairs, %d fields each in %.3fms",
            len(matched_pairs),
            len(fields),
            elapsed * 1000,
        )
        return all_comparisons

    # ------------------------------------------------------------------
    # 10. summarize_comparisons
    # ------------------------------------------------------------------

    def summarize_comparisons(
        self,
        comparisons: List[FieldComparison],
    ) -> ComparisonSummary:
        """Produce an aggregated summary of field comparisons.

        Counts matches, within-tolerance, mismatches, missing, and
        incomparable results.  Computes the match rate as:
        ``(matches + within_tolerance) / total_fields``.

        Args:
            comparisons: List of FieldComparison results from a single
                record pair or aggregated from multiple pairs.

        Returns:
            ComparisonSummary with counts and match_rate.

        Example:
            >>> engine = ComparisonEngine()
            >>> fc1 = engine.compare_numeric(100, 100)
            >>> fc2 = engine.compare_string("a", "b")
            >>> summary = engine.summarize_comparisons([fc1, fc2])
            >>> assert summary.total_fields == 2
        """
        total = len(comparisons)
        matches = 0
        within_tol = 0
        mismatches = 0
        missing = 0
        incomparable = 0
        severity_counts: Dict[str, int] = {}

        for fc in comparisons:
            status = fc.status
            if status == ComparisonResult.MATCH.value:
                matches += 1
            elif status == ComparisonResult.WITHIN_TOLERANCE.value:
                within_tol += 1
            elif status == ComparisonResult.MISMATCH.value:
                mismatches += 1
            elif status in (
                ComparisonResult.MISSING_LEFT.value,
                ComparisonResult.MISSING_RIGHT.value,
                ComparisonResult.MISSING_BOTH.value,
            ):
                missing += 1
            elif status == ComparisonResult.INCOMPARABLE.value:
                incomparable += 1

            sev = fc.severity
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        match_rate = 0.0
        if total > 0:
            match_rate = (matches + within_tol) / total

        summary = ComparisonSummary(
            total_fields=total,
            matches=matches,
            within_tolerance=within_tol,
            mismatches=mismatches,
            missing=missing,
            incomparable=incomparable,
            match_rate=round(match_rate, 6),
            severity_counts=severity_counts,
            provenance_hash=self._compute_provenance(
                "summarize_comparisons",
                {"total": total, "matches": matches},
                {"match_rate": match_rate},
            ),
        )

        logger.debug(
            "Comparison summary: total=%d matches=%d within_tol=%d "
            "mismatches=%d missing=%d incomparable=%d rate=%.4f",
            total, matches, within_tol, mismatches, missing,
            incomparable, match_rate,
        )
        return summary

    # ------------------------------------------------------------------
    # 11. classify_severity
    # ------------------------------------------------------------------

    def classify_severity(
        self,
        field_comparison: FieldComparison,
        critical_pct: float = 50.0,
        high_pct: float = 20.0,
        medium_pct: float = 5.0,
    ) -> DiscrepancySeverity:
        """Classify the severity of a field discrepancy.

        Uses the relative difference percentage to assign a severity
        level.  Non-numeric or missing comparisons are classified as
        INFO (for missing) or based on mismatch status.

        Args:
            field_comparison: The FieldComparison to classify.
            critical_pct: Minimum relative diff % for CRITICAL.
            high_pct: Minimum relative diff % for HIGH.
            medium_pct: Minimum relative diff % for MEDIUM.

        Returns:
            DiscrepancySeverity enum value.

        Example:
            >>> engine = ComparisonEngine()
            >>> fc = engine.compare_numeric(100.0, 200.0)
            >>> sev = engine.classify_severity(fc)
            >>> assert sev == DiscrepancySeverity.CRITICAL
        """
        status = field_comparison.status

        # Exact match or both missing: no discrepancy
        if status == ComparisonResult.MATCH.value:
            return DiscrepancySeverity.NONE

        if status == ComparisonResult.MISSING_BOTH.value:
            return DiscrepancySeverity.INFO

        if status in (
            ComparisonResult.MISSING_LEFT.value,
            ComparisonResult.MISSING_RIGHT.value,
        ):
            return DiscrepancySeverity.MEDIUM

        if status == ComparisonResult.INCOMPARABLE.value:
            return DiscrepancySeverity.INFO

        # within_tolerance or mismatch: use relative diff
        rel_diff = field_comparison.relative_diff_pct
        if rel_diff is not None:
            if rel_diff >= critical_pct:
                return DiscrepancySeverity.CRITICAL
            if rel_diff >= high_pct:
                return DiscrepancySeverity.HIGH
            if rel_diff >= medium_pct:
                return DiscrepancySeverity.MEDIUM
            if rel_diff > 0:
                return DiscrepancySeverity.LOW
            return DiscrepancySeverity.NONE

        # Non-numeric mismatch without relative diff
        if status == ComparisonResult.MISMATCH.value:
            return DiscrepancySeverity.MEDIUM

        if status == ComparisonResult.WITHIN_TOLERANCE.value:
            return DiscrepancySeverity.LOW

        return DiscrepancySeverity.INFO

    # ------------------------------------------------------------------
    # 12. _parse_date (private)
    # ------------------------------------------------------------------

    def _parse_date(
        self,
        value: Any,
        formats: Tuple[str, ...] = _DEFAULT_DATE_FORMATS,
    ) -> Optional[date]:
        """Parse a value into a date object trying multiple formats.

        Handles datetime, date, and string inputs.  For strings, tries
        each format in order and returns the first successful parse.

        Args:
            value: Value to parse (str, date, or datetime).
            formats: Tuple of strptime format strings to try.

        Returns:
            A date object or None if parsing fails.
        """
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value

        if not isinstance(value, str):
            value = str(value)

        value_str = value.strip()
        if not value_str:
            return None

        for fmt in formats:
            try:
                parsed_dt = datetime.strptime(value_str, fmt)
                return parsed_dt.date()
            except (ValueError, TypeError):
                continue

        logger.debug("Could not parse date from '%s' with any known format", value_str)
        return None

    # ------------------------------------------------------------------
    # 13. _convert_unit (private)
    # ------------------------------------------------------------------

    def _convert_unit(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        custom_conversions: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Optional[float]:
        """Convert a value from one unit to another.

        First checks the built-in UNIT_CONVERSIONS table, then any
        custom conversion tables.  Both units must be in the same
        dimension for conversion to succeed.

        Args:
            value: Numeric value to convert.
            from_unit: Source unit label.
            to_unit: Target unit label.
            custom_conversions: Optional custom conversion tables.

        Returns:
            Converted float value or None if conversion is not possible.
        """
        if not from_unit or not to_unit:
            return value if from_unit == to_unit else None

        from_lower = from_unit.lower().strip()
        to_lower = to_unit.lower().strip()

        # Same unit: no conversion needed
        if from_lower == to_lower:
            return value

        # Look up both units in the combined conversion tables
        all_conversions = dict(UNIT_CONVERSIONS)
        if custom_conversions:
            for dim, units in custom_conversions.items():
                if dim in all_conversions:
                    merged = dict(all_conversions[dim])
                    merged.update(units)
                    all_conversions[dim] = merged
                else:
                    all_conversions[dim] = dict(units)

        # Find the dimension and multipliers for both units
        from_info = self._find_unit_info(from_lower, all_conversions)
        to_info = self._find_unit_info(to_lower, all_conversions)

        if from_info is None or to_info is None:
            return None

        from_dim, from_mult = from_info
        to_dim, to_mult = to_info

        if from_dim != to_dim:
            logger.debug(
                "Cannot convert between dimensions: %s (%s) -> %s (%s)",
                from_unit, from_dim, to_unit, to_dim,
            )
            return None

        # Convert: value * from_multiplier / to_multiplier
        # from_multiplier converts from_unit to canonical
        # to_multiplier converts to_unit to canonical
        # So canonical = value * from_mult
        # target = canonical / to_mult
        canonical_value = value * from_mult
        return canonical_value / to_mult

    def _to_canonical_unit(
        self,
        value: float,
        unit: str,
        custom_conversions: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Optional[float]:
        """Convert a value to its canonical unit.

        Looks up the unit in the conversion tables and multiplies by
        the appropriate factor to convert to the dimension's canonical
        unit (e.g., kg for mass, kWh for energy).

        Args:
            value: Numeric value to convert.
            unit: Source unit label.
            custom_conversions: Optional custom conversion tables.

        Returns:
            Converted float value or None if the unit is not found.
        """
        if not unit:
            return value

        unit_lower = unit.lower().strip()

        all_conversions = dict(UNIT_CONVERSIONS)
        if custom_conversions:
            for dim, units in custom_conversions.items():
                if dim in all_conversions:
                    merged = dict(all_conversions[dim])
                    merged.update(units)
                    all_conversions[dim] = merged
                else:
                    all_conversions[dim] = dict(units)

        info = self._find_unit_info(unit_lower, all_conversions)
        if info is None:
            return None

        _, multiplier = info
        return value * multiplier

    def _find_unit_info(
        self,
        unit_lower: str,
        all_conversions: Dict[str, Dict[str, float]],
    ) -> Optional[Tuple[str, float]]:
        """Find the dimension and multiplier for a unit label.

        Searches through all conversion tables (case-insensitive)
        for the given unit name.

        Args:
            unit_lower: Lowercase unit label to look up.
            all_conversions: Combined built-in and custom conversion tables.

        Returns:
            Tuple of (dimension_name, multiplier) or None if not found.
        """
        for dim, units in all_conversions.items():
            for unit_name, multiplier in units.items():
                if unit_name.lower() == unit_lower:
                    return (dim, multiplier)
        return None

    # ------------------------------------------------------------------
    # 14. _convert_currency (private)
    # ------------------------------------------------------------------

    def _convert_currency(
        self,
        value: float,
        from_currency: str,
        to_currency: str,
        exchange_rates: Dict[str, float],
    ) -> Optional[float]:
        """Convert a monetary value between currencies.

        Uses the exchange_rates dictionary where each key is a currency
        code and the value is the rate to convert 1 unit of that
        currency to the base currency (USD by default).

        Args:
            value: Monetary amount to convert.
            from_currency: Source currency code (uppercase).
            to_currency: Target currency code (uppercase).
            exchange_rates: Mapping of currency codes to base rates.

        Returns:
            Converted float value or None if the rate is not available.
        """
        if from_currency == to_currency:
            return value

        from_rate = exchange_rates.get(from_currency)
        to_rate = exchange_rates.get(to_currency)

        if from_rate is None:
            logger.debug(
                "Missing exchange rate for source currency: %s",
                from_currency,
            )
            return None

        if to_rate is None:
            logger.debug(
                "Missing exchange rate for target currency: %s",
                to_currency,
            )
            return None

        if to_rate == 0.0:
            logger.warning(
                "Exchange rate for target currency %s is zero",
                to_currency,
            )
            return None

        # value_in_base = value * from_rate
        # value_in_target = value_in_base / to_rate
        return (value * from_rate) / to_rate

    # ------------------------------------------------------------------
    # 15. _compute_provenance (private)
    # ------------------------------------------------------------------

    def _compute_provenance(
        self,
        operation: str,
        input_data: Any,
        output_data: Any,
    ) -> str:
        """Compute a SHA-256 provenance hash for an operation.

        Combines the operation name, input data, output data, and
        current UTC timestamp into a deterministic hash.

        Args:
            operation: Name of the operation performed.
            input_data: Input data for the operation.
            output_data: Output data from the operation.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        timestamp = _utcnow().isoformat()
        payload = {
            "operation": operation,
            "input": input_data,
            "output": output_data,
            "timestamp": timestamp,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compare_field(
        self,
        field_name: str,
        value_a: Any,
        value_b: Any,
        field_type: FieldType,
        rule: ToleranceRule,
    ) -> FieldComparison:
        """Route a field comparison to the appropriate type-specific method.

        Args:
            field_name: Name of the field being compared.
            value_a: Value from source A.
            value_b: Value from source B.
            field_type: Type of comparison to perform.
            rule: Tolerance rule for this field.

        Returns:
            FieldComparison result.
        """
        if field_type == FieldType.NUMERIC:
            fc = self.compare_numeric(
                value_a=value_a,
                value_b=value_b,
                tolerance_abs=rule.tolerance_abs,
                tolerance_pct=rule.tolerance_pct,
                rounding_digits=rule.rounding_digits,
                field_name=field_name,
            )
        elif field_type == FieldType.STRING:
            fc = self.compare_string(
                value_a=value_a,
                value_b=value_b,
                case_sensitive=rule.case_sensitive,
                strip_whitespace=rule.strip_whitespace,
                field_name=field_name,
            )
        elif field_type == FieldType.DATE:
            fc = self.compare_date(
                value_a=value_a,
                value_b=value_b,
                max_days_diff=rule.max_days_diff,
                date_format_a=rule.date_format_a,
                date_format_b=rule.date_format_b,
                field_name=field_name,
            )
        elif field_type == FieldType.BOOLEAN:
            fc = self.compare_boolean(
                value_a=value_a,
                value_b=value_b,
                field_name=field_name,
            )
        elif field_type == FieldType.CATEGORICAL:
            fc = self.compare_categorical(
                value_a=value_a,
                value_b=value_b,
                synonyms=rule.synonyms,
                field_name=field_name,
            )
        elif field_type == FieldType.CURRENCY:
            fc = self.compare_currency(
                value_a=value_a,
                value_b=value_b,
                currency_a=rule.currency_a,
                currency_b=rule.currency_b,
                tolerance_pct=rule.tolerance_pct,
                field_name=field_name,
            )
        elif field_type == FieldType.UNIT_VALUE:
            fc = self.compare_unit_value(
                value_a=value_a,
                value_b=value_b,
                unit_a=rule.unit_a,
                unit_b=rule.unit_b,
                tolerance_pct=rule.tolerance_pct,
                tolerance_abs=rule.tolerance_abs,
                field_name=field_name,
            )
        else:
            # Fallback to string comparison for unknown types
            logger.warning(
                "Unknown field type '%s' for field '%s', falling back to string",
                field_type, field_name,
            )
            fc = self.compare_string(
                value_a=value_a,
                value_b=value_b,
                case_sensitive=rule.case_sensitive,
                strip_whitespace=rule.strip_whitespace,
                field_name=field_name,
            )

        # Classify severity
        severity = self.classify_severity(fc)
        fc.severity = severity.value

        return fc

    def _evaluate_numeric_tolerance(
        self,
        abs_diff: float,
        rel_diff_pct: float,
        tolerance_abs: Optional[float],
        tolerance_pct: Optional[float],
    ) -> ComparisonResult:
        """Evaluate a numeric difference against tolerance thresholds.

        Both absolute and percentage differences must be zero for an
        exact MATCH.  If either absolute or percentage tolerance is
        provided and the difference is within the specified threshold,
        the result is WITHIN_TOLERANCE.  Otherwise MISMATCH.

        Args:
            abs_diff: Absolute difference between values.
            rel_diff_pct: Relative difference as percentage.
            tolerance_abs: Maximum absolute tolerance (or None).
            tolerance_pct: Maximum percentage tolerance (or None).

        Returns:
            ComparisonResult status enum value.
        """
        # Exact match check (within floating-point precision)
        if abs_diff < 1e-12:
            return ComparisonResult.MATCH

        # Check tolerances
        within_abs = True
        within_pct = True

        if tolerance_abs is not None:
            within_abs = abs_diff <= tolerance_abs

        if tolerance_pct is not None:
            within_pct = rel_diff_pct <= tolerance_pct

        # Both tolerances specified: must satisfy both
        if tolerance_abs is not None and tolerance_pct is not None:
            if within_abs and within_pct:
                return ComparisonResult.WITHIN_TOLERANCE
            return ComparisonResult.MISMATCH

        # Single tolerance specified: must satisfy it
        if tolerance_abs is not None:
            if within_abs:
                return ComparisonResult.WITHIN_TOLERANCE
            return ComparisonResult.MISMATCH

        if tolerance_pct is not None:
            if within_pct:
                return ComparisonResult.WITHIN_TOLERANCE
            return ComparisonResult.MISMATCH

        # No tolerance specified: any non-zero diff is mismatch
        return ComparisonResult.MISMATCH

    def _normalize_string(
        self,
        value: str,
        case_sensitive: bool,
        strip_whitespace: bool,
    ) -> str:
        """Normalize a string for comparison.

        Args:
            value: String to normalize.
            case_sensitive: If False, convert to lowercase.
            strip_whitespace: If True, strip leading/trailing whitespace.

        Returns:
            Normalized string.
        """
        result = value
        if strip_whitespace:
            result = result.strip()
        if not case_sensitive:
            result = result.lower()
        return result

    def _are_synonymous(
        self,
        str_a: str,
        str_b: str,
        synonyms: Dict[str, List[str]],
    ) -> bool:
        """Check if two strings are synonymous.

        Builds a bi-directional synonym lookup from the provided mapping
        and checks whether both values resolve to the same canonical term.

        Args:
            str_a: First string (lowercase).
            str_b: Second string (lowercase).
            synonyms: Mapping of canonical term to list of synonyms.

        Returns:
            True if the values are synonymous, False otherwise.
        """
        if not synonyms:
            return False

        # Build reverse lookup: synonym -> canonical
        reverse: Dict[str, str] = {}
        for canonical, syn_list in synonyms.items():
            canonical_lower = canonical.lower()
            reverse[canonical_lower] = canonical_lower
            for syn in syn_list:
                reverse[syn.lower()] = canonical_lower

        # Resolve both values to canonical form
        canonical_a = reverse.get(str_a, str_a)
        canonical_b = reverse.get(str_b, str_b)

        return canonical_a == canonical_b

    def _format_numeric_message(
        self,
        float_a: float,
        float_b: float,
        abs_diff: float,
        rel_diff_pct: float,
        status: ComparisonResult,
    ) -> str:
        """Format a human-readable numeric comparison message.

        Args:
            float_a: Parsed numeric value A.
            float_b: Parsed numeric value B.
            abs_diff: Absolute difference.
            rel_diff_pct: Relative difference percentage.
            status: Comparison result status.

        Returns:
            Formatted message string.
        """
        if status == ComparisonResult.MATCH:
            return f"Exact numeric match: {float_a}"
        if status == ComparisonResult.WITHIN_TOLERANCE:
            return (
                f"Within tolerance: {float_a} vs {float_b} "
                f"(diff={abs_diff:.6g}, {rel_diff_pct:.2f}%)"
            )
        return (
            f"Numeric mismatch: {float_a} vs {float_b} "
            f"(diff={abs_diff:.6g}, {rel_diff_pct:.2f}%)"
        )

    def _make_comparison(
        self,
        field_name: str = "",
        field_type: str = "",
        value_a: Any = None,
        value_b: Any = None,
        normalized_a: Any = None,
        normalized_b: Any = None,
        status: Union[ComparisonResult, str] = ComparisonResult.MATCH,
        absolute_diff: Optional[float] = None,
        relative_diff_pct: Optional[float] = None,
        severity: Union[DiscrepancySeverity, str] = DiscrepancySeverity.NONE,
        message: str = "",
    ) -> FieldComparison:
        """Create a FieldComparison with provenance hash.

        Centralised factory method to avoid duplication across
        comparison methods.

        Args:
            field_name: Name of the field compared.
            field_type: Type of comparison performed.
            value_a: Original value from source A.
            value_b: Original value from source B.
            normalized_a: Normalised value from source A.
            normalized_b: Normalised value from source B.
            status: Comparison outcome.
            absolute_diff: Absolute numeric difference.
            relative_diff_pct: Relative difference percentage.
            severity: Discrepancy severity.
            message: Human-readable message.

        Returns:
            Populated FieldComparison instance.
        """
        status_val = status.value if isinstance(status, ComparisonResult) else status
        severity_val = (
            severity.value if isinstance(severity, DiscrepancySeverity) else severity
        )

        provenance_hash = self._compute_provenance(
            operation=f"compare_{field_type}",
            input_data={
                "field_name": field_name,
                "value_a": value_a,
                "value_b": value_b,
            },
            output_data={
                "status": status_val,
                "absolute_diff": absolute_diff,
                "relative_diff_pct": relative_diff_pct,
            },
        )

        return FieldComparison(
            field_name=field_name,
            field_type=field_type,
            value_a=value_a,
            value_b=value_b,
            normalized_a=normalized_a,
            normalized_b=normalized_b,
            status=status_val,
            absolute_diff=absolute_diff,
            relative_diff_pct=relative_diff_pct,
            severity=severity_val,
            message=message,
            provenance_hash=provenance_hash,
        )

    def _record_metric_and_provenance(
        self,
        operation: str,
        start_time: float,
        result: FieldComparison,
    ) -> None:
        """Record metric observation and provenance for a comparison.

        Args:
            operation: Name of the comparison operation.
            start_time: Monotonic start time from time.monotonic().
            result: The FieldComparison result.
        """
        elapsed = time.monotonic() - start_time
        _inc_comparisons(operation, 1)
        _observe_duration(elapsed)

        if self._provenance is not None:
            try:
                input_hash = _build_hash({
                    "field": result.field_name,
                    "a": result.value_a,
                    "b": result.value_b,
                })
                output_hash = _build_hash({
                    "status": result.status,
                    "diff": result.absolute_diff,
                })
                self._provenance.add_to_chain(
                    operation=operation,
                    input_hash=input_hash,
                    output_hash=output_hash,
                    metadata={"field_name": result.field_name},
                )
            except Exception:
                logger.debug(
                    "Provenance recording skipped for %s", operation,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def comparison_count(self) -> int:
        """Return the total number of individual comparisons performed."""
        return self._comparison_count

    @property
    def provenance_chain_length(self) -> int:
        """Return the length of the provenance chain."""
        if self._provenance is not None:
            try:
                return self._provenance.get_chain_length()
            except Exception:
                return 0
        return 0
