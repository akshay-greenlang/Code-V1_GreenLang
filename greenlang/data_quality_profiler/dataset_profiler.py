# -*- coding: utf-8 -*-
"""
Dataset Profiler Engine - AGENT-DATA-010: Data Quality Profiler (GL-DATA-X-013)

Schema inference and statistical profiling per column. Detects data types,
computes descriptive statistics (min, max, mean, median, stddev, percentiles,
skewness, kurtosis), cardinality analysis, and format-pattern recognition
for 13 data types and 20+ regex patterns.

Zero-Hallucination Guarantees:
    - All statistics use deterministic Python arithmetic (statistics module)
    - Data type detection uses rule-based regex matching only
    - Schema hashing uses SHA-256 for reproducibility
    - No ML/LLM calls in profiling path
    - Provenance recorded for every profile mutation

Supported Data Types (13):
    string, integer, float, boolean, date, datetime, email, url, phone,
    ip_address, uuid, json_str, unknown

Example:
    >>> from greenlang.data_quality_profiler.dataset_profiler import DatasetProfiler
    >>> profiler = DatasetProfiler()
    >>> data = [
    ...     {"name": "Alice", "age": 30, "email": "alice@example.com"},
    ...     {"name": "Bob", "age": 25, "email": "bob@example.com"},
    ... ]
    >>> profile = profiler.profile(data, "users")
    >>> print(profile["dataset_name"], profile["row_count"])

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
import statistics
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetProfiler",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "PRF") -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        String of the form ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash for a profiling operation.

    Args:
        operation: Name of the operation (e.g. 'profile', 'profile_column').
        data_repr: Serialised representation of the data involved.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Constants - Data Type Detection
# ---------------------------------------------------------------------------

# Canonical data type names
DATA_TYPE_STRING = "string"
DATA_TYPE_INTEGER = "integer"
DATA_TYPE_FLOAT = "float"
DATA_TYPE_BOOLEAN = "boolean"
DATA_TYPE_DATE = "date"
DATA_TYPE_DATETIME = "datetime"
DATA_TYPE_EMAIL = "email"
DATA_TYPE_URL = "url"
DATA_TYPE_PHONE = "phone"
DATA_TYPE_IP_ADDRESS = "ip_address"
DATA_TYPE_UUID = "uuid"
DATA_TYPE_JSON_STR = "json_str"
DATA_TYPE_UNKNOWN = "unknown"

ALL_DATA_TYPES = frozenset({
    DATA_TYPE_STRING, DATA_TYPE_INTEGER, DATA_TYPE_FLOAT, DATA_TYPE_BOOLEAN,
    DATA_TYPE_DATE, DATA_TYPE_DATETIME, DATA_TYPE_EMAIL, DATA_TYPE_URL,
    DATA_TYPE_PHONE, DATA_TYPE_IP_ADDRESS, DATA_TYPE_UUID, DATA_TYPE_JSON_STR,
    DATA_TYPE_UNKNOWN,
})


# ---------------------------------------------------------------------------
# Pattern Regexes (20+)
# ---------------------------------------------------------------------------

_RE_EMAIL = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)
_RE_PHONE_INTL = re.compile(
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
_RE_DATE_ISO_SLASH = re.compile(
    r"^\d{4}/(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])$"
)
_RE_DATE_DOT = re.compile(
    r"^(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\.\d{4}$"
)
_RE_DATETIME_ISO = re.compile(
    r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+\-]\d{2}:?\d{2})?$"
)
_RE_DATETIME_US = re.compile(
    r"^(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/\d{4}"
    r"\s+\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM|am|pm)?$"
)
_RE_CURRENCY = re.compile(
    r"^[\$\u20ac\u00a3\u00a5]\s?[\d,]+\.?\d*$"
)
_RE_PERCENTAGE = re.compile(
    r"^-?\d+\.?\d*\s*%$"
)
_RE_HEX_COLOR = re.compile(
    r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$"
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
_RE_CREDIT_CARD = re.compile(
    r"^\d{13,19}$"
)
_RE_INTEGER = re.compile(
    r"^-?\d+$"
)
_RE_FLOAT = re.compile(
    r"^-?\d+\.\d+$"
)

# Boolean string representations
_BOOLEAN_TRUE = frozenset({"true", "yes", "1", "t", "y", "on"})
_BOOLEAN_FALSE = frozenset({"false", "no", "0", "f", "n", "off"})
_BOOLEAN_ALL = _BOOLEAN_TRUE | _BOOLEAN_FALSE

# Pattern name -> regex mapping for detect_patterns
_PATTERN_REGISTRY: Dict[str, re.Pattern[str]] = {
    "email": _RE_EMAIL,
    "phone": _RE_PHONE_INTL,
    "url": _RE_URL,
    "ipv4": _RE_IPV4,
    "ipv6": _RE_IPV6,
    "uuid": _RE_UUID,
    "date_iso": _RE_DATE_ISO,
    "date_us": _RE_DATE_US,
    "date_eu": _RE_DATE_EU,
    "date_iso_slash": _RE_DATE_ISO_SLASH,
    "date_dot": _RE_DATE_DOT,
    "datetime_iso": _RE_DATETIME_ISO,
    "datetime_us": _RE_DATETIME_US,
    "currency": _RE_CURRENCY,
    "percentage": _RE_PERCENTAGE,
    "hex_color": _RE_HEX_COLOR,
    "zip_us": _RE_ZIP_US,
    "zip_uk": _RE_ZIP_UK,
    "country_iso2": _RE_COUNTRY_ISO2,
    "country_iso3": _RE_COUNTRY_ISO3,
    "latitude": _RE_LATITUDE,
    "longitude": _RE_LONGITUDE,
    "credit_card": _RE_CREDIT_CARD,
}

# Percentile levels to compute
_PERCENTILE_LEVELS = (25, 50, 75, 95, 99)


# ---------------------------------------------------------------------------
# Statistical Helpers (pure Python, no numpy/scipy)
# ---------------------------------------------------------------------------


def _safe_mean(values: List[float]) -> float:
    """Compute mean, returning 0.0 for empty lists.

    Args:
        values: List of numeric values.

    Returns:
        Arithmetic mean or 0.0.
    """
    if not values:
        return 0.0
    return statistics.mean(values)


def _safe_median(values: List[float]) -> float:
    """Compute median, returning 0.0 for empty lists.

    Args:
        values: List of numeric values.

    Returns:
        Median or 0.0.
    """
    if not values:
        return 0.0
    return statistics.median(values)


def _safe_stdev(values: List[float]) -> float:
    """Compute sample standard deviation, 0.0 for < 2 values.

    Args:
        values: List of numeric values.

    Returns:
        Sample standard deviation or 0.0.
    """
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _safe_variance(values: List[float]) -> float:
    """Compute sample variance, 0.0 for < 2 values.

    Args:
        values: List of numeric values.

    Returns:
        Sample variance or 0.0.
    """
    if len(values) < 2:
        return 0.0
    return statistics.variance(values)


def _percentile(sorted_values: List[float], p: float) -> float:
    """Compute the p-th percentile using linear interpolation.

    Args:
        sorted_values: Pre-sorted list of numeric values.
        p: Percentile in [0, 100].

    Returns:
        Interpolated percentile value, or 0.0 for empty input.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1


def _skewness(values: List[float]) -> float:
    """Compute sample skewness (Fisher's definition).

    Uses the adjusted Fisher-Pearson standardised moment coefficient:
    G1 = (n / ((n-1)(n-2))) * sum(((x_i - mean) / std)^3)

    Args:
        values: List of numeric values.

    Returns:
        Skewness coefficient or 0.0 if insufficient data.
    """
    n = len(values)
    if n < 3:
        return 0.0
    mean_val = statistics.mean(values)
    std_val = _safe_stdev(values)
    if std_val == 0.0:
        return 0.0
    m3 = sum(((x - mean_val) / std_val) ** 3 for x in values)
    return (n / ((n - 1) * (n - 2))) * m3


def _kurtosis(values: List[float]) -> float:
    """Compute excess kurtosis (Fisher's definition).

    Uses the formula:
    K = ((n(n+1)) / ((n-1)(n-2)(n-3))) * sum(((x_i - mean)/std)^4)
        - (3(n-1)^2) / ((n-2)(n-3))

    Args:
        values: List of numeric values.

    Returns:
        Excess kurtosis or 0.0 if insufficient data.
    """
    n = len(values)
    if n < 4:
        return 0.0
    mean_val = statistics.mean(values)
    std_val = _safe_stdev(values)
    if std_val == 0.0:
        return 0.0
    m4 = sum(((x - mean_val) / std_val) ** 4 for x in values)
    term1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    term2 = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return term1 * m4 - term2


def _is_json_string(value: str) -> bool:
    """Check if a string is valid JSON (object or array).

    Args:
        value: String to test.

    Returns:
        True if the string parses as a JSON object or array.
    """
    stripped = value.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return False
    try:
        json.loads(stripped)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _try_parse_number(value: str) -> Optional[float]:
    """Try to parse a string as a number.

    Args:
        value: String to parse.

    Returns:
        Float value or None if not a number.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# DatasetProfiler Engine
# ---------------------------------------------------------------------------


class DatasetProfiler:
    """Schema inference and statistical profiling engine.

    Profiles datasets by computing per-column statistics including data type
    detection, descriptive statistics (mean, median, stddev, percentiles,
    skewness, kurtosis), cardinality analysis, and pattern detection using
    20+ regex patterns across 13 data types.

    Thread-safe: all mutations to internal storage are protected by a
    threading lock. SHA-256 provenance hashes are computed on every profile
    creation for full audit trail support.

    Attributes:
        _config: Configuration dictionary.
        _lock: Threading lock for thread-safe storage access.
        _profiles: In-memory storage of computed profiles.
        _stats: Aggregate profiling statistics.

    Example:
        >>> profiler = DatasetProfiler()
        >>> data = [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]
        >>> profile = profiler.profile(data, "test")
        >>> assert profile["profile_id"].startswith("PRF-")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DatasetProfiler.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``max_sample_size``: int, max rows to sample (default 100000)
                - ``top_n_values``: int, top-N for cardinality (default 10)
                - ``pattern_sample_size``: int, rows for pattern detect (default 1000)
        """
        self._config = config or {}
        self._max_sample_size: int = self._config.get("max_sample_size", 100_000)
        self._top_n: int = self._config.get("top_n_values", 10)
        self._pattern_sample_size: int = self._config.get("pattern_sample_size", 1000)
        self._lock = threading.Lock()
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, Any] = {
            "profiles_created": 0,
            "columns_profiled": 0,
            "total_rows_profiled": 0,
            "total_profiling_time_ms": 0.0,
        }
        logger.info(
            "DatasetProfiler initialized: max_sample=%d, top_n=%d",
            self._max_sample_size, self._top_n,
        )

    # ------------------------------------------------------------------
    # Public API - Full Dataset Profiling
    # ------------------------------------------------------------------

    def profile(
        self,
        data: List[Dict[str, Any]],
        dataset_name: str,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Profile an entire dataset and return a DatasetProfile dict.

        Computes per-column statistics, detects data types, computes
        cardinality and patterns, and records provenance.

        Args:
            data: List of row dictionaries to profile.
            dataset_name: Human-readable dataset name.
            columns: Optional subset of columns to profile. If None,
                all columns from the first row are used.

        Returns:
            DatasetProfile dictionary with keys: profile_id, dataset_name,
            row_count, column_count, columns (per-column profiles),
            schema_hash, memory_estimate_bytes, provenance_hash, created_at.

        Raises:
            ValueError: If data is empty.
        """
        start = time.monotonic()
        if not data:
            raise ValueError("Cannot profile empty dataset")

        profile_id = _generate_id("PRF")

        # Determine columns
        all_keys = columns if columns else list(data[0].keys())
        row_count = len(data)

        # Sample if needed
        sample = data[:self._max_sample_size] if row_count > self._max_sample_size else data

        # Profile each column
        column_profiles: Dict[str, Dict[str, Any]] = {}
        for col in all_keys:
            values = [row.get(col) for row in sample]
            column_profiles[col] = self.profile_column(values, col)

        # Schema hash
        columns_info = [
            {"name": col, "type": column_profiles[col].get("inferred_type", DATA_TYPE_UNKNOWN)}
            for col in all_keys
        ]
        schema_hash = self.get_schema_hash(columns_info)

        # Memory estimate
        memory_bytes = self.estimate_memory(data)

        # Provenance
        provenance_data = json.dumps({
            "dataset_name": dataset_name,
            "row_count": row_count,
            "columns": all_keys,
            "schema_hash": schema_hash,
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("profile", provenance_data)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        result: Dict[str, Any] = {
            "profile_id": profile_id,
            "dataset_name": dataset_name,
            "row_count": row_count,
            "column_count": len(all_keys),
            "columns": column_profiles,
            "schema_hash": schema_hash,
            "memory_estimate_bytes": memory_bytes,
            "provenance_hash": provenance_hash,
            "profiling_time_ms": round(elapsed_ms, 2),
            "sampled": row_count > self._max_sample_size,
            "sample_size": len(sample),
            "created_at": _utcnow().isoformat(),
        }

        # Store and update stats
        with self._lock:
            self._profiles[profile_id] = result
            self._stats["profiles_created"] += 1
            self._stats["columns_profiled"] += len(all_keys)
            self._stats["total_rows_profiled"] += row_count
            self._stats["total_profiling_time_ms"] += elapsed_ms

        logger.info(
            "Profile created: id=%s, dataset=%s, rows=%d, cols=%d, time=%.1fms",
            profile_id, dataset_name, row_count, len(all_keys), elapsed_ms,
        )
        return result

    def profile_column(
        self,
        values: List[Any],
        column_name: str,
    ) -> Dict[str, Any]:
        """Profile a single column and return a ColumnProfile dict.

        Args:
            values: List of values for this column (may contain None).
            column_name: Name of the column.

        Returns:
            ColumnProfile dict with keys: column_name, inferred_type,
            total_count, null_count, null_rate, distinct_count,
            statistics, cardinality, patterns, provenance_hash.
        """
        total = len(values)
        non_null = [v for v in values if v is not None and v != ""]
        null_count = total - len(non_null)
        null_rate = null_count / total if total > 0 else 0.0

        # Infer data type
        inferred_type = self.infer_data_type(non_null) if non_null else DATA_TYPE_UNKNOWN

        # Compute statistics
        stats = self.compute_statistics(non_null, inferred_type)

        # Compute cardinality
        cardinality = self.compute_cardinality(non_null)

        # Detect patterns (on string representations)
        str_values = [str(v) for v in non_null[:self._pattern_sample_size]]
        patterns = self.detect_patterns(str_values)

        provenance_data = json.dumps({
            "column_name": column_name,
            "total": total,
            "null_count": null_count,
            "inferred_type": inferred_type,
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("profile_column", provenance_data)

        return {
            "column_name": column_name,
            "inferred_type": inferred_type,
            "total_count": total,
            "null_count": null_count,
            "null_rate": round(null_rate, 4),
            "distinct_count": cardinality.get("unique_count", 0),
            "statistics": stats,
            "cardinality": cardinality,
            "patterns": patterns,
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # Data Type Detection
    # ------------------------------------------------------------------

    def infer_data_type(self, values: List[Any]) -> str:
        """Infer the dominant data type from a list of non-null values.

        Uses a voting approach: samples up to 1000 values, classifies each,
        and returns the type with the highest vote count.

        Args:
            values: List of non-null values to classify.

        Returns:
            One of the 13 canonical data type strings.
        """
        if not values:
            return DATA_TYPE_UNKNOWN

        sample = values[:1000]
        votes: Dict[str, int] = {}

        for v in sample:
            detected = self._classify_value(v)
            votes[detected] = votes.get(detected, 0) + 1

        if not votes:
            return DATA_TYPE_UNKNOWN

        # Return the type with the highest vote count
        return max(votes, key=lambda t: votes[t])

    def _classify_value(self, value: Any) -> str:
        """Classify a single value into one of the 13 data types.

        Args:
            value: A non-null value to classify.

        Returns:
            Data type string.
        """
        # Python native type checks first
        if isinstance(value, bool):
            return DATA_TYPE_BOOLEAN
        if isinstance(value, int):
            return DATA_TYPE_INTEGER
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return DATA_TYPE_FLOAT
            return DATA_TYPE_FLOAT

        # Convert to string for regex-based detection
        s = str(value).strip()
        if not s:
            return DATA_TYPE_UNKNOWN

        # Boolean strings
        if s.lower() in _BOOLEAN_ALL:
            return DATA_TYPE_BOOLEAN

        # UUID
        if _RE_UUID.match(s):
            return DATA_TYPE_UUID

        # Email
        if _RE_EMAIL.match(s):
            return DATA_TYPE_EMAIL

        # URL
        if _RE_URL.match(s):
            return DATA_TYPE_URL

        # IP addresses
        if _RE_IPV4.match(s):
            return DATA_TYPE_IP_ADDRESS
        if _RE_IPV6.match(s) or _RE_IPV6_COMPRESSED.match(s):
            return DATA_TYPE_IP_ADDRESS

        # Datetime (before date so more specific matches first)
        if _RE_DATETIME_ISO.match(s) or _RE_DATETIME_US.match(s):
            return DATA_TYPE_DATETIME

        # Date patterns
        if (_RE_DATE_ISO.match(s) or _RE_DATE_US.match(s) or
                _RE_DATE_EU.match(s) or _RE_DATE_ISO_SLASH.match(s) or
                _RE_DATE_DOT.match(s)):
            return DATA_TYPE_DATE

        # Phone
        if _RE_PHONE_INTL.match(s) and len(s) >= 7:
            return DATA_TYPE_PHONE

        # JSON string
        if _is_json_string(s):
            return DATA_TYPE_JSON_STR

        # Integer
        if _RE_INTEGER.match(s):
            return DATA_TYPE_INTEGER

        # Float
        if _RE_FLOAT.match(s):
            return DATA_TYPE_FLOAT

        # Default to string
        return DATA_TYPE_STRING

    # ------------------------------------------------------------------
    # Statistics Computation
    # ------------------------------------------------------------------

    def compute_statistics(
        self,
        values: List[Any],
        inferred_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute descriptive statistics for a list of values.

        For numeric types: min, max, mean, median, stddev, variance,
        percentiles (p25, p50, p75, p95, p99), skewness, kurtosis.
        For string types: min_length, max_length, avg_length, most_common.

        Args:
            values: List of non-null values.
            inferred_type: Optional pre-inferred type to guide computation.

        Returns:
            Dictionary of computed statistics.
        """
        if not values:
            return {"count": 0}

        detected = inferred_type or self.infer_data_type(values)
        result: Dict[str, Any] = {"count": len(values)}

        if detected in (DATA_TYPE_INTEGER, DATA_TYPE_FLOAT):
            result.update(self._compute_numeric_stats(values))
        else:
            result.update(self._compute_string_stats(values))

        return result

    def _compute_numeric_stats(self, values: List[Any]) -> Dict[str, Any]:
        """Compute numeric statistics for a column.

        Args:
            values: List of values (will be converted to floats).

        Returns:
            Dict with numeric statistics.
        """
        nums: List[float] = []
        for v in values:
            parsed = _try_parse_number(str(v)) if not isinstance(v, (int, float)) else float(v)
            if parsed is not None and not math.isnan(parsed) and not math.isinf(parsed):
                nums.append(parsed)

        if not nums:
            return {"numeric_count": 0}

        sorted_nums = sorted(nums)
        n = len(nums)
        mean_val = _safe_mean(nums)
        median_val = _safe_median(nums)
        std_val = _safe_stdev(nums)
        var_val = _safe_variance(nums)
        skew_val = _skewness(nums)
        kurt_val = _kurtosis(nums)

        return {
            "numeric_count": n,
            "min": sorted_nums[0],
            "max": sorted_nums[-1],
            "mean": round(mean_val, 6),
            "median": round(median_val, 6),
            "stddev": round(std_val, 6),
            "variance": round(var_val, 6),
            "p25": round(_percentile(sorted_nums, 25), 6),
            "p50": round(_percentile(sorted_nums, 50), 6),
            "p75": round(_percentile(sorted_nums, 75), 6),
            "p95": round(_percentile(sorted_nums, 95), 6),
            "p99": round(_percentile(sorted_nums, 99), 6),
            "skewness": round(skew_val, 6),
            "kurtosis": round(kurt_val, 6),
            "sum": round(sum(nums), 6),
            "range": round(sorted_nums[-1] - sorted_nums[0], 6),
        }

    def _compute_string_stats(self, values: List[Any]) -> Dict[str, Any]:
        """Compute string-oriented statistics for a column.

        Args:
            values: List of values (converted to strings).

        Returns:
            Dict with string statistics.
        """
        str_vals = [str(v) for v in values]
        lengths = [len(s) for s in str_vals]

        if not lengths:
            return {"string_count": 0}

        counter = Counter(str_vals)
        most_common = counter.most_common(self._top_n)

        return {
            "string_count": len(str_vals),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_length": round(_safe_mean([float(l) for l in lengths]), 2),
            "empty_count": sum(1 for s in str_vals if not s.strip()),
            "most_common": [
                {"value": val, "count": cnt} for val, cnt in most_common
            ],
        }

    # ------------------------------------------------------------------
    # Cardinality
    # ------------------------------------------------------------------

    def compute_cardinality(self, values: List[Any]) -> Dict[str, Any]:
        """Compute cardinality metrics for a list of values.

        Args:
            values: List of non-null values.

        Returns:
            Dict with unique_count, cardinality_ratio, is_unique,
            is_constant, most_common (top 10).
        """
        if not values:
            return {
                "unique_count": 0,
                "cardinality_ratio": 0.0,
                "is_unique": False,
                "is_constant": True,
                "most_common": [],
            }

        str_vals = [str(v) for v in values]
        counter = Counter(str_vals)
        unique_count = len(counter)
        total = len(str_vals)
        ratio = unique_count / total if total > 0 else 0.0

        most_common = counter.most_common(self._top_n)

        return {
            "unique_count": unique_count,
            "cardinality_ratio": round(ratio, 4),
            "is_unique": unique_count == total,
            "is_constant": unique_count <= 1,
            "most_common": [
                {"value": val, "count": cnt} for val, cnt in most_common
            ],
        }

    # ------------------------------------------------------------------
    # Pattern Detection
    # ------------------------------------------------------------------

    def detect_patterns(self, values: List[str]) -> Dict[str, Any]:
        """Detect format patterns present in string values.

        Checks all 20+ registered patterns and reports which ones
        match a significant portion of the values.

        Args:
            values: List of string values to test.

        Returns:
            Dict mapping pattern_name -> {match_count, match_rate, examples}.
        """
        if not values:
            return {}

        total = len(values)
        pattern_results: Dict[str, Any] = {}

        for pattern_name, regex in _PATTERN_REGISTRY.items():
            matches: List[str] = []
            for v in values:
                if regex.match(v):
                    matches.append(v)

            if matches:
                match_rate = len(matches) / total
                pattern_results[pattern_name] = {
                    "match_count": len(matches),
                    "match_rate": round(match_rate, 4),
                    "examples": matches[:3],
                }

        return pattern_results

    # ------------------------------------------------------------------
    # Schema Hashing
    # ------------------------------------------------------------------

    def get_schema_hash(self, columns_info: List[Dict[str, Any]]) -> str:
        """Compute a SHA-256 hash of a schema definition.

        The hash is deterministic for the same set of column names
        and types regardless of input order (sorted by name).

        Args:
            columns_info: List of dicts with 'name' and 'type' keys.

        Returns:
            Hex-encoded SHA-256 digest of the schema.
        """
        sorted_cols = sorted(columns_info, key=lambda c: c.get("name", ""))
        schema_str = json.dumps(sorted_cols, sort_keys=True, default=str)
        return hashlib.sha256(schema_str.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Memory Estimation
    # ------------------------------------------------------------------

    def estimate_memory(self, data: List[Dict[str, Any]]) -> int:
        """Estimate memory usage of a dataset in bytes.

        Uses a heuristic: serialise rows to JSON and sum byte lengths.
        Adds overhead per row for Python object bookkeeping.

        Args:
            data: List of row dictionaries.

        Returns:
            Estimated memory usage in bytes.
        """
        if not data:
            return 0

        # Sample up to 100 rows for estimation
        sample_size = min(100, len(data))
        sample = data[:sample_size]

        total_sample_bytes = 0
        for row in sample:
            total_sample_bytes += len(json.dumps(row, default=str).encode("utf-8"))

        avg_row_bytes = total_sample_bytes / sample_size
        # Python dict overhead estimate: ~240 bytes per dict + 60 bytes per key
        avg_keys = len(data[0]) if data else 0
        overhead = 240 + (60 * avg_keys)

        estimated = int((avg_row_bytes + overhead) * len(data))
        return estimated

    # ------------------------------------------------------------------
    # Profile Storage
    # ------------------------------------------------------------------

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored profile by ID.

        Args:
            profile_id: The profile identifier.

        Returns:
            Profile dict or None if not found.
        """
        with self._lock:
            return self._profiles.get(profile_id)

    def list_profiles(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List stored profiles with pagination.

        Args:
            limit: Maximum number of profiles to return.
            offset: Number of profiles to skip.

        Returns:
            List of profile dicts sorted by creation time descending.
        """
        with self._lock:
            all_profiles = sorted(
                self._profiles.values(),
                key=lambda p: p.get("created_at", ""),
                reverse=True,
            )
            return all_profiles[offset:offset + limit]

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a stored profile.

        Args:
            profile_id: The profile identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if profile_id in self._profiles:
                del self._profiles[profile_id]
                logger.info("Profile deleted: %s", profile_id)
                return True
            return False

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate profiling statistics.

        Returns:
            Dictionary with counters and averages for all profiling
            operations performed by this engine instance.
        """
        with self._lock:
            created = self._stats["profiles_created"]
            avg_time = (
                self._stats["total_profiling_time_ms"] / created
                if created > 0 else 0.0
            )
            return {
                "profiles_created": created,
                "columns_profiled": self._stats["columns_profiled"],
                "total_rows_profiled": self._stats["total_rows_profiled"],
                "total_profiling_time_ms": round(
                    self._stats["total_profiling_time_ms"], 2
                ),
                "avg_profiling_time_ms": round(avg_time, 2),
                "stored_profiles": len(self._profiles),
                "timestamp": _utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_column_values(
        self,
        data: List[Dict[str, Any]],
        column: str,
    ) -> List[Any]:
        """Extract all values for a column from data rows.

        Args:
            data: Row dictionaries.
            column: Column name.

        Returns:
            List of values (may include None).
        """
        return [row.get(column) for row in data]
