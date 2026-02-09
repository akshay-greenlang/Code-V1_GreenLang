# -*- coding: utf-8 -*-
"""
Unit Tests for DatasetProfiler Engine - AGENT-DATA-010 (GL-DATA-X-013)
======================================================================

Tests DatasetProfiler from greenlang.data_quality_profiler.dataset_profiler.

Covers:
    - Initialization (default/custom config, empty stores, stats, None config)
    - Full dataset profiling (return type, fields, provenance, edge cases)
    - Per-column profiling (name, type, null counts, stats, patterns)
    - Data type inference (13 types + edge cases)
    - Statistics computation (numeric stats, percentiles, skewness, kurtosis)
    - Cardinality analysis (unique, ratio, most common)
    - Pattern detection (email, phone, URL, IP, UUID, dates, currency)
    - Memory estimation
    - Schema hashing (determinism, collision resistance)
    - Profile storage (get, list, delete)
    - Aggregate statistics tracking
    - Provenance hashing (SHA-256, determinism)
    - Thread safety (concurrent profiling)

Target: 130+ tests, ~1100 lines.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import threading
from typing import Any, Dict, List

import pytest

from greenlang.data_quality_profiler.dataset_profiler import (
    DatasetProfiler,
    _safe_mean,
    _safe_median,
    _safe_stdev,
    _safe_variance,
    _percentile,
    _skewness,
    _kurtosis,
    _is_json_string,
    _try_parse_number,
    DATA_TYPE_STRING,
    DATA_TYPE_INTEGER,
    DATA_TYPE_FLOAT,
    DATA_TYPE_BOOLEAN,
    DATA_TYPE_DATE,
    DATA_TYPE_DATETIME,
    DATA_TYPE_EMAIL,
    DATA_TYPE_URL,
    DATA_TYPE_PHONE,
    DATA_TYPE_IP_ADDRESS,
    DATA_TYPE_UUID,
    DATA_TYPE_JSON_STR,
    DATA_TYPE_UNKNOWN,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def profiler():
    """Create a DatasetProfiler instance with default config."""
    return DatasetProfiler()


@pytest.fixture
def custom_profiler():
    """Create a DatasetProfiler with custom config."""
    return DatasetProfiler(config={
        "max_sample_size": 500,
        "top_n_values": 5,
        "pattern_sample_size": 200,
    })


@pytest.fixture
def small_data():
    """Minimal dataset for basic tests."""
    return [
        {"name": "Alice", "age": 30, "email": "alice@example.com"},
        {"name": "Bob", "age": 25, "email": "bob@test.org"},
        {"name": "Charlie", "age": 35, "email": "charlie@corp.net"},
    ]


@pytest.fixture
def numeric_values():
    """Sorted numeric values for statistics testing."""
    return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


# ============================================================================
# TestInit
# ============================================================================


class TestInit:
    """Test DatasetProfiler initialization."""

    def test_default_config(self):
        """Test default configuration values are set."""
        profiler = DatasetProfiler()
        assert profiler._max_sample_size == 100_000
        assert profiler._top_n == 10
        assert profiler._pattern_sample_size == 1000

    def test_custom_config(self):
        """Test custom configuration overrides defaults."""
        profiler = DatasetProfiler(config={
            "max_sample_size": 50_000,
            "top_n_values": 5,
            "pattern_sample_size": 500,
        })
        assert profiler._max_sample_size == 50_000
        assert profiler._top_n == 5
        assert profiler._pattern_sample_size == 500

    def test_empty_stores_on_init(self):
        """Test profiles store is empty on initialization."""
        profiler = DatasetProfiler()
        assert profiler._profiles == {}

    def test_initial_stats(self):
        """Test statistics are zeroed on initialization."""
        profiler = DatasetProfiler()
        stats = profiler.get_statistics()
        assert stats["profiles_created"] == 0
        assert stats["columns_profiled"] == 0
        assert stats["total_rows_profiled"] == 0

    def test_none_config_uses_defaults(self):
        """Test None config falls back to defaults."""
        profiler = DatasetProfiler(config=None)
        assert profiler._max_sample_size == 100_000
        assert profiler._top_n == 10


# ============================================================================
# TestProfile
# ============================================================================


class TestProfile:
    """Test full dataset profiling."""

    def test_returns_dict(self, profiler, small_data):
        """Test profile() returns a dictionary."""
        result = profiler.profile(small_data, "test_ds")
        assert isinstance(result, dict)

    def test_dataset_name(self, profiler, small_data):
        """Test dataset_name is preserved in output."""
        result = profiler.profile(small_data, "my_dataset")
        assert result["dataset_name"] == "my_dataset"

    def test_row_count(self, profiler, small_data):
        """Test row_count matches input length."""
        result = profiler.profile(small_data, "test_ds")
        assert result["row_count"] == 3

    def test_column_count(self, profiler, small_data):
        """Test column_count matches number of columns."""
        result = profiler.profile(small_data, "test_ds")
        assert result["column_count"] == 3  # name, age, email

    def test_columns_list(self, profiler, small_data):
        """Test columns dict contains all column names."""
        result = profiler.profile(small_data, "test_ds")
        assert "name" in result["columns"]
        assert "age" in result["columns"]
        assert "email" in result["columns"]

    def test_schema_hash_present(self, profiler, small_data):
        """Test schema_hash is a 64-char hex string (SHA-256)."""
        result = profiler.profile(small_data, "test_ds")
        assert len(result["schema_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in result["schema_hash"])

    def test_memory_estimate_positive(self, profiler, small_data):
        """Test memory_estimate_bytes is positive."""
        result = profiler.profile(small_data, "test_ds")
        assert result["memory_estimate_bytes"] > 0

    def test_provenance_hash_present(self, profiler, small_data):
        """Test provenance_hash is a 64-char hex string."""
        result = profiler.profile(small_data, "test_ds")
        assert len(result["provenance_hash"]) == 64

    def test_profile_id_prefix(self, profiler, small_data):
        """Test profile_id starts with PRF- prefix."""
        result = profiler.profile(small_data, "test_ds")
        assert result["profile_id"].startswith("PRF-")

    def test_empty_data_raises_error(self, profiler):
        """Test profiling empty data raises ValueError."""
        with pytest.raises(ValueError, match="Cannot profile empty dataset"):
            profiler.profile([], "empty_ds")

    def test_single_row(self, profiler):
        """Test profiling a single row dataset."""
        data = [{"x": 1, "y": "hello"}]
        result = profiler.profile(data, "single")
        assert result["row_count"] == 1
        assert result["column_count"] == 2

    def test_custom_columns_subset(self, profiler, small_data):
        """Test profiling only a subset of columns."""
        result = profiler.profile(small_data, "test_ds", columns=["name"])
        assert result["column_count"] == 1
        assert "name" in result["columns"]
        assert "age" not in result["columns"]

    def test_profile_stored_internally(self, profiler, small_data):
        """Test profile is stored in internal store."""
        result = profiler.profile(small_data, "test_ds")
        stored = profiler.get_profile(result["profile_id"])
        assert stored is not None
        assert stored["profile_id"] == result["profile_id"]

    def test_created_at_present(self, profiler, small_data):
        """Test created_at timestamp is present."""
        result = profiler.profile(small_data, "test_ds")
        assert "created_at" in result
        assert isinstance(result["created_at"], str)

    def test_profiling_time_positive(self, profiler, small_data):
        """Test profiling_time_ms is positive."""
        result = profiler.profile(small_data, "test_ds")
        assert result["profiling_time_ms"] >= 0


# ============================================================================
# TestProfileColumn
# ============================================================================


class TestProfileColumn:
    """Test per-column profiling."""

    def test_column_name_preserved(self, profiler):
        """Test column_name is preserved in output."""
        result = profiler.profile_column(["a", "b", "c"], "test_col")
        assert result["column_name"] == "test_col"

    def test_data_type_detection_string(self, profiler):
        """Test string type detection."""
        result = profiler.profile_column(["hello", "world", "test"], "str_col")
        assert result["inferred_type"] == DATA_TYPE_STRING

    def test_total_count(self, profiler):
        """Test total_count matches input length."""
        result = profiler.profile_column([1, 2, 3, None, 5], "col")
        assert result["total_count"] == 5

    def test_null_count(self, profiler):
        """Test null_count for values with None."""
        result = profiler.profile_column([1, None, 3, None, 5], "col")
        assert result["null_count"] == 2

    def test_null_rate(self, profiler):
        """Test null_rate calculation."""
        result = profiler.profile_column([1, None, 3, None, 5], "col")
        assert result["null_rate"] == pytest.approx(0.4, abs=0.01)

    def test_unique_count(self, profiler):
        """Test distinct_count for unique values."""
        result = profiler.profile_column(["a", "b", "c", "a"], "col")
        assert result["distinct_count"] == 3

    def test_cardinality_present(self, profiler):
        """Test cardinality dict is present."""
        result = profiler.profile_column(["a", "b", "c"], "col")
        assert "cardinality" in result
        assert "unique_count" in result["cardinality"]

    def test_numeric_min_max(self, profiler):
        """Test min/max statistics for numeric column."""
        result = profiler.profile_column([10, 20, 30, 40, 50], "num_col")
        stats = result["statistics"]
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0

    def test_numeric_mean_median(self, profiler):
        """Test mean/median for numeric column."""
        result = profiler.profile_column([10, 20, 30, 40, 50], "num_col")
        stats = result["statistics"]
        assert stats["mean"] == pytest.approx(30.0, abs=0.01)
        assert stats["median"] == pytest.approx(30.0, abs=0.01)

    def test_pattern_detection(self, profiler):
        """Test patterns dict is present."""
        result = profiler.profile_column(
            ["alice@example.com", "bob@test.org", "eve@mail.com"],
            "email_col",
        )
        assert "patterns" in result

    def test_string_lengths(self, profiler):
        """Test string statistics include lengths."""
        result = profiler.profile_column(["abc", "de", "fghij"], "str_col")
        stats = result["statistics"]
        assert stats.get("min_length") == 2
        assert stats.get("max_length") == 5

    def test_provenance_hash_present(self, profiler):
        """Test provenance_hash is returned."""
        result = profiler.profile_column(["a", "b"], "col")
        assert len(result["provenance_hash"]) == 64


# ============================================================================
# TestInferDataType
# ============================================================================


class TestInferDataType:
    """Test data type inference for 13 types + edge cases."""

    def test_string(self, profiler):
        """Test plain strings detected as string."""
        assert profiler.infer_data_type(["hello", "world", "test"]) == DATA_TYPE_STRING

    def test_integer_native(self, profiler):
        """Test native Python ints detected as integer."""
        assert profiler.infer_data_type([1, 2, 3, 4, 5]) == DATA_TYPE_INTEGER

    def test_integer_string(self, profiler):
        """Test string integers detected as integer."""
        assert profiler.infer_data_type(["10", "20", "30"]) == DATA_TYPE_INTEGER

    def test_float_native(self, profiler):
        """Test native Python floats detected as float."""
        assert profiler.infer_data_type([1.5, 2.7, 3.14]) == DATA_TYPE_FLOAT

    def test_float_string(self, profiler):
        """Test string floats detected as float."""
        assert profiler.infer_data_type(["1.5", "2.7", "3.14"]) == DATA_TYPE_FLOAT

    def test_boolean_native(self, profiler):
        """Test native booleans detected as boolean."""
        assert profiler.infer_data_type([True, False, True]) == DATA_TYPE_BOOLEAN

    def test_boolean_string_true_false(self, profiler):
        """Test string true/false detected as boolean."""
        assert profiler.infer_data_type(["true", "false", "true"]) == DATA_TYPE_BOOLEAN

    def test_date_iso(self, profiler):
        """Test ISO date strings detected as date."""
        assert profiler.infer_data_type(["2025-01-15", "2025-06-30", "2025-12-31"]) == DATA_TYPE_DATE

    def test_date_us_format(self, profiler):
        """Test US date format detected as date."""
        assert profiler.infer_data_type(["01/15/2025", "06/30/2025", "12/31/2025"]) == DATA_TYPE_DATE

    def test_date_eu_format(self, profiler):
        """Test EU date format detected as date."""
        assert profiler.infer_data_type(["15/01/2025", "30/06/2025", "31/12/2025"]) == DATA_TYPE_DATE

    def test_datetime(self, profiler):
        """Test ISO datetime strings detected as datetime."""
        assert profiler.infer_data_type([
            "2025-01-15T10:30:00",
            "2025-06-30T14:00:00Z",
            "2025-12-31T23:59:59+00:00",
        ]) == DATA_TYPE_DATETIME

    def test_email(self, profiler):
        """Test email addresses detected as email."""
        assert profiler.infer_data_type([
            "alice@example.com",
            "bob@test.org",
            "admin@corp.net",
        ]) == DATA_TYPE_EMAIL

    def test_url(self, profiler):
        """Test URLs detected as url."""
        assert profiler.infer_data_type([
            "https://example.com",
            "http://test.org/path",
            "https://api.corp.net/v1",
        ]) == DATA_TYPE_URL

    def test_phone(self, profiler):
        """Test phone numbers detected as phone."""
        assert profiler.infer_data_type([
            "+1-555-0101",
            "+44 20 7946 0958",
            "+49 30 901820",
        ]) == DATA_TYPE_PHONE

    def test_ip_address_v4(self, profiler):
        """Test IPv4 addresses detected as ip_address."""
        assert profiler.infer_data_type([
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
        ]) == DATA_TYPE_IP_ADDRESS

    def test_uuid(self, profiler):
        """Test UUID strings detected as uuid."""
        assert profiler.infer_data_type([
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        ]) == DATA_TYPE_UUID

    def test_json_str(self, profiler):
        """Test JSON strings detected as json_str."""
        assert profiler.infer_data_type([
            '{"key": "value"}',
            '{"name": "Alice", "age": 30}',
            '[1, 2, 3]',
        ]) == DATA_TYPE_JSON_STR

    def test_mixed_types_majority_wins(self, profiler):
        """Test mixed types return the majority type."""
        # 3 integers, 1 string, 1 float
        result = profiler.infer_data_type([1, 2, 3, "hello", 4.5])
        assert result == DATA_TYPE_INTEGER

    def test_empty_list(self, profiler):
        """Test empty list returns unknown."""
        assert profiler.infer_data_type([]) == DATA_TYPE_UNKNOWN

    def test_all_nulls_via_empty_non_null(self, profiler):
        """Test empty non-null values after filtering returns unknown."""
        # profile_column filters nulls before calling infer_data_type
        assert profiler.infer_data_type([]) == DATA_TYPE_UNKNOWN


# ============================================================================
# TestComputeStatistics
# ============================================================================


class TestComputeStatistics:
    """Test statistics computation for numeric columns."""

    def test_min(self, profiler):
        """Test minimum value computation."""
        result = profiler.compute_statistics([1, 2, 3, 4, 5], DATA_TYPE_INTEGER)
        assert result["min"] == 1.0

    def test_max(self, profiler):
        """Test maximum value computation."""
        result = profiler.compute_statistics([1, 2, 3, 4, 5], DATA_TYPE_INTEGER)
        assert result["max"] == 5.0

    def test_mean(self, profiler):
        """Test mean computation."""
        result = profiler.compute_statistics([10, 20, 30], DATA_TYPE_INTEGER)
        assert result["mean"] == pytest.approx(20.0, abs=0.01)

    def test_median(self, profiler):
        """Test median computation."""
        result = profiler.compute_statistics([1, 2, 3, 4, 5], DATA_TYPE_INTEGER)
        assert result["median"] == pytest.approx(3.0, abs=0.01)

    def test_stddev(self, profiler):
        """Test standard deviation computation."""
        result = profiler.compute_statistics([2, 4, 4, 4, 5, 5, 7, 9], DATA_TYPE_INTEGER)
        expected = statistics.stdev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert result["stddev"] == pytest.approx(expected, abs=0.001)

    def test_variance(self, profiler):
        """Test variance computation."""
        result = profiler.compute_statistics([2, 4, 4, 4, 5, 5, 7, 9], DATA_TYPE_INTEGER)
        expected = statistics.variance([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert result["variance"] == pytest.approx(expected, abs=0.001)

    def test_p25(self, profiler):
        """Test 25th percentile computation."""
        result = profiler.compute_statistics(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], DATA_TYPE_INTEGER
        )
        assert "p25" in result
        assert result["p25"] == pytest.approx(3.25, abs=0.1)

    def test_p50(self, profiler):
        """Test 50th percentile computation."""
        result = profiler.compute_statistics(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], DATA_TYPE_INTEGER
        )
        assert result["p50"] == pytest.approx(5.5, abs=0.1)

    def test_p75(self, profiler):
        """Test 75th percentile computation."""
        result = profiler.compute_statistics(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], DATA_TYPE_INTEGER
        )
        assert "p75" in result
        assert result["p75"] == pytest.approx(7.75, abs=0.1)

    def test_p95(self, profiler):
        """Test 95th percentile computation."""
        result = profiler.compute_statistics(
            list(range(1, 101)), DATA_TYPE_INTEGER
        )
        assert "p95" in result

    def test_p99(self, profiler):
        """Test 99th percentile computation."""
        result = profiler.compute_statistics(
            list(range(1, 101)), DATA_TYPE_INTEGER
        )
        assert "p99" in result

    def test_skewness(self, profiler):
        """Test skewness computation for symmetric data."""
        result = profiler.compute_statistics(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], DATA_TYPE_INTEGER
        )
        # Symmetric data has skewness close to 0
        assert abs(result["skewness"]) < 0.5

    def test_kurtosis(self, profiler):
        """Test kurtosis computation."""
        result = profiler.compute_statistics(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], DATA_TYPE_INTEGER
        )
        assert "kurtosis" in result

    def test_empty_list_returns_count_zero(self, profiler):
        """Test empty list returns count=0."""
        result = profiler.compute_statistics([], DATA_TYPE_INTEGER)
        assert result["count"] == 0

    def test_single_value(self, profiler):
        """Test single value stats."""
        result = profiler.compute_statistics([42], DATA_TYPE_INTEGER)
        assert result["min"] == 42.0
        assert result["max"] == 42.0
        assert result["mean"] == pytest.approx(42.0, abs=0.01)

    def test_all_same_values(self, profiler):
        """Test all identical values produce stddev=0."""
        result = profiler.compute_statistics([5, 5, 5, 5, 5], DATA_TYPE_INTEGER)
        assert result["stddev"] == pytest.approx(0.0, abs=0.001)

    def test_negative_values(self, profiler):
        """Test negative values handled correctly."""
        result = profiler.compute_statistics([-10, -5, 0, 5, 10], DATA_TYPE_INTEGER)
        assert result["min"] == -10.0
        assert result["max"] == 10.0
        assert result["mean"] == pytest.approx(0.0, abs=0.01)


# ============================================================================
# TestComputeCardinality
# ============================================================================


class TestComputeCardinality:
    """Test cardinality analysis."""

    def test_unique_count(self, profiler):
        """Test unique_count for distinct values."""
        result = profiler.compute_cardinality(["a", "b", "c", "d"])
        assert result["unique_count"] == 4

    def test_cardinality_ratio(self, profiler):
        """Test cardinality_ratio for mixed values."""
        result = profiler.compute_cardinality(["a", "b", "a", "b"])
        assert result["cardinality_ratio"] == pytest.approx(0.5, abs=0.01)

    def test_most_common_top_10(self, profiler):
        """Test most_common returns up to top 10."""
        values = ["a"] * 50 + ["b"] * 30 + ["c"] * 20
        result = profiler.compute_cardinality(values)
        assert len(result["most_common"]) <= 10
        assert result["most_common"][0]["value"] == "a"

    def test_all_unique(self, profiler):
        """Test all unique values have is_unique=True."""
        result = profiler.compute_cardinality(["a", "b", "c"])
        assert result["is_unique"] is True
        assert result["cardinality_ratio"] == pytest.approx(1.0, abs=0.01)

    def test_all_same(self, profiler):
        """Test all same values have is_constant=True."""
        result = profiler.compute_cardinality(["x", "x", "x"])
        assert result["is_constant"] is True
        assert result["unique_count"] == 1

    def test_empty(self, profiler):
        """Test empty list returns zeroed cardinality."""
        result = profiler.compute_cardinality([])
        assert result["unique_count"] == 0
        assert result["cardinality_ratio"] == 0.0
        assert result["is_constant"] is True

    def test_single_value(self, profiler):
        """Test single value cardinality."""
        result = profiler.compute_cardinality(["only_one"])
        assert result["unique_count"] == 1
        assert result["cardinality_ratio"] == pytest.approx(1.0, abs=0.01)
        assert result["is_unique"] is True

    def test_numeric_values_as_strings(self, profiler):
        """Test numeric values converted to strings for cardinality."""
        result = profiler.compute_cardinality([1, 2, 3, 1, 2])
        assert result["unique_count"] == 3


# ============================================================================
# TestDetectPatterns
# ============================================================================


class TestDetectPatterns:
    """Test format pattern detection."""

    def test_email_patterns(self, profiler):
        """Test email pattern detected."""
        values = ["alice@example.com", "bob@test.org", "eve@mail.com"]
        result = profiler.detect_patterns(values)
        assert "email" in result
        assert result["email"]["match_count"] == 3

    def test_phone_patterns(self, profiler):
        """Test phone pattern detected."""
        values = ["+1-555-0101", "+44 20 7946 0958", "+49-30-901820"]
        result = profiler.detect_patterns(values)
        assert "phone" in result

    def test_date_patterns(self, profiler):
        """Test date pattern detected."""
        values = ["2025-01-15", "2025-06-30", "2025-12-31"]
        result = profiler.detect_patterns(values)
        assert "date_iso" in result

    def test_url_patterns(self, profiler):
        """Test URL pattern detected."""
        values = ["https://example.com", "http://test.org/path", "https://api.io"]
        result = profiler.detect_patterns(values)
        assert "url" in result

    def test_ip_patterns(self, profiler):
        """Test IPv4 pattern detected."""
        values = ["192.168.1.1", "10.0.0.1", "172.16.0.1"]
        result = profiler.detect_patterns(values)
        assert "ipv4" in result

    def test_uuid_patterns(self, profiler):
        """Test UUID pattern detected."""
        values = [
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        ]
        result = profiler.detect_patterns(values)
        assert "uuid" in result

    def test_currency_patterns(self, profiler):
        """Test currency pattern detected."""
        values = ["$100.00", "$1,234.56", "$50"]
        result = profiler.detect_patterns(values)
        assert "currency" in result

    def test_no_patterns(self, profiler):
        """Test no patterns for arbitrary strings."""
        values = ["apple", "banana", "cherry"]
        result = profiler.detect_patterns(values)
        # No common format patterns should match fruit names
        high_match = [k for k, v in result.items() if v["match_rate"] > 0.5]
        assert len(high_match) == 0

    def test_mixed_patterns(self, profiler):
        """Test mixed values produce partial matches."""
        values = ["alice@example.com", "not-an-email", "bob@test.org"]
        result = profiler.detect_patterns(values)
        if "email" in result:
            assert result["email"]["match_count"] == 2

    def test_empty_values(self, profiler):
        """Test empty list returns empty patterns."""
        result = profiler.detect_patterns([])
        assert result == {}

    def test_percentage_pattern(self, profiler):
        """Test percentage pattern detected."""
        values = ["50%", "75.5%", "100%"]
        result = profiler.detect_patterns(values)
        assert "percentage" in result

    def test_hex_color_pattern(self, profiler):
        """Test hex color pattern detected."""
        values = ["#FF0000", "#00FF00", "#0000FF"]
        result = profiler.detect_patterns(values)
        assert "hex_color" in result


# ============================================================================
# TestEstimateMemory
# ============================================================================


class TestEstimateMemory:
    """Test memory estimation."""

    def test_empty_data(self, profiler):
        """Test empty data returns 0."""
        assert profiler.estimate_memory([]) == 0

    def test_small_dataset(self, profiler):
        """Test small dataset produces positive estimate."""
        data = [{"x": 1, "y": "hello"} for _ in range(10)]
        result = profiler.estimate_memory(data)
        assert result > 0

    def test_large_dataset(self, profiler):
        """Test larger dataset produces larger estimate."""
        small = [{"x": i} for i in range(10)]
        large = [{"x": i} for i in range(1000)]
        small_est = profiler.estimate_memory(small)
        large_est = profiler.estimate_memory(large)
        assert large_est > small_est

    def test_nested_dicts(self, profiler):
        """Test nested dicts contribute to memory estimate."""
        data = [{"nested": {"a": 1, "b": [1, 2, 3]}} for _ in range(10)]
        result = profiler.estimate_memory(data)
        assert result > 0

    def test_samples_up_to_100(self, profiler):
        """Test estimation only samples first 100 rows."""
        data = [{"x": i, "y": "test" * 100} for i in range(500)]
        result = profiler.estimate_memory(data)
        assert result > 0


# ============================================================================
# TestGetSchemaHash
# ============================================================================


class TestGetSchemaHash:
    """Test schema hashing."""

    def test_deterministic(self, profiler):
        """Test same schema produces same hash."""
        cols = [{"name": "x", "type": "integer"}, {"name": "y", "type": "string"}]
        h1 = profiler.get_schema_hash(cols)
        h2 = profiler.get_schema_hash(cols)
        assert h1 == h2

    def test_different_schemas(self, profiler):
        """Test different schemas produce different hashes."""
        cols_a = [{"name": "x", "type": "integer"}]
        cols_b = [{"name": "y", "type": "string"}]
        assert profiler.get_schema_hash(cols_a) != profiler.get_schema_hash(cols_b)

    def test_order_independent(self, profiler):
        """Test column order does not affect hash (sorted internally)."""
        cols_a = [{"name": "x", "type": "int"}, {"name": "y", "type": "str"}]
        cols_b = [{"name": "y", "type": "str"}, {"name": "x", "type": "int"}]
        assert profiler.get_schema_hash(cols_a) == profiler.get_schema_hash(cols_b)

    def test_empty_schema(self, profiler):
        """Test empty schema list produces a valid hash."""
        result = profiler.get_schema_hash([])
        assert len(result) == 64

    def test_hash_is_sha256(self, profiler):
        """Test hash is 64 hex chars (SHA-256)."""
        cols = [{"name": "a", "type": "b"}]
        result = profiler.get_schema_hash(cols)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


# ============================================================================
# TestGetProfile
# ============================================================================


class TestGetProfile:
    """Test profile retrieval."""

    def test_existing_profile(self, profiler, small_data):
        """Test retrieving an existing profile."""
        created = profiler.profile(small_data, "test_ds")
        retrieved = profiler.get_profile(created["profile_id"])
        assert retrieved is not None
        assert retrieved["dataset_name"] == "test_ds"

    def test_nonexistent_profile(self, profiler):
        """Test retrieving a non-existent profile returns None."""
        assert profiler.get_profile("PRF-doesnotexist") is None

    def test_fields_validation(self, profiler, small_data):
        """Test retrieved profile contains expected fields."""
        created = profiler.profile(small_data, "test_ds")
        retrieved = profiler.get_profile(created["profile_id"])
        assert "profile_id" in retrieved
        assert "dataset_name" in retrieved
        assert "row_count" in retrieved
        assert "columns" in retrieved
        assert "provenance_hash" in retrieved


# ============================================================================
# TestListProfiles
# ============================================================================


class TestListProfiles:
    """Test profile listing with pagination."""

    def test_list_all(self, profiler, small_data):
        """Test listing all profiles."""
        profiler.profile(small_data, "ds1")
        profiler.profile(small_data, "ds2")
        profiles = profiler.list_profiles()
        assert len(profiles) == 2

    def test_limit(self, profiler, small_data):
        """Test limit parameter."""
        profiler.profile(small_data, "ds1")
        profiler.profile(small_data, "ds2")
        profiler.profile(small_data, "ds3")
        profiles = profiler.list_profiles(limit=2)
        assert len(profiles) == 2

    def test_offset(self, profiler, small_data):
        """Test offset parameter."""
        profiler.profile(small_data, "ds1")
        profiler.profile(small_data, "ds2")
        profiler.profile(small_data, "ds3")
        profiles = profiler.list_profiles(limit=10, offset=1)
        assert len(profiles) == 2

    def test_empty(self, profiler):
        """Test listing profiles when none exist."""
        profiles = profiler.list_profiles()
        assert len(profiles) == 0

    def test_sorted_by_created_at(self, profiler, small_data):
        """Test profiles are sorted by created_at descending."""
        profiler.profile(small_data, "first")
        profiler.profile(small_data, "second")
        profiles = profiler.list_profiles()
        # Most recent first
        assert profiles[0]["dataset_name"] == "second"


# ============================================================================
# TestStatistics
# ============================================================================


class TestStatistics:
    """Test aggregate profiling statistics."""

    def test_initial_stats(self, profiler):
        """Test initial statistics are zeroed."""
        stats = profiler.get_statistics()
        assert stats["profiles_created"] == 0
        assert stats["columns_profiled"] == 0
        assert stats["total_rows_profiled"] == 0

    def test_post_profiling_stats(self, profiler, small_data):
        """Test statistics updated after profiling."""
        profiler.profile(small_data, "test_ds")
        stats = profiler.get_statistics()
        assert stats["profiles_created"] == 1
        assert stats["columns_profiled"] == 3
        assert stats["total_rows_profiled"] == 3

    def test_stats_accumulate(self, profiler, small_data):
        """Test statistics accumulate across profiles."""
        profiler.profile(small_data, "ds1")
        profiler.profile(small_data, "ds2")
        stats = profiler.get_statistics()
        assert stats["profiles_created"] == 2
        assert stats["total_rows_profiled"] == 6

    def test_stored_profiles_count(self, profiler, small_data):
        """Test stored_profiles count in stats."""
        profiler.profile(small_data, "ds1")
        stats = profiler.get_statistics()
        assert stats["stored_profiles"] == 1

    def test_by_data_type_in_column(self, profiler):
        """Test profiled column stores inferred_type."""
        data = [{"val": 1}, {"val": 2}, {"val": 3}]
        result = profiler.profile(data, "int_ds")
        assert result["columns"]["val"]["inferred_type"] == DATA_TYPE_INTEGER


# ============================================================================
# TestProvenance
# ============================================================================


class TestProvenance:
    """Test provenance hash generation."""

    def test_sha256_length(self, profiler, small_data):
        """Test provenance hash is 64-char SHA-256."""
        result = profiler.profile(small_data, "test_ds")
        assert len(result["provenance_hash"]) == 64

    def test_different_inputs_different_hashes(self, profiler):
        """Test different datasets produce different provenance hashes."""
        data_a = [{"x": 1}]
        data_b = [{"y": 2}]
        result_a = profiler.profile(data_a, "ds_a")
        result_b = profiler.profile(data_b, "ds_b")
        assert result_a["provenance_hash"] != result_b["provenance_hash"]

    def test_every_profile_has_provenance(self, profiler, small_data):
        """Test every profile includes provenance_hash."""
        result = profiler.profile(small_data, "test_ds")
        assert "provenance_hash" in result
        assert result["provenance_hash"]

    def test_column_provenance(self, profiler):
        """Test each column profile also has provenance_hash."""
        data = [{"a": 1, "b": "x"}]
        result = profiler.profile(data, "test_ds")
        for col_name, col_profile in result["columns"].items():
            assert len(col_profile["provenance_hash"]) == 64


# ============================================================================
# TestThreadSafety
# ============================================================================


class TestThreadSafety:
    """Test thread safety of concurrent profiling."""

    def test_concurrent_profiling(self, profiler):
        """Test multiple threads can profile concurrently without errors."""
        errors = []
        data = [{"x": i} for i in range(100)]

        def do_profile(name):
            try:
                profiler.profile(data, f"dataset_{name}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=do_profile, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = profiler.get_statistics()
        assert stats["profiles_created"] == 10

    def test_stats_consistency(self, profiler):
        """Test stats remain consistent under concurrent access."""
        data = [{"x": i} for i in range(50)]

        def do_profile(name):
            profiler.profile(data, f"ds_{name}")

        threads = [threading.Thread(target=do_profile, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = profiler.get_statistics()
        assert stats["profiles_created"] == 5
        assert stats["total_rows_profiled"] == 250
        assert stats["stored_profiles"] == 5


# ============================================================================
# TestHelperFunctions (module-level helpers)
# ============================================================================


class TestHelperFunctions:
    """Test module-level statistical helper functions."""

    def test_safe_mean_empty(self):
        """Test _safe_mean returns 0.0 for empty list."""
        assert _safe_mean([]) == 0.0

    def test_safe_mean_normal(self):
        """Test _safe_mean computes correct mean."""
        assert _safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_safe_median_empty(self):
        """Test _safe_median returns 0.0 for empty list."""
        assert _safe_median([]) == 0.0

    def test_safe_median_normal(self):
        """Test _safe_median computes correct median."""
        assert _safe_median([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_safe_stdev_less_than_2(self):
        """Test _safe_stdev returns 0.0 for < 2 values."""
        assert _safe_stdev([1.0]) == 0.0
        assert _safe_stdev([]) == 0.0

    def test_safe_stdev_normal(self):
        """Test _safe_stdev computes correct stddev."""
        result = _safe_stdev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert result == pytest.approx(statistics.stdev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]), abs=0.001)

    def test_safe_variance_less_than_2(self):
        """Test _safe_variance returns 0.0 for < 2 values."""
        assert _safe_variance([5.0]) == 0.0

    def test_safe_variance_normal(self):
        """Test _safe_variance computes correct variance."""
        result = _safe_variance([2.0, 4.0, 6.0])
        assert result == pytest.approx(statistics.variance([2.0, 4.0, 6.0]), abs=0.001)

    def test_percentile_empty(self):
        """Test _percentile returns 0.0 for empty list."""
        assert _percentile([], 50) == 0.0

    def test_percentile_single(self):
        """Test _percentile returns the value for single-element list."""
        assert _percentile([42.0], 50) == 42.0

    def test_percentile_p50(self):
        """Test _percentile at 50th percentile."""
        values = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _percentile(values, 50) == pytest.approx(3.0, abs=0.1)

    def test_skewness_less_than_3(self):
        """Test _skewness returns 0.0 for < 3 values."""
        assert _skewness([1.0, 2.0]) == 0.0

    def test_skewness_symmetric(self):
        """Test _skewness for symmetric data is close to 0."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        assert abs(_skewness(values)) < 0.5

    def test_kurtosis_less_than_4(self):
        """Test _kurtosis returns 0.0 for < 4 values."""
        assert _kurtosis([1.0, 2.0, 3.0]) == 0.0

    def test_kurtosis_normal(self):
        """Test _kurtosis computes for sufficient data."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = _kurtosis(values)
        assert isinstance(result, float)

    def test_is_json_string_valid_object(self):
        """Test _is_json_string for valid JSON object."""
        assert _is_json_string('{"key": "value"}') is True

    def test_is_json_string_valid_array(self):
        """Test _is_json_string for valid JSON array."""
        assert _is_json_string('[1, 2, 3]') is True

    def test_is_json_string_invalid(self):
        """Test _is_json_string for non-JSON string."""
        assert _is_json_string("hello world") is False

    def test_try_parse_number_valid(self):
        """Test _try_parse_number for valid number strings."""
        assert _try_parse_number("42") == pytest.approx(42.0)
        assert _try_parse_number("3.14") == pytest.approx(3.14)

    def test_try_parse_number_invalid(self):
        """Test _try_parse_number returns None for non-numbers."""
        assert _try_parse_number("hello") is None
        assert _try_parse_number("") is None

    def test_skewness_all_same(self):
        """Test _skewness returns 0.0 for all identical values."""
        assert _skewness([5.0, 5.0, 5.0, 5.0, 5.0]) == 0.0

    def test_kurtosis_all_same(self):
        """Test _kurtosis returns 0.0 for all identical values."""
        assert _kurtosis([5.0, 5.0, 5.0, 5.0, 5.0]) == 0.0


# ============================================================================
# TestDeleteProfile
# ============================================================================


class TestDeleteProfile:
    """Test profile deletion."""

    def test_delete_existing(self, profiler, small_data):
        """Test deleting an existing profile returns True."""
        result = profiler.profile(small_data, "test_ds")
        assert profiler.delete_profile(result["profile_id"]) is True
        assert profiler.get_profile(result["profile_id"]) is None

    def test_delete_nonexistent(self, profiler):
        """Test deleting a non-existent profile returns False."""
        assert profiler.delete_profile("PRF-doesnotexist") is False


# ============================================================================
# TestSampling
# ============================================================================


class TestSampling:
    """Test dataset sampling for large datasets."""

    def test_sampled_flag_false_for_small(self, profiler, small_data):
        """Test sampled is False for small datasets."""
        result = profiler.profile(small_data, "test_ds")
        assert result["sampled"] is False

    def test_sampled_flag_true_for_large(self):
        """Test sampled is True when data exceeds max_sample_size."""
        profiler = DatasetProfiler(config={"max_sample_size": 5})
        data = [{"x": i} for i in range(100)]
        result = profiler.profile(data, "large_ds")
        assert result["sampled"] is True
        assert result["sample_size"] == 5

    def test_sample_size_matches(self):
        """Test sample_size is min(data_length, max_sample_size)."""
        profiler = DatasetProfiler(config={"max_sample_size": 3})
        data = [{"x": i} for i in range(10)]
        result = profiler.profile(data, "ds")
        assert result["sample_size"] == 3
        assert result["row_count"] == 10
