# -*- coding: utf-8 -*-
"""
Unit Tests for RecordFingerprinter Engine - AGENT-DATA-011 Batch 2

Comprehensive test suite for the RecordFingerprinter engine covering:
- SHA-256 fingerprint determinism and correctness
- SimHash locality-sensitive hashing
- MinHash signature generation and Jaccard estimation
- Batch fingerprinting
- Type-aware field normalization (STRING, NUMERIC, DATE, BOOLEAN, CATEGORICAL)
- Edge cases (None, empty, unicode, special characters)
- Provenance hash generation
- Thread-safe statistics tracking

Target: 130+ tests, 85%+ coverage.

Author: GreenLang QA Team
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
"""

from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from greenlang.duplicate_detector.models import (
    FieldType,
    FingerprintAlgorithm,
    RecordFingerprint,
)
from greenlang.duplicate_detector.record_fingerprinter import (
    RecordFingerprinter,
    _MINHASH_MAX_HASH,
    _MINHASH_PRIME,
    _SIMHASH_NGRAM_SIZE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> RecordFingerprinter:
    """Create a fresh RecordFingerprinter instance."""
    return RecordFingerprinter()


@pytest.fixture
def simple_record() -> Dict[str, Any]:
    """Create a simple test record."""
    return {"id": "rec-001", "name": "Alice Smith", "email": "alice@example.com"}


@pytest.fixture
def numeric_record() -> Dict[str, Any]:
    """Create a record with numeric fields."""
    return {"id": "rec-num", "amount": 1234.56, "quantity": 100}


@pytest.fixture
def date_record() -> Dict[str, Any]:
    """Create a record with date fields."""
    return {"id": "rec-date", "created": "2025-06-15", "updated": "2025-07-20T10:30:00"}


@pytest.fixture
def boolean_record() -> Dict[str, Any]:
    """Create a record with boolean fields."""
    return {"id": "rec-bool", "active": True, "verified": False}


@pytest.fixture
def batch_records() -> List[Dict[str, Any]]:
    """Create a batch of records for testing."""
    return [
        {"id": "b-001", "name": "Alice Smith", "email": "alice@example.com"},
        {"id": "b-002", "name": "Bob Jones", "email": "bob@example.com"},
        {"id": "b-003", "name": "Charlie Brown", "email": "charlie@example.com"},
        {"id": "b-004", "name": "Diana Prince", "email": "diana@example.com"},
        {"id": "b-005", "name": "Eve Wilson", "email": "eve@example.com"},
    ]


@pytest.fixture
def unicode_record() -> Dict[str, Any]:
    """Create a record with unicode characters."""
    return {"id": "rec-uni", "name": "Muller", "city": "Zurich"}


@pytest.fixture
def field_types_map() -> Dict[str, FieldType]:
    """Create field type mapping for tests."""
    return {
        "name": FieldType.STRING,
        "amount": FieldType.NUMERIC,
        "created": FieldType.DATE,
        "active": FieldType.BOOLEAN,
        "category": FieldType.CATEGORICAL,
    }


# ===========================================================================
# Test Class: SHA-256 Fingerprinting
# ===========================================================================


class TestFingerprintSHA256:
    """Tests for SHA-256 fingerprinting algorithm."""

    def test_fingerprint_sha256_basic(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """SHA-256 fingerprint of a basic record returns a valid RecordFingerprint."""
        result = engine.fingerprint_record(
            record=simple_record,
            field_set=["name", "email"],
            algorithm=FingerprintAlgorithm.SHA256,
            record_id="rec-001",
        )
        assert isinstance(result, RecordFingerprint)
        assert result.record_id == "rec-001"
        assert result.algorithm == FingerprintAlgorithm.SHA256
        assert len(result.fingerprint_hash) == 64  # SHA-256 hex digest
        assert result.normalized_fields is True

    def test_fingerprint_sha256_deterministic(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """Same input produces same SHA-256 fingerprint hash (determinism)."""
        fp1 = engine.fingerprint_record(
            simple_record, ["name", "email"], FingerprintAlgorithm.SHA256, "rec-001",
        )
        fp2 = engine.fingerprint_record(
            simple_record, ["name", "email"], FingerprintAlgorithm.SHA256, "rec-001",
        )
        assert fp1.fingerprint_hash == fp2.fingerprint_hash

    def test_fingerprint_sha256_different_records(self, engine: RecordFingerprinter):
        """Different records produce different SHA-256 fingerprints."""
        rec_a = {"name": "Alice", "email": "alice@example.com"}
        rec_b = {"name": "Bob", "email": "bob@example.com"}
        fp_a = engine.fingerprint_record(rec_a, ["name", "email"], FingerprintAlgorithm.SHA256)
        fp_b = engine.fingerprint_record(rec_b, ["name", "email"], FingerprintAlgorithm.SHA256)
        assert fp_a.fingerprint_hash != fp_b.fingerprint_hash

    def test_fingerprint_sha256_field_order_invariant(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """Field order does not affect SHA-256 fingerprint (sorted internally)."""
        fp1 = engine.fingerprint_record(
            simple_record, ["name", "email"], FingerprintAlgorithm.SHA256, "rec-001",
        )
        fp2 = engine.fingerprint_record(
            simple_record, ["email", "name"], FingerprintAlgorithm.SHA256, "rec-001",
        )
        assert fp1.fingerprint_hash == fp2.fingerprint_hash

    def test_fingerprint_sha256_subset_fields(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """Using a subset of fields produces different fingerprint than full fields."""
        fp_name_only = engine.fingerprint_record(
            simple_record, ["name"], FingerprintAlgorithm.SHA256, "rec-001",
        )
        fp_both = engine.fingerprint_record(
            simple_record, ["name", "email"], FingerprintAlgorithm.SHA256, "rec-001",
        )
        assert fp_name_only.fingerprint_hash != fp_both.fingerprint_hash

    def test_fingerprint_sha256_field_set_stored(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """Fingerprint result stores the sorted field set."""
        result = engine.fingerprint_record(
            simple_record, ["email", "name"], FingerprintAlgorithm.SHA256,
        )
        assert result.field_set == ["email", "name"]

    def test_fingerprint_sha256_provenance_hash_present(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """Fingerprint result contains a non-empty provenance hash."""
        result = engine.fingerprint_record(
            simple_record, ["name"], FingerprintAlgorithm.SHA256,
        )
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_fingerprint_sha256_missing_field_treated_as_empty(self, engine: RecordFingerprinter):
        """Missing fields in record are treated as empty string after normalization."""
        rec = {"name": "Alice"}
        result = engine.fingerprint_record(rec, ["name", "nonexistent"], FingerprintAlgorithm.SHA256)
        assert result.fingerprint_hash != ""

    def test_fingerprint_sha256_case_normalization(self, engine: RecordFingerprinter):
        """String fields are lowercased during normalization for SHA-256."""
        rec_upper = {"name": "ALICE SMITH"}
        rec_lower = {"name": "alice smith"}
        fp_upper = engine.fingerprint_record(rec_upper, ["name"], FingerprintAlgorithm.SHA256)
        fp_lower = engine.fingerprint_record(rec_lower, ["name"], FingerprintAlgorithm.SHA256)
        assert fp_upper.fingerprint_hash == fp_lower.fingerprint_hash

    def test_fingerprint_sha256_whitespace_normalization(self, engine: RecordFingerprinter):
        """Extra whitespace is collapsed during normalization."""
        rec_spaces = {"name": "  alice   smith  "}
        rec_clean = {"name": "alice smith"}
        fp_spaces = engine.fingerprint_record(rec_spaces, ["name"], FingerprintAlgorithm.SHA256)
        fp_clean = engine.fingerprint_record(rec_clean, ["name"], FingerprintAlgorithm.SHA256)
        assert fp_spaces.fingerprint_hash == fp_clean.fingerprint_hash

    def test_fingerprint_sha256_auto_record_id(self, engine: RecordFingerprinter):
        """Record ID is auto-generated when not provided and no 'id' field exists."""
        rec = {"name": "Alice"}
        result = engine.fingerprint_record(rec, ["name"], FingerprintAlgorithm.SHA256)
        assert result.record_id != ""

    def test_fingerprint_sha256_uses_id_field_if_present(self, engine: RecordFingerprinter):
        """Record ID defaults to the 'id' field if present and record_id not specified."""
        rec = {"id": "my-id-123", "name": "Alice"}
        result = engine.fingerprint_record(rec, ["name"], FingerprintAlgorithm.SHA256)
        assert result.record_id == "my-id-123"

    def test_fingerprint_sha256_created_at_present(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """Fingerprint result has a created_at timestamp."""
        result = engine.fingerprint_record(simple_record, ["name"], FingerprintAlgorithm.SHA256)
        assert result.created_at is not None
        assert isinstance(result.created_at, datetime)


# ===========================================================================
# Test Class: SimHash Fingerprinting
# ===========================================================================


class TestFingerprintSimHash:
    """Tests for SimHash locality-sensitive hashing."""

    def test_fingerprint_simhash_returns_valid_result(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """SimHash fingerprint returns a valid RecordFingerprint."""
        result = engine.fingerprint_record(
            simple_record, ["name", "email"], FingerprintAlgorithm.SIMHASH,
        )
        assert isinstance(result, RecordFingerprint)
        assert result.algorithm == FingerprintAlgorithm.SIMHASH
        assert result.fingerprint_hash != ""

    def test_fingerprint_simhash_deterministic(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """Same input produces same SimHash (determinism)."""
        fp1 = engine.fingerprint_record(
            simple_record, ["name"], FingerprintAlgorithm.SIMHASH, "rec-001",
        )
        fp2 = engine.fingerprint_record(
            simple_record, ["name"], FingerprintAlgorithm.SIMHASH, "rec-001",
        )
        assert fp1.fingerprint_hash == fp2.fingerprint_hash

    def test_compute_simhash_empty_string(self, engine: RecordFingerprinter):
        """SimHash of empty string returns 0."""
        assert engine.compute_simhash("") == 0

    def test_compute_simhash_short_string(self, engine: RecordFingerprinter):
        """SimHash of short string (less than n-gram size) returns valid value."""
        result = engine.compute_simhash("ab")
        assert isinstance(result, int)

    def test_compute_simhash_returns_integer(self, engine: RecordFingerprinter):
        """SimHash returns an integer value."""
        result = engine.compute_simhash("hello world")
        assert isinstance(result, int)
        assert result >= 0

    def test_compute_simhash_similar_strings_have_close_hashes(self, engine: RecordFingerprinter):
        """Similar strings produce SimHash values with few differing bits."""
        hash_a = engine.compute_simhash("alice smith")
        hash_b = engine.compute_simhash("alise smith")
        # Hamming distance should be small for similar strings
        xor = hash_a ^ hash_b
        hamming = bin(xor).count("1")
        # For similar strings, hamming distance should be less than half the bits
        assert hamming < 32, f"Hamming distance {hamming} is too large for similar strings"

    def test_compute_simhash_different_strings_differ(self, engine: RecordFingerprinter):
        """Very different strings produce different SimHash values."""
        hash_a = engine.compute_simhash("the quick brown fox jumps over the lazy dog")
        hash_b = engine.compute_simhash("1234567890 abcdef ghijklmnop")
        assert hash_a != hash_b

    def test_compute_simhash_identical_strings_match(self, engine: RecordFingerprinter):
        """Identical strings produce the same SimHash."""
        text = "sustainability carbon emissions report"
        assert engine.compute_simhash(text) == engine.compute_simhash(text)

    def test_compute_simhash_custom_bit_width(self, engine: RecordFingerprinter):
        """SimHash respects custom hash bit width."""
        hash_32 = engine.compute_simhash("test string", hash_bits=32)
        hash_64 = engine.compute_simhash("test string", hash_bits=64)
        # 32-bit hash should fit in 32 bits
        assert hash_32 < (1 << 32)
        assert isinstance(hash_64, int)

    def test_fingerprint_simhash_hex_format(self, engine: RecordFingerprinter):
        """SimHash fingerprint is formatted as hex string."""
        rec = {"name": "test"}
        result = engine.fingerprint_record(rec, ["name"], FingerprintAlgorithm.SIMHASH)
        # Default 64-bit hash -> 16 hex chars
        assert len(result.fingerprint_hash) == 16

    def test_compute_simhash_single_character(self, engine: RecordFingerprinter):
        """SimHash of a single character returns valid value."""
        result = engine.compute_simhash("a")
        assert isinstance(result, int)

    def test_compute_simhash_unicode(self, engine: RecordFingerprinter):
        """SimHash handles unicode characters."""
        result = engine.compute_simhash("muller zurich")
        assert isinstance(result, int)
        assert result >= 0


# ===========================================================================
# Test Class: MinHash Fingerprinting
# ===========================================================================


class TestFingerprintMinHash:
    """Tests for MinHash signature generation and Jaccard estimation."""

    def test_fingerprint_minhash_returns_valid_result(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """MinHash fingerprint returns a valid RecordFingerprint."""
        result = engine.fingerprint_record(
            simple_record, ["name", "email"], FingerprintAlgorithm.MINHASH,
        )
        assert isinstance(result, RecordFingerprint)
        assert result.algorithm == FingerprintAlgorithm.MINHASH
        assert len(result.fingerprint_hash) == 64  # SHA-256 of signature

    def test_fingerprint_minhash_deterministic(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """Same input produces same MinHash fingerprint (determinism)."""
        fp1 = engine.fingerprint_record(
            simple_record, ["name"], FingerprintAlgorithm.MINHASH, "rec-001",
        )
        fp2 = engine.fingerprint_record(
            simple_record, ["name"], FingerprintAlgorithm.MINHASH, "rec-001",
        )
        assert fp1.fingerprint_hash == fp2.fingerprint_hash

    def test_compute_minhash_empty_string(self, engine: RecordFingerprinter):
        """MinHash of empty string returns all max-hash signature."""
        sig = engine.compute_minhash("", num_hashes=10)
        assert len(sig) == 10
        assert all(v == _MINHASH_MAX_HASH for v in sig)

    def test_compute_minhash_returns_correct_length(self, engine: RecordFingerprinter):
        """MinHash signature length matches num_hashes."""
        sig = engine.compute_minhash("hello world", num_hashes=64)
        assert len(sig) == 64
        sig = engine.compute_minhash("hello world", num_hashes=128)
        assert len(sig) == 128

    def test_compute_minhash_values_are_integers(self, engine: RecordFingerprinter):
        """All MinHash signature values are non-negative integers."""
        sig = engine.compute_minhash("test string", num_hashes=32)
        for v in sig:
            assert isinstance(v, int)
            assert v >= 0

    def test_compute_minhash_deterministic(self, engine: RecordFingerprinter):
        """Same input produces same MinHash signature."""
        sig1 = engine.compute_minhash("sustainability report", num_hashes=50)
        sig2 = engine.compute_minhash("sustainability report", num_hashes=50)
        assert sig1 == sig2

    def test_compute_minhash_short_string(self, engine: RecordFingerprinter):
        """MinHash of a string shorter than n-gram size returns valid signature."""
        sig = engine.compute_minhash("ab", num_hashes=10)
        assert len(sig) == 10

    def test_compute_minhash_different_strings_differ(self, engine: RecordFingerprinter):
        """Different strings produce different MinHash signatures."""
        sig_a = engine.compute_minhash("alice smith carbon report", num_hashes=64)
        sig_b = engine.compute_minhash("xyz 12345 completely different", num_hashes=64)
        # Not all values should match for very different strings
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        assert matches < len(sig_a)

    def test_compute_minhash_signature_values_less_than_prime(self, engine: RecordFingerprinter):
        """All MinHash signature values are less than the Mersenne prime."""
        sig = engine.compute_minhash("test minhash values", num_hashes=128)
        for v in sig:
            assert v <= _MINHASH_MAX_HASH

    def test_estimate_jaccard_identical_signatures(self, engine: RecordFingerprinter):
        """Jaccard estimation of identical signatures returns 1.0."""
        sig = engine.compute_minhash("same text", num_hashes=64)
        assert engine.estimate_jaccard(sig, sig) == 1.0

    def test_estimate_jaccard_similar_strings(self, engine: RecordFingerprinter):
        """Jaccard estimation of similar strings returns high similarity."""
        sig_a = engine.compute_minhash("alice smith from new york", num_hashes=128)
        sig_b = engine.compute_minhash("alice smith from new jersey", num_hashes=128)
        jaccard = engine.estimate_jaccard(sig_a, sig_b)
        assert 0.0 <= jaccard <= 1.0
        # Similar strings should have moderate to high Jaccard
        assert jaccard > 0.3

    def test_estimate_jaccard_completely_different(self, engine: RecordFingerprinter):
        """Jaccard estimation of very different strings returns low similarity."""
        sig_a = engine.compute_minhash("abcdefghijklmnopqrstuvwxyz full", num_hashes=128)
        sig_b = engine.compute_minhash("1234567890 numeric only data x", num_hashes=128)
        jaccard = engine.estimate_jaccard(sig_a, sig_b)
        assert 0.0 <= jaccard <= 1.0
        assert jaccard < 0.5

    def test_estimate_jaccard_different_length_raises(self, engine: RecordFingerprinter):
        """Jaccard estimation raises ValueError for different-length signatures."""
        sig_a = [1, 2, 3]
        sig_b = [1, 2]
        with pytest.raises(ValueError, match="Signature lengths must match"):
            engine.estimate_jaccard(sig_a, sig_b)

    def test_estimate_jaccard_empty_signatures(self, engine: RecordFingerprinter):
        """Jaccard estimation of empty signatures returns 0.0."""
        assert engine.estimate_jaccard([], []) == 0.0

    def test_estimate_jaccard_in_range(self, engine: RecordFingerprinter):
        """All Jaccard estimates are in the range [0.0, 1.0]."""
        texts = [
            "hello world carbon emissions",
            "sustainability energy report",
            "cbam regulation compliance eu",
            "deforestation monitoring data",
        ]
        sigs = [engine.compute_minhash(t, num_hashes=64) for t in texts]
        for i in range(len(sigs)):
            for j in range(len(sigs)):
                j_val = engine.estimate_jaccard(sigs[i], sigs[j])
                assert 0.0 <= j_val <= 1.0

    def test_minhash_coefficient_initialization(self, engine: RecordFingerprinter):
        """MinHash coefficients are initialized with deterministic seed."""
        engine.compute_minhash("test", num_hashes=10)
        assert engine._minhash_initialized is True
        assert len(engine._minhash_a) >= 10
        assert len(engine._minhash_b) >= 10

    def test_minhash_coefficient_reuse(self, engine: RecordFingerprinter):
        """Coefficients are reused across invocations (not regenerated)."""
        engine.compute_minhash("test", num_hashes=50)
        a_first = engine._minhash_a[:50]
        engine.compute_minhash("another test", num_hashes=50)
        a_second = engine._minhash_a[:50]
        assert a_first == a_second

    def test_minhash_coefficient_expansion(self, engine: RecordFingerprinter):
        """Coefficients are expanded if more hashes are requested."""
        engine.compute_minhash("test", num_hashes=10)
        assert len(engine._minhash_a) >= 10
        engine.compute_minhash("test", num_hashes=200)
        assert len(engine._minhash_a) >= 200


# ===========================================================================
# Test Class: Batch Fingerprinting
# ===========================================================================


class TestFingerprintBatch:
    """Tests for batch fingerprinting."""

    def test_fingerprint_batch_sha256(self, engine: RecordFingerprinter, batch_records: List[Dict[str, Any]]):
        """Batch SHA-256 fingerprinting processes all records."""
        results = engine.fingerprint_batch(
            batch_records, ["name", "email"], FingerprintAlgorithm.SHA256,
        )
        assert len(results) == 5
        assert all(isinstance(r, RecordFingerprint) for r in results)

    def test_fingerprint_batch_simhash(self, engine: RecordFingerprinter, batch_records: List[Dict[str, Any]]):
        """Batch SimHash fingerprinting processes all records."""
        results = engine.fingerprint_batch(
            batch_records, ["name"], FingerprintAlgorithm.SIMHASH,
        )
        assert len(results) == 5

    def test_fingerprint_batch_minhash(self, engine: RecordFingerprinter, batch_records: List[Dict[str, Any]]):
        """Batch MinHash fingerprinting processes all records."""
        results = engine.fingerprint_batch(
            batch_records, ["name"], FingerprintAlgorithm.MINHASH,
        )
        assert len(results) == 5

    def test_fingerprint_batch_uses_id_field(self, engine: RecordFingerprinter, batch_records: List[Dict[str, Any]]):
        """Batch fingerprinting uses the specified id_field."""
        results = engine.fingerprint_batch(
            batch_records, ["name"], FingerprintAlgorithm.SHA256, id_field="id",
        )
        assert results[0].record_id == "b-001"
        assert results[4].record_id == "b-005"

    def test_fingerprint_batch_auto_id_when_missing(self, engine: RecordFingerprinter):
        """Batch fingerprinting auto-generates record IDs when id_field is missing."""
        records = [{"name": "Alice"}, {"name": "Bob"}]
        results = engine.fingerprint_batch(records, ["name"], FingerprintAlgorithm.SHA256, id_field="id")
        assert results[0].record_id == "rec-0"
        assert results[1].record_id == "rec-1"

    def test_fingerprint_batch_empty_raises(self, engine: RecordFingerprinter):
        """Batch fingerprinting with empty list raises ValueError."""
        with pytest.raises(ValueError, match="records list must not be empty"):
            engine.fingerprint_batch([], ["name"], FingerprintAlgorithm.SHA256)

    def test_fingerprint_batch_all_unique_hashes(self, engine: RecordFingerprinter, batch_records: List[Dict[str, Any]]):
        """All batch fingerprints should be unique for distinct records."""
        results = engine.fingerprint_batch(
            batch_records, ["name", "email"], FingerprintAlgorithm.SHA256,
        )
        hashes = [r.fingerprint_hash for r in results]
        assert len(set(hashes)) == len(hashes)

    def test_fingerprint_batch_identical_records_same_hash(self, engine: RecordFingerprinter):
        """Identical records in a batch produce identical fingerprint hashes."""
        records = [
            {"id": "a", "name": "Alice"},
            {"id": "b", "name": "Alice"},
        ]
        results = engine.fingerprint_batch(records, ["name"], FingerprintAlgorithm.SHA256)
        assert results[0].fingerprint_hash == results[1].fingerprint_hash

    def test_fingerprint_batch_with_field_types(self, engine: RecordFingerprinter):
        """Batch fingerprinting with field type specifications."""
        records = [
            {"id": "1", "amount": 100.5},
            {"id": "2", "amount": 200.3},
        ]
        results = engine.fingerprint_batch(
            records, ["amount"], FingerprintAlgorithm.SHA256,
            field_types={"amount": FieldType.NUMERIC},
        )
        assert len(results) == 2
        assert results[0].fingerprint_hash != results[1].fingerprint_hash

    def test_fingerprint_batch_large_set(self, engine: RecordFingerprinter):
        """Batch fingerprinting handles a large number of records."""
        records = [{"id": f"rec-{i}", "name": f"Person {i}"} for i in range(200)]
        results = engine.fingerprint_batch(records, ["name"], FingerprintAlgorithm.SHA256)
        assert len(results) == 200


# ===========================================================================
# Test Class: Field Normalization
# ===========================================================================


class TestFieldNormalization:
    """Tests for type-aware field normalization."""

    # -- STRING normalization --

    def test_normalize_string_basic(self, engine: RecordFingerprinter):
        """STRING normalization lowercases and strips."""
        assert engine.normalize_field("  HELLO WORLD  ", FieldType.STRING) == "hello world"

    def test_normalize_string_collapse_whitespace(self, engine: RecordFingerprinter):
        """STRING normalization collapses multiple whitespace."""
        assert engine.normalize_field("hello   world", FieldType.STRING) == "hello world"

    def test_normalize_string_tabs_and_newlines(self, engine: RecordFingerprinter):
        """STRING normalization handles tabs and newlines."""
        assert engine.normalize_field("hello\t\nworld", FieldType.STRING) == "hello world"

    def test_normalize_string_empty(self, engine: RecordFingerprinter):
        """STRING normalization of empty string returns empty."""
        assert engine.normalize_field("", FieldType.STRING) == ""

    def test_normalize_string_none(self, engine: RecordFingerprinter):
        """STRING normalization of None returns empty string."""
        assert engine.normalize_field(None, FieldType.STRING) == ""

    def test_normalize_string_number_input(self, engine: RecordFingerprinter):
        """STRING normalization converts numbers to string."""
        assert engine.normalize_field(42, FieldType.STRING) == "42"

    def test_normalize_string_unicode(self, engine: RecordFingerprinter):
        """STRING normalization handles unicode characters."""
        result = engine.normalize_field("Cafe Resume", FieldType.STRING)
        assert result == "cafe resume"

    def test_normalize_string_special_chars(self, engine: RecordFingerprinter):
        """STRING normalization preserves special characters."""
        result = engine.normalize_field("test@email.com!", FieldType.STRING)
        assert result == "test@email.com!"

    # -- NUMERIC normalization --

    def test_normalize_numeric_integer(self, engine: RecordFingerprinter):
        """NUMERIC normalization of integer rounds to 6 decimals."""
        assert engine.normalize_field(42, FieldType.NUMERIC) == "42.000000"

    def test_normalize_numeric_float(self, engine: RecordFingerprinter):
        """NUMERIC normalization of float rounds to 6 decimals."""
        assert engine.normalize_field(3.14159, FieldType.NUMERIC) == "3.141590"

    def test_normalize_numeric_string_input(self, engine: RecordFingerprinter):
        """NUMERIC normalization parses numeric strings."""
        assert engine.normalize_field("123.456", FieldType.NUMERIC) == "123.456000"

    def test_normalize_numeric_zero(self, engine: RecordFingerprinter):
        """NUMERIC normalization of zero."""
        assert engine.normalize_field(0, FieldType.NUMERIC) == "0.000000"

    def test_normalize_numeric_negative(self, engine: RecordFingerprinter):
        """NUMERIC normalization of negative number."""
        assert engine.normalize_field(-99.5, FieldType.NUMERIC) == "-99.500000"

    def test_normalize_numeric_none(self, engine: RecordFingerprinter):
        """NUMERIC normalization of None returns empty string."""
        assert engine.normalize_field(None, FieldType.NUMERIC) == ""

    def test_normalize_numeric_non_numeric_string(self, engine: RecordFingerprinter):
        """NUMERIC normalization of non-numeric string returns stripped string."""
        result = engine.normalize_field("not-a-number", FieldType.NUMERIC)
        assert result == "not-a-number"

    def test_normalize_numeric_large_value(self, engine: RecordFingerprinter):
        """NUMERIC normalization handles large values."""
        result = engine.normalize_field(1e15, FieldType.NUMERIC)
        assert "1000000000000000" in result

    # -- DATE normalization --

    def test_normalize_date_iso_format(self, engine: RecordFingerprinter):
        """DATE normalization of ISO date format."""
        assert engine.normalize_field("2025-06-15", FieldType.DATE) == "2025-06-15"

    def test_normalize_date_iso_datetime(self, engine: RecordFingerprinter):
        """DATE normalization of ISO datetime extracts date part."""
        assert engine.normalize_field("2025-06-15T10:30:00", FieldType.DATE) == "2025-06-15"

    def test_normalize_date_us_format(self, engine: RecordFingerprinter):
        """DATE normalization of US date format (MM/DD/YYYY)."""
        assert engine.normalize_field("06/15/2025", FieldType.DATE) == "2025-06-15"

    def test_normalize_date_eu_dash_format(self, engine: RecordFingerprinter):
        """DATE normalization of EU dash format (DD-MM-YYYY)."""
        assert engine.normalize_field("15-06-2025", FieldType.DATE) == "2025-06-15"

    def test_normalize_date_eu_slash_format(self, engine: RecordFingerprinter):
        """DATE normalization of EU slash format (DD/MM/YYYY)."""
        # Note: this matches the US format first (MM/DD/YYYY)
        # So 15/06/2025 would fail MM/DD/YYYY (month=15) and try DD/MM/YYYY
        assert engine.normalize_field("15/06/2025", FieldType.DATE) == "2025-06-15"

    def test_normalize_date_iso_with_z(self, engine: RecordFingerprinter):
        """DATE normalization of ISO format with Z suffix."""
        assert engine.normalize_field("2025-06-15T10:30:00Z", FieldType.DATE) == "2025-06-15"

    def test_normalize_date_none(self, engine: RecordFingerprinter):
        """DATE normalization of None returns empty string."""
        assert engine.normalize_field(None, FieldType.DATE) == ""

    def test_normalize_date_invalid(self, engine: RecordFingerprinter):
        """DATE normalization of unrecognized format returns lowercased input."""
        result = engine.normalize_field("not-a-date", FieldType.DATE)
        assert result == "not-a-date"

    def test_normalize_date_empty(self, engine: RecordFingerprinter):
        """DATE normalization of empty string."""
        result = engine.normalize_field("", FieldType.DATE)
        assert result == ""

    # -- BOOLEAN normalization --

    def test_normalize_boolean_true(self, engine: RecordFingerprinter):
        """BOOLEAN normalization of True."""
        assert engine.normalize_field(True, FieldType.BOOLEAN) == "true"

    def test_normalize_boolean_false(self, engine: RecordFingerprinter):
        """BOOLEAN normalization of False."""
        assert engine.normalize_field(False, FieldType.BOOLEAN) == "false"

    def test_normalize_boolean_string_true(self, engine: RecordFingerprinter):
        """BOOLEAN normalization of string 'true'."""
        assert engine.normalize_field("true", FieldType.BOOLEAN) == "true"

    def test_normalize_boolean_string_false(self, engine: RecordFingerprinter):
        """BOOLEAN normalization of string 'false'."""
        assert engine.normalize_field("false", FieldType.BOOLEAN) == "false"

    def test_normalize_boolean_1(self, engine: RecordFingerprinter):
        """BOOLEAN normalization of string '1'."""
        assert engine.normalize_field("1", FieldType.BOOLEAN) == "true"

    def test_normalize_boolean_yes(self, engine: RecordFingerprinter):
        """BOOLEAN normalization of string 'yes'."""
        assert engine.normalize_field("yes", FieldType.BOOLEAN) == "true"

    def test_normalize_boolean_y(self, engine: RecordFingerprinter):
        """BOOLEAN normalization of string 'y'."""
        assert engine.normalize_field("y", FieldType.BOOLEAN) == "true"

    def test_normalize_boolean_t(self, engine: RecordFingerprinter):
        """BOOLEAN normalization of string 't'."""
        assert engine.normalize_field("t", FieldType.BOOLEAN) == "true"

    def test_normalize_boolean_no(self, engine: RecordFingerprinter):
        """BOOLEAN normalization of string 'no' returns false."""
        assert engine.normalize_field("no", FieldType.BOOLEAN) == "false"

    def test_normalize_boolean_0(self, engine: RecordFingerprinter):
        """BOOLEAN normalization of string '0'."""
        assert engine.normalize_field("0", FieldType.BOOLEAN) == "false"

    def test_normalize_boolean_none(self, engine: RecordFingerprinter):
        """BOOLEAN normalization of None returns empty string."""
        assert engine.normalize_field(None, FieldType.BOOLEAN) == ""

    # -- CATEGORICAL normalization --

    def test_normalize_categorical_basic(self, engine: RecordFingerprinter):
        """CATEGORICAL normalization lowercases and strips."""
        assert engine.normalize_field("  HIGH  ", FieldType.CATEGORICAL) == "high"

    def test_normalize_categorical_mixed_case(self, engine: RecordFingerprinter):
        """CATEGORICAL normalization handles mixed case."""
        assert engine.normalize_field("MeDiUm", FieldType.CATEGORICAL) == "medium"

    def test_normalize_categorical_none(self, engine: RecordFingerprinter):
        """CATEGORICAL normalization of None returns empty string."""
        assert engine.normalize_field(None, FieldType.CATEGORICAL) == ""

    def test_normalize_categorical_number(self, engine: RecordFingerprinter):
        """CATEGORICAL normalization of number converts to string."""
        assert engine.normalize_field(42, FieldType.CATEGORICAL) == "42"

    def test_normalize_categorical_preserves_spaces(self, engine: RecordFingerprinter):
        """CATEGORICAL normalization does not collapse internal whitespace."""
        # _normalize_categorical only does strip().lower() -- does not collapse
        result = engine.normalize_field("category  a", FieldType.CATEGORICAL)
        assert result == "category  a"


# ===========================================================================
# Test Class: compute_sha256 direct
# ===========================================================================


class TestComputeSHA256:
    """Tests for the direct compute_sha256 method."""

    def test_compute_sha256_known_vector(self, engine: RecordFingerprinter):
        """SHA-256 of known input matches expected hash."""
        # SHA-256 of empty string is a well-known value
        expected = hashlib.sha256(b"").hexdigest()
        assert engine.compute_sha256("") == expected

    def test_compute_sha256_hello(self, engine: RecordFingerprinter):
        """SHA-256 of 'hello' matches expected."""
        expected = hashlib.sha256(b"hello").hexdigest()
        assert engine.compute_sha256("hello") == expected

    def test_compute_sha256_deterministic(self, engine: RecordFingerprinter):
        """SHA-256 is deterministic (same input = same output)."""
        text = "sustainability carbon emissions 2025"
        h1 = engine.compute_sha256(text)
        h2 = engine.compute_sha256(text)
        assert h1 == h2

    def test_compute_sha256_different_inputs_differ(self, engine: RecordFingerprinter):
        """Different inputs produce different SHA-256 hashes."""
        h1 = engine.compute_sha256("input_a")
        h2 = engine.compute_sha256("input_b")
        assert h1 != h2

    def test_compute_sha256_hex_length(self, engine: RecordFingerprinter):
        """SHA-256 hex digest is always 64 characters."""
        for text in ["", "a", "hello world", "x" * 10000]:
            assert len(engine.compute_sha256(text)) == 64

    def test_compute_sha256_unicode(self, engine: RecordFingerprinter):
        """SHA-256 handles unicode correctly."""
        result = engine.compute_sha256("cafe resume")
        assert len(result) == 64


# ===========================================================================
# Test Class: Error Handling / Edge Cases
# ===========================================================================


class TestFingerprintErrors:
    """Tests for error handling and edge cases."""

    def test_fingerprint_empty_record_raises(self, engine: RecordFingerprinter):
        """Fingerprinting an empty record raises ValueError."""
        with pytest.raises(ValueError, match="record must not be empty"):
            engine.fingerprint_record({}, ["name"], FingerprintAlgorithm.SHA256)

    def test_fingerprint_empty_field_set_raises(self, engine: RecordFingerprinter):
        """Fingerprinting with empty field_set raises ValueError."""
        with pytest.raises(ValueError, match="field_set must not be empty"):
            engine.fingerprint_record({"name": "Alice"}, [], FingerprintAlgorithm.SHA256)

    def test_fingerprint_record_with_none_values(self, engine: RecordFingerprinter):
        """Record with None field values still produces fingerprint."""
        rec = {"name": None, "email": "test@example.com"}
        result = engine.fingerprint_record(rec, ["name", "email"], FingerprintAlgorithm.SHA256)
        assert result.fingerprint_hash != ""

    def test_fingerprint_record_with_all_none_values(self, engine: RecordFingerprinter):
        """Record with all None values produces fingerprint (all empty after normalization)."""
        rec = {"name": None, "email": None}
        result = engine.fingerprint_record(rec, ["name", "email"], FingerprintAlgorithm.SHA256)
        assert result.fingerprint_hash != ""

    def test_fingerprint_record_with_empty_strings(self, engine: RecordFingerprinter):
        """Record with empty string values produces fingerprint."""
        rec = {"name": "", "email": ""}
        result = engine.fingerprint_record(rec, ["name", "email"], FingerprintAlgorithm.SHA256)
        assert result.fingerprint_hash != ""

    def test_fingerprint_record_with_mixed_types(self, engine: RecordFingerprinter):
        """Record with mixed field types still produces fingerprint."""
        rec = {"name": "Alice", "age": 30, "active": True, "score": 3.14}
        result = engine.fingerprint_record(rec, ["name", "age", "active", "score"], FingerprintAlgorithm.SHA256)
        assert result.fingerprint_hash != ""

    def test_fingerprint_record_with_field_types(self, engine: RecordFingerprinter):
        """Fingerprinting with field type specifications."""
        rec = {"amount": 100.5, "active": True}
        ftypes = {"amount": FieldType.NUMERIC, "active": FieldType.BOOLEAN}
        result = engine.fingerprint_record(
            rec, ["amount", "active"], FingerprintAlgorithm.SHA256, field_types=ftypes,
        )
        assert result.fingerprint_hash != ""

    def test_fingerprint_collision_identical_records(self, engine: RecordFingerprinter):
        """Identical records produce identical fingerprints (collision by design)."""
        rec1 = {"name": "Alice", "email": "alice@test.com"}
        rec2 = {"name": "Alice", "email": "alice@test.com"}
        fp1 = engine.fingerprint_record(rec1, ["name", "email"], FingerprintAlgorithm.SHA256)
        fp2 = engine.fingerprint_record(rec2, ["name", "email"], FingerprintAlgorithm.SHA256)
        assert fp1.fingerprint_hash == fp2.fingerprint_hash

    def test_fingerprint_uniqueness_different_records(self, engine: RecordFingerprinter):
        """Different records produce unique fingerprints."""
        records = [
            {"name": f"Person_{i}", "email": f"p{i}@test.com"}
            for i in range(50)
        ]
        fps = [
            engine.fingerprint_record(r, ["name", "email"], FingerprintAlgorithm.SHA256)
            for r in records
        ]
        hashes = [fp.fingerprint_hash for fp in fps]
        assert len(set(hashes)) == 50


# ===========================================================================
# Test Class: N-gram Generation
# ===========================================================================


class TestNgramGeneration:
    """Tests for the internal _generate_ngrams method."""

    def test_generate_ngrams_basic(self, engine: RecordFingerprinter):
        """3-gram generation from 'hello' produces correct shingles."""
        result = engine._generate_ngrams("hello", 3)
        assert result == ["hel", "ell", "llo"]

    def test_generate_ngrams_short_string(self, engine: RecordFingerprinter):
        """N-gram of string shorter than n returns [text]."""
        result = engine._generate_ngrams("ab", 3)
        assert result == ["ab"]

    def test_generate_ngrams_exact_length(self, engine: RecordFingerprinter):
        """N-gram of string with exact length n returns one shingle."""
        result = engine._generate_ngrams("abc", 3)
        assert result == ["abc"]

    def test_generate_ngrams_empty(self, engine: RecordFingerprinter):
        """N-gram of empty string returns empty list."""
        result = engine._generate_ngrams("", 3)
        assert result == []

    def test_generate_ngrams_single_char(self, engine: RecordFingerprinter):
        """N-gram of single character returns [char]."""
        result = engine._generate_ngrams("a", 3)
        assert result == ["a"]

    def test_generate_ngrams_bigrams(self, engine: RecordFingerprinter):
        """2-gram generation produces correct shingles."""
        result = engine._generate_ngrams("abcd", 2)
        assert result == ["ab", "bc", "cd"]


# ===========================================================================
# Test Class: Statistics Tracking
# ===========================================================================


class TestFingerprintStatistics:
    """Tests for thread-safe statistics tracking."""

    def test_initial_statistics(self, engine: RecordFingerprinter):
        """Initial statistics are all zero."""
        stats = engine.get_statistics()
        assert stats["engine_name"] == "RecordFingerprinter"
        assert stats["invocations"] == 0
        assert stats["successes"] == 0
        assert stats["failures"] == 0
        assert stats["total_duration_ms"] == 0.0
        assert stats["avg_duration_ms"] == 0.0
        assert stats["last_invoked_at"] is None

    def test_statistics_after_success(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """Statistics increment after successful fingerprinting."""
        engine.fingerprint_record(simple_record, ["name"], FingerprintAlgorithm.SHA256)
        stats = engine.get_statistics()
        assert stats["invocations"] == 1
        assert stats["successes"] == 1
        assert stats["failures"] == 0
        assert stats["total_duration_ms"] >= 0  # May be 0 on fast machines
        assert stats["last_invoked_at"] is not None

    def test_statistics_after_failure(self, engine: RecordFingerprinter):
        """Statistics increment after failed fingerprinting."""
        with pytest.raises(ValueError):
            engine.fingerprint_record({}, ["name"], FingerprintAlgorithm.SHA256)
        stats = engine.get_statistics()
        assert stats["invocations"] == 1
        assert stats["successes"] == 0
        assert stats["failures"] == 1

    def test_statistics_after_multiple_operations(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """Statistics accumulate across multiple operations."""
        for _ in range(5):
            engine.fingerprint_record(simple_record, ["name"], FingerprintAlgorithm.SHA256)
        stats = engine.get_statistics()
        assert stats["invocations"] == 5
        assert stats["successes"] == 5

    def test_reset_statistics(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """Reset clears all statistics."""
        engine.fingerprint_record(simple_record, ["name"], FingerprintAlgorithm.SHA256)
        engine.reset_statistics()
        stats = engine.get_statistics()
        assert stats["invocations"] == 0
        assert stats["successes"] == 0
        assert stats["failures"] == 0
        assert stats["total_duration_ms"] == 0.0
        assert stats["last_invoked_at"] is None

    def test_statistics_avg_duration(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """Average duration is computed correctly."""
        for _ in range(3):
            engine.fingerprint_record(simple_record, ["name"], FingerprintAlgorithm.SHA256)
        stats = engine.get_statistics()
        expected_avg = stats["total_duration_ms"] / 3
        assert abs(stats["avg_duration_ms"] - expected_avg) < 0.01

    def test_statistics_thread_safety(self, engine: RecordFingerprinter):
        """Statistics remain consistent under concurrent access."""
        record = {"name": "test"}
        errors: List[str] = []

        def worker():
            try:
                for _ in range(20):
                    engine.fingerprint_record(record, ["name"], FingerprintAlgorithm.SHA256)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = engine.get_statistics()
        assert stats["invocations"] == 100
        assert stats["successes"] == 100
        assert stats["failures"] == 0


# ===========================================================================
# Test Class: Provenance
# ===========================================================================


class TestFingerprintProvenance:
    """Tests for provenance hash generation."""

    def test_provenance_hash_is_sha256(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """Provenance hash is a 64-character hex SHA-256 digest."""
        result = engine.fingerprint_record(simple_record, ["name"], FingerprintAlgorithm.SHA256)
        assert len(result.provenance_hash) == 64
        # Should be valid hex
        int(result.provenance_hash, 16)

    def test_provenance_hash_not_empty(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """Every fingerprint has a non-empty provenance hash."""
        for algo in FingerprintAlgorithm:
            result = engine.fingerprint_record(simple_record, ["name"], algo)
            assert result.provenance_hash != ""

    def test_provenance_hash_different_for_different_algorithms(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """Provenance hashes differ when using different algorithms on same record."""
        fps = {}
        for algo in FingerprintAlgorithm:
            fps[algo] = engine.fingerprint_record(
                simple_record, ["name"], algo, record_id="rec-001",
            )
        # Provenance includes algorithm name, so they should differ
        prov_hashes = [fp.provenance_hash for fp in fps.values()]
        assert len(set(prov_hashes)) == len(prov_hashes)

    def test_provenance_hash_valid_hex(self, engine: RecordFingerprinter, simple_record: Dict[str, Any]):
        """Provenance hash contains only valid hex characters."""
        result = engine.fingerprint_record(simple_record, ["name"], FingerprintAlgorithm.SHA256)
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)


# ===========================================================================
# Test Class: Fingerprint with Field Types
# ===========================================================================


class TestFingerprintWithFieldTypes:
    """Tests for fingerprinting with explicit field type specifications."""

    def test_numeric_field_type_normalization(self, engine: RecordFingerprinter):
        """NUMERIC field type normalizes float representation."""
        rec1 = {"amount": 100.5}
        rec2 = {"amount": "100.5"}
        ft = {"amount": FieldType.NUMERIC}
        fp1 = engine.fingerprint_record(rec1, ["amount"], FingerprintAlgorithm.SHA256, field_types=ft)
        fp2 = engine.fingerprint_record(rec2, ["amount"], FingerprintAlgorithm.SHA256, field_types=ft)
        assert fp1.fingerprint_hash == fp2.fingerprint_hash

    def test_boolean_field_type_normalization(self, engine: RecordFingerprinter):
        """BOOLEAN field type normalizes various truthy values."""
        rec1 = {"active": True}
        rec2 = {"active": "yes"}
        ft = {"active": FieldType.BOOLEAN}
        fp1 = engine.fingerprint_record(rec1, ["active"], FingerprintAlgorithm.SHA256, field_types=ft)
        fp2 = engine.fingerprint_record(rec2, ["active"], FingerprintAlgorithm.SHA256, field_types=ft)
        assert fp1.fingerprint_hash == fp2.fingerprint_hash

    def test_date_field_type_normalization(self, engine: RecordFingerprinter):
        """DATE field type normalizes different date formats."""
        rec1 = {"created": "2025-06-15"}
        rec2 = {"created": "06/15/2025"}
        ft = {"created": FieldType.DATE}
        fp1 = engine.fingerprint_record(rec1, ["created"], FingerprintAlgorithm.SHA256, field_types=ft)
        fp2 = engine.fingerprint_record(rec2, ["created"], FingerprintAlgorithm.SHA256, field_types=ft)
        assert fp1.fingerprint_hash == fp2.fingerprint_hash

    def test_default_field_type_is_string(self, engine: RecordFingerprinter):
        """Fields without explicit type default to STRING."""
        rec = {"name": "ALICE"}
        fp_no_type = engine.fingerprint_record(rec, ["name"], FingerprintAlgorithm.SHA256)
        fp_with_type = engine.fingerprint_record(
            rec, ["name"], FingerprintAlgorithm.SHA256, field_types={"name": FieldType.STRING},
        )
        assert fp_no_type.fingerprint_hash == fp_with_type.fingerprint_hash

    def test_mixed_field_types(self, engine: RecordFingerprinter):
        """Fingerprinting with mixed field types."""
        rec = {"name": "Alice", "amount": 100.5, "active": True}
        ft = {
            "name": FieldType.STRING,
            "amount": FieldType.NUMERIC,
            "active": FieldType.BOOLEAN,
        }
        result = engine.fingerprint_record(
            rec, ["name", "amount", "active"], FingerprintAlgorithm.SHA256, field_types=ft,
        )
        assert result.fingerprint_hash != ""
        assert len(result.fingerprint_hash) == 64


# ===========================================================================
# Test Class: hash_to_int helper
# ===========================================================================


class TestHashToInt:
    """Tests for _hash_to_int internal method."""

    def test_hash_to_int_returns_integer(self, engine: RecordFingerprinter):
        """_hash_to_int returns an integer."""
        result = engine._hash_to_int("test", 64)
        assert isinstance(result, int)

    def test_hash_to_int_respects_bit_width(self, engine: RecordFingerprinter):
        """_hash_to_int result fits within specified bit width."""
        for bits in [8, 16, 32, 64]:
            result = engine._hash_to_int("test", bits)
            assert result < (1 << bits)

    def test_hash_to_int_deterministic(self, engine: RecordFingerprinter):
        """_hash_to_int is deterministic."""
        assert engine._hash_to_int("hello", 64) == engine._hash_to_int("hello", 64)

    def test_hash_to_int_different_inputs(self, engine: RecordFingerprinter):
        """_hash_to_int produces different values for different inputs."""
        h1 = engine._hash_to_int("alpha", 64)
        h2 = engine._hash_to_int("beta", 64)
        assert h1 != h2
