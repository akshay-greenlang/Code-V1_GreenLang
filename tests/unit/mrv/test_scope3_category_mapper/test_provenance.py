# -*- coding: utf-8 -*-
"""
Unit tests for Provenance module (AGENT-MRV-029)

40 tests covering:
- Stage hash computation (deterministic, divergent inputs)
- Chain hash computation (order-dependent)
- Provenance record creation
- Full 10-stage chain building
- Chain verification (valid and tampered)
- Serialization (Decimal, datetime, Enum)
- SHA-256 format validation

The provenance module provides bit-perfect reproducible audit trails
for all classification and boundary decisions.

Author: GL-TestEngineer
Date: March 2026
"""

import hashlib
import json
import re
import pytest
from decimal import Decimal
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.scope3_category_mapper.provenance import (
        ProvenanceTracker,
        ProvenanceEntry,
        ProvenanceChain,
        compute_stage_hash,
        compute_chain_hash,
        create_provenance_record,
        build_chain,
        verify_chain,
        serialize_for_hash,
    )
    PROVENANCE_AVAILABLE = True
except (ImportError, AttributeError):
    PROVENANCE_AVAILABLE = False

# Fallback: use the hashing utilities from completeness_screener
try:
    from greenlang.agents.mrv.scope3_category_mapper.completeness_screener import (
        _serialize_for_hash,
        _compute_hash,
    )
    HASH_UTILS_AVAILABLE = True
except ImportError:
    HASH_UTILS_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not PROVENANCE_AVAILABLE,
    reason="Provenance module not available",
)

_SKIP_HASH = pytest.mark.skipif(
    not HASH_UTILS_AVAILABLE,
    reason="Hash utilities not available",
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def sample_stage_data() -> Dict[str, Any]:
    """Sample data for a pipeline stage."""
    return {
        "stage": "classification",
        "record_id": "SPD-001",
        "primary_category": "cat_1_purchased_goods",
        "confidence": Decimal("0.92"),
        "method": "naics_lookup",
        "timestamp": "2025-03-15T12:00:00+00:00",
    }


@pytest.fixture
def sample_stage_data_alt() -> Dict[str, Any]:
    """Alternative stage data for divergence testing."""
    return {
        "stage": "classification",
        "record_id": "SPD-002",
        "primary_category": "cat_6_business_travel",
        "confidence": Decimal("0.88"),
        "method": "keyword_match",
        "timestamp": "2025-03-15T12:00:01+00:00",
    }


@pytest.fixture
def ten_stage_data_list() -> list:
    """Data for all 10 pipeline stages."""
    stages = [
        "input_validation", "source_classification", "code_lookup",
        "classification", "boundary_determination", "double_counting_check",
        "completeness_screening", "compliance_assessment",
        "provenance_sealing", "output_formatting",
    ]
    return [
        {
            "stage": stage,
            "stage_index": i,
            "record_count": 10,
            "status": "completed",
            "timestamp": f"2025-03-15T12:00:{i:02d}+00:00",
        }
        for i, stage in enumerate(stages)
    ]


# ==============================================================================
# STAGE HASH TESTS
# ==============================================================================


@_SKIP
class TestComputeStageHash:
    """Test compute_stage_hash function."""

    def test_compute_stage_hash_deterministic(self, sample_stage_data):
        """Same data produces same hash."""
        h1 = compute_stage_hash(sample_stage_data)
        h2 = compute_stage_hash(sample_stage_data)
        assert h1 == h2

    def test_compute_stage_hash_different_data(
        self, sample_stage_data, sample_stage_data_alt
    ):
        """Different data produces different hash."""
        h1 = compute_stage_hash(sample_stage_data)
        h2 = compute_stage_hash(sample_stage_data_alt)
        assert h1 != h2

    def test_compute_stage_hash_sha256_format(self, sample_stage_data):
        """Hash is 64-char lowercase hex (SHA-256)."""
        h = compute_stage_hash(sample_stage_data)
        assert re.match(r"^[a-f0-9]{64}$", h)

    def test_compute_stage_hash_empty_data(self):
        """Empty dict produces a valid hash."""
        h = compute_stage_hash({})
        assert len(h) == 64


# ==============================================================================
# CHAIN HASH TESTS
# ==============================================================================


@_SKIP
class TestComputeChainHash:
    """Test compute_chain_hash function."""

    def test_compute_chain_hash(self, ten_stage_data_list):
        """Chain hash over 10 stages produces valid hash."""
        hashes = [compute_stage_hash(s) for s in ten_stage_data_list]
        chain_hash = compute_chain_hash(hashes)
        assert len(chain_hash) == 64

    def test_chain_hash_order_matters(self, ten_stage_data_list):
        """Reversing stage order changes the chain hash."""
        hashes = [compute_stage_hash(s) for s in ten_stage_data_list]
        chain_fwd = compute_chain_hash(hashes)
        chain_rev = compute_chain_hash(list(reversed(hashes)))
        assert chain_fwd != chain_rev

    def test_chain_hash_deterministic(self, ten_stage_data_list):
        """Same hashes in same order produce same chain hash."""
        hashes = [compute_stage_hash(s) for s in ten_stage_data_list]
        c1 = compute_chain_hash(hashes)
        c2 = compute_chain_hash(hashes)
        assert c1 == c2

    def test_chain_hash_single_stage(self, sample_stage_data):
        """Chain hash with single stage is valid."""
        h = compute_stage_hash(sample_stage_data)
        chain = compute_chain_hash([h])
        assert len(chain) == 64


# ==============================================================================
# PROVENANCE RECORD TESTS
# ==============================================================================


@_SKIP
class TestCreateProvenanceRecord:
    """Test create_provenance_record function."""

    def test_create_provenance_record(self, sample_stage_data):
        """Create a provenance record with valid fields."""
        record = create_provenance_record(
            stage="classification",
            input_data=sample_stage_data,
            output_data={"classified": True},
            previous_hash="0" * 64,
        )
        assert record is not None
        assert record.stage == "classification"
        assert len(record.input_hash) == 64
        assert len(record.output_hash) == 64

    def test_provenance_record_has_timestamp(self, sample_stage_data):
        """Provenance record has an ISO 8601 timestamp."""
        record = create_provenance_record(
            stage="classification",
            input_data=sample_stage_data,
            output_data={},
            previous_hash="0" * 64,
        )
        assert "T" in record.timestamp

    def test_provenance_record_chain_hash(self, sample_stage_data):
        """Provenance record includes a chain hash."""
        record = create_provenance_record(
            stage="classification",
            input_data=sample_stage_data,
            output_data={},
            previous_hash="0" * 64,
        )
        assert len(record.chain_hash) == 64

    def test_provenance_record_previous_hash(self, sample_stage_data):
        """Provenance record links to previous hash."""
        prev = "a" * 64
        record = create_provenance_record(
            stage="boundary",
            input_data=sample_stage_data,
            output_data={},
            previous_hash=prev,
        )
        assert record.previous_hash == prev


# ==============================================================================
# FULL CHAIN TESTS
# ==============================================================================


@_SKIP
class TestBuildFullChain:
    """Test building a full 10-stage provenance chain."""

    def test_build_full_chain_10_stages(self, ten_stage_data_list):
        """Build chain with 10 stages."""
        chain = build_chain(ten_stage_data_list)
        assert len(chain.entries) == 10

    def test_chain_entries_linked(self, ten_stage_data_list):
        """Each entry's previous_hash matches prior entry's chain_hash."""
        chain = build_chain(ten_stage_data_list)
        for i in range(1, len(chain.entries)):
            assert chain.entries[i].previous_hash == chain.entries[i - 1].chain_hash

    def test_chain_root_is_zero(self, ten_stage_data_list):
        """First entry's previous_hash is all zeros (genesis)."""
        chain = build_chain(ten_stage_data_list)
        assert chain.entries[0].previous_hash == "0" * 64

    def test_chain_has_final_hash(self, ten_stage_data_list):
        """Chain has a final root hash."""
        chain = build_chain(ten_stage_data_list)
        assert len(chain.root_hash) == 64


# ==============================================================================
# CHAIN VERIFICATION TESTS
# ==============================================================================


@_SKIP
class TestVerifyChain:
    """Test chain verification and tamper detection."""

    def test_verify_chain_valid(self, ten_stage_data_list):
        """Valid chain verifies successfully."""
        chain = build_chain(ten_stage_data_list)
        assert verify_chain(chain) is True

    def test_verify_chain_tampered(self, ten_stage_data_list):
        """Tampered chain fails verification."""
        chain = build_chain(ten_stage_data_list)
        # Tamper with middle entry
        tampered = chain.entries[5]
        # Create a new entry with different data
        if hasattr(tampered, "_replace"):
            chain.entries[5] = tampered._replace(input_hash="f" * 64)
        else:
            # Mutable object path
            try:
                chain.entries[5].input_hash = "f" * 64
            except (AttributeError, TypeError):
                pytest.skip("Cannot tamper with immutable entries")
        assert verify_chain(chain) is False

    def test_verify_empty_chain(self):
        """Empty chain verifies as valid (vacuously true)."""
        chain = build_chain([])
        assert verify_chain(chain) is True

    def test_verify_single_entry_chain(self, sample_stage_data):
        """Single-entry chain verifies."""
        chain = build_chain([sample_stage_data])
        assert verify_chain(chain) is True


# ==============================================================================
# SERIALIZATION TESTS
# ==============================================================================


class TestSerialization:
    """Test serialization utilities for hashing."""

    @_SKIP
    def test_serialize_deterministic(self):
        """Serialization is deterministic (sorted keys)."""
        data = {"b": 2, "a": 1, "c": 3}
        s1 = serialize_for_hash(data)
        s2 = serialize_for_hash(data)
        assert s1 == s2

    @_SKIP
    def test_serialize_decimal(self):
        """Decimal values are serialized as strings."""
        data = {"value": Decimal("123.456")}
        s = serialize_for_hash(data)
        assert "123.456" in s

    @_SKIP
    def test_serialize_datetime(self):
        """datetime values are serialized as ISO strings."""
        dt = datetime(2025, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
        data = {"ts": dt}
        s = serialize_for_hash(data)
        assert "2025-03-15" in s

    @_SKIP
    def test_serialize_enum(self):
        """Enum values are serialized as their .value."""

        class Color(str, Enum):
            RED = "red"
            BLUE = "blue"

        data = {"color": Color.RED}
        s = serialize_for_hash(data)
        assert "red" in s

    @_SKIP_HASH
    def test_serialize_for_hash_from_screener_deterministic(self):
        """_serialize_for_hash from completeness_screener is deterministic."""
        data = {"z": 9, "a": 1, "m": 5}
        s1 = _serialize_for_hash(data)
        s2 = _serialize_for_hash(data)
        assert s1 == s2

    @_SKIP_HASH
    def test_serialize_for_hash_decimal(self):
        """_serialize_for_hash handles Decimal."""
        data = {"val": Decimal("99.99")}
        s = _serialize_for_hash(data)
        assert "99.99" in s

    @_SKIP_HASH
    def test_serialize_for_hash_datetime(self):
        """_serialize_for_hash handles datetime."""
        dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        data = {"dt": dt}
        s = _serialize_for_hash(data)
        assert "2026-01-01" in s

    @_SKIP_HASH
    def test_serialize_for_hash_enum(self):
        """_serialize_for_hash handles Enum."""

        class Status(str, Enum):
            OK = "ok"

        data = {"status": Status.OK}
        s = _serialize_for_hash(data)
        assert "ok" in s


# ==============================================================================
# SHA-256 FORMAT TESTS
# ==============================================================================


class TestSHA256Format:
    """Test SHA-256 hash format and properties."""

    @_SKIP_HASH
    def test_compute_hash_sha256(self):
        """_compute_hash returns valid SHA-256 hex string."""
        h = _compute_hash({"test": "data"})
        assert re.match(r"^[a-f0-9]{64}$", h)

    @_SKIP_HASH
    def test_compute_hash_deterministic(self):
        """Same input -> same hash."""
        data = {"key": "value", "num": 42}
        h1 = _compute_hash(data)
        h2 = _compute_hash(data)
        assert h1 == h2

    @_SKIP_HASH
    def test_compute_hash_different_input(self):
        """Different input -> different hash."""
        h1 = _compute_hash({"a": 1})
        h2 = _compute_hash({"a": 2})
        assert h1 != h2

    @_SKIP_HASH
    def test_compute_hash_matches_hashlib(self):
        """_compute_hash matches direct hashlib SHA-256."""
        data = {"hello": "world"}
        serialized = _serialize_for_hash(data)
        expected = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        actual = _compute_hash(data)
        assert actual == expected

    @_SKIP_HASH
    def test_hash_empty_dict(self):
        """Hash of empty dict is valid."""
        h = _compute_hash({})
        assert len(h) == 64

    @_SKIP_HASH
    def test_hash_nested_dict(self):
        """Hash handles nested dicts."""
        data = {"outer": {"inner": "value", "count": 5}}
        h = _compute_hash(data)
        assert len(h) == 64

    @_SKIP_HASH
    def test_hash_list_input(self):
        """Hash handles list input."""
        data = [1, 2, 3, "four"]
        h = _compute_hash(data)
        assert len(h) == 64

    @_SKIP_HASH
    def test_hash_with_decimal_and_datetime(self):
        """Hash handles mixed Decimal and datetime."""
        data = {
            "amount": Decimal("1234.56"),
            "ts": datetime(2025, 6, 15, tzinfo=timezone.utc),
            "label": "test",
        }
        h = _compute_hash(data)
        assert len(h) == 64

    @_SKIP_HASH
    def test_hash_boolean_values(self):
        """Hash handles boolean values."""
        h = _compute_hash({"flag": True, "other": False})
        assert len(h) == 64

    @_SKIP_HASH
    def test_hash_none_value(self):
        """Hash handles None values."""
        h = _compute_hash({"key": None})
        assert len(h) == 64

    @_SKIP_HASH
    def test_hash_large_decimal(self):
        """Hash handles large Decimal values."""
        data = {"big": Decimal("999999999999999.99999999")}
        h = _compute_hash(data)
        assert len(h) == 64

    @_SKIP_HASH
    def test_hash_unicode_string(self):
        """Hash handles unicode strings."""
        data = {"name": "Muller GmbH & Co. KG"}
        h = _compute_hash(data)
        assert len(h) == 64
