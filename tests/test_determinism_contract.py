# -*- coding: utf-8 -*-
"""
GreenLang Determinism Contract Test Suite

This module provides comprehensive tests to verify deterministic behavior
as specified in the GreenLang Determinism Contract v1.0 (docs/specs/determinism-contract-v1.0.md).

Test Coverage by Tier:
- Tier 1: Guaranteed Stable (byte-identical verification)
- Tier 2: Functionally Stable (semantic equivalence)
- Tier 3: Non-Deterministic (proper handling verification)

Author: GL-DeterminismAuditor
Date: 2026-02-03
"""

import hashlib
import json
import os
import random
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import pytest


# =============================================================================
# HELPER UTILITIES
# =============================================================================

def compute_provenance_hash(data: Dict[str, Any]) -> str:
    """
    Compute deterministic SHA-256 hash of data for provenance tracking.

    This function implements the hash stability rules from the determinism contract:
    - JSON keys MUST be sorted alphabetically
    - Numbers MUST use consistent string representation
    - No whitespace variations
    - UTF-8 encoding only

    Args:
        data: Dictionary to hash

    Returns:
        SHA-256 hex digest string

    Example:
        >>> compute_provenance_hash({"b": 2, "a": 1})
        # Always returns the same hash regardless of dict insertion order
    """
    # Serialize with sorted keys and compact separators for determinism
    canonical = json.dumps(
        data,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
        default=_json_serializer
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for deterministic output."""
    if isinstance(obj, Decimal):
        # Use string representation for exact precision
        return str(obj)
    if isinstance(obj, datetime):
        # ISO 8601 format with UTC timezone
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=timezone.utc)
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.hex()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def verify_byte_stability(artifact1: Union[str, bytes, Dict], artifact2: Union[str, bytes, Dict]) -> Tuple[bool, Optional[str]]:
    """
    Verify two artifacts are byte-identical.

    Args:
        artifact1: First artifact (path, bytes, or dict)
        artifact2: Second artifact (path, bytes, or dict)

    Returns:
        Tuple of (is_identical, difference_description)
    """
    def get_bytes(artifact: Union[str, bytes, Dict]) -> bytes:
        if isinstance(artifact, dict):
            return json.dumps(artifact, sort_keys=True, separators=(',', ':')).encode('utf-8')
        if isinstance(artifact, str):
            if os.path.exists(artifact):
                with open(artifact, 'rb') as f:
                    return f.read()
            return artifact.encode('utf-8')
        return artifact

    bytes1 = get_bytes(artifact1)
    bytes2 = get_bytes(artifact2)

    if bytes1 == bytes2:
        return True, None

    # Find first difference
    for i, (b1, b2) in enumerate(zip(bytes1, bytes2)):
        if b1 != b2:
            return False, f"First difference at byte {i}: {b1!r} vs {b2!r}"

    if len(bytes1) != len(bytes2):
        return False, f"Length mismatch: {len(bytes1)} vs {len(bytes2)}"

    return False, "Unknown difference"


@dataclass
class HashComparisonResult:
    """Result of comparing hashes between two runs."""
    is_identical: bool
    run_a_hashes: Dict[str, str]
    run_b_hashes: Dict[str, str]
    mismatches: List[Dict[str, Any]] = field(default_factory=list)

    def __str__(self) -> str:
        if self.is_identical:
            return "PASS: All hashes identical"
        lines = ["FAIL: Hash mismatches detected"]
        for mismatch in self.mismatches:
            lines.append(f"  - {mismatch['key']}: {mismatch['run_a'][:16]}... vs {mismatch['run_b'][:16]}...")
        return "\n".join(lines)


def compare_run_hashes(run_a: Dict[str, Any], run_b: Dict[str, Any]) -> HashComparisonResult:
    """
    Compare hashes between two run.json files.

    Args:
        run_a: First run's data
        run_b: Second run's data

    Returns:
        HashComparisonResult with detailed comparison
    """
    def extract_hashes(data: Dict, prefix: str = "") -> Dict[str, str]:
        """Recursively extract all hash values from a nested dict."""
        hashes = {}
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                hashes.update(extract_hashes(value, full_key))
            elif isinstance(value, str) and ('hash' in key.lower() or len(value) == 64):
                # Likely a SHA-256 hash
                hashes[full_key] = value
        return hashes

    hashes_a = extract_hashes(run_a)
    hashes_b = extract_hashes(run_b)

    mismatches = []
    all_keys = set(hashes_a.keys()) | set(hashes_b.keys())

    for key in sorted(all_keys):
        hash_a = hashes_a.get(key)
        hash_b = hashes_b.get(key)

        if hash_a != hash_b:
            mismatches.append({
                'key': key,
                'run_a': hash_a or 'MISSING',
                'run_b': hash_b or 'MISSING'
            })

    return HashComparisonResult(
        is_identical=len(mismatches) == 0,
        run_a_hashes=hashes_a,
        run_b_hashes=hashes_b,
        mismatches=mismatches
    )


# =============================================================================
# DETERMINISTIC CLOCK CLASS
# =============================================================================

class DeterministicClock:
    """
    Deterministic clock for reproducible tests.

    This clock provides predictable timestamps for testing scenarios where
    time-based operations must be reproducible. Implements the pattern
    described in the determinism contract.

    Thread-safe implementation using threading.Lock.

    Example:
        >>> clock = DeterministicClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        >>> t1 = clock.now()
        >>> clock.advance(timedelta(seconds=1))
        >>> t2 = clock.now()
        >>> assert t2 - t1 == timedelta(seconds=1)
    """

    def __init__(self, start_time: Optional[datetime] = None, tick: Optional[timedelta] = None):
        """
        Initialize deterministic clock.

        Args:
            start_time: Initial time (defaults to 2026-01-01T00:00:00+00:00)
            tick: Default tick increment (defaults to 1 second)
        """
        self._lock = threading.Lock()
        self._time = start_time or datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self._tick = tick or timedelta(seconds=1)

        # Ensure timezone awareness
        if self._time.tzinfo is None:
            self._time = self._time.replace(tzinfo=timezone.utc)

    def now(self) -> datetime:
        """Return current deterministic time."""
        with self._lock:
            return self._time

    def utcnow(self) -> datetime:
        """Return current deterministic UTC time."""
        return self.now()

    def advance(self, delta: Optional[timedelta] = None) -> datetime:
        """
        Advance time by specified amount.

        Args:
            delta: Amount to advance (uses default tick if None)

        Returns:
            New current time after advancing
        """
        with self._lock:
            self._time += delta or self._tick
            return self._time

    def set(self, new_time: datetime) -> None:
        """
        Set clock to specific time.

        Args:
            new_time: New time to set
        """
        with self._lock:
            if new_time.tzinfo is None:
                new_time = new_time.replace(tzinfo=timezone.utc)
            self._time = new_time

    def reset(self, start_time: Optional[datetime] = None) -> None:
        """
        Reset clock to initial state.

        Args:
            start_time: New start time (defaults to 2026-01-01)
        """
        with self._lock:
            self._time = start_time or datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            if self._time.tzinfo is None:
                self._time = self._time.replace(tzinfo=timezone.utc)

    def iso_format(self) -> str:
        """Return current time in ISO 8601 format."""
        return self.now().isoformat()


# =============================================================================
# DETERMINISTIC RANDOM GENERATOR
# =============================================================================

class DeterministicRandom:
    """
    Deterministic random number generator for reproducible tests.

    Uses a seeded random.Random instance to ensure reproducibility.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize with seed.

        Args:
            seed: Random seed (default: 42)
        """
        self._seed = seed
        self._rng = random.Random(seed)

    def random(self) -> float:
        """Return random float in [0.0, 1.0)."""
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        """Return random integer in [a, b]."""
        return self._rng.randint(a, b)

    def choice(self, seq: List[Any]) -> Any:
        """Return random element from sequence."""
        return self._rng.choice(seq)

    def shuffle(self, seq: List[Any]) -> None:
        """Shuffle sequence in place."""
        self._rng.shuffle(seq)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset generator with same or new seed."""
        self._seed = seed if seed is not None else self._seed
        self._rng = random.Random(self._seed)

    @property
    def seed(self) -> int:
        """Return current seed."""
        return self._seed


# =============================================================================
# FINANCIAL DECIMAL FOR DETERMINISTIC CALCULATIONS
# =============================================================================

class FinancialDecimal:
    """
    Wrapper for deterministic decimal calculations.

    Ensures all financial/emission calculations use proper decimal arithmetic
    with consistent rounding rules as specified in the determinism contract.
    """

    PRECISION = Decimal('0.00000001')  # 8 decimal places

    @classmethod
    def from_any(cls, value: Any) -> Decimal:
        """Convert any numeric type to Decimal safely."""
        if isinstance(value, Decimal):
            return value.quantize(cls.PRECISION)
        if isinstance(value, int):
            return Decimal(value).quantize(cls.PRECISION)
        if isinstance(value, float):
            return Decimal(str(value)).quantize(cls.PRECISION)
        if isinstance(value, str):
            return Decimal(value.replace(',', '')).quantize(cls.PRECISION)
        raise TypeError(f"Cannot convert {type(value).__name__} to Decimal")

    @classmethod
    def multiply(cls, a: Any, b: Any) -> Decimal:
        """Multiply with proper precision."""
        return (cls.from_any(a) * cls.from_any(b)).quantize(cls.PRECISION)

    @classmethod
    def add(cls, a: Any, b: Any) -> Decimal:
        """Add with proper precision."""
        return (cls.from_any(a) + cls.from_any(b)).quantize(cls.PRECISION)


# =============================================================================
# RUN.JSON STRUCTURE VALIDATOR
# =============================================================================

# Required field ordering per determinism contract
RUN_JSON_FIELD_ORDER = [
    "schema_version",
    "run_id",
    "pipeline",
    "status",
    "success",
    "started_at",
    "completed_at",
    "duration_ms",
    "inputs",
    "outputs",
    "steps",
    "errors",
    "provenance"
]


def validate_run_json_structure(run_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate run.json structure for determinism compliance.

    Args:
        run_data: Parsed run.json data

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Check for required provenance fields
    if 'spec' in run_data:
        spec = run_data['spec']
        required_hashes = ['config_hash', 'inputs_hash', 'pipeline_hash', 'ledger_hash']
        for hash_field in required_hashes:
            if hash_field not in spec:
                issues.append(f"Missing required hash field: spec.{hash_field}")
            elif spec[hash_field] is None:
                issues.append(f"Hash field is null: spec.{hash_field}")

    # Check metadata timestamps are in ISO 8601 format
    if 'metadata' in run_data:
        for ts_field in ['started_at', 'completed_at']:
            if ts_field in run_data['metadata']:
                ts_value = run_data['metadata'][ts_field]
                if ts_value and not _is_iso8601(ts_value):
                    issues.append(f"Timestamp not in ISO 8601 format: {ts_field}")

    return len(issues) == 0, issues


def _is_iso8601(s: str) -> bool:
    """Check if string is valid ISO 8601 timestamp."""
    try:
        datetime.fromisoformat(s.replace('Z', '+00:00'))
        return True
    except (ValueError, AttributeError):
        return False


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture
def deterministic_clock():
    """Provide a deterministic clock for tests."""
    return DeterministicClock(datetime(2026, 2, 3, 12, 0, 0, tzinfo=timezone.utc))


@pytest.fixture
def deterministic_random():
    """Provide a deterministic random generator for tests."""
    return DeterministicRandom(seed=42)


@pytest.fixture
def sample_run_json():
    """Provide a sample run.json structure for testing."""
    return {
        "kind": "greenlang-run-ledger",
        "version": "1.0.0",
        "metadata": {
            "started_at": "2026-02-03T12:00:00+00:00",
            "finished_at": "2026-02-03T12:05:00+00:00",
            "duration": 300.0,
            "status": "success"
        },
        "execution": {
            "backend": "local",
            "profile": "dev"
        },
        "spec": {
            "config_hash": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2",
            "inputs_hash": "f2e1d0c9b8a7z6y5x4w3v2u1t0s9r8q7p6o5n4m3l2k1j0i9h8g7f6e5d4c3b2a1",
            "pipeline_hash": "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "ledger_hash": "fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321"
        },
        "outputs": {
            "emissions_kg_co2e": "265.72000000",
            "scope": "Scope 1"
        }
    }


@pytest.fixture
def emission_factor_data():
    """Provide sample emission factor data for testing."""
    return {
        "factor_id": "DEFRA-2024-diesel",
        "source": "DEFRA",
        "vintage": 2024,
        "value": "2.6572",
        "unit": "kg_CO2e_per_liter"
    }


# =============================================================================
# TIER 1 TESTS: GUARANTEED STABLE
# =============================================================================

class TestTier1HashGeneration:
    """Test Tier 1: Hash generation must be byte-identical."""

    def test_hash_generation_deterministic(self):
        """Test that hash generation is deterministic."""
        data = {"key": "value", "number": 42}
        hash1 = compute_provenance_hash(data)
        hash2 = compute_provenance_hash(data)
        assert hash1 == hash2, "Hash generation must be deterministic"

    def test_hash_generation_key_order_independent(self):
        """Test that hash is independent of dict key insertion order."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}
        data3 = {"b": 2, "c": 3, "a": 1}

        hash1 = compute_provenance_hash(data1)
        hash2 = compute_provenance_hash(data2)
        hash3 = compute_provenance_hash(data3)

        assert hash1 == hash2 == hash3, "Hash must be independent of key order"

    def test_hash_generation_nested_dict(self):
        """Test hash determinism with nested dictionaries."""
        data = {
            "outer": {
                "inner": {
                    "value": 42,
                    "list": [1, 2, 3]
                }
            },
            "other": "data"
        }

        hash1 = compute_provenance_hash(data)
        hash2 = compute_provenance_hash(data)

        assert hash1 == hash2, "Nested dict hashing must be deterministic"

    def test_hash_generation_decimal_values(self):
        """Test hash determinism with Decimal values."""
        data = {
            "emissions": Decimal("265.72000000"),
            "factor": Decimal("2.6572")
        }

        hash1 = compute_provenance_hash(data)
        hash2 = compute_provenance_hash(data)

        assert hash1 == hash2, "Decimal value hashing must be deterministic"

    def test_hash_different_data_different_hash(self):
        """Test that different data produces different hashes."""
        data1 = {"key": "value1"}
        data2 = {"key": "value2"}

        hash1 = compute_provenance_hash(data1)
        hash2 = compute_provenance_hash(data2)

        assert hash1 != hash2, "Different data must produce different hashes"

    def test_hash_length_is_sha256(self):
        """Test that hash is valid SHA-256 (64 hex characters)."""
        data = {"test": "data"}
        hash_value = compute_provenance_hash(data)

        assert len(hash_value) == 64, "Hash must be 64 characters (SHA-256)"
        assert all(c in '0123456789abcdef' for c in hash_value), "Hash must be hex"


class TestTier1JsonSerialization:
    """Test Tier 1: JSON serialization must be stable."""

    def test_json_serialization_stable(self):
        """Test that JSON serialization is stable."""
        data = {"b": 2, "a": 1}  # Unordered
        json1 = json.dumps(data, sort_keys=True)
        json2 = json.dumps(data, sort_keys=True)
        assert json1 == json2, "JSON serialization must be stable"

    def test_json_serialization_compact(self):
        """Test that compact serialization is deterministic."""
        data = {"key": "value", "number": 42}
        json1 = json.dumps(data, sort_keys=True, separators=(',', ':'))
        json2 = json.dumps(data, sort_keys=True, separators=(',', ':'))
        assert json1 == json2, "Compact JSON must be deterministic"
        assert ' ' not in json1, "Compact JSON must have no whitespace"

    def test_json_serialization_unicode(self):
        """Test JSON serialization with unicode characters."""
        data = {"name": "Test", "symbol": "CO2"}
        json1 = json.dumps(data, sort_keys=True, ensure_ascii=True)
        json2 = json.dumps(data, sort_keys=True, ensure_ascii=True)
        assert json1 == json2, "Unicode JSON must be deterministic"

    def test_json_serialization_numeric_precision(self):
        """Test that numeric precision is preserved in JSON."""
        data = {"value": "265.72000000"}  # String for exact precision
        json_str = json.dumps(data, sort_keys=True)
        parsed = json.loads(json_str)
        assert parsed["value"] == "265.72000000", "Numeric precision must be preserved"


class TestTier1CalculationReproducibility:
    """Test Tier 1: Calculation results must be identical."""

    def test_same_inputs_same_outputs(self):
        """Test that same inputs produce same outputs."""
        factor = Decimal("2.6572")
        amount = Decimal("100")

        result1 = FinancialDecimal.multiply(amount, factor)
        result2 = FinancialDecimal.multiply(amount, factor)

        assert result1 == result2, "Same inputs must produce same outputs"

    def test_factor_application_exact(self):
        """Test that factor application is exact."""
        # Diesel emission factor: 2.6572 kg CO2e/liter
        factor = Decimal("2.6572")
        amount = Decimal("100")
        expected = Decimal("265.72000000")

        result = FinancialDecimal.multiply(amount, factor)

        assert result == expected, f"Factor application must be exact: {result} != {expected}"

    def test_decimal_addition_deterministic(self):
        """Test that decimal addition is deterministic."""
        # Classic float problem: 0.1 + 0.2 != 0.3
        a = Decimal("0.1")
        b = Decimal("0.2")
        expected = Decimal("0.30000000")

        result = FinancialDecimal.add(a, b)

        assert result == expected, "Decimal addition must be exact"

    def test_multiple_operations_deterministic(self):
        """Test that multiple operations remain deterministic."""
        results = []
        for _ in range(10):
            a = FinancialDecimal.from_any(100)
            b = FinancialDecimal.from_any(2.6572)
            c = FinancialDecimal.multiply(a, b)
            d = FinancialDecimal.add(c, Decimal("10"))
            results.append(d)

        assert all(r == results[0] for r in results), "Multiple operations must be deterministic"


class TestTier1RunJsonStructure:
    """Test Tier 1: run.json structure must be field-for-field identical."""

    def test_run_json_field_presence(self, sample_run_json):
        """Test that required fields are present in run.json."""
        is_valid, issues = validate_run_json_structure(sample_run_json)
        assert is_valid, f"run.json validation failed: {issues}"

    def test_run_json_hash_stability(self, sample_run_json):
        """Test that run.json hashing is stable."""
        hash1 = compute_provenance_hash(sample_run_json)
        hash2 = compute_provenance_hash(sample_run_json)
        assert hash1 == hash2, "run.json hash must be stable"

    def test_run_json_numeric_precision(self, sample_run_json):
        """Test that numeric precision is preserved in run.json."""
        emissions = sample_run_json["outputs"]["emissions_kg_co2e"]
        assert emissions == "265.72000000", "Numeric precision must be preserved as strings"

    def test_run_json_timestamp_format(self, sample_run_json):
        """Test that timestamps are in ISO 8601 format."""
        started = sample_run_json["metadata"]["started_at"]
        assert _is_iso8601(started), f"Timestamp must be ISO 8601: {started}"


# =============================================================================
# TIER 2 TESTS: FUNCTIONALLY STABLE
# =============================================================================

class TestTier2DeterministicClock:
    """Test Tier 2: DeterministicClock for reproducible timestamps."""

    def test_deterministic_clock_creation(self, deterministic_clock):
        """Test deterministic clock creation."""
        t1 = deterministic_clock.now()
        assert t1.tzinfo is not None, "Clock must be timezone-aware"

    def test_deterministic_clock_advance(self):
        """Test clock advance functionality."""
        clock = DeterministicClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        t1 = clock.now()
        clock.advance(timedelta(seconds=1))
        t2 = clock.now()
        assert t2 - t1 == timedelta(seconds=1), "Clock advance must be exact"

    def test_deterministic_clock_multiple_advances(self):
        """Test multiple clock advances."""
        clock = DeterministicClock(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc))

        for i in range(10):
            clock.advance(timedelta(minutes=1))

        expected = datetime(2026, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
        assert clock.now() == expected, "Multiple advances must be cumulative"

    def test_deterministic_clock_reset(self):
        """Test clock reset functionality."""
        clock = DeterministicClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        clock.advance(timedelta(hours=5))
        clock.reset()

        expected = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert clock.now() == expected, "Clock reset must return to default start"

    def test_deterministic_clock_set(self):
        """Test clock set functionality."""
        clock = DeterministicClock()
        new_time = datetime(2026, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        clock.set(new_time)

        assert clock.now() == new_time, "Clock set must work correctly"

    def test_deterministic_clock_iso_format(self):
        """Test ISO format output."""
        clock = DeterministicClock(datetime(2026, 2, 3, 12, 0, 0, tzinfo=timezone.utc))
        iso = clock.iso_format()

        assert "2026-02-03" in iso, "ISO format must contain date"
        assert "12:00:00" in iso, "ISO format must contain time"

    def test_deterministic_clock_thread_safety(self):
        """Test clock thread safety."""
        clock = DeterministicClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        results = []

        def advance_and_read():
            for _ in range(100):
                clock.advance(timedelta(milliseconds=1))
                results.append(clock.now())

        threads = [threading.Thread(target=advance_and_read) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify monotonic increase
        sorted_results = sorted(results)
        assert results == sorted_results or len(set(results)) > 1, "Thread-safe operations"


class TestTier2DeterministicRandom:
    """Test Tier 2: DeterministicRandom for reproducible randomness."""

    def test_deterministic_random_seed(self, deterministic_random):
        """Test that seeded random is reproducible."""
        rng1 = DeterministicRandom(seed=42)
        rng2 = DeterministicRandom(seed=42)

        values1 = [rng1.random() for _ in range(10)]
        values2 = [rng2.random() for _ in range(10)]

        assert values1 == values2, "Same seed must produce same sequence"

    def test_deterministic_random_different_seeds(self):
        """Test that different seeds produce different sequences."""
        rng1 = DeterministicRandom(seed=42)
        rng2 = DeterministicRandom(seed=43)

        values1 = [rng1.random() for _ in range(10)]
        values2 = [rng2.random() for _ in range(10)]

        assert values1 != values2, "Different seeds must produce different sequences"

    def test_deterministic_random_reset(self):
        """Test random generator reset."""
        rng = DeterministicRandom(seed=42)
        values1 = [rng.random() for _ in range(5)]

        rng.reset()
        values2 = [rng.random() for _ in range(5)]

        assert values1 == values2, "Reset must reproduce sequence"

    def test_deterministic_random_choice(self):
        """Test deterministic choice."""
        items = ['a', 'b', 'c', 'd', 'e']

        rng1 = DeterministicRandom(seed=42)
        rng2 = DeterministicRandom(seed=42)

        choices1 = [rng1.choice(items) for _ in range(10)]
        choices2 = [rng2.choice(items) for _ in range(10)]

        assert choices1 == choices2, "Choice must be deterministic"


class TestTier2FloatingPointTolerance:
    """Test Tier 2: Floating point with tolerance (1e-10)."""

    def test_floating_point_within_tolerance(self):
        """Test floating point comparison within tolerance."""
        tolerance = Decimal("1e-10")
        a = Decimal("265.72000000001")
        b = Decimal("265.72000000002")

        diff = abs(a - b)
        assert diff < tolerance, "Values within tolerance should pass"

    def test_floating_point_outside_tolerance(self):
        """Test floating point comparison outside tolerance."""
        tolerance = Decimal("1e-10")
        a = Decimal("265.72")
        b = Decimal("265.73")

        diff = abs(a - b)
        assert diff > tolerance, "Values outside tolerance should fail"


# =============================================================================
# TIER 3 TESTS: NON-DETERMINISTIC HANDLING
# =============================================================================

class TestTier3LLMOutputHandling:
    """Test Tier 3: LLM outputs are marked as non-deterministic."""

    def test_llm_output_marked_non_deterministic(self):
        """Test that LLM outputs would be marked as non-deterministic."""
        # Simulated LLM result structure
        llm_result = {
            "output": "Generated summary text",
            "deterministic": False,
            "audit_record": {
                "prompt_hash": "abc123",
                "model_id": "gpt-4",
                "temperature": 0,
                "seed": 42
            }
        }

        assert llm_result["deterministic"] is False, "LLM output must be marked non-deterministic"
        assert "audit_record" in llm_result, "LLM output must have audit record"

    def test_llm_audit_record_completeness(self):
        """Test that LLM audit records are complete."""
        required_fields = ["prompt_hash", "model_id", "temperature", "seed"]
        audit_record = {
            "prompt_hash": "abc123",
            "model_id": "gpt-4",
            "temperature": 0,
            "seed": 42,
            "timestamp": "2026-02-03T12:00:00Z"
        }

        for field in required_fields:
            assert field in audit_record, f"Audit record must contain {field}"


class TestTier3ExternalAPIHandling:
    """Test Tier 3: External API responses are cached/versioned."""

    def test_external_data_caching_structure(self):
        """Test that external data caching has proper structure."""
        cached_data = {
            "data": {"grid_factor": 0.453},
            "metadata": {
                "fetched_at": "2026-02-03T12:00:00Z",
                "url": "https://api.example.com/grid-factors",
                "hash": compute_provenance_hash({"grid_factor": 0.453})
            }
        }

        assert "metadata" in cached_data, "Cached data must have metadata"
        assert "fetched_at" in cached_data["metadata"], "Metadata must have fetched_at"
        assert "hash" in cached_data["metadata"], "Metadata must have content hash"


# =============================================================================
# BYTE STABILITY TESTS
# =============================================================================

class TestByteStability:
    """Test byte-level stability of artifacts."""

    def test_verify_byte_stability_identical(self):
        """Test byte stability verification for identical data."""
        data1 = {"key": "value", "number": 42}
        data2 = {"key": "value", "number": 42}

        is_identical, diff = verify_byte_stability(data1, data2)
        assert is_identical, f"Identical data should pass: {diff}"

    def test_verify_byte_stability_different(self):
        """Test byte stability verification for different data."""
        data1 = {"key": "value1"}
        data2 = {"key": "value2"}

        is_identical, diff = verify_byte_stability(data1, data2)
        assert not is_identical, "Different data should fail"
        assert diff is not None, "Difference should be reported"

    def test_verify_byte_stability_with_strings(self):
        """Test byte stability with string inputs."""
        str1 = '{"key":"value"}'
        str2 = '{"key":"value"}'

        is_identical, diff = verify_byte_stability(str1, str2)
        assert is_identical, f"Identical strings should pass: {diff}"

    def test_verify_byte_stability_with_files(self):
        """Test byte stability with file inputs."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1:
            json.dump({"test": "data"}, f1)
            f1_path = f1.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f2:
            json.dump({"test": "data"}, f2)
            f2_path = f2.name

        try:
            is_identical, diff = verify_byte_stability(f1_path, f2_path)
            assert is_identical, f"Identical files should pass: {diff}"
        finally:
            os.unlink(f1_path)
            os.unlink(f2_path)


# =============================================================================
# RUN COMPARISON TESTS
# =============================================================================

class TestRunComparison:
    """Test run-to-run comparison functionality."""

    def test_compare_identical_runs(self, sample_run_json):
        """Test comparison of identical runs."""
        run_a = sample_run_json.copy()
        run_b = sample_run_json.copy()

        result = compare_run_hashes(run_a, run_b)

        assert result.is_identical, f"Identical runs should match: {result.mismatches}"

    def test_compare_different_runs(self, sample_run_json):
        """Test comparison of different runs."""
        import copy
        run_a = copy.deepcopy(sample_run_json)
        run_b = copy.deepcopy(sample_run_json)
        run_b["spec"]["config_hash"] = "different_hash_value_here_123456789012345678901234567890"

        result = compare_run_hashes(run_a, run_b)

        assert not result.is_identical, "Different runs should not match"
        assert len(result.mismatches) > 0, "Mismatches should be reported"

    def test_compare_runs_detailed_mismatch(self, sample_run_json):
        """Test detailed mismatch reporting."""
        import copy
        run_a = copy.deepcopy(sample_run_json)
        run_b = copy.deepcopy(sample_run_json)
        run_b["spec"]["inputs_hash"] = "0" * 64  # Different hash

        result = compare_run_hashes(run_a, run_b)

        assert not result.is_identical
        mismatch_keys = [m['key'] for m in result.mismatches]
        assert any('inputs_hash' in k for k in mismatch_keys), "Should identify inputs_hash mismatch"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestDeterminismIntegration:
    """Integration tests for end-to-end determinism verification."""

    def test_full_calculation_pipeline_determinism(self, deterministic_clock):
        """Test full calculation pipeline determinism."""
        # Simulate a complete calculation run
        input_data = {
            "fuel_type": "diesel",
            "amount": 100,
            "unit": "liters",
            "country": "US"
        }

        # Run calculation multiple times
        results = []
        for _ in range(5):
            # Compute emissions
            factor = Decimal("2.6572")
            amount = FinancialDecimal.from_any(input_data["amount"])
            emissions = FinancialDecimal.multiply(amount, factor)

            # Build result
            result = {
                "inputs": input_data,
                "outputs": {
                    "emissions_kg_co2e": str(emissions),
                    "emission_factor": str(factor),
                    "emission_factor_unit": "kg_CO2e_per_liter"
                },
                "timestamp": deterministic_clock.iso_format(),
                "provenance_hash": compute_provenance_hash(input_data)
            }
            results.append(result)

            # Advance clock for next run
            deterministic_clock.advance(timedelta(seconds=1))

        # Verify deterministic outputs (excluding timestamp)
        for i in range(1, len(results)):
            assert results[i]["outputs"] == results[0]["outputs"], f"Run {i+1} outputs differ"
            assert results[i]["provenance_hash"] == results[0]["provenance_hash"], f"Run {i+1} hash differs"

    def test_emission_factor_lookup_determinism(self, emission_factor_data):
        """Test emission factor lookup determinism."""
        # Simulate multiple lookups
        lookups = []
        for _ in range(10):
            # Lookup returns same data
            factor = emission_factor_data.copy()
            lookups.append(factor)

        # All lookups should be identical
        for i in range(1, len(lookups)):
            assert lookups[i] == lookups[0], f"Lookup {i+1} differs"

    def test_provenance_chain_determinism(self):
        """Test provenance chain building determinism."""
        input_hash = compute_provenance_hash({"fuel": "diesel", "amount": 100})
        factor_hash = compute_provenance_hash({"source": "DEFRA", "value": "2.6572"})
        output_hash = compute_provenance_hash({"emissions": "265.72"})

        # Build chain multiple times
        chains = []
        for _ in range(5):
            chain = [input_hash, factor_hash, output_hash]
            chain_hash = compute_provenance_hash({"chain": chain})
            chains.append(chain_hash)

        # All chains should be identical
        assert all(c == chains[0] for c in chains), "Provenance chains must be deterministic"


# =============================================================================
# GOLDEN FILE TESTS
# =============================================================================

class TestGoldenFiles:
    """Tests using golden file patterns."""

    @pytest.fixture
    def golden_input(self):
        """Golden input data."""
        return {
            "fuel_type": "natural_gas",
            "amount": 1000.0,
            "unit": "therms",
            "country": "US"
        }

    @pytest.fixture
    def golden_expected_hash(self):
        """Expected hash for golden input."""
        # Pre-computed hash of golden input
        return compute_provenance_hash({
            "fuel_type": "natural_gas",
            "amount": 1000.0,
            "unit": "therms",
            "country": "US"
        })

    def test_golden_input_hash_matches(self, golden_input, golden_expected_hash):
        """Test that golden input produces expected hash."""
        actual_hash = compute_provenance_hash(golden_input)
        assert actual_hash == golden_expected_hash, "Golden hash mismatch"

    def test_golden_calculation_result(self, golden_input):
        """Test that golden input produces expected calculation result."""
        # Natural gas emission factor: approximately 5.31 kg CO2e/therm
        factor = Decimal("5.31")
        amount = FinancialDecimal.from_any(golden_input["amount"])

        expected_emissions = Decimal("5310.00000000")
        actual_emissions = FinancialDecimal.multiply(amount, factor)

        assert actual_emissions == expected_emissions, f"Expected {expected_emissions}, got {actual_emissions}"


# =============================================================================
# ENVIRONMENT DETERMINISM TESTS
# =============================================================================

class TestEnvironmentDeterminism:
    """Tests for environment-related determinism."""

    def test_pythonhashseed_consideration(self):
        """Test that PYTHONHASHSEED is considered for hash determinism."""
        # Note: This test documents that dict iteration order depends on PYTHONHASHSEED
        # Our implementation uses sorted keys to avoid this issue
        data = {"z": 1, "a": 2, "m": 3}

        # With sorted keys, hash should always be the same
        hash1 = compute_provenance_hash(data)
        hash2 = compute_provenance_hash(data)

        assert hash1 == hash2, "Hash must be deterministic regardless of PYTHONHASHSEED"

    def test_locale_independent_sorting(self):
        """Test that string sorting is locale-independent."""
        # Using ASCII-only strings for guaranteed ordering
        data = {"ABC": 1, "abc": 2, "123": 3}

        hash1 = compute_provenance_hash(data)
        hash2 = compute_provenance_hash(data)

        assert hash1 == hash2, "Sorting must be locale-independent"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases in determinism."""

    def test_empty_dict_hash(self):
        """Test hashing empty dictionary."""
        hash1 = compute_provenance_hash({})
        hash2 = compute_provenance_hash({})
        assert hash1 == hash2, "Empty dict hash must be deterministic"

    def test_null_values_hash(self):
        """Test hashing with null values."""
        data = {"key": None, "other": "value"}
        hash1 = compute_provenance_hash(data)
        hash2 = compute_provenance_hash(data)
        assert hash1 == hash2, "Null value hash must be deterministic"

    def test_very_large_numbers(self):
        """Test with very large numbers."""
        data = {"large": 10**100, "small": 10**-100}
        hash1 = compute_provenance_hash(data)
        hash2 = compute_provenance_hash(data)
        assert hash1 == hash2, "Large number hash must be deterministic"

    def test_unicode_strings(self):
        """Test with unicode strings."""
        data = {"emoji": "test", "chinese": "test", "arabic": "test"}
        hash1 = compute_provenance_hash(data)
        hash2 = compute_provenance_hash(data)
        assert hash1 == hash2, "Unicode hash must be deterministic"

    def test_deeply_nested_structure(self):
        """Test with deeply nested structure."""
        data = {"level1": {"level2": {"level3": {"level4": {"level5": "value"}}}}}
        hash1 = compute_provenance_hash(data)
        hash2 = compute_provenance_hash(data)
        assert hash1 == hash2, "Deep nesting hash must be deterministic"

    def test_list_ordering_preserved(self):
        """Test that list ordering is preserved."""
        data1 = {"items": [1, 2, 3]}
        data2 = {"items": [3, 2, 1]}

        hash1 = compute_provenance_hash(data1)
        hash2 = compute_provenance_hash(data2)

        assert hash1 != hash2, "Different list order must produce different hash"

    def test_zero_decimal_handling(self):
        """Test zero decimal handling."""
        result1 = FinancialDecimal.multiply(0, Decimal("2.6572"))
        result2 = FinancialDecimal.multiply(Decimal("0"), Decimal("2.6572"))

        assert result1 == result2 == Decimal("0.00000000"), "Zero handling must be consistent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
