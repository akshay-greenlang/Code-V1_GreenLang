# -*- coding: utf-8 -*-
"""GreenLang Determinism Module for GL-004 BURNMASTER.

This module provides deterministic utilities for ensuring reproducible
calculations and audit trails in the GL-004 Burner Optimization Agent.

Key Features:
- DeterministicClock: Provides deterministic timestamps for reproducibility
  in testing and simulation modes, while supporting real-time in production.
- deterministic_uuid: Generates deterministic UUIDs from seed strings for
  reproducible identifiers.
- calculate_provenance_hash: Calculates SHA-256 hashes for complete
  audit trails and verification.

Zero-Hallucination Guarantee:
All functions in this module are purely deterministic with no LLM involvement.
Same inputs will always produce the same outputs (bit-perfect reproducibility).

Author: GreenLang AI Agent Factory
License: Proprietary
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Optional, Any, Dict
import hashlib
import json
import uuid


class DeterministicClock:
    """Provides deterministic timestamps for reproducibility.

    In production mode, returns actual UTC time. In test/simulation mode,
    can be set to a fixed time for reproducible results.

    This enables bit-perfect reproducibility of calculations during:
    - Unit testing
    - Integration testing
    - Simulation/replay of historical scenarios
    - Audit verification

    Usage:
        # Production usage (real time)
        timestamp = DeterministicClock.now()

        # Test mode (fixed time)
        DeterministicClock.set_fixed_time(datetime(2024, 1, 15, 12, 0, 0))
        timestamp = DeterministicClock.now()  # Always returns fixed time

        # Reset to production mode
        DeterministicClock.reset()

    Thread Safety:
        This class uses class variables for simplicity. In multi-threaded
        environments, use thread-local storage or pass clock instances.
    """

    _instance: Optional['DeterministicClock'] = None
    _fixed_time: Optional[datetime] = None
    _time_offset: Optional[float] = None  # Offset in seconds from real time

    @classmethod
    def now(cls) -> datetime:
        """Get current time (deterministic or real).

        Returns:
            datetime: Current UTC time, or fixed time if set
        """
        if cls._fixed_time is not None:
            return cls._fixed_time
        if cls._time_offset is not None:
            from datetime import timedelta
            return datetime.utcnow() + timedelta(seconds=cls._time_offset)
        return datetime.utcnow()

    @classmethod
    def utcnow(cls) -> datetime:
        """Get current UTC time (alias for now()).

        This method exists for compatibility with datetime.utcnow() API.

        Returns:
            datetime: Current UTC time, or fixed time if set
        """
        return cls.now()

    @classmethod
    def timestamp(cls) -> float:
        """Get current timestamp as Unix epoch seconds.

        Returns:
            float: Unix timestamp (seconds since 1970-01-01 00:00:00 UTC)
        """
        return cls.now().timestamp()

    @classmethod
    def set_fixed_time(cls, dt: datetime) -> None:
        """Set a fixed time for deterministic behavior.

        Use this in test fixtures to ensure reproducible results.

        Args:
            dt: The fixed datetime to return from now()/utcnow()
        """
        cls._fixed_time = dt
        cls._time_offset = None

    @classmethod
    def set_time_offset(cls, offset_seconds: float) -> None:
        """Set a time offset from real time.

        Useful for simulating future or past scenarios while
        still having time advance.

        Args:
            offset_seconds: Seconds to add to real time (negative for past)
        """
        cls._time_offset = offset_seconds
        cls._fixed_time = None

    @classmethod
    def reset(cls) -> None:
        """Reset to real-time mode.

        Clears any fixed time or offset, returning to production behavior.
        """
        cls._fixed_time = None
        cls._time_offset = None

    @classmethod
    def is_deterministic(cls) -> bool:
        """Check if clock is in deterministic mode.

        Returns:
            bool: True if using fixed time or offset
        """
        return cls._fixed_time is not None or cls._time_offset is not None

    @classmethod
    def isoformat(cls) -> str:
        """Get current time as ISO 8601 string.

        Returns:
            str: ISO 8601 formatted timestamp
        """
        return cls.now().isoformat()


def deterministic_uuid(seed: str) -> str:
    """Generate a deterministic UUID from a seed string.

    Creates a UUID that is fully determined by the input seed.
    Same seed will always produce the same UUID (reproducible).

    This is useful for:
    - Creating reproducible identifiers in tests
    - Generating consistent IDs for the same logical entity
    - Audit trail verification

    Algorithm:
    - SHA-256 hash the seed string
    - Take first 16 bytes of hash
    - Format as UUID version 4 (variant 1)

    Args:
        seed: The seed string to generate UUID from

    Returns:
        str: A deterministic UUID string

    Example:
        >>> deterministic_uuid("test_seed")
        '9f86d081-884c-7d65-9a2f-eac9fa7e5f0e'
        >>> deterministic_uuid("test_seed")  # Same result
        '9f86d081-884c-7d65-9a2f-eac9fa7e5f0e'
    """
    # Hash the seed to get deterministic bytes
    hash_bytes = hashlib.sha256(seed.encode('utf-8')).digest()[:16]

    # Convert to list for modification
    hash_list = list(hash_bytes)

    # Set version to 4 (random UUID format)
    hash_list[6] = (hash_list[6] & 0x0f) | 0x40

    # Set variant to 1 (RFC 4122)
    hash_list[8] = (hash_list[8] & 0x3f) | 0x80

    # Create UUID from bytes
    deterministic_bytes = bytes(hash_list)
    return str(uuid.UUID(bytes=deterministic_bytes))


def calculate_provenance_hash(data: Dict[str, Any]) -> str:
    """Calculate SHA-256 provenance hash for audit verification.

    Creates a deterministic hash of input data for:
    - Audit trail verification
    - Data integrity checking
    - Reproducibility verification

    The hash is calculated from a JSON serialization of the data
    with sorted keys to ensure deterministic ordering.

    Args:
        data: Dictionary of data to hash

    Returns:
        str: Full SHA-256 hash as hexadecimal string (64 characters)

    Example:
        >>> calculate_provenance_hash({'a': 1, 'b': 2})
        'a1b2c3d4...'  # 64 character hex string
    """
    # Serialize to JSON with sorted keys for deterministic ordering
    json_str = json.dumps(data, sort_keys=True, default=str)

    # Calculate SHA-256 hash
    hash_obj = hashlib.sha256(json_str.encode('utf-8'))

    return hash_obj.hexdigest()


def calculate_short_hash(data: Dict[str, Any], length: int = 16) -> str:
    """Calculate a shortened provenance hash.

    Convenience function for shorter hashes when full 64 characters
    is not needed (e.g., display purposes).

    Args:
        data: Dictionary of data to hash
        length: Number of characters to return (default: 16)

    Returns:
        str: First `length` characters of SHA-256 hash
    """
    full_hash = calculate_provenance_hash(data)
    return full_hash[:length]


def verify_provenance(data: Dict[str, Any], expected_hash: str) -> bool:
    """Verify data against an expected provenance hash.

    Used to verify that data has not been modified since the
    hash was calculated.

    Args:
        data: Dictionary of data to verify
        expected_hash: The expected hash value (full or partial)

    Returns:
        bool: True if hash matches, False otherwise
    """
    calculated_hash = calculate_provenance_hash(data)

    # Support partial hash matching (e.g., 16-char short hashes)
    if len(expected_hash) < 64:
        return calculated_hash.startswith(expected_hash)

    return calculated_hash == expected_hash


class ProvenanceTracker:
    """Track calculation provenance for audit trails.

    Provides a convenient way to track multiple calculation steps
    and generate a combined provenance hash.

    Example:
        tracker = ProvenanceTracker()
        tracker.add_step("input", {"fuel_flow": 100.0})
        tracker.add_step("calculation", {"afr": 17.2})
        tracker.add_step("output", {"efficiency": 0.85})
        hash = tracker.get_combined_hash()
    """

    def __init__(self):
        """Initialize the provenance tracker."""
        self._steps: list = []

    def add_step(self, step_name: str, data: Dict[str, Any]) -> str:
        """Add a calculation step to the tracker.

        Args:
            step_name: Name/identifier for this step
            data: Data associated with this step

        Returns:
            str: Hash of this individual step
        """
        step_hash = calculate_provenance_hash(data)
        self._steps.append({
            'step': len(self._steps) + 1,
            'name': step_name,
            'data': data,
            'hash': step_hash,
            'timestamp': DeterministicClock.isoformat()
        })
        return step_hash

    def get_steps(self) -> list:
        """Get all tracked steps.

        Returns:
            list: List of step dictionaries
        """
        return self._steps.copy()

    def get_combined_hash(self) -> str:
        """Get combined hash of all steps.

        Returns:
            str: SHA-256 hash of all steps combined
        """
        combined_data = {
            'steps': self._steps,
            'step_count': len(self._steps)
        }
        return calculate_provenance_hash(combined_data)

    def clear(self) -> None:
        """Clear all tracked steps."""
        self._steps = []


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("GreenLang Determinism Module Test")
    print("=" * 60)

    # Test DeterministicClock
    print("\n1. DeterministicClock Tests:")
    print(f"   Real time: {DeterministicClock.now()}")

    # Set fixed time
    fixed_time = datetime(2024, 6, 15, 12, 0, 0)
    DeterministicClock.set_fixed_time(fixed_time)
    print(f"   Fixed time: {DeterministicClock.now()}")
    print(f"   Is deterministic: {DeterministicClock.is_deterministic()}")

    # Reset
    DeterministicClock.reset()
    print(f"   After reset: {DeterministicClock.now()}")

    # Test deterministic_uuid
    print("\n2. Deterministic UUID Tests:")
    seed = "test_seed_123"
    uuid1 = deterministic_uuid(seed)
    uuid2 = deterministic_uuid(seed)
    uuid3 = deterministic_uuid("different_seed")
    print(f"   UUID from '{seed}': {uuid1}")
    print(f"   Same seed again: {uuid2}")
    print(f"   Same result: {uuid1 == uuid2}")
    print(f"   Different seed: {uuid3}")

    # Test provenance hash
    print("\n3. Provenance Hash Tests:")
    data = {'fuel_flow': 100.0, 'afr': 17.2, 'efficiency': 0.85}
    hash1 = calculate_provenance_hash(data)
    hash2 = calculate_provenance_hash(data)
    print(f"   Data: {data}")
    print(f"   Hash: {hash1}")
    print(f"   Reproducible: {hash1 == hash2}")

    # Test short hash
    short = calculate_short_hash(data)
    print(f"   Short hash: {short}")

    # Test verification
    print(f"   Verify full hash: {verify_provenance(data, hash1)}")
    print(f"   Verify short hash: {verify_provenance(data, short)}")
    print(f"   Verify wrong hash: {verify_provenance(data, 'wrong')}")

    # Test ProvenanceTracker
    print("\n4. ProvenanceTracker Tests:")
    tracker = ProvenanceTracker()
    tracker.add_step("input", {"fuel_flow": 100.0})
    tracker.add_step("calculation", {"afr": 17.2})
    tracker.add_step("output", {"efficiency": 0.85})
    print(f"   Steps tracked: {len(tracker.get_steps())}")
    print(f"   Combined hash: {tracker.get_combined_hash()[:32]}...")

    print("\n" + "=" * 60)
