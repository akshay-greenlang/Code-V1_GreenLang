"""
GreenLang Determinism Module - Zero-Hallucination Utilities

This module provides deterministic operations for GreenLang agents to ensure
reproducible, auditable calculations in regulatory compliance systems.

Key Features:
- Deterministic timestamp generation for testing and reproducibility
- Deterministic UUID generation based on input data (SHA-256)
- SHA-256 provenance hashing for audit trails
- Validation of calculation reproducibility
- Thread-safe operations

Example:
    >>> from greenlang.determinism import DeterministicClock, deterministic_uuid
    >>> clock = DeterministicClock(test_mode=True)
    >>> clock.set_time("2024-01-01T00:00:00Z")
    >>> timestamp = clock.now()
    >>> uuid = deterministic_uuid("steam_trap_inspection_123")
"""

import hashlib
import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from threading import Lock
from collections import OrderedDict

logger = logging.getLogger(__name__)


class DeterministicClock:
    """
    Deterministic clock for reproducible timestamp generation.

    This class provides deterministic timestamps for testing and reproducibility.
    In test mode, it returns a fixed timestamp that can be set manually.
    In production mode, it returns the current UTC time.

    Thread-safe implementation ensures consistent timestamps in multi-threaded
    environments.

    Attributes:
        test_mode: Whether the clock is in test mode
        fixed_time: The fixed timestamp for test mode

    Example:
        >>> clock = DeterministicClock(test_mode=True)
        >>> clock.set_time("2024-01-01T00:00:00Z")
        >>> timestamp = clock.now()
        >>> assert timestamp.isoformat() == "2024-01-01T00:00:00+00:00"
    """

    def __init__(self, test_mode: bool = False):
        """
        Initialize DeterministicClock.

        Args:
            test_mode: If True, use fixed timestamps. If False, use system time.
        """
        self.test_mode = test_mode
        self.fixed_time: Optional[datetime] = None
        self._lock = Lock()  # Thread-safe operations
        logger.info(f"DeterministicClock initialized (test_mode={test_mode})")

    def now(self) -> datetime:
        """
        Get current timestamp (deterministic in test mode).

        Returns:
            Current timestamp (UTC timezone-aware)

        Raises:
            ValueError: If test_mode is True but no time has been set
        """
        with self._lock:
            if self.test_mode:
                if self.fixed_time is None:
                    raise ValueError(
                        "Test mode enabled but no time set. Call set_time() first."
                    )
                return self.fixed_time
            else:
                return datetime.now(timezone.utc)

    def set_time(self, timestamp: Union[str, datetime]) -> None:
        """
        Set fixed timestamp for test mode.

        Args:
            timestamp: ISO format string or datetime object

        Raises:
            ValueError: If not in test mode or invalid timestamp format
        """
        with self._lock:
            if not self.test_mode:
                raise ValueError("Cannot set time when not in test_mode")

            if isinstance(timestamp, str):
                # Parse ISO format string
                try:
                    parsed_time = datetime.fromisoformat(
                        timestamp.replace("Z", "+00:00")
                    )
                    # Ensure UTC timezone
                    if parsed_time.tzinfo is None:
                        parsed_time = parsed_time.replace(tzinfo=timezone.utc)
                    self.fixed_time = parsed_time
                except ValueError as e:
                    raise ValueError(f"Invalid timestamp format: {timestamp}") from e
            elif isinstance(timestamp, datetime):
                # Ensure UTC timezone
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                self.fixed_time = timestamp
            else:
                raise ValueError(
                    f"Timestamp must be str or datetime, got {type(timestamp)}"
                )

            logger.debug(f"Clock time set to {self.fixed_time.isoformat()}")

    def advance(self, seconds: float = 0, minutes: float = 0, hours: float = 0) -> None:
        """
        Advance the fixed time by a specified duration (test mode only).

        Args:
            seconds: Seconds to advance
            minutes: Minutes to advance
            hours: Hours to advance

        Raises:
            ValueError: If not in test mode or no time has been set
        """
        with self._lock:
            if not self.test_mode:
                raise ValueError("Cannot advance time when not in test_mode")

            if self.fixed_time is None:
                raise ValueError("No time set. Call set_time() first.")

            from datetime import timedelta
            delta = timedelta(seconds=seconds, minutes=minutes, hours=hours)
            self.fixed_time = self.fixed_time + delta

            logger.debug(f"Clock advanced to {self.fixed_time.isoformat()}")

    def reset(self) -> None:
        """Reset clock to production mode and clear fixed time."""
        with self._lock:
            self.test_mode = False
            self.fixed_time = None
            logger.debug("Clock reset to production mode")


def deterministic_uuid(input_data: Union[str, Dict, List, Any]) -> str:
    """
    Generate deterministic UUID based on input data using SHA-256.

    This function creates reproducible UUIDs using SHA-256 hashing of input data.
    Same input always produces same UUID, enabling testing and audit trails.

    Args:
        input_data: Input data (string, dict, list, or any JSON-serializable object)

    Returns:
        Deterministic UUID string (RFC 4122 version 5 format)

    Example:
        >>> uuid1 = deterministic_uuid("steam_trap_inspection_123")
        >>> uuid2 = deterministic_uuid("steam_trap_inspection_123")
        >>> assert uuid1 == uuid2  # Same input = same UUID
        >>> uuid3 = deterministic_uuid({"trap_id": "ST-001", "type": "acoustic"})
        >>> assert uuid3 != uuid1  # Different input = different UUID
    """
    # Convert input to canonical string representation
    if isinstance(input_data, str):
        canonical_str = input_data
    elif isinstance(input_data, (dict, list)):
        # Sort dict keys for determinism
        canonical_str = json.dumps(
            input_data, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
    else:
        # Try to convert to JSON
        try:
            canonical_str = json.dumps(
                input_data, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        except (TypeError, ValueError):
            # Fall back to string representation
            canonical_str = str(input_data)

    # Generate SHA-256 hash
    hash_bytes = hashlib.sha256(canonical_str.encode("utf-8")).digest()

    # Create UUID5 (RFC 4122) from hash
    # Use first 16 bytes of SHA-256 hash
    uuid_bytes = hash_bytes[:16]

    # Set version (5) and variant bits per RFC 4122
    uuid_bytes = bytearray(uuid_bytes)
    uuid_bytes[6] = (uuid_bytes[6] & 0x0F) | 0x50  # Version 5
    uuid_bytes[8] = (uuid_bytes[8] & 0x3F) | 0x80  # Variant 10

    # Convert to UUID string
    uuid_obj = uuid.UUID(bytes=bytes(uuid_bytes))

    logger.debug(f"Generated deterministic UUID: {uuid_obj} from input length {len(canonical_str)}")
    return str(uuid_obj)


def calculate_provenance_hash(data: Union[Dict, List, str, Any]) -> str:
    """
    Calculate SHA-256 provenance hash for audit trail.

    This function creates a cryptographic hash of data for provenance tracking.
    Used to verify data integrity and create immutable audit trails.

    The hash is deterministic - same input always produces same output.

    Args:
        data: Data to hash (dict, list, string, or JSON-serializable object)

    Returns:
        SHA-256 hash as hexadecimal string (64 characters)

    Example:
        >>> data = {"trap_id": "ST-001", "status": "failed_open", "energy_loss": 15000}
        >>> hash1 = calculate_provenance_hash(data)
        >>> hash2 = calculate_provenance_hash(data)
        >>> assert hash1 == hash2  # Deterministic
        >>> assert len(hash1) == 64  # SHA-256 = 256 bits = 64 hex chars
    """
    # Convert data to canonical JSON string
    if isinstance(data, str):
        canonical_str = data
    elif isinstance(data, (dict, list)):
        # Sort dict keys for determinism
        canonical_str = json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
    else:
        # Try to convert to JSON
        try:
            canonical_str = json.dumps(
                data, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        except (TypeError, ValueError):
            # Fall back to string representation
            canonical_str = str(data)

    # Calculate SHA-256 hash
    hash_obj = hashlib.sha256(canonical_str.encode("utf-8"))
    hash_hex = hash_obj.hexdigest()

    logger.debug(f"Calculated provenance hash: {hash_hex[:16]}... (length: {len(canonical_str)} bytes)")
    return hash_hex


class DeterminismValidator:
    """
    Validator for calculation reproducibility and determinism.

    This class validates that calculations are reproducible by comparing
    provenance hashes between runs and detecting non-deterministic operations.

    Thread-safe implementation ensures consistent validation in multi-threaded
    environments.

    Attributes:
        reference_hashes: Dictionary of operation names to expected hashes
        validation_failures: List of validation failures

    Example:
        >>> validator = DeterminismValidator()
        >>> data = {"trap_id": "ST-001", "energy_loss": 15000}
        >>> hash1 = calculate_provenance_hash(data)
        >>> validator.register_hash("inspection_123", hash1)
        >>> # Later, validate same calculation
        >>> is_valid = validator.validate_hash("inspection_123", hash1)
        >>> assert is_valid is True
    """

    def __init__(self):
        """Initialize DeterminismValidator."""
        self.reference_hashes: Dict[str, str] = OrderedDict()
        self.validation_failures: List[Dict[str, str]] = []
        self._lock = Lock()  # Thread-safe operations
        logger.info("DeterminismValidator initialized")

    def register_hash(self, operation_name: str, hash_value: str) -> None:
        """
        Register a reference hash for an operation.

        Args:
            operation_name: Name of the operation (e.g., "acoustic_analysis_ST001")
            hash_value: Expected hash value (SHA-256 hex string)
        """
        with self._lock:
            self.reference_hashes[operation_name] = hash_value
            logger.debug(f"Registered hash for '{operation_name}': {hash_value[:16]}...")

    def validate_hash(
        self, operation_name: str, hash_value: str, auto_register: bool = False
    ) -> bool:
        """
        Validate hash against registered reference.

        Args:
            operation_name: Name of the operation
            hash_value: Hash to validate
            auto_register: If True, register hash if not found

        Returns:
            True if hash matches reference, False otherwise
        """
        with self._lock:
            if operation_name not in self.reference_hashes:
                if auto_register:
                    logger.warning(
                        f"No reference hash for '{operation_name}', auto-registering"
                    )
                    self.reference_hashes[operation_name] = hash_value
                    return True
                else:
                    logger.error(f"No reference hash found for '{operation_name}'")
                    self.validation_failures.append(
                        {
                            "operation": operation_name,
                            "reason": "no_reference_hash",
                            "hash": hash_value[:16] + "...",
                        }
                    )
                    return False

            reference_hash = self.reference_hashes[operation_name]
            is_valid = hash_value == reference_hash

            if is_valid:
                logger.debug(f"Hash validation PASSED for '{operation_name}'")
            else:
                logger.error(
                    f"Hash validation FAILED for '{operation_name}': "
                    f"expected {reference_hash[:16]}..., got {hash_value[:16]}..."
                )
                self.validation_failures.append(
                    {
                        "operation": operation_name,
                        "reason": "hash_mismatch",
                        "expected": reference_hash[:16] + "...",
                        "actual": hash_value[:16] + "...",
                    }
                )

            return is_valid

    def get_failures(self) -> List[Dict[str, str]]:
        """
        Get list of validation failures.

        Returns:
            List of failure dictionaries with operation name, reason, and details
        """
        with self._lock:
            return list(self.validation_failures)

    def reset(self) -> None:
        """Reset validator state (clear hashes and failures)."""
        with self._lock:
            self.reference_hashes.clear()
            self.validation_failures.clear()
            logger.debug("Validator state reset")

    def summary(self) -> Dict[str, Any]:
        """
        Get validation summary.

        Returns:
            Dictionary with total operations, failures, and success rate
        """
        with self._lock:
            total_operations = len(self.reference_hashes)
            total_failures = len(self.validation_failures)
            success_rate = (
                (total_operations - total_failures) / total_operations * 100
                if total_operations > 0
                else 100.0
            )

            return {
                "total_operations": total_operations,
                "total_failures": total_failures,
                "success_rate": success_rate,
                "all_passed": total_failures == 0,
            }


# Convenience functions for common use cases

def create_inspection_uuid(trap_id: str, timestamp: datetime) -> str:
    """
    Create deterministic UUID for steam trap inspection.

    Args:
        trap_id: Steam trap identifier
        timestamp: Inspection timestamp

    Returns:
        Deterministic UUID string
    """
    data = {
        "trap_id": trap_id,
        "timestamp": timestamp.isoformat(),
        "type": "inspection"
    }
    return deterministic_uuid(data)


def create_audit_hash(trap_id: str, inspection_data: Dict[str, Any]) -> str:
    """
    Create provenance hash for audit trail.

    Args:
        trap_id: Steam trap identifier
        inspection_data: Inspection results and metadata

    Returns:
        SHA-256 hash for audit trail
    """
    audit_data = {
        "trap_id": trap_id,
        "inspection": inspection_data,
    }
    return calculate_provenance_hash(audit_data)
