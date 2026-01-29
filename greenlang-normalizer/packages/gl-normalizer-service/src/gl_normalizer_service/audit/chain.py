"""
Hash Chaining for Tamper-Evident Audit Trails in GL-FOUND-X-003.

This module provides hash chaining utilities for creating tamper-evident
audit trails. Each event's hash is computed from its content plus the
previous event's hash, creating an unbreakable chain.

Key Features:
    - SHA-256 hash computation with deterministic serialization
    - Chain hash linking (event_hash includes prev_event_hash)
    - Chain verification for detecting tampering
    - Batch verification for performance
    - ChainIntegrityError for integrity violations

Security Properties:
    - Any modification to a historical event breaks the chain
    - Deletion of events creates gaps detectable by verification
    - Reordering events invalidates subsequent hashes
    - Chain provides non-repudiation for audit compliance

Example:
    >>> from gl_normalizer_service.audit.chain import compute_chain_hash, verify_chain
    >>> event = {"event_id": "evt-001", "status": "success", ...}
    >>> prev_hash = None  # First event in chain
    >>> event_hash = compute_chain_hash(event, prev_hash)
    >>> events = [event1, event2, event3]
    >>> is_valid = verify_chain(events)

NFR Compliance:
    - NFR-035: Tamper-evident audit hashing with hash chaining
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ChainIntegrityError(Exception):
    """
    Exception raised when chain integrity verification fails.

    This exception indicates that the audit trail has been tampered with
    or corrupted. The attributes provide details about where and how
    the integrity violation was detected.

    Attributes:
        event_id: ID of the event where integrity failed.
        event_index: Index of the event in the chain (0-based).
        expected_hash: Expected hash value (computed or linked).
        actual_hash: Actual hash value found in the event.
        violation_type: Type of violation (hash_mismatch, link_broken, gap_detected).
        message: Human-readable error message.

    Example:
        >>> try:
        ...     verify_chain(events)
        ... except ChainIntegrityError as e:
        ...     print(f"Tampering detected at event {e.event_id}")
        ...     print(f"Expected: {e.expected_hash}")
        ...     print(f"Actual: {e.actual_hash}")
    """

    def __init__(
        self,
        event_id: str,
        event_index: int,
        expected_hash: Optional[str],
        actual_hash: Optional[str],
        violation_type: str,
        message: str,
    ):
        self.event_id = event_id
        self.event_index = event_index
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        self.violation_type = violation_type
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for logging/serialization.

        Returns:
            Dictionary representation of the error.
        """
        return {
            "event_id": self.event_id,
            "event_index": self.event_index,
            "expected_hash": self.expected_hash,
            "actual_hash": self.actual_hash,
            "violation_type": self.violation_type,
            "message": str(self),
        }


# Hash algorithm and prefix
HASH_ALGORITHM = "sha256"
HASH_PREFIX = "sha256:"


def compute_chain_hash(
    event: Dict[str, Any],
    previous_hash: Optional[str],
) -> str:
    """
    Compute the chain hash for an audit event.

    The chain hash is computed by:
    1. Extracting all event fields except event_hash
    2. Including the previous_hash in the hashable data
    3. Serializing to JSON with deterministic ordering
    4. Computing SHA-256 hash of the JSON bytes

    Args:
        event: Complete audit event dictionary.
        previous_hash: Hash of the previous event in the chain (None for first event).

    Returns:
        SHA-256 hash with "sha256:" prefix.

    Example:
        >>> event = {
        ...     "event_id": "norm-evt-001",
        ...     "event_ts": "2026-01-30T12:00:00Z",
        ...     "org_id": "org-acme",
        ...     "status": "success",
        ...     "measurements": [...],
        ...     "entities": [...],
        ... }
        >>> hash_value = compute_chain_hash(event, previous_hash=None)
        >>> assert hash_value.startswith("sha256:")
    """
    # Create hashable data excluding event_hash (circular reference)
    hashable_data = {
        k: v for k, v in event.items()
        if k != "event_hash"
    }

    # Include previous hash in the chain
    hashable_data["prev_event_hash"] = previous_hash

    # Serialize with deterministic ordering
    json_str = json.dumps(
        hashable_data,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_serializer,
    )

    # Compute SHA-256 hash
    hash_bytes = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
    chain_hash = f"{HASH_PREFIX}{hash_bytes}"

    logger.debug(
        "Computed chain hash for event %s: %s (prev=%s)",
        event.get("event_id", "unknown"),
        chain_hash[:30] + "...",
        previous_hash[:30] + "..." if previous_hash else None,
    )

    return chain_hash


def verify_chain(events: List[Dict[str, Any]]) -> bool:
    """
    Verify the integrity of a chain of audit events.

    Checks that:
    1. Each event's event_hash matches the computed hash
    2. Each event's prev_event_hash matches the previous event's event_hash
    3. The chain is continuous without gaps

    Args:
        events: List of audit events in chronological order.

    Returns:
        True if the chain is valid.

    Raises:
        ChainIntegrityError: If any integrity violation is detected.

    Example:
        >>> events = [event1, event2, event3]
        >>> try:
        ...     is_valid = verify_chain(events)
        ...     print("Chain integrity verified")
        ... except ChainIntegrityError as e:
        ...     print(f"Integrity violation: {e}")
    """
    if not events:
        logger.debug("Empty event list, chain is trivially valid")
        return True

    logger.info("Verifying chain integrity for %d events", len(events))

    previous_hash: Optional[str] = None

    for index, event in enumerate(events):
        event_id = event.get("event_id", f"unknown-{index}")
        stored_prev_hash = event.get("prev_event_hash")
        stored_event_hash = event.get("event_hash")

        # Check 1: prev_event_hash links correctly to previous event
        if stored_prev_hash != previous_hash:
            error = ChainIntegrityError(
                event_id=event_id,
                event_index=index,
                expected_hash=previous_hash,
                actual_hash=stored_prev_hash,
                violation_type="link_broken",
                message=(
                    f"Chain link broken at event {event_id} (index {index}): "
                    f"prev_event_hash mismatch. "
                    f"Expected '{previous_hash}', got '{stored_prev_hash}'"
                ),
            )
            logger.error(
                "Chain integrity violation: link broken at event %s",
                event_id,
            )
            raise error

        # Check 2: Recompute and verify event_hash
        computed_hash = compute_chain_hash(event, previous_hash)

        if computed_hash != stored_event_hash:
            error = ChainIntegrityError(
                event_id=event_id,
                event_index=index,
                expected_hash=computed_hash,
                actual_hash=stored_event_hash,
                violation_type="hash_mismatch",
                message=(
                    f"Hash mismatch at event {event_id} (index {index}): "
                    f"computed '{computed_hash}', stored '{stored_event_hash}'"
                ),
            )
            logger.error(
                "Chain integrity violation: hash mismatch at event %s",
                event_id,
            )
            raise error

        # Update previous_hash for next iteration
        previous_hash = stored_event_hash

    logger.info("Chain integrity verified for %d events", len(events))
    return True


def verify_chain_batch(
    events: List[Dict[str, Any]],
    expected_first_prev_hash: Optional[str] = None,
) -> Tuple[bool, Optional[ChainIntegrityError]]:
    """
    Verify chain integrity with explicit first hash expectation.

    This variant is useful for verifying a subset of the chain where
    you know what the previous hash should be.

    Args:
        events: List of audit events in chronological order.
        expected_first_prev_hash: Expected prev_event_hash of the first event.

    Returns:
        Tuple of (is_valid, error_or_none).

    Example:
        >>> # Verify events 100-200, knowing event 99's hash
        >>> is_valid, error = verify_chain_batch(
        ...     events[100:200],
        ...     expected_first_prev_hash=events[99]["event_hash"],
        ... )
    """
    try:
        if events and expected_first_prev_hash is not None:
            first_prev = events[0].get("prev_event_hash")
            if first_prev != expected_first_prev_hash:
                return False, ChainIntegrityError(
                    event_id=events[0].get("event_id", "unknown"),
                    event_index=0,
                    expected_hash=expected_first_prev_hash,
                    actual_hash=first_prev,
                    violation_type="link_broken",
                    message=(
                        f"First event's prev_event_hash mismatch: "
                        f"expected '{expected_first_prev_hash}', got '{first_prev}'"
                    ),
                )

        verify_chain(events)
        return True, None

    except ChainIntegrityError as e:
        return False, e


def verify_single_event(
    event: Dict[str, Any],
    expected_prev_hash: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Verify integrity of a single event.

    Args:
        event: Event to verify.
        expected_prev_hash: Expected prev_event_hash value (optional).

    Returns:
        Tuple of (is_valid, error_message_or_none).

    Example:
        >>> is_valid, error = verify_single_event(event, prev_hash)
        >>> if not is_valid:
        ...     print(f"Verification failed: {error}")
    """
    event_id = event.get("event_id", "unknown")
    stored_prev_hash = event.get("prev_event_hash")
    stored_event_hash = event.get("event_hash")

    # Check prev_hash if expected value provided
    if expected_prev_hash is not None:
        if stored_prev_hash != expected_prev_hash:
            return False, (
                f"prev_event_hash mismatch for {event_id}: "
                f"expected '{expected_prev_hash}', got '{stored_prev_hash}'"
            )

    # Recompute and verify event_hash
    computed_hash = compute_chain_hash(event, stored_prev_hash)

    if computed_hash != stored_event_hash:
        return False, (
            f"event_hash mismatch for {event_id}: "
            f"computed '{computed_hash}', stored '{stored_event_hash}'"
        )

    return True, None


def compute_payload_hash(
    measurements: List[Any],
    entities: List[Any],
) -> str:
    """
    Compute SHA-256 hash of the audit payload.

    The payload hash covers measurements and entities for integrity
    verification independent of chain linking.

    Args:
        measurements: List of measurement audit records.
        entities: List of entity audit records.

    Returns:
        SHA-256 hash with "sha256:" prefix.

    Example:
        >>> payload_hash = compute_payload_hash(measurements, entities)
        >>> assert payload_hash.startswith("sha256:")
    """
    # Normalize to dicts if Pydantic models
    normalized_measurements = _normalize_to_dicts(measurements)
    normalized_entities = _normalize_to_dicts(entities)

    # Create payload structure
    payload = {
        "measurements": normalized_measurements,
        "entities": normalized_entities,
    }

    # Serialize and hash
    json_str = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_serializer,
    )

    hash_bytes = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
    return f"{HASH_PREFIX}{hash_bytes}"


def create_genesis_hash(org_id: str) -> str:
    """
    Create a genesis hash for starting a new chain.

    The genesis hash is a deterministic hash based on the org_id,
    providing a known starting point for the chain.

    Args:
        org_id: Organization ID for the chain.

    Returns:
        SHA-256 genesis hash.

    Example:
        >>> genesis = create_genesis_hash("org-acme")
        >>> # First event should have prev_event_hash = genesis
    """
    genesis_data = {
        "type": "genesis",
        "org_id": org_id,
        "version": "1.0.0",
    }

    json_str = json.dumps(genesis_data, sort_keys=True, separators=(",", ":"))
    hash_bytes = hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    genesis_hash = f"{HASH_PREFIX}{hash_bytes}"
    logger.info("Created genesis hash for org %s: %s", org_id, genesis_hash[:30] + "...")

    return genesis_hash


def _normalize_to_dicts(items: List[Any]) -> List[Dict[str, Any]]:
    """
    Normalize a list of items to dictionaries.

    Handles both Pydantic models and plain dictionaries.

    Args:
        items: List of items to normalize.

    Returns:
        List of dictionaries.
    """
    result = []
    for item in items:
        if hasattr(item, "model_dump"):
            result.append(item.model_dump(mode="json"))
        elif isinstance(item, dict):
            result.append(item)
        else:
            result.append(dict(item))
    return result


def _json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for non-standard types.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-serializable representation.

    Raises:
        TypeError: If object cannot be serialized.
    """
    if isinstance(obj, datetime):
        return obj.isoformat() + "Z" if obj.tzinfo is None else obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
