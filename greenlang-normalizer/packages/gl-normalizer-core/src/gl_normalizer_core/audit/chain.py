"""
Hash chain generator for GL-FOUND-X-003 audit events.

This module provides tamper-evident hash chaining for audit events,
ensuring that any modification to historical audit records can be
detected. The implementation follows NFR-035 (tamper-evident audit
hashing with hash chaining).

Key Features:
    - UUID generation for event_id
    - SHA-256 payload hashing
    - Event hash linking with prev_event_hash
    - Chain integrity verification
    - Thread-safe chain management

Security Considerations:
    - Hashes are computed using SHA-256 (cryptographically secure)
    - JSON serialization is deterministic (sorted keys, no whitespace)
    - Chain verification detects any tampering

Example:
    >>> from gl_normalizer_core.audit.chain import HashChainGenerator
    >>> generator = HashChainGenerator()
    >>> event_id = generator.generate_event_id()
    >>> payload_hash = generator.compute_payload_hash(measurements, entities)
    >>> event_hash = generator.compute_event_hash(event_data, prev_hash)
"""

import hashlib
import json
import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ChainState(BaseModel):
    """
    State of the hash chain for a specific organization or scope.

    Attributes:
        scope_id: Identifier for the chain scope (e.g., org_id).
        last_event_id: ID of the last event in the chain.
        last_event_hash: Hash of the last event in the chain.
        chain_length: Number of events in the chain.
        created_at: Timestamp when the chain was created.
        updated_at: Timestamp of the last update.
    """

    scope_id: str
    last_event_id: Optional[str] = None
    last_event_hash: Optional[str] = None
    chain_length: int = 0
    created_at: datetime
    updated_at: datetime


class ChainIntegrityError(Exception):
    """
    Exception raised when chain integrity verification fails.

    Attributes:
        event_id: ID of the event where integrity failed.
        expected_hash: Expected prev_event_hash value.
        actual_hash: Actual prev_event_hash value found.
        message: Detailed error message.
    """

    def __init__(
        self,
        event_id: str,
        expected_hash: Optional[str],
        actual_hash: Optional[str],
        message: str,
    ):
        self.event_id = event_id
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(message)


class HashChainGenerator:
    """
    Generator for tamper-evident hash chains.

    Provides methods for generating event IDs, computing payload hashes,
    computing event hashes with chain linking, and verifying chain integrity.

    Thread Safety:
        This class is thread-safe for concurrent event generation within
        a single scope. Different scopes can be processed in parallel.

    Attributes:
        _chain_states: Dictionary of chain states by scope_id.
        _lock: Threading lock for state updates.

    Example:
        >>> generator = HashChainGenerator()
        >>> event_id = generator.generate_event_id()
        >>> payload_hash = generator.compute_payload_hash(
        ...     measurements=[...],
        ...     entities=[...]
        ... )
        >>> event_hash = generator.compute_event_hash(
        ...     event_data={"event_id": event_id, ...},
        ...     prev_event_hash=None
        ... )
    """

    # Prefix for event IDs
    EVENT_ID_PREFIX = "norm-evt"

    # Hash algorithm
    HASH_ALGORITHM = "sha256"

    # Hash prefix for display
    HASH_PREFIX = "sha256:"

    def __init__(self):
        """Initialize the hash chain generator."""
        self._chain_states: Dict[str, ChainState] = {}
        self._lock = threading.Lock()

    def generate_event_id(self, prefix: Optional[str] = None) -> str:
        """
        Generate a unique event ID using UUID4.

        Args:
            prefix: Optional prefix override (default: "norm-evt").

        Returns:
            Unique event ID in format "{prefix}-{uuid}".

        Example:
            >>> generator = HashChainGenerator()
            >>> event_id = generator.generate_event_id()
            >>> assert event_id.startswith("norm-evt-")
        """
        prefix = prefix or self.EVENT_ID_PREFIX
        unique_id = uuid.uuid4().hex[:12]
        event_id = f"{prefix}-{unique_id}"
        logger.debug("Generated event_id: %s", event_id)
        return event_id

    def compute_payload_hash(
        self,
        measurements: List[Any],
        entities: List[Any],
    ) -> str:
        """
        Compute SHA-256 hash of the payload (measurements + entities).

        The payload is serialized to JSON with deterministic ordering
        to ensure reproducible hashes.

        Args:
            measurements: List of MeasurementAudit records (as dicts or models).
            entities: List of EntityAudit records (as dicts or models).

        Returns:
            SHA-256 hash with "sha256:" prefix.

        Example:
            >>> generator = HashChainGenerator()
            >>> payload_hash = generator.compute_payload_hash(
            ...     measurements=[{"field": "energy", "raw_value": 100}],
            ...     entities=[{"field": "fuel", "raw_name": "Diesel"}]
            ... )
            >>> assert payload_hash.startswith("sha256:")
        """
        # Normalize to dicts if Pydantic models
        normalized_measurements = self._normalize_to_dicts(measurements)
        normalized_entities = self._normalize_to_dicts(entities)

        # Create payload structure
        payload = {
            "measurements": normalized_measurements,
            "entities": normalized_entities,
        }

        # Compute hash
        payload_hash = self._compute_hash(payload)
        logger.debug(
            "Computed payload_hash for %d measurements, %d entities: %s",
            len(measurements),
            len(entities),
            payload_hash[:20] + "...",
        )
        return payload_hash

    def compute_event_hash(
        self,
        event_data: Dict[str, Any],
        prev_event_hash: Optional[str] = None,
    ) -> str:
        """
        Compute SHA-256 hash of the complete event including chain link.

        The event hash includes:
        - All event fields (excluding event_hash itself)
        - The prev_event_hash for chain linking

        Args:
            event_data: Complete event data as dictionary.
            prev_event_hash: Hash of the previous event in the chain.

        Returns:
            SHA-256 hash with "sha256:" prefix.

        Example:
            >>> generator = HashChainGenerator()
            >>> event_hash = generator.compute_event_hash(
            ...     event_data={"event_id": "norm-evt-001", "status": "success"},
            ...     prev_event_hash=None
            ... )
            >>> assert event_hash.startswith("sha256:")
        """
        # Create hashable data excluding event_hash (circular reference)
        hashable_data = {
            k: v for k, v in event_data.items() if k != "event_hash"
        }

        # Ensure prev_event_hash is included
        hashable_data["prev_event_hash"] = prev_event_hash

        # Compute hash
        event_hash = self._compute_hash(hashable_data)
        logger.debug(
            "Computed event_hash for event_id=%s: %s",
            event_data.get("event_id", "unknown"),
            event_hash[:20] + "...",
        )
        return event_hash

    def link_event(
        self,
        scope_id: str,
        event_id: str,
        event_hash: str,
    ) -> Optional[str]:
        """
        Link a new event to the chain and return the previous hash.

        This method is thread-safe and atomic. It:
        1. Retrieves the current last event hash for the scope
        2. Updates the chain state with the new event
        3. Returns the previous event hash for linking

        Args:
            scope_id: Identifier for the chain scope (e.g., org_id).
            event_id: ID of the new event being added.
            event_hash: Hash of the new event.

        Returns:
            Hash of the previous event in the chain (None if first event).

        Example:
            >>> generator = HashChainGenerator()
            >>> prev_hash = generator.link_event(
            ...     scope_id="org-acme",
            ...     event_id="norm-evt-001",
            ...     event_hash="sha256:abc123..."
            ... )
            >>> assert prev_hash is None  # First event in chain
        """
        with self._lock:
            now = datetime.utcnow()

            # Get or create chain state
            if scope_id not in self._chain_states:
                self._chain_states[scope_id] = ChainState(
                    scope_id=scope_id,
                    created_at=now,
                    updated_at=now,
                )
                logger.info("Created new hash chain for scope: %s", scope_id)

            state = self._chain_states[scope_id]
            prev_hash = state.last_event_hash

            # Update state
            state.last_event_id = event_id
            state.last_event_hash = event_hash
            state.chain_length += 1
            state.updated_at = now

            logger.debug(
                "Linked event %s to chain %s (length=%d, prev_hash=%s)",
                event_id,
                scope_id,
                state.chain_length,
                prev_hash[:20] + "..." if prev_hash else None,
            )

            return prev_hash

    def get_chain_state(self, scope_id: str) -> Optional[ChainState]:
        """
        Get the current state of a hash chain.

        Args:
            scope_id: Identifier for the chain scope.

        Returns:
            ChainState if exists, None otherwise.

        Example:
            >>> generator = HashChainGenerator()
            >>> state = generator.get_chain_state("org-acme")
        """
        with self._lock:
            return self._chain_states.get(scope_id)

    def get_prev_event_hash(self, scope_id: str) -> Optional[str]:
        """
        Get the hash of the last event in a chain.

        Args:
            scope_id: Identifier for the chain scope.

        Returns:
            Hash of the last event, or None if chain is empty.

        Example:
            >>> generator = HashChainGenerator()
            >>> prev_hash = generator.get_prev_event_hash("org-acme")
        """
        with self._lock:
            state = self._chain_states.get(scope_id)
            return state.last_event_hash if state else None

    def verify_chain_integrity(
        self,
        events: List[Dict[str, Any]],
    ) -> Tuple[bool, Optional[ChainIntegrityError]]:
        """
        Verify the integrity of a sequence of events.

        Checks that:
        1. Each event's event_hash is correctly computed
        2. Each event's prev_event_hash matches the previous event's event_hash
        3. The chain is continuous without gaps

        Args:
            events: List of events in chronological order.

        Returns:
            Tuple of (is_valid, error_or_none).

        Example:
            >>> generator = HashChainGenerator()
            >>> events = [event1, event2, event3]
            >>> is_valid, error = generator.verify_chain_integrity(events)
            >>> if not is_valid:
            ...     print(f"Integrity violation at event {error.event_id}")
        """
        if not events:
            return True, None

        logger.info("Verifying chain integrity for %d events", len(events))

        prev_hash: Optional[str] = None

        for i, event in enumerate(events):
            event_id = event.get("event_id", f"unknown-{i}")
            stored_prev_hash = event.get("prev_event_hash")
            stored_event_hash = event.get("event_hash")

            # Verify prev_event_hash links correctly
            if stored_prev_hash != prev_hash:
                error = ChainIntegrityError(
                    event_id=event_id,
                    expected_hash=prev_hash,
                    actual_hash=stored_prev_hash,
                    message=(
                        f"Chain integrity violation at event {event_id}: "
                        f"prev_event_hash mismatch. Expected {prev_hash}, "
                        f"got {stored_prev_hash}"
                    ),
                )
                logger.error(
                    "Chain integrity violation at event %s: prev_hash mismatch",
                    event_id,
                )
                return False, error

            # Recompute and verify event_hash
            computed_hash = self.compute_event_hash(event, prev_hash)
            if computed_hash != stored_event_hash:
                error = ChainIntegrityError(
                    event_id=event_id,
                    expected_hash=computed_hash,
                    actual_hash=stored_event_hash,
                    message=(
                        f"Chain integrity violation at event {event_id}: "
                        f"event_hash mismatch. Computed {computed_hash}, "
                        f"stored {stored_event_hash}"
                    ),
                )
                logger.error(
                    "Chain integrity violation at event %s: event_hash mismatch",
                    event_id,
                )
                return False, error

            # Update prev_hash for next iteration
            prev_hash = stored_event_hash

        logger.info("Chain integrity verified successfully for %d events", len(events))
        return True, None

    def verify_single_event(
        self,
        event: Dict[str, Any],
        expected_prev_hash: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify integrity of a single event.

        Args:
            event: Event to verify.
            expected_prev_hash: Expected prev_event_hash value.

        Returns:
            Tuple of (is_valid, error_message_or_none).

        Example:
            >>> generator = HashChainGenerator()
            >>> is_valid, error = generator.verify_single_event(event)
        """
        event_id = event.get("event_id", "unknown")
        stored_prev_hash = event.get("prev_event_hash")
        stored_event_hash = event.get("event_hash")

        # Check prev_hash if expected value provided
        if expected_prev_hash is not None and stored_prev_hash != expected_prev_hash:
            return False, (
                f"prev_event_hash mismatch for {event_id}: "
                f"expected {expected_prev_hash}, got {stored_prev_hash}"
            )

        # Recompute and verify event_hash
        computed_hash = self.compute_event_hash(event, stored_prev_hash)
        if computed_hash != stored_event_hash:
            return False, (
                f"event_hash mismatch for {event_id}: "
                f"computed {computed_hash}, stored {stored_event_hash}"
            )

        return True, None

    def _compute_hash(self, data: Any) -> str:
        """
        Compute SHA-256 hash of data with deterministic serialization.

        Args:
            data: Data to hash (will be JSON serialized).

        Returns:
            SHA-256 hash with "sha256:" prefix.
        """
        # Serialize with deterministic ordering
        json_str = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            default=self._json_serializer,
        )

        # Compute hash
        hash_bytes = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
        return f"{self.HASH_PREFIX}{hash_bytes}"

    def _normalize_to_dicts(self, items: List[Any]) -> List[Dict[str, Any]]:
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

    @staticmethod
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


# Singleton instance for convenience
_default_generator: Optional[HashChainGenerator] = None
_generator_lock = threading.Lock()


def get_default_generator() -> HashChainGenerator:
    """
    Get the default hash chain generator singleton.

    Returns:
        The default HashChainGenerator instance.

    Example:
        >>> from gl_normalizer_core.audit.chain import get_default_generator
        >>> generator = get_default_generator()
        >>> event_id = generator.generate_event_id()
    """
    global _default_generator
    if _default_generator is None:
        with _generator_lock:
            if _default_generator is None:
                _default_generator = HashChainGenerator()
    return _default_generator


def generate_event_id(prefix: Optional[str] = None) -> str:
    """
    Convenience function to generate an event ID using the default generator.

    Args:
        prefix: Optional prefix override.

    Returns:
        Unique event ID.

    Example:
        >>> from gl_normalizer_core.audit.chain import generate_event_id
        >>> event_id = generate_event_id()
    """
    return get_default_generator().generate_event_id(prefix)


def compute_payload_hash(
    measurements: List[Any],
    entities: List[Any],
) -> str:
    """
    Convenience function to compute payload hash using the default generator.

    Args:
        measurements: List of measurement records.
        entities: List of entity records.

    Returns:
        SHA-256 payload hash.

    Example:
        >>> from gl_normalizer_core.audit.chain import compute_payload_hash
        >>> hash_value = compute_payload_hash(measurements=[], entities=[])
    """
    return get_default_generator().compute_payload_hash(measurements, entities)


def compute_event_hash(
    event_data: Dict[str, Any],
    prev_event_hash: Optional[str] = None,
) -> str:
    """
    Convenience function to compute event hash using the default generator.

    Args:
        event_data: Event data dictionary.
        prev_event_hash: Previous event hash for chain linking.

    Returns:
        SHA-256 event hash.

    Example:
        >>> from gl_normalizer_core.audit.chain import compute_event_hash
        >>> hash_value = compute_event_hash(event_data={}, prev_event_hash=None)
    """
    return get_default_generator().compute_event_hash(event_data, prev_event_hash)


def verify_chain_integrity(
    events: List[Dict[str, Any]],
) -> Tuple[bool, Optional[ChainIntegrityError]]:
    """
    Convenience function to verify chain integrity using the default generator.

    Args:
        events: List of events in chronological order.

    Returns:
        Tuple of (is_valid, error_or_none).

    Example:
        >>> from gl_normalizer_core.audit.chain import verify_chain_integrity
        >>> is_valid, error = verify_chain_integrity(events)
    """
    return get_default_generator().verify_chain_integrity(events)
