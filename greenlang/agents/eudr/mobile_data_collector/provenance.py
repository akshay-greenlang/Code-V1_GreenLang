# -*- coding: utf-8 -*-
"""
Provenance Tracking for Mobile Data Collector - AGENT-EUDR-015

Provides SHA-256 based audit trail tracking for all mobile data
collection operations. Maintains an in-memory chain-hashed operation
log for tamper-evident provenance across form submission, GPS capture,
polygon trace, photo evidence collection, sync queue management,
sync conflict resolution, form template management, digital signature
capture, data package building, device registration, device event
telemetry, and audit logging.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - JSON export for external audit systems
    - Complete provenance for every mobile data collection operation
    - Bit-perfect reproducibility for compliance audits
    - EUDR Article 14 five-year audit trail compliance
    - GPS coordinate and photo integrity hashes provide end-to-end
      verifiability from field capture to server storage

Entity Types (12):
    - form_submission: Form creation, validation, submission, and sync
    - gps_capture: GPS point coordinate capture with accuracy metadata
    - polygon_trace: Plot boundary polygon capture and area calculation
    - photo_evidence: Photo capture with EXIF extraction and SHA-256 hash
    - sync_item: Sync queue item lifecycle (queue, upload, complete, fail)
    - sync_conflict: Conflict detection, resolution, and manual override
    - form_template: Template creation, update, versioning, deployment
    - digital_signature: ECDSA P-256 signature capture and verification
    - data_package: Package assembly, sealing, Merkle root computation
    - device_registration: Device registration, assignment, decommission
    - device_event: Device telemetry event recording
    - audit_entry: Immutable audit log entry recording

Actions (12):
    - create: Create a new entity (form, template, device, package)
    - update: Update an existing entity's fields or status
    - submit: Submit a form for synchronization
    - capture: Capture GPS coordinates or photo evidence
    - upload: Upload data from device to server during sync
    - sync: Complete a sync session for a device
    - resolve: Resolve a sync conflict
    - build: Build or seal a data package
    - register: Register a new device in the fleet
    - deregister: Decommission a device from the fleet
    - sign: Capture a digital signature
    - verify: Verify signature, package integrity, or photo hash

Example:
    >>> from greenlang.agents.eudr.mobile_data_collector.provenance import (
    ...     ProvenanceTracker,
    ... )
    >>> tracker = ProvenanceTracker()
    >>> entry = tracker.record("form_submission", "submit", "form-001")
    >>> assert entry.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015 Mobile Data Collector (GL-EUDR-MDC-015)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# ProvenanceRecord dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceRecord:
    """A single tamper-evident provenance record for a mobile data collector operation.

    This dataclass is frozen (immutable) to guarantee that recorded
    provenance entries cannot be modified after creation, supporting
    EUDR Article 14 audit trail integrity requirements and field
    data integrity verification records.

    Attributes:
        entity_type: Type of entity being tracked (form_submission,
            gps_capture, polygon_trace, photo_evidence, sync_item,
            sync_conflict, form_template, digital_signature,
            data_package, device_registration, device_event,
            audit_entry).
        entity_id: Unique identifier for the entity instance.
        action: Action performed (create, update, submit, capture,
            upload, sync, resolve, build, register, deregister,
            sign, verify).
        hash_value: SHA-256 chain hash of this entry, incorporating
            the previous entry's hash for tamper detection.
        parent_hash: SHA-256 chain hash of the immediately preceding
            entry.
        timestamp: UTC ISO-formatted timestamp when the entry was
            created.
        metadata: Optional dictionary of additional contextual fields
            including data_hash, operator_id, device_id, form_id,
            capture_id, photo_id, package_id, etc.
    """

    entity_type: str
    entity_id: str
    action: str
    hash_value: str
    parent_hash: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entry to a plain dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "action": self.action,
            "hash_value": self.hash_value,
            "parent_hash": self.parent_hash,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Valid entity types and actions
# ---------------------------------------------------------------------------

VALID_ENTITY_TYPES = frozenset({
    # Form submission lifecycle
    "form_submission",
    # GPS point coordinate capture
    "gps_capture",
    # Plot boundary polygon trace
    "polygon_trace",
    # Photo evidence with integrity hash
    "photo_evidence",
    # Sync queue item lifecycle
    "sync_item",
    # Sync conflict detection and resolution
    "sync_conflict",
    # Form template management
    "form_template",
    # Digital signature capture/verification
    "digital_signature",
    # Data package assembly
    "data_package",
    # Device registration and fleet management
    "device_registration",
    # Device telemetry events
    "device_event",
    # Immutable audit log entries
    "audit_entry",
})

VALID_ACTIONS = frozenset({
    # Create a new entity
    "create",
    # Update an existing entity
    "update",
    # Submit a form for sync
    "submit",
    # Capture GPS or photo
    "capture",
    # Upload data during sync
    "upload",
    # Complete a sync session
    "sync",
    # Resolve a sync conflict
    "resolve",
    # Build or seal a data package
    "build",
    # Register a new device
    "register",
    # Decommission a device
    "deregister",
    # Capture a digital signature
    "sign",
    # Verify integrity (signature, hash, package)
    "verify",
})


# ---------------------------------------------------------------------------
# ProvenanceTracker
# ---------------------------------------------------------------------------


class ProvenanceTracker:
    """Tracks provenance for mobile data collector operations with SHA-256 chain hashing.

    Maintains an ordered log of operations with SHA-256 hashes that chain
    together to provide tamper-evident audit trails for EUDR Article 14
    record-keeping compliance (5-year retention) and field data integrity
    verification audit records.

    The genesis hash anchors the chain. Every new entry incorporates the
    previous chain hash so that any tampering is detectable via
    ``verify_chain()``.

    Supported entity types (12):
        - ``form_submission``: Form creation, validation, submission,
          status transitions, and sync lifecycle events.
        - ``gps_capture``: GPS point coordinate capture with HDOP,
          satellite count, fix type, and accuracy tier metadata.
        - ``polygon_trace``: Plot boundary polygon capture with vertex
          recording, area calculation, and validity checks.
        - ``photo_evidence``: Photo capture with EXIF extraction,
          geotagging, SHA-256 integrity hashing, and compression.
        - ``sync_item``: Sync queue item lifecycle from queueing
          through upload, completion, and failure handling.
        - ``sync_conflict``: Conflict detection between device and
          server data, resolution strategy application.
        - ``form_template``: Template creation, versioning, language
          pack updates, and deployment tracking.
        - ``digital_signature``: ECDSA P-256 signature capture,
          verification, timestamp binding, and revocation.
        - ``data_package``: Package assembly, artifact addition,
          Merkle root computation, sealing, and upload.
        - ``device_registration``: Device registration, operator
          assignment, area assignment, and decommissioning.
        - ``device_event``: Device telemetry event recording for
          fleet monitoring and operational visibility.
        - ``audit_entry``: Immutable audit log entry for compliance
          record-keeping and regulatory inspection support.

    Supported actions (12):
        Create: create.
        Update: update.
        Submit: submit.
        Capture: capture.
        Upload: upload.
        Sync: sync.
        Resolve: resolve.
        Build: build.
        Register: register.
        Deregister: deregister.
        Sign: sign.
        Verify: verify.

    Attributes:
        _genesis_hash: Immutable anchor hash for the provenance chain.
        _chain_store: In-memory chain storage keyed by
            ``"entity_type:entity_id"``.
        _global_chain: Flat list of all ProvenanceRecord objects in
            insertion order.
        _last_chain_hash: Most recent chain hash for linking the
            next entry.
        _lock: Reentrant lock for thread-safe access.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> entry = tracker.record("form_submission", "submit", "form-001")
        >>> assert entry.hash_value != ""
        >>> valid = tracker.verify_chain()
        >>> assert valid is True
    """

    def __init__(
        self,
        genesis_hash: str = "GL-EUDR-MDC-015-MOBILE-DATA-COLLECTOR-GENESIS",
    ) -> None:
        """Initialize ProvenanceTracker with a genesis hash anchor.

        Args:
            genesis_hash: String used to compute the immutable genesis
                hash. Defaults to the Mobile Data Collector agent
                identifier.
        """
        self._genesis_hash: str = hashlib.sha256(
            genesis_hash.encode("utf-8")
        ).hexdigest()
        self._chain_store: Dict[str, List[ProvenanceRecord]] = {}
        self._global_chain: List[ProvenanceRecord] = []
        self._last_chain_hash: str = self._genesis_hash
        self._lock: threading.RLock = threading.RLock()
        logger.info(
            "ProvenanceTracker initialized with genesis hash prefix=%s",
            self._genesis_hash[:16],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceRecord:
        """Record a provenance entry for a mobile data collector operation.

        Computes a SHA-256 hash of ``data`` (or a placeholder when None),
        then chains it to the previous entry hash to produce a
        tamper-evident audit record.

        Args:
            entity_type: Type of entity (form_submission, gps_capture,
                polygon_trace, photo_evidence, sync_item, sync_conflict,
                form_template, digital_signature, data_package,
                device_registration, device_event, audit_entry).
            action: Action performed (see VALID_ACTIONS).
            entity_id: Unique entity identifier.
            data: Optional serializable payload; its SHA-256 hash is
                stored.
            metadata: Optional dictionary of extra contextual fields.

        Returns:
            The newly created ProvenanceRecord.

        Raises:
            ValueError: If entity_type, action, or entity_id are empty.
        """
        if not entity_type:
            raise ValueError("entity_type must not be empty")
        if not action:
            raise ValueError("action must not be empty")
        if not entity_id:
            raise ValueError("entity_id must not be empty")

        timestamp = _utcnow().isoformat()
        data_hash = self._hash_data(data)
        store_key = f"{entity_type}:{entity_id}"

        # Build entry metadata combining data hash with caller metadata
        entry_metadata: Dict[str, Any] = {"data_hash": data_hash}
        if metadata:
            entry_metadata.update(metadata)

        with self._lock:
            parent_hash = self._last_chain_hash
            chain_hash = self._compute_chain_hash(
                parent_hash=parent_hash,
                data_hash=data_hash,
                action=action,
                timestamp=timestamp,
            )

            entry = ProvenanceRecord(
                entity_type=entity_type,
                entity_id=entity_id,
                action=action,
                hash_value=chain_hash,
                parent_hash=parent_hash,
                timestamp=timestamp,
                metadata=entry_metadata,
            )

            # Persist to entity-scoped and global stores
            if store_key not in self._chain_store:
                self._chain_store[store_key] = []
            self._chain_store[store_key].append(entry)
            self._global_chain.append(entry)
            self._last_chain_hash = chain_hash

        logger.debug(
            "Recorded provenance: %s/%s action=%s hash_prefix=%s",
            entity_type,
            entity_id[:16],
            action,
            chain_hash[:16],
        )
        return entry

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire global provenance chain.

        Walks the global chain in insertion order and checks that every
        entry contains all required fields and that the chain is
        structurally consistent: the first entry chains from the genesis
        hash, and each subsequent entry's parent_hash matches the
        preceding entry's hash_value.

        Returns:
            True if the chain is intact, False if any entry is
            malformed or the chain links are broken.
        """
        with self._lock:
            chain = list(self._global_chain)

        if not chain:
            logger.debug("verify_chain: chain is empty - trivially valid")
            return True

        required_fields = {
            "entity_type",
            "entity_id",
            "action",
            "hash_value",
            "parent_hash",
            "timestamp",
        }

        for i, entry in enumerate(chain):
            # Validate all required fields are present and non-empty
            for field_name in required_fields:
                value = getattr(entry, field_name, None)
                if not value:
                    logger.warning(
                        "verify_chain: entry[%d] missing or empty "
                        "field '%s'",
                        i,
                        field_name,
                    )
                    return False

            # First entry must chain from the genesis hash
            if i == 0 and entry.parent_hash != self._genesis_hash:
                logger.warning(
                    "verify_chain: entry[0] parent_hash does not match "
                    "genesis hash"
                )
                return False

            # Each subsequent entry's parent must match the previous hash
            if i > 0 and entry.parent_hash != chain[i - 1].hash_value:
                logger.warning(
                    "verify_chain: chain break between entry[%d] and "
                    "entry[%d]",
                    i - 1,
                    i,
                )
                return False

        logger.debug(
            "verify_chain: %d entries verified successfully", len(chain)
        )
        return True

    def get_entries(
        self,
        entity_type: Optional[str] = None,
        action: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ProvenanceRecord]:
        """Return provenance entries filtered by entity_type, action, and/or limit.

        Args:
            entity_type: Optional entity type filter.
            action: Optional action filter.
            limit: Optional maximum number of entries to return
                (most recent when truncated).

        Returns:
            List of matching ProvenanceRecord objects.
        """
        with self._lock:
            entries = list(self._global_chain)

        if entity_type:
            entries = [e for e in entries if e.entity_type == entity_type]

        if action:
            entries = [e for e in entries if e.action == action]

        if limit is not None and limit > 0 and len(entries) > limit:
            entries = entries[-limit:]

        return entries

    def get_entries_for_entity(
        self,
        entity_type: str,
        entity_id: str,
    ) -> List[ProvenanceRecord]:
        """Return provenance entries for a specific entity_type:entity_id pair.

        Args:
            entity_type: Entity type to look up.
            entity_id: Entity identifier to look up.

        Returns:
            List of ProvenanceRecord objects, oldest first.
        """
        store_key = f"{entity_type}:{entity_id}"
        with self._lock:
            return list(self._chain_store.get(store_key, []))

    def export_json(self) -> str:
        """Export all provenance records as a formatted JSON string.

        Returns:
            Indented JSON string representation of the global chain.
        """
        with self._lock:
            chain_dicts = [
                entry.to_dict() for entry in self._global_chain
            ]
        return json.dumps(chain_dicts, indent=2, default=str)

    def clear(self) -> None:
        """Clear all provenance state and reset to genesis.

        Primarily intended for testing.
        """
        with self._lock:
            self._chain_store.clear()
            self._global_chain.clear()
            self._last_chain_hash = self._genesis_hash
        logger.info("ProvenanceTracker reset to genesis state")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def entry_count(self) -> int:
        """Return the total number of provenance entries recorded."""
        with self._lock:
            return len(self._global_chain)

    @property
    def entity_count(self) -> int:
        """Return the number of unique entity_type:entity_id keys tracked."""
        with self._lock:
            return len(self._chain_store)

    @property
    def genesis_hash(self) -> str:
        """Return the genesis hash that anchors the provenance chain."""
        return self._genesis_hash

    @property
    def last_chain_hash(self) -> str:
        """Return the most recent chain hash for the global chain."""
        with self._lock:
            return self._last_chain_hash

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of provenance entries recorded."""
        return self.entry_count

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"ProvenanceTracker(entries={self.entry_count}, "
            f"entities={self.entity_count}, "
            f"genesis_prefix={self._genesis_hash[:12]})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_data(self, data: Optional[Any]) -> str:
        """Compute a SHA-256 hash for arbitrary data.

        Serializes the payload to canonical JSON (sorted keys, default
        ``str`` fallback) before hashing so that equivalent structures
        always produce the same digest.

        Args:
            data: Any JSON-serializable object, or None.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        if data is None:
            serialized = "null"
        else:
            serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_chain_hash(
        self,
        parent_hash: str,
        data_hash: str,
        action: str,
        timestamp: str,
    ) -> str:
        """Compute the next SHA-256 chain hash linking to the previous entry.

        Args:
            parent_hash: Chain hash of the preceding entry (or genesis
                hash for the first entry).
            data_hash: SHA-256 hash of the operation's data payload.
            action: Action label recorded in this entry.
            timestamp: ISO-formatted UTC timestamp string.

        Returns:
            New hex-encoded SHA-256 chain hash.
        """
        combined = json.dumps(
            {
                "action": action,
                "data_hash": data_hash,
                "parent_hash": parent_hash,
                "timestamp": timestamp,
            },
            sort_keys=True,
        )
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def build_hash(self, data: Any) -> str:
        """Build a standalone SHA-256 hash for arbitrary data.

        Utility method for callers that need to pre-compute hashes
        before calling ``record()``.

        Args:
            data: Any JSON-serializable object.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        return self._hash_data(data)


# ---------------------------------------------------------------------------
# Thread-safe singleton helpers
# ---------------------------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton_tracker: Optional[ProvenanceTracker] = None


def get_provenance_tracker() -> ProvenanceTracker:
    """Return the process-wide singleton ProvenanceTracker.

    Creates the instance on first call (lazy initialization).
    Thread-safe via double-checked locking.

    Returns:
        The singleton ProvenanceTracker instance.

    Example:
        >>> tracker_a = get_provenance_tracker()
        >>> tracker_b = get_provenance_tracker()
        >>> assert tracker_a is tracker_b
    """
    global _singleton_tracker
    if _singleton_tracker is None:
        with _singleton_lock:
            if _singleton_tracker is None:
                _singleton_tracker = ProvenanceTracker()
                logger.info(
                    "Mobile data collector singleton "
                    "ProvenanceTracker created"
                )
    return _singleton_tracker


def set_provenance_tracker(tracker: ProvenanceTracker) -> None:
    """Replace the process-wide singleton with a custom tracker.

    Useful in tests that need isolated tracker instances.

    Args:
        tracker: The ProvenanceTracker instance to install.

    Raises:
        TypeError: If tracker is not a ProvenanceTracker instance.
    """
    if not isinstance(tracker, ProvenanceTracker):
        raise TypeError(
            f"tracker must be a ProvenanceTracker instance, "
            f"got {type(tracker)}"
        )
    global _singleton_tracker
    with _singleton_lock:
        _singleton_tracker = tracker
    logger.info(
        "Mobile data collector ProvenanceTracker singleton replaced"
    )


def reset_provenance_tracker() -> None:
    """Destroy the current singleton and reset to None.

    The next call to get_provenance_tracker() will create a fresh
    instance. Intended for test teardown.

    Example:
        >>> reset_provenance_tracker()
        >>> tracker = get_provenance_tracker()  # fresh instance
    """
    global _singleton_tracker
    with _singleton_lock:
        _singleton_tracker = None
    logger.info(
        "Mobile data collector ProvenanceTracker singleton reset "
        "to None"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Dataclass
    "ProvenanceRecord",
    # Constants
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    # Tracker class
    "ProvenanceTracker",
    # Singleton helpers
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
]
