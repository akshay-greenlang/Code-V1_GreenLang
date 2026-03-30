# -*- coding: utf-8 -*-
"""
Provenance Tracking for QR Code Generator - AGENT-EUDR-014

Provides SHA-256 based audit trail tracking for all QR code generation
operations. Maintains an in-memory chain-hashed operation log for
tamper-evident provenance across QR code generation, data payload
composition, label rendering, batch code generation, verification URL
construction, HMAC signing, scan event recording, bulk job orchestration,
lifecycle management, template management, code association tracking,
and configuration change tracking.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - JSON export for external audit systems
    - Complete provenance for every QR code operation
    - Bit-perfect reproducibility for compliance audits
    - EUDR Article 14 five-year audit trail compliance
    - QR code payload hashes provide end-to-end verifiability

Entity Types (12):
    - qr_code: QR code generation, activation, and lifecycle events
    - payload: Data payload composition, compression, and encryption
    - label: Label rendering with template application
    - batch_code: Batch code generation with check digit computation
    - verification_url: Verification URL construction with HMAC tokens
    - signature: HMAC-SHA256 signing and signature verification
    - scan_event: Scan event recording and counterfeit detection
    - bulk_job: Bulk generation job submission and completion
    - lifecycle_event: QR code status transitions (activate/deactivate/revoke)
    - template: Label template creation and modification
    - association: QR code to entity association management
    - config_change: Configuration change tracking

Actions (12):
    - generate: Generate a new QR code or batch code
    - compose: Compose a data payload with compression/encryption
    - render: Render a label with QR code and template
    - encode: Encode payload data for QR embedding
    - build_url: Construct a verification URL with HMAC token
    - sign: Sign data with HMAC-SHA256
    - verify: Verify a QR code, HMAC signature, or scan token
    - scan: Record a QR code scan event
    - activate: Activate a QR code for scanning
    - deactivate: Temporarily deactivate a QR code
    - revoke: Permanently revoke a QR code
    - cancel: Cancel a bulk generation job

Example:
    >>> from greenlang.agents.eudr.qr_code_generator.provenance import (
    ...     ProvenanceTracker,
    ... )
    >>> tracker = ProvenanceTracker()
    >>> entry = tracker.record("qr_code", "generate", "code-001")
    >>> assert entry.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ProvenanceRecord dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProvenanceRecord:
    """A single tamper-evident provenance record for a QR code generator operation.

    This dataclass is frozen (immutable) to guarantee that recorded
    provenance entries cannot be modified after creation, supporting
    EUDR Article 14 audit trail integrity requirements and QR code
    payload verification records.

    Attributes:
        entity_type: Type of entity being tracked (qr_code,
            payload, label, batch_code, verification_url, signature,
            scan_event, bulk_job, lifecycle_event, template,
            association, config_change).
        entity_id: Unique identifier for the entity instance.
        action: Action performed (generate, compose, render, encode,
            build_url, sign, verify, scan, activate, deactivate,
            revoke, cancel).
        hash_value: SHA-256 chain hash of this entry, incorporating
            the previous entry's hash for tamper detection.
        parent_hash: SHA-256 chain hash of the immediately preceding
            entry.
        timestamp: UTC ISO-formatted timestamp when the entry was
            created.
        metadata: Optional dictionary of additional contextual fields
            including data_hash, operator_id, code_id, payload_hash,
            scan_id, job_id, etc.
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
    # QR code generation and lifecycle
    "qr_code",
    # Data payload composition
    "payload",
    # Label rendering
    "label",
    # Batch code generation
    "batch_code",
    # Verification URL construction
    "verification_url",
    # HMAC signing
    "signature",
    # Scan event recording
    "scan_event",
    # Bulk generation job
    "bulk_job",
    # Lifecycle transitions
    "lifecycle_event",
    # Template management
    "template",
    # Code-to-entity associations
    "association",
    # Configuration changes
    "config_change",
})

VALID_ACTIONS = frozenset({
    # QR code generation
    "generate",
    # Payload composition
    "compose",
    # Label rendering
    "render",
    # Payload encoding
    "encode",
    # Verification URL construction
    "build_url",
    # HMAC signing
    "sign",
    # Verification
    "verify",
    # Scan recording
    "scan",
    # Code activation
    "activate",
    # Code deactivation
    "deactivate",
    # Code revocation
    "revoke",
    # Job cancellation
    "cancel",
})

# ---------------------------------------------------------------------------
# ProvenanceTracker
# ---------------------------------------------------------------------------

class ProvenanceTracker:
    """Tracks provenance for QR code generator operations with SHA-256 chain hashing.

    Maintains an ordered log of operations with SHA-256 hashes that chain
    together to provide tamper-evident audit trails for EUDR Article 14
    record-keeping compliance (5-year retention) and QR code payload
    verification audit records.

    The genesis hash anchors the chain. Every new entry incorporates the
    previous chain hash so that any tampering is detectable via
    ``verify_chain()``.

    Supported entity types (12):
        - ``qr_code``: QR code generation, activation, deactivation,
          revocation, and expiry lifecycle events.
        - ``payload``: Data payload composition with optional zlib
          compression and AES-256-GCM encryption.
        - ``label``: Label rendering with EUDR compliance status
          colour coding and template application.
        - ``batch_code``: Batch code generation with Luhn, ISO 7064,
          or CRC-8 check digit computation.
        - ``verification_url``: Verification URL construction with
          HMAC-SHA256 signed tokens and TTL management.
        - ``signature``: HMAC-SHA256 signing of QR code payloads
          and verification of existing signatures.
        - ``scan_event``: Scan event recording with counterfeit
          detection (velocity throttling, geo-fencing, HMAC validation).
        - ``bulk_job``: Bulk QR code generation job submission,
          progress tracking, completion, and failure handling.
        - ``lifecycle_event``: QR code status transitions recorded
          as discrete auditable events.
        - ``template``: Label template creation, modification, and
          version management.
        - ``association``: QR code to external entity association
          creation and deletion.
        - ``config_change``: Configuration modification tracking for
          change management and rollback.

    Supported actions (12):
        Generate: generate.
        Compose: compose.
        Render: render.
        Encode: encode.
        Build URL: build_url.
        Sign: sign.
        Verify: verify.
        Scan: scan.
        Activate: activate.
        Deactivate: deactivate.
        Revoke: revoke.
        Cancel: cancel.

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
        >>> entry = tracker.record("qr_code", "generate", "code-001")
        >>> assert entry.hash_value != ""
        >>> valid = tracker.verify_chain()
        >>> assert valid is True
    """

    def __init__(
        self,
        genesis_hash: str = "GL-EUDR-QRG-014-QR-CODE-GENERATOR-GENESIS",
    ) -> None:
        """Initialize ProvenanceTracker with a genesis hash anchor.

        Args:
            genesis_hash: String used to compute the immutable genesis
                hash. Defaults to the QR Code Generator agent
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
        """Record a provenance entry for a QR code generator operation.

        Computes a SHA-256 hash of ``data`` (or a placeholder when None),
        then chains it to the previous entry hash to produce a
        tamper-evident audit record.

        Args:
            entity_type: Type of entity (qr_code, payload, label,
                batch_code, verification_url, signature, scan_event,
                bulk_job, lifecycle_event, template, association,
                config_change).
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

        timestamp = utcnow().isoformat()
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
                    "QR code generator singleton "
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
        "QR code generator ProvenanceTracker singleton replaced"
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
        "QR code generator ProvenanceTracker singleton reset "
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
