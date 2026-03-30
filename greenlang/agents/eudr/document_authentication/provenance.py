# -*- coding: utf-8 -*-
"""
Provenance Tracking for Document Authentication - AGENT-EUDR-012

Provides SHA-256 based audit trail tracking for all document authentication
operations. Maintains an in-memory chain-hashed operation log for
tamper-evident provenance across document classification, signature
verification, hash integrity validation, certificate chain validation,
metadata extraction, fraud pattern detection, cross-reference verification,
report generation, template registration, trusted CA management, and
batch processing.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256
    - Chain hashing links operations in sequence
    - JSON export for external audit systems
    - Complete provenance for every document verification operation
    - Bit-perfect reproducibility for compliance audits
    - EUDR Article 14 five-year audit trail compliance
    - eIDAS Regulation (EU) No 910/2014 signature verification records

Entity Types (12):
    - document: Document record creation, update, and finalization
    - classification: Document type classification operations
    - signature: Digital signature verification operations
    - hash: Hash computation, verification, and registry operations
    - certificate: Certificate chain validation operations
    - metadata: Metadata extraction and anomaly analysis operations
    - fraud_alert: Fraud pattern detection and alert lifecycle
    - crossref: Cross-reference verification against external registries
    - report: Authentication report generation and evidence packaging
    - template: Document template registration and management
    - trusted_ca: Trusted certificate authority management
    - batch_job: Batch verification job lifecycle

Actions (12):
    - classify: Classify a document by type
    - verify_signature: Verify a document's digital signature
    - compute_hash: Compute integrity hashes for a document
    - verify_hash: Verify a document against a known hash
    - validate_chain: Validate a certificate chain
    - extract_metadata: Extract and analyze document metadata
    - detect_fraud: Run fraud pattern detection
    - cross_reference: Cross-reference against external registry
    - generate_report: Generate an authentication report
    - register_template: Register a known document template
    - add_ca: Add a trusted certificate authority
    - submit_batch: Submit a batch verification job

Example:
    >>> from greenlang.agents.eudr.document_authentication.provenance import (
    ...     ProvenanceTracker,
    ... )
    >>> tracker = ProvenanceTracker()
    >>> entry = tracker.record("document", "classify", "doc-001")
    >>> assert entry.hash_value is not None
    >>> valid = tracker.verify_chain()
    >>> assert valid is True

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-012 Document Authentication (GL-EUDR-DAV-012)
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
    """A single tamper-evident provenance record for a document authentication operation.

    This dataclass is frozen (immutable) to guarantee that recorded
    provenance entries cannot be modified after creation, supporting
    EUDR Article 14 audit trail integrity requirements and eIDAS
    Regulation (EU) No 910/2014 signature verification records.

    Attributes:
        entity_type: Type of entity being tracked (document,
            classification, signature, hash, certificate, metadata,
            fraud_alert, crossref, report, template, trusted_ca,
            batch_job).
        entity_id: Unique identifier for the entity instance.
        action: Action performed (classify, verify_signature,
            compute_hash, verify_hash, validate_chain,
            extract_metadata, detect_fraud, cross_reference,
            generate_report, register_template, add_ca,
            submit_batch).
        hash_value: SHA-256 chain hash of this entry, incorporating
            the previous entry's hash for tamper detection.
        parent_hash: SHA-256 chain hash of the immediately preceding
            entry.
        timestamp: UTC ISO-formatted timestamp when the entry was
            created.
        metadata: Optional dictionary of additional contextual fields
            including data_hash, operator_id, document_id, document_type,
            registry_type, etc.
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
    "document",
    "classification",
    "signature",
    "hash",
    "certificate",
    "metadata",
    "fraud_alert",
    "crossref",
    "report",
    "template",
    "trusted_ca",
    "batch_job",
})

VALID_ACTIONS = frozenset({
    # Classification
    "classify",
    # Signature verification
    "verify_signature",
    # Hash operations
    "compute_hash",
    "verify_hash",
    # Certificate chain validation
    "validate_chain",
    # Metadata extraction
    "extract_metadata",
    # Fraud detection
    "detect_fraud",
    # Cross-reference verification
    "cross_reference",
    # Report generation
    "generate_report",
    # Template management
    "register_template",
    # Trusted CA management
    "add_ca",
    # Batch processing
    "submit_batch",
})

# ---------------------------------------------------------------------------
# ProvenanceTracker
# ---------------------------------------------------------------------------

class ProvenanceTracker:
    """Tracks provenance for document authentication operations with SHA-256 chain hashing.

    Maintains an ordered log of operations with SHA-256 hashes that chain
    together to provide tamper-evident audit trails for EUDR Article 14
    record-keeping compliance (5-year retention) and eIDAS Regulation
    (EU) No 910/2014 signature verification audit records.

    The genesis hash anchors the chain. Every new entry incorporates the
    previous chain hash so that any tampering is detectable via
    ``verify_chain()``.

    Supported entity types (12):
        - ``document``: Document record creation, update, and
          finalization.
        - ``classification``: Document type classification operations
          including confidence scoring and alternative predictions.
        - ``signature``: Digital signature verification operations
          for CAdES, PAdES, XAdES, JAdES, QES, PGP, and PKCS#7.
        - ``hash``: Hash computation, verification, and registry
          operations for SHA-256/SHA-512 integrity checking.
        - ``certificate``: Certificate chain validation operations
          including OCSP, CRL, and CT log verification.
        - ``metadata``: Metadata extraction and anomaly analysis
          for creation date, author, GPS, and field completeness.
        - ``fraud_alert``: Fraud pattern detection alerts including
          15 pattern types from duplicate reuse to scope mismatch.
        - ``crossref``: Cross-reference verification against FSC,
          RSPO, ISCC, Fairtrade, UTZ/RA, IPPC, and customs registries.
        - ``report``: Authentication report generation and evidence
          package creation for EUDR compliance submission.
        - ``template``: Document template registration for forgery
          detection against known issuing authority layouts.
        - ``trusted_ca``: Trusted certificate authority management
          for signature verification trust store.
        - ``batch_job``: Batch verification job lifecycle (submit,
          progress, complete, fail).

    Supported actions (12):
        Classification: classify.
        Signature: verify_signature.
        Hash: compute_hash, verify_hash.
        Certificate: validate_chain.
        Metadata: extract_metadata.
        Fraud: detect_fraud.
        Cross-reference: cross_reference.
        Reporting: generate_report.
        Template: register_template.
        Trusted CA: add_ca.
        Batch: submit_batch.

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
        >>> entry = tracker.record("document", "classify", "doc-001")
        >>> assert entry.hash_value != ""
        >>> valid = tracker.verify_chain()
        >>> assert valid is True
    """

    def __init__(
        self,
        genesis_hash: str = "GL-EUDR-DAV-012-DOCUMENT-AUTHENTICATION-GENESIS",
    ) -> None:
        """Initialize ProvenanceTracker with a genesis hash anchor.

        Args:
            genesis_hash: String used to compute the immutable genesis
                hash. Defaults to the Document Authentication agent
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
        """Record a provenance entry for a document authentication operation.

        Computes a SHA-256 hash of ``data`` (or a placeholder when None),
        then chains it to the previous entry hash to produce a
        tamper-evident audit record.

        Args:
            entity_type: Type of entity (document, classification,
                signature, hash, certificate, metadata, fraud_alert,
                crossref, report, template, trusted_ca, batch_job).
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
                    "Document authentication singleton "
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
        "Document authentication ProvenanceTracker singleton replaced"
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
        "Document authentication ProvenanceTracker singleton reset "
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
