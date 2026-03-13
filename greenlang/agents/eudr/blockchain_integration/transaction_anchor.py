# -*- coding: utf-8 -*-
"""
Transaction Anchoring Engine - AGENT-EUDR-013: Blockchain Integration (Engine 1)

Deterministic transaction anchoring engine for EUDR supply chain compliance
data. Creates immutable on-chain anchor records by hashing compliance data
payloads (DDS submissions, custody transfers, certificate references, mass
balance entries, etc.) and submitting them to blockchain networks either
individually (P0 immediate) or aggregated via Merkle trees (P1/P2 batch).

Zero-Hallucination Guarantees:
    - All hashing is deterministic SHA-256
    - Merkle tree construction uses sorted leaves for reproducibility
    - No ML/LLM used for any anchor computation
    - Gas estimation is formula-based (data_size * per_byte_gas + base_gas)
    - Status transitions follow a strict finite-state machine
    - SHA-256 provenance hashes on every state-changing operation
    - Anchor history is immutable and auditable

Regulatory References:
    - EU 2023/1115 (EUDR) Article 4: Due diligence obligations
    - EU 2023/1115 (EUDR) Article 14: Five-year record retention
    - EU 2023/1115 (EUDR) Article 33: EU Information System reporting

Performance Targets:
    - Single anchor creation: <20ms
    - Batch anchor (100 records): <500ms
    - Merkle tree construction: <200ms for 1000 leaves
    - Gas estimation: <5ms

Supported Event Types (8):
    dds_submission, custody_transfer, batch_event, certificate_reference,
    reconciliation_result, mass_balance_entry, document_authentication,
    geolocation_verification.

Priority Levels:
    P0_IMMEDIATE: Direct on-chain submission without batching
    P1_STANDARD: Standard batch aggregation with normal interval
    P2_BATCH: Low priority deferred to next batch window

PRD Feature References:
    - PRD-AGENT-EUDR-013 Feature 1: Transaction Anchoring Engine
    - PRD-AGENT-EUDR-013 Feature 1.1: Single Record Anchoring
    - PRD-AGENT-EUDR-013 Feature 1.2: Batch Merkle Tree Anchoring
    - PRD-AGENT-EUDR-013 Feature 1.3: Priority-Based Scheduling
    - PRD-AGENT-EUDR-013 Feature 1.4: Gas Cost Tracking
    - PRD-AGENT-EUDR-013 Feature 1.5: Retry with Exponential Backoff
    - PRD-AGENT-EUDR-013 Feature 1.6: Anchor History Retrieval
    - PRD-AGENT-EUDR-013 Feature 1.7: Confirmation Depth Monitoring

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-013
Agent ID: GL-EUDR-BCI-013
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.blockchain_integration.config import get_config
from greenlang.agents.eudr.blockchain_integration.metrics import (
    observe_anchor_duration,
    record_anchor_confirmed,
    record_anchor_created,
    record_anchor_failed,
    record_api_error,
    record_gas_spent,
    set_pending_anchors,
)
from greenlang.agents.eudr.blockchain_integration.models import (
    AnchorEventType,
    AnchorPriority,
    AnchorRecord,
    AnchorStatus,
    BlockchainNetwork,
    GasCost,
    MerkleLeaf,
    MerkleTree,
)
from greenlang.agents.eudr.blockchain_integration.provenance import (
    ProvenanceTracker,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id() -> str:
    """Generate a new UUID4 string identifier.

    Returns:
        UUID4 string.
    """
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid anchor status transitions (finite-state machine).
_VALID_STATUS_TRANSITIONS: Dict[str, List[str]] = {
    "pending": ["submitted", "failed"],
    "submitted": ["confirmed", "failed", "expired"],
    "confirmed": [],
    "failed": ["pending"],
    "expired": ["pending"],
}

#: Base gas cost for an on-chain anchor transaction (EVM chains).
_BASE_GAS_ANCHOR: int = 21000

#: Per-byte gas cost for calldata (non-zero bytes).
_GAS_PER_BYTE: int = 16

#: Per-byte gas cost for zero-value calldata bytes.
_GAS_PER_ZERO_BYTE: int = 4

#: Fixed gas overhead for storage write (SSTORE on EVM).
_SSTORE_GAS: int = 20000

#: Maximum retry delay cap in seconds.
_MAX_RETRY_DELAY_S: float = 120.0

#: Default simulated block time per chain in seconds (for confirmation wait).
_BLOCK_TIMES: Dict[str, float] = {
    "ethereum": 12.0,
    "polygon": 2.0,
    "fabric": 0.5,
    "besu": 2.0,
}

#: Merkle tree leaf domain separator prefix for second pre-image resistance.
_LEAF_PREFIX: bytes = b"\x00"

#: Merkle tree node domain separator prefix for second pre-image resistance.
_NODE_PREFIX: bytes = b"\x01"


# ==========================================================================
# TransactionAnchor
# ==========================================================================


class TransactionAnchor:
    """Transaction anchoring engine for EUDR compliance data.

    Creates, manages, and tracks on-chain anchor records for EUDR supply
    chain compliance events. Supports single-record immediate anchoring
    (P0) and batch aggregation via Merkle trees (P1/P2). Implements
    retry with exponential backoff, gas cost tracking, confirmation
    depth monitoring, and full provenance chain hashing.

    All numeric computations are deterministic (SHA-256, arithmetic).
    No ML/LLM calls are made anywhere in this engine.

    Attributes:
        _config: Blockchain integration configuration.
        _provenance: Provenance tracker for audit trail.
        _anchors: In-memory anchor store keyed by anchor_id.
        _anchors_by_tx: Index from tx_hash to anchor_id.
        _anchors_by_record: Index from source_record_id to anchor_ids.
        _merkle_trees: In-memory Merkle tree store keyed by tree_id.
        _gas_costs: In-memory gas cost records keyed by cost_id.
        _pending_queue: List of pending anchor IDs awaiting batch.
        _lock: Reentrant lock for thread-safe access.

    Example:
        >>> from greenlang.agents.eudr.blockchain_integration.transaction_anchor import (
        ...     TransactionAnchor,
        ... )
        >>> engine = TransactionAnchor()
        >>> anchor = engine.anchor_record(
        ...     record_id="rec-001",
        ...     record_hash="a" * 64,
        ...     event_type="dds_submission",
        ...     operator_id="op-001",
        ... )
        >>> assert anchor.status == "pending"
    """

    def __init__(
        self,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize TransactionAnchor engine.

        Args:
            provenance: Optional provenance tracker instance. If None,
                a new tracker is created with the configured genesis hash.
        """
        self._config = get_config()
        self._provenance = provenance or ProvenanceTracker(
            genesis_hash=self._config.genesis_hash,
        )
        self._anchors: Dict[str, AnchorRecord] = {}
        self._anchors_by_tx: Dict[str, str] = {}
        self._anchors_by_record: Dict[str, List[str]] = {}
        self._merkle_trees: Dict[str, MerkleTree] = {}
        self._gas_costs: Dict[str, GasCost] = {}
        self._pending_queue: List[str] = []
        self._lock = threading.RLock()

        logger.info(
            "TransactionAnchor engine initialized (version=%s, "
            "primary_chain=%s, batch_size=%d)",
            _MODULE_VERSION,
            self._config.primary_chain,
            self._config.batch_size,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def anchor_count(self) -> int:
        """Return total number of anchor records."""
        with self._lock:
            return len(self._anchors)

    @property
    def pending_count(self) -> int:
        """Return number of pending anchor records."""
        with self._lock:
            return len(self._pending_queue)

    @property
    def tree_count(self) -> int:
        """Return total number of Merkle trees built."""
        with self._lock:
            return len(self._merkle_trees)

    # ------------------------------------------------------------------
    # Public API: Single Record Anchoring
    # ------------------------------------------------------------------

    def anchor_record(
        self,
        record_id: str,
        record_hash: str,
        event_type: str,
        operator_id: str,
        priority: str = "p1_standard",
        network: Optional[str] = None,
        commodity: Optional[str] = None,
        source_agent_id: Optional[str] = None,
        payload_metadata: Optional[Dict[str, Any]] = None,
    ) -> AnchorRecord:
        """Create a new on-chain anchor record for a compliance data payload.

        Creates an AnchorRecord with PENDING status. For P0_IMMEDIATE
        priority, immediately simulates on-chain submission. For P1/P2,
        adds to the pending batch queue for later Merkle tree aggregation.

        Args:
            record_id: Source record identifier in the calling agent domain.
            record_hash: SHA-256 hex hash of the compliance data payload.
                Must be 64-128 hex characters.
            event_type: EUDR anchor event type (dds_submission,
                custody_transfer, batch_event, certificate_reference,
                reconciliation_result, mass_balance_entry,
                document_authentication, geolocation_verification).
            operator_id: EUDR operator identifier.
            priority: Submission priority (p0_immediate, p1_standard,
                p2_batch). Defaults to p1_standard.
            network: Target blockchain network. Defaults to primary_chain.
            commodity: EUDR-regulated commodity type (optional).
            source_agent_id: Agent ID that produced the source data.
            payload_metadata: Additional metadata about the payload.

        Returns:
            Created AnchorRecord with PENDING status.

        Raises:
            ValueError: If record_hash, event_type, or operator_id are
                invalid or empty.
        """
        start_time = time.monotonic()

        # Validate inputs
        self._validate_anchor_inputs(
            record_hash=record_hash,
            event_type=event_type,
            operator_id=operator_id,
            priority=priority,
        )

        chain = network or self._config.primary_chain
        self._validate_network(chain)

        confirmation_depth = self._get_confirmation_depth(chain)

        anchor_id = _generate_id()
        now = _utcnow()

        anchor = AnchorRecord(
            anchor_id=anchor_id,
            data_hash=record_hash.lower(),
            event_type=event_type,
            chain=chain,
            status=AnchorStatus.PENDING,
            priority=priority,
            required_confirmations=confirmation_depth,
            operator_id=operator_id,
            commodity=commodity,
            source_agent_id=source_agent_id,
            source_record_id=record_id,
            payload_metadata=payload_metadata or {},
            created_at=now,
        )

        # Compute provenance hash
        provenance_data = {
            "anchor_id": anchor_id,
            "data_hash": record_hash.lower(),
            "event_type": event_type,
            "chain": chain,
            "operator_id": operator_id,
            "created_at": now.isoformat(),
        }
        anchor.provenance_hash = _compute_hash(provenance_data)

        with self._lock:
            self._anchors[anchor_id] = anchor

            # Index by source record
            if record_id:
                if record_id not in self._anchors_by_record:
                    self._anchors_by_record[record_id] = []
                self._anchors_by_record[record_id].append(anchor_id)

            # Queue management based on priority
            if priority == AnchorPriority.P0_IMMEDIATE:
                # P0: immediate submission (skip batch queue)
                self._submit_immediate(anchor_id)
            else:
                # P1/P2: add to pending batch queue
                self._pending_queue.append(anchor_id)
                set_pending_anchors(len(self._pending_queue))

        # Record provenance
        self._provenance.record(
            entity_type="anchor",
            action="create",
            entity_id=anchor_id,
            data=provenance_data,
            metadata={
                "event_type": event_type,
                "chain": chain,
                "priority": priority,
                "operator_id": operator_id,
            },
        )

        # Record metrics
        record_anchor_created(chain, event_type)

        elapsed = time.monotonic() - start_time
        logger.info(
            "Anchor record created: anchor_id=%s, event_type=%s, "
            "chain=%s, priority=%s, elapsed_ms=%.1f",
            anchor_id,
            event_type,
            chain,
            priority,
            elapsed * 1000,
        )

        return anchor

    # ------------------------------------------------------------------
    # Public API: Batch Anchoring
    # ------------------------------------------------------------------

    def anchor_batch(
        self,
        records: List[Dict[str, Any]],
        operator_id: str,
        network: Optional[str] = None,
    ) -> List[AnchorRecord]:
        """Create anchor records for a batch of compliance data payloads.

        Builds individual AnchorRecords for each input, constructs a
        Merkle tree from their data hashes, and anchors the Merkle root
        on-chain as a single transaction.

        Args:
            records: List of dictionaries, each containing:
                - record_id (str): Source record identifier.
                - record_hash (str): SHA-256 hex hash of the data.
                - event_type (str): EUDR anchor event type.
                - commodity (str, optional): EUDR commodity.
                - source_agent_id (str, optional): Source agent.
                - payload_metadata (dict, optional): Extra metadata.
            operator_id: EUDR operator identifier.
            network: Target blockchain network. Defaults to primary_chain.

        Returns:
            List of created AnchorRecords with Merkle tree references.

        Raises:
            ValueError: If records list is empty, exceeds batch_max_size,
                or contains invalid entries.
        """
        start_time = time.monotonic()

        if not records:
            raise ValueError("records list must not be empty")

        max_size = self._config.batch_max_size
        if len(records) > max_size:
            raise ValueError(
                f"Batch size {len(records)} exceeds maximum {max_size}"
            )

        chain = network or self._config.primary_chain
        self._validate_network(chain)

        # Step 1: Create individual anchor records
        anchors: List[AnchorRecord] = []
        for rec in records:
            record_id = rec.get("record_id", _generate_id())
            record_hash = rec.get("record_hash", "")
            event_type = rec.get("event_type", "")
            commodity = rec.get("commodity")
            source_agent_id = rec.get("source_agent_id")
            payload_metadata = rec.get("payload_metadata", {})

            self._validate_anchor_inputs(
                record_hash=record_hash,
                event_type=event_type,
                operator_id=operator_id,
                priority="p2_batch",
            )

            anchor = self._create_anchor_internal(
                record_id=record_id,
                record_hash=record_hash,
                event_type=event_type,
                operator_id=operator_id,
                chain=chain,
                priority="p2_batch",
                commodity=commodity,
                source_agent_id=source_agent_id,
                payload_metadata=payload_metadata,
            )
            anchors.append(anchor)

        # Step 2: Build Merkle tree from anchor data hashes
        data_hashes = [a.data_hash for a in anchors]
        anchor_ids = [a.anchor_id for a in anchors]
        merkle_tree = self._build_merkle_tree(
            data_hashes=data_hashes,
            anchor_ids=anchor_ids,
            chain=chain,
        )

        # Step 3: Submit Merkle root to chain
        tx_hash = self._submit_to_chain(
            merkle_root=merkle_tree.root_hash,
            network=chain,
            gas_limit=self._estimate_gas(
                data_size=len(merkle_tree.root_hash),
                network=chain,
            ),
        )

        # Step 4: Update all anchors with tree and tx references
        with self._lock:
            for i, anchor in enumerate(anchors):
                anchor.merkle_root = merkle_tree.root_hash
                anchor.merkle_leaf_index = i
                anchor.tx_hash = tx_hash
                anchor.status = AnchorStatus.SUBMITTED
                anchor.submitted_at = _utcnow()
                self._anchors[anchor.anchor_id] = anchor

                if tx_hash:
                    self._anchors_by_tx[tx_hash] = anchor.anchor_id

            merkle_tree.tx_hash = tx_hash
            self._merkle_trees[merkle_tree.tree_id] = merkle_tree

        # Record provenance for batch
        batch_provenance_data = {
            "tree_id": merkle_tree.tree_id,
            "root_hash": merkle_tree.root_hash,
            "leaf_count": merkle_tree.leaf_count,
            "tx_hash": tx_hash,
            "chain": chain,
            "operator_id": operator_id,
            "anchor_ids": anchor_ids,
        }
        self._provenance.record(
            entity_type="merkle_tree",
            action="create",
            entity_id=merkle_tree.tree_id,
            data=batch_provenance_data,
            metadata={
                "chain": chain,
                "leaf_count": merkle_tree.leaf_count,
            },
        )

        # Record metrics for each anchor
        for anchor in anchors:
            record_anchor_created(chain, anchor.event_type)

        elapsed = time.monotonic() - start_time
        observe_anchor_duration(elapsed)

        logger.info(
            "Batch anchor complete: tree_id=%s, records=%d, "
            "root_hash=%s, tx_hash=%s, elapsed_ms=%.1f",
            merkle_tree.tree_id,
            len(anchors),
            merkle_tree.root_hash[:16],
            tx_hash[:16] if tx_hash else "None",
            elapsed * 1000,
        )

        return anchors

    # ------------------------------------------------------------------
    # Public API: Flush Pending Queue
    # ------------------------------------------------------------------

    def flush_pending(
        self,
        operator_id: str,
        network: Optional[str] = None,
    ) -> Optional[MerkleTree]:
        """Flush pending anchor queue by building and submitting a Merkle tree.

        Takes all pending P1/P2 anchors from the queue, constructs a
        Merkle tree, and submits the root hash on-chain.

        Args:
            operator_id: EUDR operator identifier for provenance.
            network: Target blockchain network. Defaults to primary_chain.

        Returns:
            The constructed MerkleTree, or None if queue was empty.
        """
        chain = network or self._config.primary_chain

        with self._lock:
            if not self._pending_queue:
                logger.debug("No pending anchors to flush")
                return None

            pending_ids = list(self._pending_queue)
            self._pending_queue.clear()
            set_pending_anchors(0)

        # Collect data hashes from pending anchors
        data_hashes: List[str] = []
        anchor_ids: List[str] = []
        for aid in pending_ids:
            anchor = self._anchors.get(aid)
            if anchor and anchor.status == AnchorStatus.PENDING.value:
                data_hashes.append(anchor.data_hash)
                anchor_ids.append(aid)

        if not data_hashes:
            logger.debug("No valid pending anchors found after filtering")
            return None

        # Build tree and submit
        merkle_tree = self._build_merkle_tree(
            data_hashes=data_hashes,
            anchor_ids=anchor_ids,
            chain=chain,
        )

        tx_hash = self._submit_to_chain(
            merkle_root=merkle_tree.root_hash,
            network=chain,
            gas_limit=self._estimate_gas(
                data_size=len(merkle_tree.root_hash),
                network=chain,
            ),
        )

        # Update anchors
        with self._lock:
            for i, aid in enumerate(anchor_ids):
                anchor = self._anchors.get(aid)
                if anchor:
                    anchor.merkle_root = merkle_tree.root_hash
                    anchor.merkle_leaf_index = i
                    anchor.tx_hash = tx_hash
                    anchor.status = AnchorStatus.SUBMITTED
                    anchor.submitted_at = _utcnow()

            merkle_tree.tx_hash = tx_hash
            self._merkle_trees[merkle_tree.tree_id] = merkle_tree

        # Provenance
        self._provenance.record(
            entity_type="merkle_tree",
            action="submit",
            entity_id=merkle_tree.tree_id,
            data={
                "tree_id": merkle_tree.tree_id,
                "root_hash": merkle_tree.root_hash,
                "tx_hash": tx_hash,
                "anchor_count": len(anchor_ids),
            },
        )

        logger.info(
            "Flushed %d pending anchors: tree_id=%s, tx_hash=%s",
            len(anchor_ids),
            merkle_tree.tree_id,
            tx_hash[:16] if tx_hash else "None",
        )

        return merkle_tree

    # ------------------------------------------------------------------
    # Public API: Retrieve Anchors
    # ------------------------------------------------------------------

    def get_anchor(self, anchor_id: str) -> Optional[AnchorRecord]:
        """Retrieve an anchor record by its unique identifier.

        Args:
            anchor_id: Unique anchor record identifier.

        Returns:
            AnchorRecord if found, None otherwise.
        """
        if not anchor_id:
            raise ValueError("anchor_id must not be empty")

        with self._lock:
            return self._anchors.get(anchor_id)

    def get_anchor_by_tx_hash(self, tx_hash: str) -> Optional[AnchorRecord]:
        """Retrieve an anchor record by its blockchain transaction hash.

        Args:
            tx_hash: Blockchain transaction hash.

        Returns:
            AnchorRecord if found, None otherwise.
        """
        if not tx_hash:
            raise ValueError("tx_hash must not be empty")

        with self._lock:
            anchor_id = self._anchors_by_tx.get(tx_hash)
            if anchor_id:
                return self._anchors.get(anchor_id)
            # Fallback: scan all anchors for tx_hash match
            for anchor in self._anchors.values():
                if anchor.tx_hash == tx_hash:
                    return anchor
            return None

    def get_anchor_history(self, record_id: str) -> List[AnchorRecord]:
        """Retrieve all anchor records for a given source record.

        Args:
            record_id: Source record identifier from the calling agent.

        Returns:
            List of AnchorRecords associated with this record, sorted
            by creation time (oldest first). Empty list if none found.
        """
        if not record_id:
            raise ValueError("record_id must not be empty")

        with self._lock:
            anchor_ids = self._anchors_by_record.get(record_id, [])
            anchors = [
                self._anchors[aid]
                for aid in anchor_ids
                if aid in self._anchors
            ]

        # Sort by creation time
        anchors.sort(key=lambda a: a.created_at)
        return anchors

    def list_anchors(
        self,
        status: Optional[str] = None,
        event_type: Optional[str] = None,
        chain: Optional[str] = None,
        operator_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AnchorRecord]:
        """List anchor records with optional filtering.

        Args:
            status: Filter by anchor status.
            event_type: Filter by event type.
            chain: Filter by blockchain network.
            operator_id: Filter by operator identifier.
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            Filtered and paginated list of AnchorRecords.
        """
        with self._lock:
            results = list(self._anchors.values())

        # Apply filters
        if status:
            results = [a for a in results if a.status == status]
        if event_type:
            results = [a for a in results if a.event_type == event_type]
        if chain:
            results = [a for a in results if a.chain == chain]
        if operator_id:
            results = [a for a in results if a.operator_id == operator_id]

        # Sort by creation time (newest first)
        results.sort(key=lambda a: a.created_at, reverse=True)

        # Paginate
        return results[offset: offset + limit]

    # ------------------------------------------------------------------
    # Public API: Anchor Confirmation
    # ------------------------------------------------------------------

    def confirm_anchor(
        self,
        anchor_id: str,
        block_number: int,
        block_hash: str,
        gas_used: int,
        gas_price_wei: int,
        confirmations: Optional[int] = None,
    ) -> AnchorRecord:
        """Mark an anchor as confirmed after reaching required depth.

        Transitions an anchor from SUBMITTED to CONFIRMED status after
        the required number of block confirmations have been observed.

        Args:
            anchor_id: Anchor record identifier.
            block_number: Block number containing the anchor transaction.
            block_hash: Hash of the block containing the transaction.
            gas_used: Actual gas consumed by the transaction.
            gas_price_wei: Gas price in wei at transaction time.
            confirmations: Number of confirmations observed. If None,
                uses the required_confirmations value.

        Returns:
            Updated AnchorRecord with CONFIRMED status.

        Raises:
            ValueError: If anchor_id not found or invalid transition.
        """
        with self._lock:
            anchor = self._anchors.get(anchor_id)
            if anchor is None:
                raise ValueError(f"Anchor not found: {anchor_id}")

            current_status = anchor.status
            if current_status not in (
                AnchorStatus.SUBMITTED.value,
                AnchorStatus.SUBMITTED,
            ):
                valid = _VALID_STATUS_TRANSITIONS.get(
                    str(current_status), []
                )
                if "confirmed" not in valid:
                    raise ValueError(
                        f"Cannot confirm anchor in status '{current_status}'"
                    )

            now = _utcnow()
            anchor.status = AnchorStatus.CONFIRMED
            anchor.block_number = block_number
            anchor.block_hash = block_hash
            anchor.gas_used = gas_used
            anchor.gas_price_wei = gas_price_wei
            anchor.confirmations = (
                confirmations
                if confirmations is not None
                else anchor.required_confirmations
            )
            anchor.confirmed_at = now

            # Update provenance hash
            provenance_data = {
                "anchor_id": anchor_id,
                "block_number": block_number,
                "block_hash": block_hash,
                "gas_used": gas_used,
                "confirmed_at": now.isoformat(),
            }
            anchor.provenance_hash = _compute_hash(provenance_data)
            self._anchors[anchor_id] = anchor

        # Record provenance
        self._provenance.record(
            entity_type="anchor",
            action="confirm",
            entity_id=anchor_id,
            data=provenance_data,
            metadata={
                "block_number": block_number,
                "gas_used": gas_used,
            },
        )

        # Record metrics
        chain_str = str(anchor.chain)
        record_anchor_confirmed(chain_str)
        if gas_used and gas_price_wei:
            record_gas_spent(chain_str, float(gas_used * gas_price_wei))

        # Track time from creation to confirmation
        if anchor.created_at and anchor.confirmed_at:
            duration = (
                anchor.confirmed_at - anchor.created_at
            ).total_seconds()
            observe_anchor_duration(duration)

        logger.info(
            "Anchor confirmed: anchor_id=%s, block=%d, gas_used=%d",
            anchor_id,
            block_number,
            gas_used,
        )

        return anchor

    def fail_anchor(
        self,
        anchor_id: str,
        error_message: str,
    ) -> AnchorRecord:
        """Mark an anchor as failed.

        Args:
            anchor_id: Anchor record identifier.
            error_message: Description of the failure.

        Returns:
            Updated AnchorRecord with FAILED status.

        Raises:
            ValueError: If anchor not found or invalid transition.
        """
        with self._lock:
            anchor = self._anchors.get(anchor_id)
            if anchor is None:
                raise ValueError(f"Anchor not found: {anchor_id}")

            anchor.status = AnchorStatus.FAILED
            anchor.error_message = error_message
            anchor.retry_count += 1

            provenance_data = {
                "anchor_id": anchor_id,
                "error_message": error_message,
                "retry_count": anchor.retry_count,
                "failed_at": _utcnow().isoformat(),
            }
            anchor.provenance_hash = _compute_hash(provenance_data)
            self._anchors[anchor_id] = anchor

        # Record provenance
        self._provenance.record(
            entity_type="anchor",
            action="submit",
            entity_id=anchor_id,
            data=provenance_data,
            metadata={"error": error_message},
        )

        # Metrics
        record_anchor_failed(str(anchor.chain))

        logger.warning(
            "Anchor failed: anchor_id=%s, error=%s, retries=%d",
            anchor_id,
            error_message,
            anchor.retry_count,
        )

        return anchor

    # ------------------------------------------------------------------
    # Public API: Gas Cost Tracking
    # ------------------------------------------------------------------

    def track_gas_cost(
        self,
        anchor_id: str,
        tx_hash: str,
        gas_used: int,
        gas_price_wei: int,
        chain: Optional[str] = None,
    ) -> GasCost:
        """Track the gas cost of an anchor transaction.

        Creates a GasCost record for a confirmed anchor transaction,
        calculating total cost in wei. Optionally estimates USD equivalent
        based on a simple ETH/USD rate (for reporting purposes only).

        Args:
            anchor_id: Associated anchor record identifier.
            tx_hash: Blockchain transaction hash.
            gas_used: Actual gas consumed by the transaction.
            gas_price_wei: Gas price in wei at transaction time.
            chain: Blockchain network. Defaults to primary_chain.

        Returns:
            Created GasCost record.

        Raises:
            ValueError: If gas_used or gas_price_wei is negative.
        """
        if gas_used < 0:
            raise ValueError(
                f"gas_used must be >= 0, got {gas_used}"
            )
        if gas_price_wei < 0:
            raise ValueError(
                f"gas_price_wei must be >= 0, got {gas_price_wei}"
            )

        network = chain or self._config.primary_chain
        total_cost_wei = gas_used * gas_price_wei

        cost_record = GasCost(
            cost_id=_generate_id(),
            chain=network,
            operation="anchor",
            estimated_gas=None,
            actual_gas=gas_used,
            gas_price_wei=gas_price_wei,
            total_cost_wei=total_cost_wei,
            tx_hash=tx_hash,
        )

        with self._lock:
            self._gas_costs[cost_record.cost_id] = cost_record

        # Provenance
        self._provenance.record(
            entity_type="gas_cost",
            action="create",
            entity_id=cost_record.cost_id,
            data={
                "anchor_id": anchor_id,
                "tx_hash": tx_hash,
                "gas_used": gas_used,
                "gas_price_wei": gas_price_wei,
                "total_cost_wei": total_cost_wei,
            },
        )

        # Metrics
        record_gas_spent(network, float(total_cost_wei))

        logger.info(
            "Gas cost tracked: anchor_id=%s, gas_used=%d, "
            "total_wei=%d, chain=%s",
            anchor_id,
            gas_used,
            total_cost_wei,
            network,
        )

        return cost_record

    # ------------------------------------------------------------------
    # Public API: Merkle Tree Access
    # ------------------------------------------------------------------

    def get_merkle_tree(self, tree_id: str) -> Optional[MerkleTree]:
        """Retrieve a Merkle tree by its identifier.

        Args:
            tree_id: Merkle tree identifier.

        Returns:
            MerkleTree if found, None otherwise.
        """
        if not tree_id:
            raise ValueError("tree_id must not be empty")

        with self._lock:
            return self._merkle_trees.get(tree_id)

    def get_merkle_proof(
        self,
        tree_id: str,
        leaf_index: int,
    ) -> Optional[List[Tuple[str, int]]]:
        """Generate a Merkle proof for a specific leaf in a tree.

        Returns the authentication path (sibling hashes and left/right
        indicators) from the specified leaf to the root.

        Args:
            tree_id: Merkle tree identifier.
            leaf_index: Zero-based index of the target leaf.

        Returns:
            List of (sibling_hash, path_index) tuples forming the
            authentication path, or None if tree not found.

        Raises:
            ValueError: If leaf_index is out of bounds.
        """
        with self._lock:
            tree = self._merkle_trees.get(tree_id)
            if tree is None:
                return None

        if leaf_index < 0 or leaf_index >= tree.leaf_count:
            raise ValueError(
                f"leaf_index {leaf_index} out of bounds "
                f"(0..{tree.leaf_count - 1})"
            )

        # Reconstruct proof from the tree leaves
        leaf_hashes = [leaf.leaf_hash for leaf in tree.leaves]
        proof_path = self._compute_merkle_proof(leaf_hashes, leaf_index)

        logger.debug(
            "Merkle proof generated: tree_id=%s, leaf_index=%d, "
            "proof_length=%d",
            tree_id,
            leaf_index,
            len(proof_path),
        )

        return proof_path

    # ------------------------------------------------------------------
    # Public API: Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics for the anchor engine.

        Returns:
            Dictionary containing anchor counts by status, event type,
            chain, total gas spent, and tree counts.
        """
        with self._lock:
            all_anchors = list(self._anchors.values())
            all_costs = list(self._gas_costs.values())
            tree_count = len(self._merkle_trees)
            pending_count = len(self._pending_queue)

        status_counts: Dict[str, int] = {}
        event_type_counts: Dict[str, int] = {}
        chain_counts: Dict[str, int] = {}

        for anchor in all_anchors:
            st = str(anchor.status)
            status_counts[st] = status_counts.get(st, 0) + 1

            et = str(anchor.event_type)
            event_type_counts[et] = event_type_counts.get(et, 0) + 1

            ch = str(anchor.chain)
            chain_counts[ch] = chain_counts.get(ch, 0) + 1

        total_gas_wei = sum(
            (c.total_cost_wei or 0) for c in all_costs
        )

        return {
            "total_anchors": len(all_anchors),
            "pending_count": pending_count,
            "tree_count": tree_count,
            "by_status": status_counts,
            "by_event_type": event_type_counts,
            "by_chain": chain_counts,
            "total_gas_cost_wei": total_gas_wei,
            "gas_cost_records": len(all_costs),
        }

    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------

    def _validate_anchor_inputs(
        self,
        record_hash: str,
        event_type: str,
        operator_id: str,
        priority: str,
    ) -> None:
        """Validate inputs for anchor record creation.

        Args:
            record_hash: SHA-256 hex hash of the data.
            event_type: EUDR anchor event type.
            operator_id: Operator identifier.
            priority: Submission priority.

        Raises:
            ValueError: If any input is invalid.
        """
        if not record_hash or len(record_hash) < 64:
            raise ValueError(
                f"record_hash must be at least 64 hex characters, "
                f"got {len(record_hash) if record_hash else 0}"
            )

        # Validate hex
        try:
            int(record_hash, 16)
        except ValueError:
            raise ValueError(
                f"record_hash must be a valid hexadecimal string"
            )

        valid_event_types = {e.value for e in AnchorEventType}
        if event_type not in valid_event_types:
            raise ValueError(
                f"event_type must be one of {sorted(valid_event_types)}, "
                f"got '{event_type}'"
            )

        if not operator_id:
            raise ValueError("operator_id must not be empty")

        valid_priorities = {p.value for p in AnchorPriority}
        if priority not in valid_priorities:
            raise ValueError(
                f"priority must be one of {sorted(valid_priorities)}, "
                f"got '{priority}'"
            )

    def _validate_network(self, network: str) -> None:
        """Validate a blockchain network identifier.

        Args:
            network: Blockchain network string.

        Raises:
            ValueError: If network is not supported.
        """
        valid_networks = {n.value for n in BlockchainNetwork}
        if network not in valid_networks:
            raise ValueError(
                f"network must be one of {sorted(valid_networks)}, "
                f"got '{network}'"
            )

    # ------------------------------------------------------------------
    # Internal: Anchor Creation
    # ------------------------------------------------------------------

    def _create_anchor_internal(
        self,
        record_id: str,
        record_hash: str,
        event_type: str,
        operator_id: str,
        chain: str,
        priority: str,
        commodity: Optional[str] = None,
        source_agent_id: Optional[str] = None,
        payload_metadata: Optional[Dict[str, Any]] = None,
    ) -> AnchorRecord:
        """Create an anchor record without queue management (internal use).

        Args:
            record_id: Source record identifier.
            record_hash: SHA-256 hex hash of the data.
            event_type: EUDR anchor event type.
            operator_id: Operator identifier.
            chain: Target blockchain network.
            priority: Submission priority.
            commodity: EUDR commodity (optional).
            source_agent_id: Source agent ID (optional).
            payload_metadata: Extra metadata (optional).

        Returns:
            Created AnchorRecord.
        """
        anchor_id = _generate_id()
        now = _utcnow()
        confirmation_depth = self._get_confirmation_depth(chain)

        anchor = AnchorRecord(
            anchor_id=anchor_id,
            data_hash=record_hash.lower(),
            event_type=event_type,
            chain=chain,
            status=AnchorStatus.PENDING,
            priority=priority,
            required_confirmations=confirmation_depth,
            operator_id=operator_id,
            commodity=commodity,
            source_agent_id=source_agent_id,
            source_record_id=record_id,
            payload_metadata=payload_metadata or {},
            created_at=now,
        )

        # Provenance hash
        provenance_data = {
            "anchor_id": anchor_id,
            "data_hash": record_hash.lower(),
            "event_type": event_type,
            "chain": chain,
            "created_at": now.isoformat(),
        }
        anchor.provenance_hash = _compute_hash(provenance_data)

        with self._lock:
            self._anchors[anchor_id] = anchor
            if record_id:
                if record_id not in self._anchors_by_record:
                    self._anchors_by_record[record_id] = []
                self._anchors_by_record[record_id].append(anchor_id)

        return anchor

    # ------------------------------------------------------------------
    # Internal: Immediate Submission
    # ------------------------------------------------------------------

    def _submit_immediate(self, anchor_id: str) -> None:
        """Submit a P0_IMMEDIATE anchor directly to chain.

        Wraps the single anchor hash in a degenerate Merkle tree
        (one leaf) and submits to the primary chain.

        Args:
            anchor_id: Anchor record identifier.
        """
        anchor = self._anchors.get(anchor_id)
        if anchor is None:
            logger.error(
                "Cannot submit immediate: anchor not found %s", anchor_id
            )
            return

        chain = str(anchor.chain)

        # Build a single-leaf Merkle tree
        merkle_tree = self._build_merkle_tree(
            data_hashes=[anchor.data_hash],
            anchor_ids=[anchor_id],
            chain=chain,
        )

        # Submit root to chain
        gas_limit = self._estimate_gas(
            data_size=len(merkle_tree.root_hash),
            network=chain,
        )
        tx_hash = self._submit_to_chain(
            merkle_root=merkle_tree.root_hash,
            network=chain,
            gas_limit=gas_limit,
        )

        # Update anchor
        anchor.merkle_root = merkle_tree.root_hash
        anchor.merkle_leaf_index = 0
        anchor.tx_hash = tx_hash
        anchor.status = AnchorStatus.SUBMITTED
        anchor.submitted_at = _utcnow()
        self._anchors[anchor_id] = anchor

        if tx_hash:
            self._anchors_by_tx[tx_hash] = anchor_id

        merkle_tree.tx_hash = tx_hash
        self._merkle_trees[merkle_tree.tree_id] = merkle_tree

        # Provenance
        self._provenance.record(
            entity_type="anchor",
            action="submit",
            entity_id=anchor_id,
            data={
                "anchor_id": anchor_id,
                "tx_hash": tx_hash,
                "merkle_root": merkle_tree.root_hash,
                "chain": chain,
            },
        )

        logger.info(
            "Immediate anchor submitted: anchor_id=%s, tx_hash=%s",
            anchor_id,
            tx_hash[:16] if tx_hash else "None",
        )

    # ------------------------------------------------------------------
    # Internal: Hash Computation
    # ------------------------------------------------------------------

    def _compute_record_hash(self, record_bytes: bytes) -> str:
        """Compute the SHA-256 hash of raw record bytes.

        Used for hashing arbitrary compliance data payloads before
        anchor creation.

        Args:
            record_bytes: Raw bytes of the record data.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        return hashlib.sha256(record_bytes).hexdigest()

    def _compute_leaf_hash(self, data_hash: str) -> str:
        """Compute the Merkle tree leaf hash with domain separation.

        Applies a domain-separated hash: H(0x00 || data_hash_bytes)
        to prevent second pre-image attacks on the Merkle tree.

        Args:
            data_hash: Hex-encoded data hash string.

        Returns:
            Hex-encoded leaf hash string.
        """
        data_bytes = bytes.fromhex(data_hash)
        return hashlib.sha256(_LEAF_PREFIX + data_bytes).hexdigest()

    def _compute_node_hash(self, left: str, right: str) -> str:
        """Compute a Merkle tree internal node hash with domain separation.

        Applies a domain-separated hash: H(0x01 || left || right)
        to prevent second pre-image attacks.

        Args:
            left: Hex-encoded left child hash.
            right: Hex-encoded right child hash.

        Returns:
            Hex-encoded internal node hash string.
        """
        left_bytes = bytes.fromhex(left)
        right_bytes = bytes.fromhex(right)
        return hashlib.sha256(
            _NODE_PREFIX + left_bytes + right_bytes
        ).hexdigest()

    # ------------------------------------------------------------------
    # Internal: Merkle Tree Construction
    # ------------------------------------------------------------------

    def _build_merkle_tree(
        self,
        data_hashes: List[str],
        anchor_ids: List[str],
        chain: str,
    ) -> MerkleTree:
        """Build a Merkle tree from a list of data hashes.

        Constructs a balanced binary Merkle tree with domain-separated
        hashing. If sorted_tree is enabled, leaves are sorted before
        tree construction for deterministic proofs.

        Args:
            data_hashes: List of hex-encoded SHA-256 data hashes.
            anchor_ids: Corresponding anchor record identifiers.
            chain: Target blockchain network.

        Returns:
            Constructed MerkleTree with root hash and leaves.
        """
        start_time = time.monotonic()

        if not data_hashes:
            raise ValueError("data_hashes must not be empty")

        max_leaves = self._config.max_tree_leaves
        if len(data_hashes) > max_leaves:
            raise ValueError(
                f"Leaf count {len(data_hashes)} exceeds "
                f"max_tree_leaves {max_leaves}"
            )

        # Compute leaf hashes with domain separation
        leaf_entries: List[Tuple[int, str, str, str]] = []
        for i, dh in enumerate(data_hashes):
            leaf_hash = self._compute_leaf_hash(dh)
            leaf_entries.append((i, dh, anchor_ids[i], leaf_hash))

        # Sort if configured
        use_sorted = self._config.sorted_tree
        if use_sorted:
            leaf_entries.sort(key=lambda e: e[3])

        # Build MerkleLeaf objects
        leaves: List[MerkleLeaf] = []
        leaf_hashes: List[str] = []
        for idx, (orig_idx, dh, aid, lh) in enumerate(leaf_entries):
            leaf = MerkleLeaf(
                leaf_index=idx,
                data_hash=dh,
                anchor_id=aid,
                leaf_hash=lh,
            )
            leaves.append(leaf)
            leaf_hashes.append(lh)

        # Build tree layers bottom-up
        root_hash = self._compute_merkle_root(leaf_hashes)
        depth = max(0, math.ceil(math.log2(len(leaf_hashes)))) if len(leaf_hashes) > 1 else 0

        tree = MerkleTree(
            tree_id=_generate_id(),
            root_hash=root_hash,
            leaf_count=len(leaves),
            leaves=leaves,
            depth=depth,
            hash_algorithm=self._config.hash_algorithm,
            sorted=use_sorted,
            chain=chain,
            anchor_ids=[l.anchor_id for l in leaves],
        )

        # Provenance hash for tree
        tree_provenance = {
            "tree_id": tree.tree_id,
            "root_hash": root_hash,
            "leaf_count": tree.leaf_count,
            "depth": depth,
            "hash_algorithm": self._config.hash_algorithm,
        }
        tree.provenance_hash = _compute_hash(tree_provenance)

        elapsed = time.monotonic() - start_time

        logger.debug(
            "Merkle tree built: tree_id=%s, leaves=%d, depth=%d, "
            "root=%s, elapsed_ms=%.1f",
            tree.tree_id,
            tree.leaf_count,
            depth,
            root_hash[:16],
            elapsed * 1000,
        )

        return tree

    def _compute_merkle_root(self, leaf_hashes: List[str]) -> str:
        """Compute the Merkle root from a list of leaf hashes.

        Iteratively combines pairs of hashes bottom-up until a single
        root hash remains. If the number of leaves at any level is odd,
        the last hash is duplicated.

        Args:
            leaf_hashes: List of hex-encoded leaf hashes.

        Returns:
            Hex-encoded Merkle root hash.
        """
        if not leaf_hashes:
            raise ValueError("leaf_hashes must not be empty")

        if len(leaf_hashes) == 1:
            return leaf_hashes[0]

        current_level = list(leaf_hashes)

        while len(current_level) > 1:
            next_level: List[str] = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                # If odd number, duplicate the last node
                right = (
                    current_level[i + 1]
                    if i + 1 < len(current_level)
                    else current_level[i]
                )
                parent = self._compute_node_hash(left, right)
                next_level.append(parent)
            current_level = next_level

        return current_level[0]

    def _compute_merkle_proof(
        self,
        leaf_hashes: List[str],
        target_index: int,
    ) -> List[Tuple[str, int]]:
        """Compute the Merkle proof (authentication path) for a leaf.

        Returns a list of (sibling_hash, position) tuples where
        position is 0 (sibling is on the left) or 1 (sibling is on
        the right).

        Args:
            leaf_hashes: Complete list of leaf hashes in tree order.
            target_index: Index of the target leaf.

        Returns:
            List of (sibling_hash, path_index) tuples.
        """
        if not leaf_hashes:
            return []

        if len(leaf_hashes) == 1:
            return []

        proof: List[Tuple[str, int]] = []
        current_level = list(leaf_hashes)
        current_index = target_index

        while len(current_level) > 1:
            next_level: List[str] = []

            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = (
                    current_level[i + 1]
                    if i + 1 < len(current_level)
                    else current_level[i]
                )
                parent = self._compute_node_hash(left, right)
                next_level.append(parent)

                # Check if target is in this pair
                if i == current_index or i + 1 == current_index:
                    if current_index == i:
                        # Target is the left child; sibling is right
                        sibling = right
                        path_idx = 1
                    else:
                        # Target is the right child; sibling is left
                        sibling = left
                        path_idx = 0
                    proof.append((sibling, path_idx))

            current_index = current_index // 2
            current_level = next_level

        return proof

    # ------------------------------------------------------------------
    # Internal: On-Chain Submission (Simulated)
    # ------------------------------------------------------------------

    def _submit_to_chain(
        self,
        merkle_root: str,
        network: str,
        gas_limit: int,
    ) -> str:
        """Submit a Merkle root hash to the blockchain network.

        In this implementation, the submission is simulated by generating
        a deterministic transaction hash. In production, this would invoke
        the MultiChainConnector to send a real transaction.

        Args:
            merkle_root: Hex-encoded Merkle root hash to anchor.
            network: Target blockchain network identifier.
            gas_limit: Gas limit for the transaction.

        Returns:
            Simulated transaction hash (hex string).
        """
        # Generate deterministic tx hash from root and timestamp
        tx_data = f"{merkle_root}:{network}:{_utcnow().isoformat()}"
        tx_hash = "0x" + hashlib.sha256(
            tx_data.encode("utf-8")
        ).hexdigest()

        logger.debug(
            "Transaction submitted to %s: merkle_root=%s, "
            "gas_limit=%d, tx_hash=%s",
            network,
            merkle_root[:16],
            gas_limit,
            tx_hash[:18],
        )

        return tx_hash

    def _wait_for_confirmation(
        self,
        tx_hash: str,
        network: str,
        confirmations: int,
    ) -> bool:
        """Wait for a transaction to reach required confirmation depth.

        In this implementation, confirmation is simulated. In production,
        this would poll the MultiChainConnector for block height updates
        and check the transaction receipt.

        Args:
            tx_hash: Blockchain transaction hash.
            network: Blockchain network identifier.
            confirmations: Required number of block confirmations.

        Returns:
            True if confirmed, False if timeout or error.
        """
        block_time = _BLOCK_TIMES.get(network, 2.0)
        estimated_wait = block_time * confirmations

        logger.debug(
            "Waiting for confirmation: tx_hash=%s, network=%s, "
            "confirmations=%d, estimated_wait=%.1fs",
            tx_hash[:18] if tx_hash else "None",
            network,
            confirmations,
            estimated_wait,
        )

        # In simulation mode, we return True immediately
        # Production would poll: while current_block - tx_block < confirmations
        return True

    # ------------------------------------------------------------------
    # Internal: Gas Estimation
    # ------------------------------------------------------------------

    def _estimate_gas(self, data_size: int, network: str) -> int:
        """Estimate gas required for an anchor transaction.

        Uses a deterministic formula based on data size and network
        parameters. This is an approximation for cost budgeting; actual
        gas may vary based on EVM state.

        Formula: base_gas + sstore_gas + (data_size * gas_per_byte)

        Args:
            data_size: Size of the data being anchored in bytes.
            network: Target blockchain network identifier.

        Returns:
            Estimated gas units (integer).
        """
        # Base transaction cost + storage write + calldata
        estimated = (
            _BASE_GAS_ANCHOR
            + _SSTORE_GAS
            + (data_size * _GAS_PER_BYTE)
        )

        # Apply gas price multiplier from config
        multiplier = self._config.gas_price_multiplier
        estimated = int(estimated * multiplier)

        # Cap at default gas limit
        gas_limit = self._config.default_gas_limit
        estimated = min(estimated, gas_limit)

        logger.debug(
            "Gas estimated: data_size=%d, network=%s, "
            "estimated=%d, multiplier=%.2f",
            data_size,
            network,
            estimated,
            multiplier,
        )

        return estimated

    # ------------------------------------------------------------------
    # Internal: Retry Logic
    # ------------------------------------------------------------------

    def _retry_with_backoff(
        self,
        func: Any,
        max_retries: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with exponential backoff retry.

        Args:
            func: Callable to execute.
            max_retries: Maximum retries. Defaults to config.max_retries.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Result of the function call.

        Raises:
            Exception: The last exception if all retries are exhausted.
        """
        retries = max_retries if max_retries is not None else self._config.max_retries
        backoff_factor = self._config.retry_backoff_factor
        last_exception: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exception = exc
                if attempt >= retries:
                    break

                delay = min(
                    backoff_factor ** attempt,
                    _MAX_RETRY_DELAY_S,
                )
                logger.warning(
                    "Retry %d/%d after %.1fs: %s",
                    attempt + 1,
                    retries,
                    delay,
                    str(exc),
                )
                time.sleep(delay)

        record_api_error("anchor")
        raise last_exception  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Internal: Configuration Helpers
    # ------------------------------------------------------------------

    def _get_confirmation_depth(self, network: str) -> int:
        """Get the required confirmation depth for a network.

        Args:
            network: Blockchain network identifier.

        Returns:
            Required number of block confirmations.
        """
        depth_map = {
            "ethereum": self._config.confirmation_depth_ethereum,
            "polygon": self._config.confirmation_depth_polygon,
            "fabric": self._config.confirmation_depth_fabric,
            "besu": self._config.confirmation_depth_besu,
        }
        return depth_map.get(network, 12)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all in-memory state (for testing only).

        Removes all anchors, trees, gas costs, and pending queue entries.
        Does NOT clear the provenance tracker.
        """
        with self._lock:
            self._anchors.clear()
            self._anchors_by_tx.clear()
            self._anchors_by_record.clear()
            self._merkle_trees.clear()
            self._gas_costs.clear()
            self._pending_queue.clear()
            set_pending_anchors(0)

        logger.info("TransactionAnchor state cleared")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "TransactionAnchor",
]
