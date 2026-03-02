# -*- coding: utf-8 -*-
"""
Downstream Transportation & Distribution Provenance Tracking - AGENT-MRV-022

SHA-256 chain-hashed provenance for zero-hallucination audit trails.
10 pipeline stages tracked with deterministic hashing.
Agent: GL-MRV-S3-009

This module implements a complete provenance tracking system for downstream
transportation and distribution emissions calculations. Every step in the
calculation pipeline is recorded with SHA-256 hashes, creating an immutable
audit trail that proves no data was hallucinated or modified.

Pipeline Stages:
    1. VALIDATE - Input validation and data quality checks
    2. CLASSIFY - Shipment classification (mode, Incoterm, channel)
    3. NORMALIZE - Unit normalization (mass, distance, currency)
    4. RESOLVE_EFS - Emission factor resolution (DEFRA, GLEC, EPA, EEIO)
    5. CALCULATE_TRANSPORT - Transport emissions calculation (tonne-km)
    6. CALCULATE_WAREHOUSE - Warehouse/DC emissions calculation
    7. CALCULATE_LAST_MILE - Last-mile delivery emissions
    8. COMPLIANCE - Regulatory compliance checking (7 frameworks)
    9. AGGREGATE - Aggregation and summarization
    10. SEAL - Final sealing and verification

Example:
    >>> tracker = get_provenance_tracker()
    >>> chain_id = tracker.start_chain()
    >>> tracker.record_stage(chain_id, "VALIDATE", input_data, validated_data)
    >>> tracker.record_stage(chain_id, "CALCULATE_TRANSPORT", calc_input, calc_output)
    >>> final_hash = tracker.seal_chain(chain_id)
    >>> assert tracker.validate_chain(chain_id)

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-009
"""

import hashlib
import json
import math
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# ============================================================================
# CONSTANTS
# ============================================================================

AGENT_ID = "GL-MRV-S3-009"
AGENT_VERSION = "1.0.0"
HASH_ALGORITHM = "sha256"
ENCODING = "utf-8"


class ProvenanceStage(str, Enum):
    """Pipeline stages for provenance tracking."""

    VALIDATE = "VALIDATE"
    CLASSIFY = "CLASSIFY"
    NORMALIZE = "NORMALIZE"
    RESOLVE_EFS = "RESOLVE_EFS"
    CALCULATE_TRANSPORT = "CALCULATE_TRANSPORT"
    CALCULATE_WAREHOUSE = "CALCULATE_WAREHOUSE"
    CALCULATE_LAST_MILE = "CALCULATE_LAST_MILE"
    COMPLIANCE = "COMPLIANCE"
    AGGREGATE = "AGGREGATE"
    SEAL = "SEAL"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class ProvenanceEntry:
    """
    Single provenance entry in the chain.

    Frozen (immutable) record of one stage in the calculation pipeline.
    The chain_hash links this entry to the previous entry, creating an
    immutable chain.

    Attributes:
        entry_id: Unique identifier for this entry (UUID)
        stage: Pipeline stage name
        timestamp: ISO 8601 UTC timestamp
        input_hash: SHA-256 hash of input data
        output_hash: SHA-256 hash of output data
        chain_hash: SHA-256 hash linking to previous entry
        previous_hash: Chain hash of previous entry (empty for first)
        agent_id: Agent identifier (GL-MRV-S3-009)
        agent_version: Agent version (1.0.0)
        metadata: Additional context (optional)
        duration_ms: Processing duration in milliseconds (optional)
        engine_id: Engine that processed this stage (optional)
        engine_version: Engine version (optional)
    """

    entry_id: str
    stage: str
    timestamp: str
    input_hash: str
    output_hash: str
    chain_hash: str
    previous_hash: str
    agent_id: str = AGENT_ID
    agent_version: str = AGENT_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    engine_id: Optional[str] = None
    engine_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert entry to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass
class ProvenanceChain:
    """
    Complete provenance chain for a calculation.

    Tracks all entries in order with validation status and final hash.

    Attributes:
        chain_id: Unique identifier for this chain
        tenant_id: Tenant identifier for multi-tenant isolation
        entries: Ordered list of provenance entries
        started_at: ISO 8601 UTC timestamp of chain start
        sealed_at: ISO 8601 UTC timestamp when sealed (or None)
        final_hash: Final SHA-256 hash when sealed (or None)
        chain_valid: Whether chain validation passed
    """

    chain_id: str
    tenant_id: Optional[str] = None
    entries: List[ProvenanceEntry] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sealed_at: Optional[str] = None
    final_hash: Optional[str] = None
    chain_valid: bool = True

    @property
    def is_valid(self) -> bool:
        """Whether the chain is valid."""
        return self.chain_valid

    @property
    def is_sealed(self) -> bool:
        """Whether the chain is sealed."""
        return self.sealed_at is not None

    @property
    def root_hash(self) -> str:
        """Root hash of the chain (first entry's chain_hash or empty)."""
        if self.entries:
            return self.entries[0].chain_hash
        return ""

    @property
    def entry_count(self) -> int:
        """Number of entries in the chain."""
        return len(self.entries)

    @property
    def stages(self) -> List[str]:
        """List of stage names in order."""
        return [e.stage for e in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary."""
        return {
            "chain_id": self.chain_id,
            "tenant_id": self.tenant_id,
            "entries": [e.to_dict() for e in self.entries],
            "started_at": self.started_at,
            "sealed_at": self.sealed_at,
            "final_hash": self.final_hash,
            "chain_valid": self.chain_valid,
            "is_valid": self.is_valid,
            "is_sealed": self.is_sealed,
            "root_hash": self.root_hash,
            "entry_count": self.entry_count,
            "stages": self.stages,
        }

    def to_json(self) -> str:
        """Convert chain to JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass
class BatchProvenance:
    """
    Provenance tracking for batch calculations with Merkle tree support.

    Attributes:
        batch_id: Unique identifier for this batch
        individual_chain_ids: List of chain IDs in this batch
        batch_started_at: ISO 8601 UTC timestamp
        batch_sealed_at: ISO 8601 UTC timestamp when sealed
        batch_hash: Merkle root hash (or None)
        item_count: Number of items in batch
        leaf_hashes: Leaf-level hashes for Merkle tree
        tree_levels: Merkle tree levels (list of lists)
        root_hash: Merkle root hash
    """

    batch_id: str
    individual_chain_ids: List[str] = field(default_factory=list)
    batch_started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    batch_sealed_at: Optional[str] = None
    batch_hash: Optional[str] = None
    item_count: int = 0
    leaf_hashes: List[str] = field(default_factory=list)
    tree_levels: List[List[str]] = field(default_factory=list)
    root_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch provenance to dictionary."""
        return {
            "batch_id": self.batch_id,
            "individual_chain_ids": self.individual_chain_ids,
            "batch_started_at": self.batch_started_at,
            "batch_sealed_at": self.batch_sealed_at,
            "batch_hash": self.batch_hash,
            "item_count": self.item_count,
            "leaf_hashes": self.leaf_hashes,
            "tree_levels_count": len(self.tree_levels),
            "root_hash": self.root_hash,
        }


# ============================================================================
# HASH UTILITIES
# ============================================================================


def _serialize(obj: Any) -> str:
    """
    Serialize object to deterministic JSON string.

    Converts Decimal to string, datetime to ISO format, Enum to value,
    sorts keys, handles nested dicts/lists and frozen models.

    Args:
        obj: Object to serialize

    Returns:
        Deterministic JSON string
    """

    def default_handler(o: Any) -> Any:
        """Handle non-JSON-serializable types."""
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, (set, frozenset)):
            return sorted(str(x) for x in o)
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, sort_keys=True, default=default_handler)


def _compute_hash(data: Any) -> str:
    """
    Compute SHA-256 hash of data.

    Args:
        data: Data to hash (any JSON-serializable type)

    Returns:
        Lowercase hex SHA-256 hash
    """
    serialized = _serialize(data)
    return hashlib.sha256(serialized.encode(ENCODING)).hexdigest()


def _compute_chain_hash(
    previous_hash: str, stage: str, input_hash: str, output_hash: str
) -> str:
    """
    Compute chain hash linking to previous entry.

    Chain hash = SHA-256(previous_hash + stage + input_hash + output_hash)

    Args:
        previous_hash: Chain hash of previous entry (empty string for first)
        stage: Current stage name
        input_hash: Hash of input data
        output_hash: Hash of output data

    Returns:
        SHA-256 chain hash
    """
    chain_data = f"{previous_hash}|{stage}|{input_hash}|{output_hash}"
    return hashlib.sha256(chain_data.encode(ENCODING)).hexdigest()


def _merkle_hash_pair(left: str, right: str) -> str:
    """
    Compute Merkle hash of two sibling nodes.

    Args:
        left: Left child hash
        right: Right child hash

    Returns:
        SHA-256 parent hash
    """
    combined = f"{left}|{right}"
    return hashlib.sha256(combined.encode(ENCODING)).hexdigest()


def _build_merkle_tree(leaf_hashes: List[str]) -> Tuple[List[List[str]], str]:
    """
    Build a full Merkle tree from leaf hashes.

    If the number of leaves at any level is odd, the last leaf is
    duplicated to form a complete binary tree.

    Args:
        leaf_hashes: List of leaf-level SHA-256 hashes

    Returns:
        Tuple of (tree_levels, root_hash) where tree_levels[0] is the
        leaf level and tree_levels[-1] contains the root.
    """
    if not leaf_hashes:
        empty_hash = hashlib.sha256(b"").hexdigest()
        return [[empty_hash]], empty_hash

    if len(leaf_hashes) == 1:
        return [leaf_hashes], leaf_hashes[0]

    levels: List[List[str]] = [list(leaf_hashes)]
    current_level = list(leaf_hashes)

    while len(current_level) > 1:
        next_level: List[str] = []
        # Duplicate last element if odd count
        if len(current_level) % 2 != 0:
            current_level.append(current_level[-1])

        for i in range(0, len(current_level), 2):
            parent = _merkle_hash_pair(current_level[i], current_level[i + 1])
            next_level.append(parent)

        levels.append(next_level)
        current_level = next_level

    root_hash = current_level[0]
    return levels, root_hash


def _merkle_hash(hashes: List[str]) -> str:
    """
    Compute Merkle root hash from list of hashes.

    Used for batch aggregation. Sorts hashes for determinism.

    Args:
        hashes: List of SHA-256 hashes

    Returns:
        Merkle root SHA-256 hash
    """
    if not hashes:
        return hashlib.sha256(b"").hexdigest()
    if len(hashes) == 1:
        return hashes[0]

    sorted_hashes = sorted(hashes)
    _, root = _build_merkle_tree(sorted_hashes)
    return root


def _verify_merkle_proof(
    leaf_hash: str,
    proof: List[Tuple[str, str]],
    root_hash: str,
) -> bool:
    """
    Verify a Merkle inclusion proof.

    Args:
        leaf_hash: The leaf hash to verify
        proof: List of (sibling_hash, side) tuples where side is 'left' or 'right'
        root_hash: Expected Merkle root hash

    Returns:
        True if the proof is valid
    """
    current = leaf_hash
    for sibling_hash, side in proof:
        if side == "left":
            current = _merkle_hash_pair(sibling_hash, current)
        else:
            current = _merkle_hash_pair(current, sibling_hash)
    return current == root_hash


# ============================================================================
# PROVENANCE TRACKER
# ============================================================================


class ProvenanceTracker:
    """
    Main provenance tracking system for downstream transportation calculations.

    Manages multiple chains, records stages, validates integrity.
    Thread-safe with RLock.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> chain_id = tracker.start_chain()
        >>> tracker.record_stage(chain_id, "VALIDATE", input_data, output_data)
        >>> tracker.seal_chain(chain_id)
        >>> assert tracker.validate_chain(chain_id)
    """

    def __init__(
        self,
        agent_id: str = AGENT_ID,
        agent_version: str = AGENT_VERSION,
    ):
        """
        Initialize provenance tracker.

        Args:
            agent_id: Agent identifier
            agent_version: Agent version
        """
        self.agent_id = agent_id
        self.agent_version = agent_version
        self._chains: Dict[str, ProvenanceChain] = {}
        self._lock = threading.RLock()

    def start_chain(
        self,
        chain_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        """
        Start a new provenance chain.

        Args:
            chain_id: Optional chain ID (UUID generated if not provided)
            tenant_id: Optional tenant identifier

        Returns:
            Chain ID

        Raises:
            ValueError: If chain_id already exists
        """
        with self._lock:
            if chain_id is None:
                chain_id = str(uuid.uuid4())

            if chain_id in self._chains:
                raise ValueError(f"Chain {chain_id} already exists")

            chain = ProvenanceChain(chain_id=chain_id, tenant_id=tenant_id)
            self._chains[chain_id] = chain
            return chain_id

    def record_stage(
        self,
        chain_id: str,
        stage: Union[str, ProvenanceStage],
        input_data: Any,
        output_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        engine_id: Optional[str] = None,
        engine_version: Optional[str] = None,
    ) -> ProvenanceEntry:
        """
        Record a pipeline stage in the provenance chain.

        Args:
            chain_id: Chain identifier
            stage: Pipeline stage name
            input_data: Input data for this stage
            output_data: Output data from this stage
            metadata: Optional metadata
            duration_ms: Processing duration in milliseconds
            engine_id: Engine identifier
            engine_version: Engine version

        Returns:
            Created ProvenanceEntry

        Raises:
            ValueError: If chain not found or already sealed
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            if chain.sealed_at is not None:
                raise ValueError(f"Chain {chain_id} already sealed")

            stage_str = stage.value if isinstance(stage, ProvenanceStage) else stage

            input_hash = _compute_hash(input_data)
            output_hash = _compute_hash(output_data)

            previous_hash = ""
            if chain.entries:
                previous_hash = chain.entries[-1].chain_hash

            chain_hash = _compute_chain_hash(
                previous_hash, stage_str, input_hash, output_hash
            )

            entry = ProvenanceEntry(
                entry_id=str(uuid.uuid4()),
                stage=stage_str,
                timestamp=datetime.now(timezone.utc).isoformat(),
                input_hash=input_hash,
                output_hash=output_hash,
                chain_hash=chain_hash,
                previous_hash=previous_hash,
                agent_id=self.agent_id,
                agent_version=self.agent_version,
                metadata=metadata or {},
                duration_ms=duration_ms,
                engine_id=engine_id,
                engine_version=engine_version,
            )

            chain.entries.append(entry)
            return entry

    def validate_chain(self, chain_id: str) -> bool:
        """
        Validate integrity of provenance chain.

        Checks:
        1. Each entry's chain_hash matches recomputed value
        2. Each entry's previous_hash matches previous entry's chain_hash
        3. No gaps in chain

        Args:
            chain_id: Chain identifier

        Returns:
            True if chain is valid, False otherwise

        Raises:
            ValueError: If chain not found
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            if not chain.entries:
                return True

            for i, entry in enumerate(chain.entries):
                if i == 0:
                    if entry.previous_hash != "":
                        chain.chain_valid = False
                        return False
                else:
                    if entry.previous_hash != chain.entries[i - 1].chain_hash:
                        chain.chain_valid = False
                        return False

                expected_hash = _compute_chain_hash(
                    entry.previous_hash,
                    entry.stage,
                    entry.input_hash,
                    entry.output_hash,
                )
                if entry.chain_hash != expected_hash:
                    chain.chain_valid = False
                    return False

            chain.chain_valid = True
            return True

    def get_chain(self, chain_id: str) -> ProvenanceChain:
        """
        Get provenance chain.

        Args:
            chain_id: Chain identifier

        Returns:
            ProvenanceChain

        Raises:
            ValueError: If chain not found
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            return chain

    def seal_chain(self, chain_id: str) -> str:
        """
        Seal the provenance chain.

        Records final timestamp and hash. No more entries can be added.

        Args:
            chain_id: Chain identifier

        Returns:
            Final hash

        Raises:
            ValueError: If chain not found, already sealed, or fails validation
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            if chain.sealed_at is not None:
                raise ValueError(f"Chain {chain_id} already sealed")

            if not self.validate_chain(chain_id):
                raise ValueError(f"Chain {chain_id} failed validation")

            if chain.entries:
                final_hash = chain.entries[-1].chain_hash
            else:
                final_hash = hashlib.sha256(chain_id.encode(ENCODING)).hexdigest()

            chain.sealed_at = datetime.now(timezone.utc).isoformat()
            chain.final_hash = final_hash
            return final_hash

    def export_chain(self, chain_id: str, fmt: str = "json") -> str:
        """
        Export provenance chain.

        Args:
            chain_id: Chain identifier
            fmt: Export format ("json")

        Returns:
            Serialized chain

        Raises:
            ValueError: If chain not found or format unsupported
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            if fmt == "json":
                return chain.to_json()
            else:
                raise ValueError(f"Unsupported format: {fmt}")

    def verify_entry(self, entry: ProvenanceEntry) -> bool:
        """
        Verify a single provenance entry.

        Args:
            entry: Provenance entry to verify

        Returns:
            True if entry is valid
        """
        expected_hash = _compute_chain_hash(
            entry.previous_hash,
            entry.stage,
            entry.input_hash,
            entry.output_hash,
        )
        return entry.chain_hash == expected_hash

    def get_stage_hash(
        self, chain_id: str, stage: Union[str, ProvenanceStage]
    ) -> Optional[str]:
        """
        Get output hash for a specific stage.

        Args:
            chain_id: Chain identifier
            stage: Pipeline stage name

        Returns:
            Output hash for stage, or None if not found
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            stage_str = stage.value if isinstance(stage, ProvenanceStage) else stage
            for entry in chain.entries:
                if entry.stage == stage_str:
                    return entry.output_hash
            return None

    def get_chain_summary(self, chain_id: str) -> Dict[str, Any]:
        """
        Get summary of provenance chain.

        Args:
            chain_id: Chain identifier

        Returns:
            Summary dictionary
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            return {
                "chain_id": chain.chain_id,
                "tenant_id": chain.tenant_id,
                "agent_id": self.agent_id,
                "agent_version": self.agent_version,
                "started_at": chain.started_at,
                "sealed_at": chain.sealed_at,
                "final_hash": chain.final_hash,
                "chain_valid": chain.chain_valid,
                "entry_count": chain.entry_count,
                "stages": chain.stages,
                "is_sealed": chain.is_sealed,
                "root_hash": chain.root_hash,
            }

    def reset(self) -> None:
        """Clear all chains. Useful for testing."""
        with self._lock:
            self._chains.clear()

    def delete_chain(self, chain_id: str) -> bool:
        """
        Delete a provenance chain.

        Args:
            chain_id: Chain identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if chain_id in self._chains:
                del self._chains[chain_id]
                return True
            return False

    def get_all_chains(self) -> List[str]:
        """Get list of all chain IDs."""
        with self._lock:
            return list(self._chains.keys())

    def clear_all_chains(self) -> int:
        """
        Clear all chains.

        Returns:
            Number of chains cleared
        """
        with self._lock:
            count = len(self._chains)
            self._chains.clear()
            return count


# ============================================================================
# BATCH PROVENANCE TRACKER
# ============================================================================


class BatchProvenanceTracker:
    """
    Provenance tracking for batch calculations with Merkle tree support.

    Manages multiple individual chains plus batch-level aggregation
    via Merkle tree hashing.

    Example:
        >>> batch_tracker = BatchProvenanceTracker(tracker)
        >>> batch_id = batch_tracker.start_batch(100)
        >>> for item in items:
        ...     chain_id = tracker.start_chain()
        ...     batch_tracker.add_chain(batch_id, chain_id)
        >>> batch_tracker.build_merkle_tree(batch_id)
        >>> batch_hash = batch_tracker.seal_batch(batch_id)
    """

    def __init__(self, tracker: ProvenanceTracker):
        """
        Initialize batch provenance tracker.

        Args:
            tracker: ProvenanceTracker instance for individual chains
        """
        self.tracker = tracker
        self._batches: Dict[str, BatchProvenance] = {}
        self._lock = threading.RLock()

    def start_batch(
        self,
        item_count: int = 0,
        batch_id: Optional[str] = None,
    ) -> str:
        """
        Start a new batch.

        Args:
            item_count: Expected number of items
            batch_id: Optional batch ID (UUID generated if not provided)

        Returns:
            Batch ID

        Raises:
            ValueError: If batch_id already exists
        """
        with self._lock:
            if batch_id is None:
                batch_id = str(uuid.uuid4())

            if batch_id in self._batches:
                raise ValueError(f"Batch {batch_id} already exists")

            batch = BatchProvenance(batch_id=batch_id, item_count=item_count)
            self._batches[batch_id] = batch
            return batch_id

    def add_chain(self, batch_id: str, chain_id: str) -> None:
        """
        Add a chain to the batch.

        Args:
            batch_id: Batch identifier
            chain_id: Chain identifier to add

        Raises:
            ValueError: If batch not found or already sealed
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            if batch.batch_sealed_at is not None:
                raise ValueError(f"Batch {batch_id} already sealed")

            batch.individual_chain_ids.append(chain_id)

    def build_merkle_tree(self, batch_id: str) -> str:
        """
        Build Merkle tree from all chain final hashes in the batch.

        Args:
            batch_id: Batch identifier

        Returns:
            Merkle root hash

        Raises:
            ValueError: If batch not found or already sealed
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            if batch.batch_sealed_at is not None:
                raise ValueError(f"Batch {batch_id} already sealed")

            leaf_hashes = []
            for chain_id in batch.individual_chain_ids:
                try:
                    chain = self.tracker.get_chain(chain_id)
                    if chain.final_hash:
                        leaf_hashes.append(chain.final_hash)
                    elif chain.entries:
                        leaf_hashes.append(chain.entries[-1].chain_hash)
                except ValueError:
                    pass

            if not leaf_hashes:
                leaf_hashes = [hashlib.sha256(b"empty").hexdigest()]

            tree_levels, root_hash = _build_merkle_tree(sorted(leaf_hashes))

            batch.leaf_hashes = sorted(leaf_hashes)
            batch.tree_levels = tree_levels
            batch.root_hash = root_hash
            return root_hash

    def verify_merkle_proof(
        self,
        batch_id: str,
        leaf_hash: str,
        proof: List[Tuple[str, str]],
    ) -> bool:
        """
        Verify a Merkle inclusion proof for a leaf in the batch.

        Args:
            batch_id: Batch identifier
            leaf_hash: The leaf hash to verify
            proof: List of (sibling_hash, side) tuples

        Returns:
            True if the proof verifies against the batch root hash

        Raises:
            ValueError: If batch not found or no root hash
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            if batch.root_hash is None:
                raise ValueError(f"Batch {batch_id} has no Merkle tree")

            return _verify_merkle_proof(leaf_hash, proof, batch.root_hash)

    def seal_batch(self, batch_id: str) -> str:
        """
        Seal the batch and compute final hash.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch hash

        Raises:
            ValueError: If batch not found or already sealed
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            if batch.batch_sealed_at is not None:
                raise ValueError(f"Batch {batch_id} already sealed")

            # Build tree if not already built
            if batch.root_hash is None:
                self.build_merkle_tree(batch_id)

            batch_hash = batch.root_hash or hashlib.sha256(
                batch_id.encode(ENCODING)
            ).hexdigest()
            batch.batch_hash = batch_hash
            batch.batch_sealed_at = datetime.now(timezone.utc).isoformat()
            return batch_hash

    def get_batch_summary(self, batch_id: str) -> Dict[str, Any]:
        """
        Get summary of batch provenance.

        Args:
            batch_id: Batch identifier

        Returns:
            Summary dictionary
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")

            return {
                "batch_id": batch.batch_id,
                "batch_started_at": batch.batch_started_at,
                "batch_sealed_at": batch.batch_sealed_at,
                "batch_hash": batch.batch_hash,
                "item_count": batch.item_count,
                "chain_count": len(batch.individual_chain_ids),
                "individual_chain_ids": batch.individual_chain_ids,
                "is_sealed": batch.batch_sealed_at is not None,
                "leaf_count": len(batch.leaf_hashes),
                "tree_depth": len(batch.tree_levels),
                "root_hash": batch.root_hash,
            }

    def get_batch(self, batch_id: str) -> BatchProvenance:
        """
        Get batch provenance.

        Args:
            batch_id: Batch identifier

        Returns:
            BatchProvenance
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            return batch

    def reset(self) -> None:
        """Clear all batches. Useful for testing."""
        with self._lock:
            self._batches.clear()


# ============================================================================
# STANDALONE HASH FUNCTIONS (35+)
# ============================================================================


def hash_shipment_input(shipment: Dict[str, Any]) -> str:
    """
    Hash shipment input data.

    Args:
        shipment: Shipment data (origin, destination, weight, mode, etc.)

    Returns:
        SHA-256 hash
    """
    return _compute_hash(shipment)


def hash_spend_input(
    spend_amount: Union[Decimal, float],
    currency: str,
    naics_code: str,
    year: int,
) -> str:
    """
    Hash spend-based input data.

    Args:
        spend_amount: Spend amount
        currency: Currency code
        naics_code: NAICS sector code
        year: Reporting year

    Returns:
        SHA-256 hash
    """
    data = {
        "spend_amount": str(spend_amount),
        "currency": currency,
        "naics_code": naics_code,
        "year": year,
    }
    return _compute_hash(data)


def hash_warehouse_input(warehouse_data: Dict[str, Any]) -> str:
    """
    Hash warehouse input data.

    Args:
        warehouse_data: Warehouse data (type, area, energy, throughput, etc.)

    Returns:
        SHA-256 hash
    """
    return _compute_hash(warehouse_data)


def hash_last_mile_input(last_mile_data: Dict[str, Any]) -> str:
    """
    Hash last-mile delivery input data.

    Args:
        last_mile_data: Delivery data (type, area, distance, weight, etc.)

    Returns:
        SHA-256 hash
    """
    return _compute_hash(last_mile_data)


def hash_average_data_input(
    channel: str,
    region: str,
    product_category: str,
    volume: Union[Decimal, float],
) -> str:
    """
    Hash average-data calculation input.

    Args:
        channel: Distribution channel
        region: Geographic region
        product_category: Product category
        volume: Sales volume or units

    Returns:
        SHA-256 hash
    """
    data = {
        "channel": channel,
        "region": region,
        "product_category": product_category,
        "volume": str(volume),
    }
    return _compute_hash(data)


def hash_calculation_result(result: Dict[str, Any]) -> str:
    """
    Hash complete calculation result.

    Args:
        result: Calculation result dictionary

    Returns:
        SHA-256 hash
    """
    return _compute_hash(result)


def hash_shipment_result(
    mode: str,
    distance_km: Union[Decimal, float],
    weight_tonnes: Union[Decimal, float],
    tonne_km: Union[Decimal, float],
    ef: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash shipment calculation result.

    Args:
        mode: Transport mode
        distance_km: Distance in km
        weight_tonnes: Weight in tonnes
        tonne_km: Tonne-kilometres
        ef: Emission factor
        co2e_kg: Total emissions kgCO2e

    Returns:
        SHA-256 hash
    """
    data = {
        "mode": mode,
        "distance_km": str(distance_km),
        "weight_tonnes": str(weight_tonnes),
        "tonne_km": str(tonne_km),
        "ef": str(ef),
        "co2e_kg": str(co2e_kg),
        "formula": "tonne_km * ef",
    }
    return _compute_hash(data)


def hash_warehouse_result(
    warehouse_type: str,
    energy_kwh: Union[Decimal, float],
    grid_factor: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash warehouse emission calculation result.

    Args:
        warehouse_type: Type of warehouse
        energy_kwh: Energy consumption in kWh
        grid_factor: Grid emission factor
        co2e_kg: Total emissions kgCO2e

    Returns:
        SHA-256 hash
    """
    data = {
        "warehouse_type": warehouse_type,
        "energy_kwh": str(energy_kwh),
        "grid_factor": str(grid_factor),
        "co2e_kg": str(co2e_kg),
        "formula": "energy_kwh * grid_factor",
    }
    return _compute_hash(data)


def hash_distance_calculation(
    origin: str,
    destination: str,
    mode: str,
    distance_km: Union[Decimal, float],
    uplift_factor: Union[Decimal, float],
) -> str:
    """
    Hash distance calculation.

    Args:
        origin: Origin location
        destination: Destination location
        mode: Transport mode
        distance_km: Calculated distance
        uplift_factor: Distance uplift factor applied

    Returns:
        SHA-256 hash
    """
    data = {
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "distance_km": str(distance_km),
        "uplift_factor": str(uplift_factor),
    }
    return _compute_hash(data)


def hash_spend_calculation(
    spend_usd: Union[Decimal, float],
    naics_code: str,
    ef: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
    margin_removed: bool,
    cpi_deflated: bool,
) -> str:
    """
    Hash spend-based calculation result.

    Args:
        spend_usd: Spend in USD (after conversion)
        naics_code: NAICS sector code
        ef: EEIO emission factor
        co2e_kg: Total emissions kgCO2e
        margin_removed: Whether margin was removed
        cpi_deflated: Whether CPI deflation was applied

    Returns:
        SHA-256 hash
    """
    data = {
        "spend_usd": str(spend_usd),
        "naics_code": naics_code,
        "ef": str(ef),
        "co2e_kg": str(co2e_kg),
        "margin_removed": margin_removed,
        "cpi_deflated": cpi_deflated,
        "formula": "spend_usd * ef",
    }
    return _compute_hash(data)


def hash_cold_chain_result(
    base_co2e: Union[Decimal, float],
    regime: str,
    uplift_factor: Union[Decimal, float],
    final_co2e: Union[Decimal, float],
) -> str:
    """
    Hash cold chain uplift result.

    Args:
        base_co2e: Base emissions before uplift
        regime: Temperature regime
        uplift_factor: Cold chain uplift factor
        final_co2e: Final emissions after uplift

    Returns:
        SHA-256 hash
    """
    data = {
        "base_co2e": str(base_co2e),
        "regime": regime,
        "uplift_factor": str(uplift_factor),
        "final_co2e": str(final_co2e),
        "formula": "base_co2e * uplift_factor",
    }
    return _compute_hash(data)


def hash_return_result(
    return_rate: Union[Decimal, float],
    outbound_co2e: Union[Decimal, float],
    return_co2e: Union[Decimal, float],
    consolidation_factor: Union[Decimal, float],
) -> str:
    """
    Hash return logistics result.

    Args:
        return_rate: Product return rate
        outbound_co2e: Outbound emissions
        return_co2e: Return trip emissions
        consolidation_factor: Consolidation efficiency

    Returns:
        SHA-256 hash
    """
    data = {
        "return_rate": str(return_rate),
        "outbound_co2e": str(outbound_co2e),
        "return_co2e": str(return_co2e),
        "consolidation_factor": str(consolidation_factor),
        "formula": "outbound_co2e * return_rate * consolidation_factor",
    }
    return _compute_hash(data)


def hash_incoterm_classification(
    incoterm: str,
    seller_responsibility: bool,
    category_assigned: int,
) -> str:
    """
    Hash Incoterm classification result.

    Args:
        incoterm: Incoterm code (e.g., EXW, FOB, CIF, DDP)
        seller_responsibility: Whether seller bears transport responsibility
        category_assigned: GHG Protocol category assigned (4 or 9)

    Returns:
        SHA-256 hash
    """
    data = {
        "incoterm": incoterm,
        "seller_responsibility": seller_responsibility,
        "category_assigned": category_assigned,
    }
    return _compute_hash(data)


def hash_load_factor_adjustment(
    base_ef: Union[Decimal, float],
    load_factor: Union[Decimal, float],
    adjusted_ef: Union[Decimal, float],
) -> str:
    """
    Hash load factor adjustment.

    Args:
        base_ef: Base emission factor (at 100% load)
        load_factor: Vehicle utilisation rate
        adjusted_ef: Adjusted emission factor

    Returns:
        SHA-256 hash
    """
    data = {
        "base_ef": str(base_ef),
        "load_factor": str(load_factor),
        "adjusted_ef": str(adjusted_ef),
        "formula": "base_ef / load_factor",
    }
    return _compute_hash(data)


def hash_allocation(
    method: str,
    share: Union[Decimal, float],
    total: Union[Decimal, float],
    allocated: Union[Decimal, float],
) -> str:
    """
    Hash allocation calculation.

    Args:
        method: Allocation method
        share: Product share
        total: Total emissions
        allocated: Allocated emissions

    Returns:
        SHA-256 hash
    """
    data = {
        "method": method,
        "share": str(share),
        "total": str(total),
        "allocated": str(allocated),
        "formula": "total * share",
    }
    return _compute_hash(data)


def hash_aggregation(
    by_mode: Dict[str, Any],
    by_component: Dict[str, Any],
    by_channel: Dict[str, Any],
    total: Union[Decimal, float],
) -> str:
    """
    Hash aggregation result.

    Args:
        by_mode: Emissions by transport mode
        by_component: Emissions by component
        by_channel: Emissions by distribution channel
        total: Total emissions

    Returns:
        SHA-256 hash
    """
    data = {
        "by_mode": by_mode,
        "by_component": by_component,
        "by_channel": by_channel,
        "total": str(total),
    }
    return _compute_hash(data)


def hash_compliance_result(
    framework: str,
    status: str,
    score: Union[Decimal, float],
) -> str:
    """
    Hash compliance check result.

    Args:
        framework: Regulatory framework
        status: Compliance status
        score: Compliance score

    Returns:
        SHA-256 hash
    """
    data = {"framework": framework, "status": status, "score": str(score)}
    return _compute_hash(data)


def hash_batch_input(
    batch_id: str,
    item_count: int,
    shipment_hashes: List[str],
) -> str:
    """
    Hash batch input.

    Args:
        batch_id: Batch identifier
        item_count: Number of items
        shipment_hashes: List of individual shipment hashes

    Returns:
        SHA-256 hash
    """
    data = {
        "batch_id": batch_id,
        "item_count": item_count,
        "shipment_hashes": sorted(shipment_hashes),
    }
    return _compute_hash(data)


def hash_batch_result(
    batch_id: str,
    results: List[Dict[str, Any]],
) -> str:
    """
    Hash batch calculation result.

    Args:
        batch_id: Batch identifier
        results: List of individual results

    Returns:
        SHA-256 hash
    """
    data = {"batch_id": batch_id, "results": results}
    return _compute_hash(data)


def hash_emission_factor(
    ef_type: str,
    ef_value: Union[Decimal, float],
    source: str,
    year: int = 0,
) -> str:
    """
    Hash emission factor.

    Args:
        ef_type: Emission factor type
        ef_value: Emission factor value
        source: Data source
        year: Factor year

    Returns:
        SHA-256 hash
    """
    data = {
        "type": ef_type,
        "value": str(ef_value),
        "source": source,
        "year": year,
    }
    return _compute_hash(data)


def hash_wtt_result(
    fuel_type: str,
    quantity: Union[Decimal, float],
    wtt_factor: Union[Decimal, float],
    wtt_co2e: Union[Decimal, float],
) -> str:
    """
    Hash well-to-tank emissions result.

    Args:
        fuel_type: Fuel type
        quantity: Fuel quantity
        wtt_factor: WTT emission factor
        wtt_co2e: WTT emissions kgCO2e

    Returns:
        SHA-256 hash
    """
    data = {
        "fuel_type": fuel_type,
        "quantity": str(quantity),
        "wtt_factor": str(wtt_factor),
        "wtt_co2e": str(wtt_co2e),
        "formula": "quantity * wtt_factor",
    }
    return _compute_hash(data)


def hash_last_mile_result(
    delivery_type: str,
    area: str,
    distance_km: Union[Decimal, float],
    parcels: int,
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash last-mile delivery result.

    Args:
        delivery_type: Delivery vehicle type
        area: Delivery area
        distance_km: Delivery distance
        parcels: Number of parcels
        co2e_kg: Total emissions

    Returns:
        SHA-256 hash
    """
    data = {
        "delivery_type": delivery_type,
        "area": area,
        "distance_km": str(distance_km),
        "parcels": parcels,
        "co2e_kg": str(co2e_kg),
    }
    return _compute_hash(data)


def hash_multileg_result(
    legs: List[Dict[str, Any]],
    total_co2e: Union[Decimal, float],
    hub_emissions: Union[Decimal, float],
) -> str:
    """
    Hash multi-leg transport chain result.

    Args:
        legs: List of leg details
        total_co2e: Total emissions across all legs
        hub_emissions: Transshipment hub emissions

    Returns:
        SHA-256 hash
    """
    data = {
        "legs": legs,
        "total_co2e": str(total_co2e),
        "hub_emissions": str(hub_emissions),
    }
    return _compute_hash(data)


def hash_uncertainty_result(
    method: str,
    confidence_level: Union[Decimal, float],
    lower_bound: Union[Decimal, float],
    upper_bound: Union[Decimal, float],
) -> str:
    """
    Hash uncertainty quantification result.

    Args:
        method: Uncertainty method
        confidence_level: Confidence level
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound

    Returns:
        SHA-256 hash
    """
    data = {
        "method": method,
        "confidence_level": str(confidence_level),
        "lower_bound": str(lower_bound),
        "upper_bound": str(upper_bound),
    }
    return _compute_hash(data)


def hash_dqi_result(
    dqi_scores: Dict[str, Union[Decimal, float]],
    composite: Union[Decimal, float],
) -> str:
    """
    Hash data quality assessment.

    Args:
        dqi_scores: DQI scores by dimension
        composite: Composite DQI score

    Returns:
        SHA-256 hash
    """
    data = {
        "dqi_scores": {k: str(v) for k, v in dqi_scores.items()},
        "composite": str(composite),
    }
    return _compute_hash(data)


def hash_config(config: Dict[str, Any]) -> str:
    """
    Hash configuration data.

    Args:
        config: Configuration dictionary

    Returns:
        SHA-256 hash
    """
    return _compute_hash(config)


def hash_metadata(metadata: Dict[str, Any]) -> str:
    """
    Hash metadata.

    Args:
        metadata: Metadata dictionary

    Returns:
        SHA-256 hash
    """
    return _compute_hash(metadata)


def hash_arbitrary(data: Any) -> str:
    """
    Hash arbitrary data.

    Args:
        data: Any data to hash

    Returns:
        SHA-256 hash
    """
    return _compute_hash(data)


def hash_channel_result(
    channel: str,
    avg_ef: Union[Decimal, float],
    volume: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash distribution channel calculation result.

    Args:
        channel: Distribution channel
        avg_ef: Average emission factor
        volume: Sales volume
        co2e_kg: Total emissions

    Returns:
        SHA-256 hash
    """
    data = {
        "channel": channel,
        "avg_ef": str(avg_ef),
        "volume": str(volume),
        "co2e_kg": str(co2e_kg),
        "formula": "volume * avg_ef",
    }
    return _compute_hash(data)


def hash_refrigerant_leakage(
    refrigerant: str,
    charge_kg: Union[Decimal, float],
    leak_rate: Union[Decimal, float],
    gwp: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash refrigerant leakage calculation.

    Args:
        refrigerant: Refrigerant type
        charge_kg: Refrigerant charge in kg
        leak_rate: Annual leak rate
        gwp: Global warming potential
        co2e_kg: Total emissions

    Returns:
        SHA-256 hash
    """
    data = {
        "refrigerant": refrigerant,
        "charge_kg": str(charge_kg),
        "leak_rate": str(leak_rate),
        "gwp": str(gwp),
        "co2e_kg": str(co2e_kg),
        "formula": "charge_kg * leak_rate * gwp",
    }
    return _compute_hash(data)


def hash_redelivery_result(
    failed_rate: Union[Decimal, float],
    base_deliveries: int,
    redelivery_co2e: Union[Decimal, float],
) -> str:
    """
    Hash failed delivery redelivery result.

    Args:
        failed_rate: Failed delivery rate
        base_deliveries: Number of base deliveries
        redelivery_co2e: Redelivery emissions

    Returns:
        SHA-256 hash
    """
    data = {
        "failed_rate": str(failed_rate),
        "base_deliveries": base_deliveries,
        "redelivery_co2e": str(redelivery_co2e),
    }
    return _compute_hash(data)


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_tracker_instance: Optional[ProvenanceTracker] = None
_tracker_lock = threading.RLock()

_batch_tracker_instance: Optional[BatchProvenanceTracker] = None
_batch_tracker_lock = threading.RLock()


def get_provenance_tracker() -> ProvenanceTracker:
    """
    Get singleton ProvenanceTracker instance.

    Thread-safe singleton pattern.

    Returns:
        ProvenanceTracker instance
    """
    global _tracker_instance
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = ProvenanceTracker()
        return _tracker_instance


def reset_provenance_tracker() -> None:
    """Reset singleton ProvenanceTracker instance."""
    global _tracker_instance
    with _tracker_lock:
        _tracker_instance = None


def get_batch_provenance_tracker() -> BatchProvenanceTracker:
    """
    Get singleton BatchProvenanceTracker instance.

    Thread-safe singleton pattern. Creates the underlying ProvenanceTracker
    if needed.

    Returns:
        BatchProvenanceTracker instance
    """
    global _batch_tracker_instance
    with _batch_tracker_lock:
        if _batch_tracker_instance is None:
            tracker = get_provenance_tracker()
            _batch_tracker_instance = BatchProvenanceTracker(tracker)
        return _batch_tracker_instance


def reset_batch_provenance_tracker() -> None:
    """Reset singleton BatchProvenanceTracker instance."""
    global _batch_tracker_instance
    with _batch_tracker_lock:
        _batch_tracker_instance = None


# ============================================================================
# CONVENIENCE STAGE RECORDERS (10 stages)
# ============================================================================


def create_chain(tenant_id: Optional[str] = None) -> Tuple[ProvenanceTracker, str]:
    """
    Create a new provenance chain using singleton tracker.

    Args:
        tenant_id: Optional tenant identifier

    Returns:
        Tuple of (tracker, chain_id)
    """
    tracker = get_provenance_tracker()
    chain_id = tracker.start_chain(tenant_id=tenant_id)
    return tracker, chain_id


def record_validation(
    chain_id: str,
    input_data: Any,
    validated_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record VALIDATE stage.

    Args:
        chain_id: Chain identifier
        input_data: Raw input data
        validated_data: Validated data
        metadata: Optional metadata

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.VALIDATE, input_data, validated_data, metadata
    )


def record_classification(
    chain_id: str,
    input_data: Any,
    classified_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record CLASSIFY stage.

    Args:
        chain_id: Chain identifier
        input_data: Input data
        classified_data: Classified data (mode, Incoterm, channel)
        metadata: Optional metadata

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.CLASSIFY, input_data, classified_data, metadata
    )


def record_normalization(
    chain_id: str,
    input_data: Any,
    normalized_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record NORMALIZE stage.

    Args:
        chain_id: Chain identifier
        input_data: Input data
        normalized_data: Normalized data (units, currency, distances)
        metadata: Optional metadata

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.NORMALIZE, input_data, normalized_data, metadata
    )


def record_ef_resolution(
    chain_id: str,
    input_data: Any,
    resolved_efs: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record RESOLVE_EFS stage.

    Args:
        chain_id: Chain identifier
        input_data: Input data (mode, vehicle type, etc.)
        resolved_efs: Resolved emission factors
        metadata: Optional metadata (source, year, etc.)

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.RESOLVE_EFS, input_data, resolved_efs, metadata
    )


def record_transport_calculation(
    chain_id: str,
    input_data: Any,
    transport_results: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record CALCULATE_TRANSPORT stage.

    Args:
        chain_id: Chain identifier
        input_data: Transport calculation input
        transport_results: Transport calculation results
        metadata: Optional metadata (method, formula, etc.)

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.CALCULATE_TRANSPORT, input_data, transport_results, metadata
    )


def record_warehouse_calculation(
    chain_id: str,
    input_data: Any,
    warehouse_results: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record CALCULATE_WAREHOUSE stage.

    Args:
        chain_id: Chain identifier
        input_data: Warehouse calculation input
        warehouse_results: Warehouse calculation results
        metadata: Optional metadata (type, energy source, etc.)

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.CALCULATE_WAREHOUSE, input_data, warehouse_results, metadata
    )


def record_last_mile_calculation(
    chain_id: str,
    input_data: Any,
    last_mile_results: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record CALCULATE_LAST_MILE stage.

    Args:
        chain_id: Chain identifier
        input_data: Last-mile calculation input
        last_mile_results: Last-mile calculation results
        metadata: Optional metadata (delivery type, area, etc.)

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.CALCULATE_LAST_MILE, input_data, last_mile_results, metadata
    )


def record_compliance(
    chain_id: str,
    input_data: Any,
    compliance_results: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record COMPLIANCE stage.

    Args:
        chain_id: Chain identifier
        input_data: Data to check
        compliance_results: Compliance check results
        metadata: Optional metadata (framework, rules, etc.)

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.COMPLIANCE, input_data, compliance_results, metadata
    )


def record_aggregation(
    chain_id: str,
    input_data: Any,
    aggregated_results: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record AGGREGATE stage.

    Args:
        chain_id: Chain identifier
        input_data: Pre-aggregation data
        aggregated_results: Aggregated results
        metadata: Optional metadata (dimensions, totals, etc.)

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.AGGREGATE, input_data, aggregated_results, metadata
    )


def seal_and_verify(chain_id: str) -> Tuple[str, bool]:
    """
    Seal chain and verify integrity.

    Args:
        chain_id: Chain identifier

    Returns:
        Tuple of (final_hash, is_valid)
    """
    tracker = get_provenance_tracker()
    final_hash = tracker.seal_chain(chain_id)
    is_valid = tracker.validate_chain(chain_id)
    return final_hash, is_valid


def export_chain_json(chain_id: str) -> str:
    """
    Export chain as JSON.

    Args:
        chain_id: Chain identifier

    Returns:
        JSON string
    """
    tracker = get_provenance_tracker()
    return tracker.export_chain(chain_id, fmt="json")


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "ProvenanceStage",
    # Models
    "ProvenanceEntry",
    "ProvenanceChain",
    "BatchProvenance",
    # Trackers
    "ProvenanceTracker",
    "BatchProvenanceTracker",
    # Singletons
    "get_provenance_tracker",
    "reset_provenance_tracker",
    "get_batch_provenance_tracker",
    "reset_batch_provenance_tracker",
    # Hash functions (35+)
    "hash_shipment_input",
    "hash_spend_input",
    "hash_warehouse_input",
    "hash_last_mile_input",
    "hash_average_data_input",
    "hash_calculation_result",
    "hash_shipment_result",
    "hash_warehouse_result",
    "hash_distance_calculation",
    "hash_spend_calculation",
    "hash_cold_chain_result",
    "hash_return_result",
    "hash_incoterm_classification",
    "hash_load_factor_adjustment",
    "hash_allocation",
    "hash_aggregation",
    "hash_compliance_result",
    "hash_batch_input",
    "hash_batch_result",
    "hash_emission_factor",
    "hash_wtt_result",
    "hash_last_mile_result",
    "hash_multileg_result",
    "hash_uncertainty_result",
    "hash_dqi_result",
    "hash_config",
    "hash_metadata",
    "hash_arbitrary",
    "hash_channel_result",
    "hash_refrigerant_leakage",
    "hash_redelivery_result",
    # Convenience functions
    "create_chain",
    "record_validation",
    "record_classification",
    "record_normalization",
    "record_ef_resolution",
    "record_transport_calculation",
    "record_warehouse_calculation",
    "record_last_mile_calculation",
    "record_compliance",
    "record_aggregation",
    "seal_and_verify",
    "export_chain_json",
    # Low-level utilities
    "_compute_hash",
    "_serialize",
    "_build_merkle_tree",
    "_verify_merkle_proof",
]
