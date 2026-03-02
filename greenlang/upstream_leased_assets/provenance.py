# -*- coding: utf-8 -*-
"""
Provenance Tracking - Upstream Leased Assets Agent (AGENT-MRV-021)

SHA-256 chain hashing for complete audit trail of every calculation.
Supports single-entry chain hashing and batch Merkle-tree aggregation.

Zero-Hallucination Guarantees:
    - Every calculation stage produces a deterministic SHA-256 hash
    - Chain hashing links each stage to the previous for tamper detection
    - Merkle tree aggregation for batch provenance verification
    - All hashing uses hashlib.sha256 with canonical JSON serialization

Pipeline Stages:
    1. VALIDATE - Input validation and data quality checks
    2. CLASSIFY - Lease classification (asset type, IFRS 16/ASC 842)
    3. NORMALIZE - Unit normalization (energy, currency, area, time)
    4. RESOLVE_EFS - Emission factor resolution (DEFRA, EPA, IEA, EEIO)
    5. CALCULATE - Leased asset emissions calculation
    6. ALLOCATE - Allocation of shared-space/fleet emissions
    7. AGGREGATE - Aggregation and summarization by asset type
    8. COMPLIANCE - Regulatory compliance checking
    9. PROVENANCE - Provenance metadata assembly and DQI scoring
    10. SEAL - Final sealing and verification

Example:
    >>> tracker = get_provenance_tracker()
    >>> chain = tracker.start_chain("tenant-001", {"source": "erp"})
    >>> tracker.record_stage(
    ...     chain.chain_id, ProvenanceStage.VALIDATE,
    ...     input_data, validated_data,
    ...     engine_id="validation-engine", engine_version="1.0.0"
    ... )
    >>> final_hash = tracker.seal_chain(chain.chain_id)
    >>> valid, errors = tracker.validate_chain(chain.chain_id)
    >>> assert valid and not errors

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-021 Upstream Leased Assets (GL-MRV-S3-008)
Status: Production Ready
"""

import hashlib
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# ============================================================================
# CONSTANTS
# ============================================================================

AGENT_ID = "GL-MRV-S3-008"
AGENT_VERSION = "1.0.0"
HASH_ALGORITHM = "sha256"
ENCODING = "utf-8"
GENESIS_HASH = "0" * 64  # Sentinel for chain start


# ============================================================================
# PROVENANCE STAGE ENUM
# ============================================================================


class ProvenanceStage(str, Enum):
    """
    Pipeline stages for upstream leased assets provenance tracking.

    Each stage corresponds to a distinct processing step in the
    upstream leased assets emissions calculation pipeline. The stages
    are ordered to reflect the typical processing flow from raw
    input validation through final sealing.

    Stages:
        VALIDATE: Input validation and data quality checks. Ensures
            all required fields are present, types are correct, and
            values are within acceptable ranges for leased asset data
            (floor area, energy consumption, lease terms, etc.).
        CLASSIFY: Lease classification. Determines asset type
            (building, vehicle, equipment, IT asset), lease type
            (operating vs finance per IFRS 16/ASC 842), and whether
            short-term/low-value exemptions apply.
        NORMALIZE: Unit normalization. Converts energy to kWh,
            currencies to reporting currency (USD), areas to sqm,
            fuel volumes to standard units, and prorate partial-year
            leases to annual equivalents.
        RESOLVE_EFS: Emission factor resolution. Selects appropriate
            emission factors from DEFRA, EPA eGRID, IEA, Energy Star
            benchmarks, or EEIO tables based on asset type, energy
            source, region, and building type.
        CALCULATE: Leased asset emissions calculation. Applies
            asset-specific, average-data, spend-based, or
            lessor-specific methods to compute CO2e for each
            leased asset.
        ALLOCATE: Allocation of shared-space/fleet emissions.
            Applies floor-area, headcount, FTE, or custom allocation
            methods to distribute shared building or fleet vehicle
            emissions to the reporting entity's leased portion.
        AGGREGATE: Aggregation and summarization. Computes totals by
            asset type, building type, region, calculation method,
            and other dimensions.
        COMPLIANCE: Regulatory compliance checking. Validates results
            against GHG Protocol Scope 3, ISO 14064, CSRD, CDP, SBTi,
            SB 253, and GRI frameworks.
        PROVENANCE: Provenance metadata assembly. Compiles data quality
            indicator (DQI) scores, uncertainty bounds, emission factor
            source citations, and methodology documentation.
        SEAL: Final sealing and verification. Computes Merkle root hash
            and marks the chain as immutable.
    """

    VALIDATE = "VALIDATE"
    CLASSIFY = "CLASSIFY"
    NORMALIZE = "NORMALIZE"
    RESOLVE_EFS = "RESOLVE_EFS"
    CALCULATE = "CALCULATE"
    ALLOCATE = "ALLOCATE"
    AGGREGATE = "AGGREGATE"
    COMPLIANCE = "COMPLIANCE"
    PROVENANCE = "PROVENANCE"
    SEAL = "SEAL"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class ProvenanceEntry:
    """
    Single provenance entry in the chain.

    This is a frozen (immutable) record of one stage in the calculation
    pipeline. The chain_hash links this entry to the previous entry,
    creating an immutable chain.

    The entry captures:
        - What went in (input_hash)
        - What came out (output_hash)
        - How it links to the previous step (chain_hash via previous_hash)
        - When it happened (timestamp)
        - Which engine produced it (engine_id, engine_version)
        - How long it took (duration_ms)
        - Optional context (metadata)

    Chain Hash Formula:
        chain_hash = SHA-256(previous_chain_hash | stage | input_hash | output_hash)

    Attributes:
        stage: Pipeline stage name (one of ProvenanceStage values)
        timestamp: ISO 8601 UTC timestamp of when this entry was created
        input_hash: SHA-256 hash of input data for this stage
        output_hash: SHA-256 hash of output data from this stage
        chain_hash: SHA-256 hash linking to previous entry, computed as
                    SHA-256(previous_hash|stage|input_hash|output_hash)
        metadata: Additional context (optional). May include calculation
                  method, emission factor source, uncertainty bounds, etc.
        duration_ms: Processing time for this stage in milliseconds
        engine_id: Identifier of the engine that performed this stage
        engine_version: Version of the engine that performed this stage
    """

    stage: ProvenanceStage
    timestamp: str
    input_hash: str
    output_hash: str
    chain_hash: str
    metadata: Dict[str, Any]
    duration_ms: float
    engine_id: str
    engine_version: str

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entry to dictionary.

        Returns:
            Dictionary representation of this entry with all fields.
            The stage field is converted to its string value.
        """
        result = asdict(self)
        # Ensure stage is serialized as its string value
        if isinstance(result.get("stage"), Enum):
            result["stage"] = result["stage"].value
        return result

    def to_json(self) -> str:
        """
        Convert entry to deterministic JSON string.

        Returns:
            JSON string with sorted keys for deterministic output.
        """
        return _canonical_json(self.to_dict())


@dataclass
class ProvenanceChain:
    """
    Complete provenance chain for a calculation.

    Tracks all entries in order, with validation status and final hash
    when sealed. Once sealed, no more entries can be added.

    The chain is mutable internally to allow appending entries and
    sealing. The ProvenanceTracker manages all mutations under lock.

    Attributes:
        chain_id: Unique identifier for this chain (UUID)
        tenant_id: Tenant identifier for multi-tenancy isolation
        agent_id: Agent identifier (GL-MRV-S3-008 for upstream leased assets)
        entries: Ordered list of provenance entries in pipeline order
        started_at: ISO 8601 UTC timestamp of chain start
        sealed_at: ISO 8601 UTC timestamp when sealed (or None)
        final_hash: Final SHA-256 hash when sealed (or None)
        is_sealed: Whether the chain has been sealed (no more additions)
    """

    chain_id: str
    tenant_id: str
    agent_id: str = AGENT_ID
    entries: List[ProvenanceEntry] = field(default_factory=list)
    started_at: str = ""
    sealed_at: Optional[str] = None
    final_hash: Optional[str] = None
    is_sealed: bool = False

    @property
    def root_hash(self) -> str:
        """
        Root hash of the chain (first entry's chain_hash or empty).

        Returns:
            Chain hash of the first entry, or empty string if no entries.
        """
        if self.entries:
            return self.entries[0].chain_hash
        return ""

    @property
    def last_hash(self) -> str:
        """
        Last chain hash in the chain.

        Returns:
            Chain hash of the last entry, or empty string if no entries.
        """
        if self.entries:
            return self.entries[-1].chain_hash
        return ""

    @property
    def entry_count(self) -> int:
        """
        Number of entries in the chain.

        Returns:
            Integer count of entries.
        """
        return len(self.entries)

    @property
    def stages_recorded(self) -> List[str]:
        """
        List of stage names recorded in this chain.

        Returns:
            List of stage name strings in order.
        """
        return [
            e.stage.value if isinstance(e.stage, ProvenanceStage) else e.stage
            for e in self.entries
        ]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chain to dictionary.

        Returns:
            Dictionary representation of the complete chain with all
            entries expanded.
        """
        return {
            "chain_id": self.chain_id,
            "tenant_id": self.tenant_id,
            "agent_id": self.agent_id,
            "entries": [e.to_dict() for e in self.entries],
            "started_at": self.started_at,
            "sealed_at": self.sealed_at,
            "final_hash": self.final_hash,
            "is_sealed": self.is_sealed,
            "entry_count": self.entry_count,
            "root_hash": self.root_hash,
            "last_hash": self.last_hash,
            "stages_recorded": self.stages_recorded,
        }

    def to_json(self) -> str:
        """
        Convert chain to deterministic JSON string.

        Returns:
            JSON string with sorted keys and indentation.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)


# ============================================================================
# HASH UTILITIES
# ============================================================================


def _canonical_json(obj: Any) -> str:
    """
    Serialize object to canonical (deterministic) JSON string.

    Converts Decimal to string, datetime to ISO format, Enum to value,
    sorts keys, and handles nested dicts/lists and frozen models. This
    is the foundation of all provenance hashing. Deterministic
    serialization ensures that the same input always produces the same
    hash, regardless of dict ordering or Python object representation.

    Args:
        obj: Object to serialize. Supports dict, list, str, int, float,
             Decimal, datetime, Enum, dataclasses with to_dict(), and
             any object with __dict__.

    Returns:
        Deterministic JSON string with sorted keys.

    Example:
        >>> _canonical_json({"b": 1, "a": 2})
        '{"a": 2, "b": 1}'
        >>> _canonical_json(Decimal("123.456"))
        '"123.456"'
    """

    def default_handler(o: Any) -> Any:
        """Handle non-JSON-serializable types."""
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, sort_keys=True, default=default_handler)


def _sha256(data: str) -> str:
    """
    Compute SHA-256 hash of a UTF-8 string.

    This is the lowest-level hashing primitive. All other hash functions
    ultimately call this method.

    Args:
        data: UTF-8 string to hash.

    Returns:
        Lowercase hex SHA-256 hash (64 characters).

    Example:
        >>> _sha256("hello")
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    return hashlib.sha256(data.encode(ENCODING)).hexdigest()


def _serialize(obj: Any) -> str:
    """
    Serialize object to deterministic JSON string.

    Alias for _canonical_json for backward compatibility and convenience.

    Args:
        obj: Object to serialize.

    Returns:
        Deterministic JSON string with sorted keys.
    """
    return _canonical_json(obj)


def _compute_hash(data: Any) -> str:
    """
    Compute SHA-256 hash of data.

    Serializes the data to a deterministic JSON string via _canonical_json,
    then computes the SHA-256 hash of the UTF-8 encoded string.

    Args:
        data: Data to hash (any JSON-serializable type, or any type
              supported by _canonical_json).

    Returns:
        Lowercase hex SHA-256 hash (64 characters).

    Example:
        >>> _compute_hash({"asset_type": "building", "floor_area_sqm": 5000.0})
        'a1b2c3...'  # 64-character hex string
    """
    serialized = _canonical_json(data)
    return _sha256(serialized)


def _compute_chain_hash(
    previous_hash: str, stage: str, input_hash: str, output_hash: str
) -> str:
    """
    Compute chain hash linking to previous entry.

    Chain hash = SHA-256(previous_hash|stage|input_hash|output_hash)

    The pipe-delimited concatenation ensures that the chain hash depends
    on all four components, preventing substitution attacks where one
    component is swapped with another.

    Args:
        previous_hash: Chain hash of previous entry (empty string for first)
        stage: Current stage name (e.g., "VALIDATE", "CALCULATE")
        input_hash: SHA-256 hash of input data
        output_hash: SHA-256 hash of output data

    Returns:
        SHA-256 chain hash (64-character hex string)

    Example:
        >>> _compute_chain_hash("", "VALIDATE", "abc123...", "def456...")
        'f1e2d3...'  # 64-character hex string
    """
    chain_data = f"{previous_hash}|{stage}|{input_hash}|{output_hash}"
    return _sha256(chain_data)


def _merkle_hash(hashes: List[str]) -> str:
    """
    Compute Merkle-tree root hash of a list of hashes.

    Implements a binary Merkle tree. If the number of hashes at any level
    is odd, the last hash is duplicated. Leaf nodes are the input hashes,
    and each parent node is SHA-256(left_child|right_child).

    For batch aggregation and chain sealing. The tree structure provides
    efficient proof-of-inclusion for any individual chain within a batch.

    Args:
        hashes: List of SHA-256 hashes to aggregate.

    Returns:
        Merkle root SHA-256 hash (64-character hex string).

    Example:
        >>> _merkle_hash(["abc123...", "def456..."])
        '789xyz...'  # 64-character hex string
        >>> _merkle_hash([])
        'e3b0c44...'  # SHA-256 of empty bytes
    """
    if not hashes:
        return hashlib.sha256(b"").hexdigest()
    if len(hashes) == 1:
        return hashes[0]

    # Sort for determinism before building tree
    current_level = sorted(hashes)

    while len(current_level) > 1:
        next_level: List[str] = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            # Duplicate last if odd
            right = current_level[i + 1] if i + 1 < len(current_level) else left
            combined = f"{left}|{right}"
            next_level.append(_sha256(combined))
        current_level = next_level

    return current_level[0]


def _merkle_proof(hashes: List[str], target_index: int) -> List[Tuple[str, str]]:
    """
    Build a Merkle proof for the hash at target_index.

    The proof is a list of (sibling_hash, direction) tuples where
    direction is 'L' or 'R' indicating which side the sibling is on.

    Args:
        hashes: Sorted list of leaf hashes used to build the tree.
        target_index: Index of the target hash in the sorted list.

    Returns:
        List of (sibling_hash, direction) tuples forming the proof path.

    Raises:
        ValueError: If target_index is out of range.
    """
    if not hashes:
        return []
    if target_index < 0 or target_index >= len(hashes):
        raise ValueError(f"target_index {target_index} out of range [0, {len(hashes)})")

    current_level = list(hashes)
    proof: List[Tuple[str, str]] = []
    idx = target_index

    while len(current_level) > 1:
        next_level: List[str] = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i + 1] if i + 1 < len(current_level) else left

            if i == idx or i + 1 == idx:
                if idx % 2 == 0:
                    # Target is on left, sibling is on right
                    sibling = right
                    proof.append((sibling, "R"))
                else:
                    # Target is on right, sibling is on left
                    sibling = left
                    proof.append((sibling, "L"))

            combined = f"{left}|{right}"
            next_level.append(_sha256(combined))

        idx = idx // 2
        current_level = next_level

    return proof


def _verify_merkle_proof(
    target_hash: str, proof: List[Tuple[str, str]], root_hash: str
) -> bool:
    """
    Verify a Merkle proof against a root hash.

    Args:
        target_hash: The leaf hash to verify.
        proof: List of (sibling_hash, direction) tuples.
        root_hash: Expected Merkle root hash.

    Returns:
        True if the proof is valid, False otherwise.
    """
    current = target_hash
    for sibling_hash, direction in proof:
        if direction == "L":
            combined = f"{sibling_hash}|{current}"
        else:
            combined = f"{current}|{sibling_hash}"
        current = _sha256(combined)
    return current == root_hash


# ============================================================================
# PROVENANCE TRACKER
# ============================================================================


class ProvenanceTracker:
    """
    Main provenance tracking system for upstream leased assets calculations.

    Manages multiple chains, records stages, validates integrity, and
    exports audit trails. Thread-safe with threading.RLock.

    Each calculation gets its own chain (identified by chain_id). Stages
    are recorded sequentially, with each stage's chain_hash depending on
    the previous stage's chain_hash, creating an immutable linked chain.

    The tracker uses RLock (reentrant lock) to allow methods that call
    other locked methods without deadlocking (e.g., seal_chain calls
    validate_chain internally).

    Example:
        >>> tracker = ProvenanceTracker()
        >>> chain = tracker.start_chain("tenant-001")
        >>> tracker.record_stage(
        ...     chain.chain_id, ProvenanceStage.VALIDATE,
        ...     input_data, output_data,
        ...     engine_id="validation-engine", engine_version="1.0.0"
        ... )
        >>> final_hash = tracker.seal_chain(chain.chain_id)
        >>> valid, errors = tracker.validate_chain(chain.chain_id)
        >>> assert valid
    """

    def __init__(
        self, agent_id: str = AGENT_ID, agent_version: str = AGENT_VERSION
    ):
        """
        Initialize provenance tracker.

        Args:
            agent_id: Agent identifier (default: GL-MRV-S3-008)
            agent_version: Agent version (default: 1.0.0)
        """
        self.agent_id = agent_id
        self.agent_version = agent_version
        self._chains: Dict[str, ProvenanceChain] = {}
        self._lock = threading.RLock()

    def start_chain(
        self,
        tenant_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        chain_id: Optional[str] = None,
    ) -> ProvenanceChain:
        """
        Start a new provenance chain.

        Creates a new empty chain with a unique identifier and associates
        it with the given tenant. The chain starts in an unsealed state,
        ready to accept stage recordings.

        Args:
            tenant_id: Tenant identifier for multi-tenancy isolation.
            metadata: Optional metadata to associate with chain creation.
                      Stored as metadata on an implicit genesis entry if
                      provided. Common fields: source, calculation_type,
                      reporting_year, asset_type.
            chain_id: Optional chain ID. If not provided, a UUID v4 is
                      generated automatically.

        Returns:
            ProvenanceChain - the newly created chain object.

        Raises:
            ValueError: If chain_id already exists in this tracker.

        Example:
            >>> chain = tracker.start_chain("tenant-001", {"year": 2025})
            >>> assert chain.tenant_id == "tenant-001"
        """
        with self._lock:
            if chain_id is None:
                chain_id = str(uuid.uuid4())

            if chain_id in self._chains:
                raise ValueError(f"Chain {chain_id} already exists")

            now = datetime.now(timezone.utc).isoformat()
            chain = ProvenanceChain(
                chain_id=chain_id,
                tenant_id=tenant_id,
                agent_id=self.agent_id,
                entries=[],
                started_at=now,
                sealed_at=None,
                final_hash=None,
                is_sealed=False,
            )
            self._chains[chain_id] = chain

            # If metadata provided, record it as context on the chain
            # We store creation metadata as a property accessible later
            if metadata:
                chain._creation_metadata = metadata  # type: ignore[attr-defined]

            return chain

    def record_stage(
        self,
        chain_id: str,
        stage: Union[str, ProvenanceStage],
        input_data: Any,
        output_data: Any,
        engine_id: str = "unknown",
        engine_version: str = "0.0.0",
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: float = 0.0,
    ) -> ProvenanceEntry:
        """
        Record a pipeline stage in the provenance chain.

        Computes SHA-256 hashes of input and output data, links to the
        previous entry via chain_hash, and appends the entry to the chain.

        Args:
            chain_id: Chain identifier.
            stage: Pipeline stage name. Can be a ProvenanceStage enum or
                   a string matching one of the stage names.
            input_data: Input data for this stage. Any JSON-serializable
                        type. Will be hashed but not stored.
            output_data: Output data from this stage. Any JSON-serializable
                         type. Will be hashed but not stored.
            engine_id: Identifier of the engine that performed this stage
                       (e.g., "building-calculator-engine", "allocation-engine").
            engine_version: Semantic version of the engine (e.g., "1.0.0").
            metadata: Optional metadata dictionary. May include calculation
                      method, emission factor source, uncertainty bounds,
                      asset type, allocation method, lease type, etc.
            duration_ms: Processing time for this stage in milliseconds.
                         If 0.0, caller did not track timing.

        Returns:
            Created ProvenanceEntry (frozen, immutable).

        Raises:
            ValueError: If chain not found or already sealed.

        Example:
            >>> entry = tracker.record_stage(
            ...     chain.chain_id,
            ...     ProvenanceStage.CALCULATE,
            ...     {"floor_area_sqm": 5000.0, "eui_kwh_sqm": 150.0},
            ...     {"co2e_kg": 45000.0},
            ...     engine_id="building-calculator-engine",
            ...     engine_version="1.0.0",
            ...     metadata={"method": "asset_specific"},
            ...     duration_ms=12.5
            ... )
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            if chain.is_sealed:
                raise ValueError(f"Chain {chain_id} already sealed")

            # Convert string to enum if needed
            if isinstance(stage, str):
                try:
                    stage_enum = ProvenanceStage(stage)
                except ValueError:
                    stage_enum = ProvenanceStage(stage)
            else:
                stage_enum = stage

            stage_str = stage_enum.value

            # Compute hashes of input and output data
            input_hash = _compute_hash(input_data)
            output_hash = _compute_hash(output_data)

            # Get previous chain hash
            previous_chain_hash = ""
            if chain.entries:
                previous_chain_hash = chain.entries[-1].chain_hash

            # Compute chain hash linking to previous entry
            chain_hash = _compute_chain_hash(
                previous_chain_hash, stage_str, input_hash, output_hash
            )

            # Create frozen entry
            entry = ProvenanceEntry(
                stage=stage_enum,
                timestamp=datetime.now(timezone.utc).isoformat(),
                input_hash=input_hash,
                output_hash=output_hash,
                chain_hash=chain_hash,
                metadata=metadata if metadata is not None else {},
                duration_ms=duration_ms,
                engine_id=engine_id,
                engine_version=engine_version,
            )

            # Append to chain
            chain.entries.append(entry)
            return entry

    def validate_chain(self, chain_id: str) -> Tuple[bool, List[str]]:
        """
        Validate integrity of provenance chain.

        Performs comprehensive integrity checks on every entry:
            1. First entry's previous chain hash must be empty string
            2. Each subsequent entry's previous hash must match the
               chain_hash of the preceding entry
            3. Each entry's chain_hash must match the recomputed value
               from (previous_hash, stage, input_hash, output_hash)
            4. Timestamps must be monotonically non-decreasing
            5. All entries must have valid stage names

        Args:
            chain_id: Chain identifier.

        Returns:
            Tuple of (is_valid, error_messages).
            - is_valid: True if chain passes all checks.
            - error_messages: List of error strings (empty if valid).

        Raises:
            ValueError: If chain not found.

        Example:
            >>> valid, errors = tracker.validate_chain(chain_id)
            >>> if not valid:
            ...     for err in errors:
            ...         print(f"INTEGRITY ERROR: {err}")
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            errors: List[str] = []
            entries = chain.entries

            if not entries:
                return True, []

            # Check each entry in sequence
            for i, entry in enumerate(entries):
                # Check 1: Previous hash linkage
                if i == 0:
                    if entry.chain_hash != _compute_chain_hash(
                        "", self._stage_value(entry.stage),
                        entry.input_hash, entry.output_hash
                    ):
                        errors.append(
                            f"Entry {i} ({self._stage_value(entry.stage)}): "
                            f"chain_hash mismatch for genesis entry"
                        )
                else:
                    expected_prev = entries[i - 1].chain_hash
                    # Recompute chain hash using previous entry's chain_hash
                    expected_chain_hash = _compute_chain_hash(
                        expected_prev,
                        self._stage_value(entry.stage),
                        entry.input_hash,
                        entry.output_hash,
                    )
                    if entry.chain_hash != expected_chain_hash:
                        errors.append(
                            f"Entry {i} ({self._stage_value(entry.stage)}): "
                            f"chain_hash mismatch. Expected {expected_chain_hash[:16]}..., "
                            f"got {entry.chain_hash[:16]}..."
                        )

                # Check 2: Validate stage is a known stage
                stage_val = self._stage_value(entry.stage)
                valid_stages = {s.value for s in ProvenanceStage}
                if stage_val not in valid_stages:
                    errors.append(
                        f"Entry {i}: unknown stage '{stage_val}'"
                    )

                # Check 3: Timestamp monotonicity
                if i > 0:
                    prev_ts = entries[i - 1].timestamp
                    curr_ts = entry.timestamp
                    if curr_ts < prev_ts:
                        errors.append(
                            f"Entry {i} ({self._stage_value(entry.stage)}): "
                            f"timestamp {curr_ts} is before previous {prev_ts}"
                        )

                # Check 4: Hash format validation (64 hex chars)
                for hash_name, hash_val in [
                    ("input_hash", entry.input_hash),
                    ("output_hash", entry.output_hash),
                    ("chain_hash", entry.chain_hash),
                ]:
                    if len(hash_val) != 64:
                        errors.append(
                            f"Entry {i} ({self._stage_value(entry.stage)}): "
                            f"{hash_name} has invalid length {len(hash_val)}, expected 64"
                        )
                    elif not all(c in "0123456789abcdef" for c in hash_val):
                        errors.append(
                            f"Entry {i} ({self._stage_value(entry.stage)}): "
                            f"{hash_name} contains non-hex characters"
                        )

            is_valid = len(errors) == 0
            return is_valid, errors

    def seal_chain(self, chain_id: str) -> str:
        """
        Seal the provenance chain.

        Records final timestamp and computes the Merkle root hash from
        all entry chain_hashes. Once sealed, no more entries can be added
        to this chain.

        The sealing process:
            1. Validates chain integrity
            2. Collects all entry chain_hashes
            3. Computes Merkle-tree aggregate hash
            4. Records sealed timestamp
            5. Marks chain as sealed
            6. Returns final hash

        Args:
            chain_id: Chain identifier.

        Returns:
            Final Merkle root hash (64-character hex string).

        Raises:
            ValueError: If chain not found, already sealed, or fails validation.

        Example:
            >>> final_hash = tracker.seal_chain(chain_id)
            >>> assert len(final_hash) == 64
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            if chain.is_sealed:
                raise ValueError(f"Chain {chain_id} already sealed")

            # Validate before sealing
            is_valid, validation_errors = self.validate_chain(chain_id)
            if not is_valid:
                error_summary = "; ".join(validation_errors[:5])
                raise ValueError(
                    f"Chain {chain_id} failed validation: {error_summary}"
                )

            # Compute Merkle root from all entry chain_hashes
            if chain.entries:
                all_hashes = [e.chain_hash for e in chain.entries]
                final_hash = _merkle_hash(all_hashes)
            else:
                final_hash = hashlib.sha256(chain_id.encode(ENCODING)).hexdigest()

            chain.sealed_at = datetime.now(timezone.utc).isoformat()
            chain.final_hash = final_hash
            chain.is_sealed = True
            return final_hash

    def get_chain(self, chain_id: str) -> Optional[ProvenanceChain]:
        """
        Get provenance chain.

        Returns the chain object if found, or None if not found. Unlike
        other methods that raise ValueError, this returns None for
        missing chains to support safe lookups.

        Args:
            chain_id: Chain identifier.

        Returns:
            ProvenanceChain if found, None otherwise.

        Example:
            >>> chain = tracker.get_chain(chain_id)
            >>> if chain is not None:
            ...     print(f"Chain has {chain.entry_count} entries")
        """
        with self._lock:
            return self._chains.get(chain_id)

    def export_chain(self, chain_id: str) -> Dict[str, Any]:
        """
        Export provenance chain as a dictionary.

        Returns the full chain data including all entries, suitable for
        JSON serialization, database storage, or API responses.

        Args:
            chain_id: Chain identifier.

        Returns:
            Dictionary representation of the complete chain.

        Raises:
            ValueError: If chain not found.

        Example:
            >>> chain_dict = tracker.export_chain(chain_id)
            >>> json_str = json.dumps(chain_dict, indent=2)
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            return chain.to_dict()

    def verify_entry(
        self, entry: ProvenanceEntry, previous_chain_hash: str = ""
    ) -> bool:
        """
        Verify a single provenance entry.

        Recomputes chain_hash from the entry's components and the given
        previous_chain_hash, then checks that it matches the stored
        chain_hash.

        Args:
            entry: Provenance entry to verify.
            previous_chain_hash: Chain hash of the previous entry in the
                                 chain. Empty string for the first entry.

        Returns:
            True if entry is valid, False otherwise.

        Example:
            >>> entry = tracker.record_stage(chain_id, "VALIDATE", inp, out)
            >>> assert tracker.verify_entry(entry, "")
        """
        stage_str = self._stage_value(entry.stage)
        expected_hash = _compute_chain_hash(
            previous_chain_hash,
            stage_str,
            entry.input_hash,
            entry.output_hash,
        )
        return entry.chain_hash == expected_hash

    def delete_chain(self, chain_id: str) -> bool:
        """
        Delete a provenance chain.

        Permanently removes the chain from the tracker. This operation
        cannot be undone.

        Args:
            chain_id: Chain identifier.

        Returns:
            True if deleted, False if not found.

        Example:
            >>> deleted = tracker.delete_chain(chain_id)
            >>> assert deleted
        """
        with self._lock:
            if chain_id in self._chains:
                del self._chains[chain_id]
                return True
            return False

    def list_chains(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ProvenanceChain]:
        """
        List provenance chains with optional tenant filtering and pagination.

        Args:
            tenant_id: Optional tenant ID to filter by. If None, returns
                       chains for all tenants.
            limit: Maximum number of chains to return (default 100).
            offset: Number of chains to skip (default 0).

        Returns:
            List of ProvenanceChain objects matching the criteria.

        Example:
            >>> chains = tracker.list_chains("tenant-001", limit=10)
            >>> for chain in chains:
            ...     print(chain.chain_id, chain.is_sealed)
        """
        with self._lock:
            all_chains = list(self._chains.values())

            # Filter by tenant if specified
            if tenant_id is not None:
                all_chains = [c for c in all_chains if c.tenant_id == tenant_id]

            # Sort by started_at for deterministic ordering
            all_chains.sort(key=lambda c: c.started_at)

            # Apply pagination
            return all_chains[offset: offset + limit]

    def get_chain_summary(self, chain_id: str) -> Dict[str, Any]:
        """
        Get summary of provenance chain.

        Returns a lightweight summary without full entry details,
        suitable for logging, monitoring, and dashboard display.

        Args:
            chain_id: Chain identifier.

        Returns:
            Summary dictionary with key metrics including chain_id,
            tenant_id, agent info, timestamps, entry count, stages,
            sealing status, and hash information.

        Raises:
            ValueError: If chain not found.

        Example:
            >>> summary = tracker.get_chain_summary(chain_id)
            >>> print(summary['entry_count'])
            10
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            return {
                "chain_id": chain.chain_id,
                "tenant_id": chain.tenant_id,
                "agent_id": chain.agent_id,
                "started_at": chain.started_at,
                "sealed_at": chain.sealed_at,
                "final_hash": chain.final_hash,
                "is_sealed": chain.is_sealed,
                "entry_count": chain.entry_count,
                "stages": chain.stages_recorded,
                "root_hash": chain.root_hash,
                "last_hash": chain.last_hash,
                "duration_total_ms": sum(e.duration_ms for e in chain.entries),
                "engines_used": list({e.engine_id for e in chain.entries}),
            }

    def get_stage_hash(
        self, chain_id: str, stage: Union[str, ProvenanceStage]
    ) -> Optional[str]:
        """
        Get output hash for a specific stage.

        Searches the chain for the first entry matching the given stage
        and returns its output_hash.

        Args:
            chain_id: Chain identifier.
            stage: Pipeline stage name (string or ProvenanceStage enum).

        Returns:
            Output hash for stage, or None if stage not found in chain.

        Raises:
            ValueError: If chain not found.

        Example:
            >>> validate_hash = tracker.get_stage_hash(chain_id, "VALIDATE")
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            target = stage.value if isinstance(stage, ProvenanceStage) else stage
            for entry in chain.entries:
                if self._stage_value(entry.stage) == target:
                    return entry.output_hash
            return None

    def reset(self) -> None:
        """
        Clear all chains tracked by this tracker instance.

        Useful for testing. Removes all chain state.

        Example:
            >>> tracker.reset()
            >>> assert len(tracker.get_all_chains()) == 0
        """
        with self._lock:
            self._chains.clear()

    def get_all_chains(self) -> List[str]:
        """
        Get list of all chain IDs.

        Returns:
            List of chain IDs managed by this tracker.

        Example:
            >>> chain_ids = tracker.get_all_chains()
        """
        with self._lock:
            return list(self._chains.keys())

    def clear_all_chains(self) -> int:
        """
        Clear all chains (for testing).

        Returns:
            Number of chains cleared.
        """
        with self._lock:
            count = len(self._chains)
            self._chains.clear()
            return count

    @staticmethod
    def _stage_value(stage: Union[str, ProvenanceStage]) -> str:
        """
        Extract string value from stage (enum or str).

        Args:
            stage: ProvenanceStage enum or string.

        Returns:
            String value of the stage.
        """
        if isinstance(stage, ProvenanceStage):
            return stage.value
        return str(stage)


# ============================================================================
# BATCH PROVENANCE TRACKER
# ============================================================================


@dataclass
class BatchProvenance:
    """
    Provenance tracking record for batch calculations.

    Tracks multiple individual chains plus batch-level Merkle tree
    aggregation. Used when processing a portfolio of upstream leased
    assets (buildings, vehicles, equipment, IT) in one batch job.

    Attributes:
        batch_id: Unique identifier for this batch (UUID).
        tenant_id: Tenant identifier for multi-tenancy isolation.
        individual_chain_ids: List of chain IDs in this batch.
        batch_started_at: ISO 8601 UTC timestamp of batch start.
        batch_sealed_at: ISO 8601 UTC timestamp when sealed (or None).
        batch_hash: Merkle-tree root hash (or None until sealed).
        batch_size: Expected number of items in batch.
        merkle_tree_levels: Number of levels in the Merkle tree (set on seal).
        merkle_leaf_hashes: Sorted leaf hashes used to build the tree.
    """

    batch_id: str
    tenant_id: str
    individual_chain_ids: List[str] = field(default_factory=list)
    batch_started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    batch_sealed_at: Optional[str] = None
    batch_hash: Optional[str] = None
    batch_size: int = 0
    merkle_tree_levels: int = 0
    merkle_leaf_hashes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert batch provenance to dictionary.

        Returns:
            Dictionary representation of the batch provenance.
        """
        return {
            "batch_id": self.batch_id,
            "tenant_id": self.tenant_id,
            "individual_chain_ids": list(self.individual_chain_ids),
            "batch_started_at": self.batch_started_at,
            "batch_sealed_at": self.batch_sealed_at,
            "batch_hash": self.batch_hash,
            "batch_size": self.batch_size,
            "chain_count": len(self.individual_chain_ids),
            "merkle_tree_levels": self.merkle_tree_levels,
            "is_sealed": self.batch_sealed_at is not None,
        }


class BatchProvenanceTracker:
    """
    Provenance tracking for batch upstream leased assets calculations.

    Manages multiple individual chains plus batch-level Merkle-tree
    aggregation. Each batch contains references to individual leased
    asset calculation chains, and produces a Merkle root hash on sealing.

    Supports Merkle proof generation and verification for efficient
    proof-of-inclusion of any individual chain within a batch.

    Thread-safe with threading.RLock.

    Example:
        >>> batch_tracker = BatchProvenanceTracker(tracker)
        >>> batch_id = batch_tracker.start_batch("tenant-001", 200)
        >>> for asset in leased_assets:
        ...     chain = tracker.start_chain("tenant-001")
        ...     # ... record stages for this asset ...
        ...     tracker.seal_chain(chain.chain_id)
        ...     batch_tracker.add_chain(batch_id, chain)
        >>> root_hash = batch_tracker.build_merkle_tree(batch_id)
        >>> assert batch_tracker.verify_merkle_proof(batch_id, chain.chain_id)
    """

    def __init__(self, tracker: ProvenanceTracker):
        """
        Initialize batch provenance tracker.

        Args:
            tracker: ProvenanceTracker instance for individual chains.
                     Used to retrieve chain final hashes during batch
                     sealing and Merkle tree construction.
        """
        self.tracker = tracker
        self._batches: Dict[str, BatchProvenance] = {}
        self._lock = threading.RLock()

    def start_batch(
        self,
        tenant_id: str,
        batch_size: int = 0,
        batch_id: Optional[str] = None,
    ) -> str:
        """
        Start a new batch.

        Args:
            tenant_id: Tenant identifier for multi-tenancy isolation.
            batch_size: Expected number of items in the batch. Used for
                        progress tracking and validation.
            batch_id: Optional batch ID. If not provided, a UUID v4 is
                      generated automatically.

        Returns:
            Batch ID (str).

        Raises:
            ValueError: If batch_id already exists.

        Example:
            >>> batch_id = batch_tracker.start_batch("tenant-001", 200)
        """
        with self._lock:
            if batch_id is None:
                batch_id = str(uuid.uuid4())

            if batch_id in self._batches:
                raise ValueError(f"Batch {batch_id} already exists")

            batch = BatchProvenance(
                batch_id=batch_id,
                tenant_id=tenant_id,
                individual_chain_ids=[],
                batch_started_at=datetime.now(timezone.utc).isoformat(),
                batch_sealed_at=None,
                batch_hash=None,
                batch_size=batch_size,
                merkle_tree_levels=0,
                merkle_leaf_hashes=[],
            )
            self._batches[batch_id] = batch
            return batch_id

    def add_chain(
        self, batch_id: str, chain: Union[ProvenanceChain, str]
    ) -> None:
        """
        Add a chain to the batch.

        Accepts either a ProvenanceChain object or a chain_id string.

        Args:
            batch_id: Batch identifier.
            chain: ProvenanceChain object or chain_id string to add.

        Raises:
            ValueError: If batch not found or already sealed.

        Example:
            >>> batch_tracker.add_chain(batch_id, chain)
            >>> batch_tracker.add_chain(batch_id, "chain-uuid-string")
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            if batch.batch_sealed_at is not None:
                raise ValueError(f"Batch {batch_id} already sealed")

            if isinstance(chain, ProvenanceChain):
                chain_id = chain.chain_id
            else:
                chain_id = chain

            batch.individual_chain_ids.append(chain_id)

    def build_merkle_tree(self, batch_id: str) -> str:
        """
        Build Merkle tree from all chain final hashes and seal the batch.

        Collects the final_hash from each chain in the batch, builds a
        binary Merkle tree, and stores the root hash. Chains that are not
        sealed or not found are skipped (their hashes are not included).

        The sorted leaf hashes are stored on the batch for later
        proof generation and verification.

        Args:
            batch_id: Batch identifier.

        Returns:
            Merkle root hash (64-character hex string).

        Raises:
            ValueError: If batch not found or already sealed.

        Example:
            >>> root_hash = batch_tracker.build_merkle_tree(batch_id)
            >>> assert len(root_hash) == 64
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            if batch.batch_sealed_at is not None:
                raise ValueError(f"Batch {batch_id} already sealed")

            # Collect final hashes from all sealed chains
            chain_hashes: List[str] = []
            for chain_id in batch.individual_chain_ids:
                chain = self.tracker.get_chain(chain_id)
                if chain is not None and chain.final_hash is not None:
                    chain_hashes.append(chain.final_hash)

            # Sort for determinism and store leaf hashes
            sorted_hashes = sorted(chain_hashes)
            batch.merkle_leaf_hashes = sorted_hashes

            # Compute tree depth
            if sorted_hashes:
                batch.merkle_tree_levels = math.ceil(math.log2(len(sorted_hashes))) + 1
            else:
                batch.merkle_tree_levels = 0

            # Build Merkle root
            root_hash = _merkle_hash(sorted_hashes)
            batch.batch_hash = root_hash
            batch.batch_sealed_at = datetime.now(timezone.utc).isoformat()
            return root_hash

    def verify_merkle_proof(self, batch_id: str, chain_id: str) -> bool:
        """
        Verify that a chain is included in the batch Merkle tree.

        Generates a Merkle proof for the given chain and verifies it
        against the batch root hash.

        Args:
            batch_id: Batch identifier.
            chain_id: Chain identifier to verify inclusion of.

        Returns:
            True if the chain is provably included in the batch,
            False otherwise.

        Raises:
            ValueError: If batch not found or batch not sealed.

        Example:
            >>> included = batch_tracker.verify_merkle_proof(batch_id, chain_id)
            >>> assert included, "Chain not in batch"
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            if batch.batch_sealed_at is None:
                raise ValueError(f"Batch {batch_id} not yet sealed")
            if batch.batch_hash is None:
                return False

            # Get chain's final hash
            chain = self.tracker.get_chain(chain_id)
            if chain is None or chain.final_hash is None:
                return False

            target_hash = chain.final_hash
            leaf_hashes = batch.merkle_leaf_hashes

            # Find target in sorted leaf hashes
            if target_hash not in leaf_hashes:
                return False

            target_index = leaf_hashes.index(target_hash)

            # Build proof and verify
            proof = _merkle_proof(leaf_hashes, target_index)
            return _verify_merkle_proof(target_hash, proof, batch.batch_hash)

    def get_batch_summary(self, batch_id: str) -> Dict[str, Any]:
        """
        Get summary of batch provenance.

        Args:
            batch_id: Batch identifier.

        Returns:
            Summary dictionary with batch metadata, chain count,
            Merkle tree info, and sealing status.

        Raises:
            ValueError: If batch not found.

        Example:
            >>> summary = batch_tracker.get_batch_summary(batch_id)
            >>> print(summary['chain_count'])
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")

            return {
                "batch_id": batch.batch_id,
                "tenant_id": batch.tenant_id,
                "batch_started_at": batch.batch_started_at,
                "batch_sealed_at": batch.batch_sealed_at,
                "batch_hash": batch.batch_hash,
                "batch_size": batch.batch_size,
                "chain_count": len(batch.individual_chain_ids),
                "individual_chain_ids": list(batch.individual_chain_ids),
                "is_sealed": batch.batch_sealed_at is not None,
                "merkle_tree_levels": batch.merkle_tree_levels,
                "merkle_leaf_count": len(batch.merkle_leaf_hashes),
            }

    def get_batch(self, batch_id: str) -> BatchProvenance:
        """
        Get batch provenance object.

        Args:
            batch_id: Batch identifier.

        Returns:
            BatchProvenance object.

        Raises:
            ValueError: If batch not found.

        Example:
            >>> batch = batch_tracker.get_batch(batch_id)
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            return batch

    def delete_batch(self, batch_id: str) -> bool:
        """
        Delete a batch provenance record.

        Args:
            batch_id: Batch identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if batch_id in self._batches:
                del self._batches[batch_id]
                return True
            return False

    def list_batches(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[BatchProvenance]:
        """
        List batch provenance records with optional filtering.

        Args:
            tenant_id: Optional tenant ID to filter by.
            limit: Maximum number of batches to return.
            offset: Number of batches to skip.

        Returns:
            List of BatchProvenance objects.
        """
        with self._lock:
            all_batches = list(self._batches.values())
            if tenant_id is not None:
                all_batches = [b for b in all_batches if b.tenant_id == tenant_id]
            all_batches.sort(key=lambda b: b.batch_started_at)
            return all_batches[offset: offset + limit]

    def reset(self) -> None:
        """
        Clear all batches (for testing).
        """
        with self._lock:
            self._batches.clear()


# ============================================================================
# STANDALONE HASH FUNCTIONS (20+)
# ============================================================================


def hash_calculation_input(input_data: Dict[str, Any]) -> str:
    """
    Hash complete calculation request.

    Args:
        input_data: Calculation request dictionary containing leased
                    asset data, method selection, and configuration.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    return _compute_hash(input_data)


def hash_building_input(building_input: Dict[str, Any]) -> str:
    """
    Hash building leased asset input data.

    Covers the building-specific input record including building type,
    floor area, energy consumption, occupancy, and lease terms.

    Args:
        building_input: Building data dictionary with keys such as
            building_type (office, warehouse, retail, data_center,
            manufacturing, healthcare, education, mixed_use),
            floor_area_sqm, leased_area_sqm, energy_kwh,
            natural_gas_kwh, district_heat_kwh, climate_zone,
            lease_start_date, lease_end_date, building_year, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_building_input({
        ...     "building_type": "office",
        ...     "floor_area_sqm": 5000.0,
        ...     "energy_kwh": 750000.0
        ... })
    """
    return _compute_hash(building_input)


def hash_vehicle_input(vehicle_input: Dict[str, Any]) -> str:
    """
    Hash vehicle leased asset input data.

    For leased fleet vehicles (company cars, trucks, vans, etc.).
    Captures vehicle characteristics that determine emissions.

    Args:
        vehicle_input: Vehicle data dictionary with keys such as
            vehicle_type (passenger_car, light_truck, heavy_truck,
            van, bus, motorcycle, electric_vehicle, hybrid),
            fuel_type (petrol, diesel, lpg, cng, electric, hybrid,
            biodiesel, ethanol, hydrogen), annual_km, fuel_consumption_l,
            vehicle_year, engine_size_cc, lease_start_date,
            lease_end_date, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_vehicle_input({
        ...     "vehicle_type": "passenger_car",
        ...     "fuel_type": "diesel",
        ...     "annual_km": 25000
        ... })
    """
    return _compute_hash(vehicle_input)


def hash_equipment_input(equipment_input: Dict[str, Any]) -> str:
    """
    Hash equipment leased asset input data.

    For leased industrial/commercial equipment (generators,
    compressors, forklifts, HVAC units, etc.).

    Args:
        equipment_input: Equipment data dictionary with keys such as
            equipment_type (generator, compressor, forklift, hvac,
            chiller, boiler, pump, conveyor, crane), rated_power_kw,
            fuel_type, annual_operating_hours, load_factor,
            energy_consumption_kwh, lease_start_date,
            lease_end_date, equipment_age_years, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_equipment_input({
        ...     "equipment_type": "generator",
        ...     "rated_power_kw": 500.0,
        ...     "fuel_type": "diesel",
        ...     "annual_operating_hours": 2000
        ... })
    """
    return _compute_hash(equipment_input)


def hash_it_asset_input(it_input: Dict[str, Any]) -> str:
    """
    Hash IT asset leased input data.

    For leased IT infrastructure (servers, storage arrays, networking
    equipment, data center space).

    Args:
        it_input: IT asset data dictionary with keys such as
            asset_subtype (server, storage, network_switch, router,
            ups, cooling_unit), rated_power_watts, annual_kwh,
            pue (power usage effectiveness), utilization_pct,
            data_center_tier, lease_start_date, lease_end_date,
            include_embodied, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_it_asset_input({
        ...     "asset_subtype": "server",
        ...     "rated_power_watts": 500,
        ...     "pue": 1.4,
        ...     "utilization_pct": 0.65
        ... })
    """
    return _compute_hash(it_input)


def hash_lease_classification(classification_input: Dict[str, Any]) -> str:
    """
    Hash lease classification input data.

    For IFRS 16 / ASC 842 lease classification determination
    (operating vs finance lease).

    Args:
        classification_input: Classification data dictionary with keys
            such as accounting_standard (IFRS_16, ASC_842, BOTH),
            lease_term_months, asset_useful_life_months,
            present_value_payments, fair_value_asset,
            transfer_ownership, purchase_option, specialized_asset,
            short_term_exempt, low_value_exempt, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_lease_classification({
        ...     "accounting_standard": "IFRS_16",
        ...     "lease_term_months": 60,
        ...     "asset_useful_life_months": 120,
        ...     "transfer_ownership": False
        ... })
    """
    return _compute_hash(classification_input)


def hash_spend_input(spend_input: Dict[str, Any]) -> str:
    """
    Hash spend-based calculation input.

    For the spend-based calculation method where lease payments
    are used with EEIO emission factors.

    Args:
        spend_input: Spend data dictionary with keys such as
            spend_amount, currency, spend_category (lease_payment,
            operating_expense, maintenance), reporting_period,
            eeio_sector, deflation_factor, margin_removal_factor,
            cpi_base_year, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_spend_input({
        ...     "spend_amount": 120000.00,
        ...     "currency": "USD",
        ...     "spend_category": "lease_payment"
        ... })
    """
    return _compute_hash(spend_input)


def hash_average_data_input(avg_input: Dict[str, Any]) -> str:
    """
    Hash average-data method input.

    For the average-data calculation method where building EUI
    benchmarks or fleet average emission factors are used instead
    of asset-specific data.

    Args:
        avg_input: Average data dictionary with keys such as
            asset_type, building_type, climate_zone, region,
            floor_area_sqm, eui_kwh_per_sqm, grid_ef_kgco2e_per_kwh,
            data_source (ENERGY_STAR, CBECS, CIBSE), data_year, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_average_data_input({
        ...     "building_type": "office",
        ...     "climate_zone": "4A",
        ...     "floor_area_sqm": 5000.0,
        ...     "eui_kwh_per_sqm": 150.0
        ... })
    """
    return _compute_hash(avg_input)


def hash_lessor_specific_input(lessor_input: Dict[str, Any]) -> str:
    """
    Hash lessor-specific calculation input.

    For the lessor-specific method where the landlord/lessor provides
    emission factors or actual energy data for the leased space.

    Args:
        lessor_input: Lessor-specific data dictionary with keys such as
            lessor_id, lessor_ef_kgco2e, lessor_data_source,
            lessor_certification (ISO_14001, LEED, BREEAM),
            reporting_boundary, coverage_pct, verification_status, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_lessor_specific_input({
        ...     "lessor_id": "LESSOR-001",
        ...     "lessor_ef_kgco2e": 45000.0,
        ...     "verification_status": "third_party_verified"
        ... })
    """
    return _compute_hash(lessor_input)


def hash_emission_factor(ef_data: Dict[str, Any]) -> str:
    """
    Hash emission factor data.

    Hashes the complete emission factor record including type, value,
    source, year, region, and any other qualifying attributes.

    Args:
        ef_data: Emission factor dictionary with keys such as
            ef_type (e.g., "grid_electricity", "natural_gas_combustion",
            "diesel_mobile"), ef_value (kgCO2e per unit), source
            (e.g., "DEFRA", "EPA_eGRID", "IEA"), year, region,
            unit, gas_breakdown (co2, ch4, n2o), wtt_included, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_emission_factor({
        ...     "ef_type": "grid_electricity",
        ...     "ef_value": "0.4172",
        ...     "source": "EPA_eGRID",
        ...     "year": 2024,
        ...     "region": "US-SRSO"
        ... })
    """
    return _compute_hash(ef_data)


def hash_calculation_result(result: Dict[str, Any]) -> str:
    """
    Hash complete calculation result.

    Args:
        result: Calculation result dictionary with all emissions
                components, totals, and metadata.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    return _compute_hash(result)


def hash_building_result(result: Dict[str, Any]) -> str:
    """
    Hash building emissions calculation result.

    Hashes the output of the building emissions calculation engine
    for a single leased building or building portion.

    Args:
        result: Building result dictionary with keys such as
            co2e_kg, co2_kg, ch4_kg, n2o_kg, building_type, method,
            floor_area_sqm, energy_kwh, grid_ef_used, natural_gas_ef_used,
            allocation_factor, partial_year_factor, wtt_co2e_kg, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_building_result({
        ...     "co2e_kg": 45000.0,
        ...     "building_type": "office",
        ...     "method": "asset_specific"
        ... })
    """
    return _compute_hash(result)


def hash_vehicle_result(result: Dict[str, Any]) -> str:
    """
    Hash vehicle emissions calculation result.

    Args:
        result: Vehicle result dictionary with keys such as
            co2e_kg, vehicle_type, fuel_type, annual_km,
            fuel_consumption_l, ef_used, wtt_co2e_kg,
            partial_year_factor, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_vehicle_result({
        ...     "co2e_kg": 3500.0,
        ...     "vehicle_type": "passenger_car",
        ...     "fuel_type": "diesel"
        ... })
    """
    return _compute_hash(result)


def hash_equipment_result(result: Dict[str, Any]) -> str:
    """
    Hash equipment emissions calculation result.

    Args:
        result: Equipment result dictionary with keys such as
            co2e_kg, equipment_type, fuel_type, operating_hours,
            load_factor, rated_power_kw, ef_used,
            partial_year_factor, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_equipment_result({
        ...     "co2e_kg": 12000.0,
        ...     "equipment_type": "generator",
        ...     "fuel_type": "diesel"
        ... })
    """
    return _compute_hash(result)


def hash_it_asset_result(result: Dict[str, Any]) -> str:
    """
    Hash IT asset emissions calculation result.

    Args:
        result: IT asset result dictionary with keys such as
            co2e_kg, asset_subtype, annual_kwh, pue,
            utilization_pct, grid_ef_used, embodied_co2e_kg,
            partial_year_factor, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_it_asset_result({
        ...     "co2e_kg": 8500.0,
        ...     "asset_subtype": "server",
        ...     "pue": 1.4
        ... })
    """
    return _compute_hash(result)


def hash_compliance_result(result: Dict[str, Any]) -> str:
    """
    Hash compliance check result.

    Args:
        result: Compliance result dictionary with keys such as
            framework (GHG_PROTOCOL, ISO_14064, CSRD, CDP, SBTi,
            SB_253, GRI), status (COMPLIANT, NON_COMPLIANT, PARTIAL),
            score (0.0-1.0), findings, recommendations, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_compliance_result({
        ...     "framework": "GHG_PROTOCOL",
        ...     "status": "COMPLIANT",
        ...     "score": 0.92
        ... })
    """
    return _compute_hash(result)


def hash_aggregation_result(result: Dict[str, Any]) -> str:
    """
    Hash aggregation and summarization result.

    Args:
        result: Aggregation result dictionary with keys such as
            total_co2e_kg, by_asset_type, by_building_type,
            by_calculation_method, by_region, asset_count,
            average_per_asset, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_aggregation_result({
        ...     "total_co2e_kg": 250000.0,
        ...     "asset_count": 45,
        ...     "by_asset_type": {"building": 180000.0, "vehicle": 35000.0}
        ... })
    """
    return _compute_hash(result)


def hash_allocation(allocation_input: Dict[str, Any]) -> str:
    """
    Hash allocation input data.

    For floor-area, headcount, FTE, or custom allocation methods
    applied to shared leased spaces.

    Args:
        allocation_input: Allocation data dictionary with keys such as
            method (FLOOR_AREA, HEADCOUNT, FTE, CUSTOM),
            total_area_sqm, leased_area_sqm, total_headcount,
            leased_headcount, allocation_factor, partial_year_factor,
            common_area_factor, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_allocation({
        ...     "method": "FLOOR_AREA",
        ...     "total_area_sqm": 10000.0,
        ...     "leased_area_sqm": 3000.0,
        ...     "allocation_factor": 0.30
        ... })
    """
    return _compute_hash(allocation_input)


def hash_batch_input(batch_items: List[Dict[str, Any]]) -> str:
    """
    Hash a batch of input items.

    Computes a single hash over an entire batch of leased asset
    input records. The list is serialized deterministically.

    Args:
        batch_items: List of input item dictionaries.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_batch_input([
        ...     {"asset_id": "BLD-001", "asset_type": "building"},
        ...     {"asset_id": "VEH-001", "asset_type": "vehicle"}
        ... ])
    """
    return _compute_hash(batch_items)


# ============================================================================
# ADDITIONAL DOMAIN-SPECIFIC HASH FUNCTIONS
# ============================================================================


def hash_asset_specific_result(
    asset_type: str,
    energy_kwh: Union[Decimal, float],
    ef: Union[Decimal, float],
    allocation_factor: Union[Decimal, float],
    partial_year_factor: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash asset-specific calculation result.

    Args:
        asset_type: Leased asset type (building, vehicle, equipment, it_asset).
        energy_kwh: Total energy consumption in kWh.
        ef: Emission factor (kgCO2e/kWh or kgCO2e/unit).
        allocation_factor: Allocation share (0.0-1.0).
        partial_year_factor: Partial-year proration factor (0.0-1.0).
        co2e_kg: Total emissions (kgCO2e).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "asset_type": asset_type,
        "energy_kwh": str(energy_kwh),
        "ef": str(ef),
        "allocation_factor": str(allocation_factor),
        "partial_year_factor": str(partial_year_factor),
        "co2e_kg": str(co2e_kg),
        "formula": "energy_kwh * ef * allocation_factor * partial_year_factor",
    }
    return _compute_hash(data)


def hash_average_data_result(
    building_type: str,
    floor_area_sqm: Union[Decimal, float],
    eui_kwh_per_sqm: Union[Decimal, float],
    grid_ef: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash average-data calculation result.

    Args:
        building_type: Building type (office, warehouse, retail, etc.).
        floor_area_sqm: Leased floor area in square metres.
        eui_kwh_per_sqm: Energy Use Intensity benchmark (kWh/sqm/year).
        grid_ef: Grid emission factor (kgCO2e/kWh).
        co2e_kg: Total emissions (kgCO2e).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "building_type": building_type,
        "floor_area_sqm": str(floor_area_sqm),
        "eui_kwh_per_sqm": str(eui_kwh_per_sqm),
        "grid_ef": str(grid_ef),
        "co2e_kg": str(co2e_kg),
        "formula": "floor_area_sqm * eui_kwh_per_sqm * grid_ef",
    }
    return _compute_hash(data)


def hash_spend_result(
    spend: Union[Decimal, float],
    currency: str,
    eeio_ef: Union[Decimal, float],
    deflation_factor: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash spend-based calculation result.

    Args:
        spend: Annual lease payment in original currency.
        currency: ISO 4217 currency code.
        eeio_ef: EEIO emission factor (kgCO2e per currency unit).
        deflation_factor: CPI deflation factor to base year.
        co2e_kg: Total emissions (kgCO2e).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "spend": str(spend),
        "currency": currency,
        "eeio_ef": str(eeio_ef),
        "deflation_factor": str(deflation_factor),
        "co2e_kg": str(co2e_kg),
        "formula": "spend * deflation_factor * eeio_ef",
    }
    return _compute_hash(data)


def hash_lessor_specific_result(
    lessor_id: str,
    lessor_ef_kgco2e: Union[Decimal, float],
    allocation_factor: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash lessor-specific calculation result.

    Args:
        lessor_id: Lessor/landlord identifier.
        lessor_ef_kgco2e: Lessor-provided emission factor or total.
        allocation_factor: Tenant allocation share (0.0-1.0).
        co2e_kg: Allocated emissions (kgCO2e).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "lessor_id": lessor_id,
        "lessor_ef_kgco2e": str(lessor_ef_kgco2e),
        "allocation_factor": str(allocation_factor),
        "co2e_kg": str(co2e_kg),
        "formula": "lessor_ef_kgco2e * allocation_factor",
    }
    return _compute_hash(data)


def hash_pue_calculation(
    it_load_kwh: Union[Decimal, float],
    pue: Union[Decimal, float],
    total_facility_kwh: Union[Decimal, float],
    grid_ef: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash PUE-based data center calculation result.

    Args:
        it_load_kwh: IT equipment electrical load (kWh).
        pue: Power Usage Effectiveness ratio.
        total_facility_kwh: Total facility energy (it_load * pue).
        grid_ef: Grid emission factor (kgCO2e/kWh).
        co2e_kg: Total emissions (kgCO2e).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "it_load_kwh": str(it_load_kwh),
        "pue": str(pue),
        "total_facility_kwh": str(total_facility_kwh),
        "grid_ef": str(grid_ef),
        "co2e_kg": str(co2e_kg),
        "formula": "it_load_kwh * pue * grid_ef",
    }
    return _compute_hash(data)


def hash_wtt_result(
    base_co2e_kg: Union[Decimal, float],
    wtt_factor: Union[Decimal, float],
    wtt_co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash well-to-tank (WTT) upstream emissions result.

    Args:
        base_co2e_kg: Base combustion/use-phase emissions (kgCO2e).
        wtt_factor: WTT emission factor ratio.
        wtt_co2e_kg: WTT upstream emissions (kgCO2e).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "base_co2e_kg": str(base_co2e_kg),
        "wtt_factor": str(wtt_factor),
        "wtt_co2e_kg": str(wtt_co2e_kg),
        "formula": "base_co2e_kg * wtt_factor",
    }
    return _compute_hash(data)


def hash_partial_year_proration(
    lease_start: str,
    lease_end: str,
    reporting_year: int,
    proration_factor: Union[Decimal, float],
) -> str:
    """
    Hash partial-year proration calculation.

    Args:
        lease_start: Lease start date (ISO 8601).
        lease_end: Lease end date (ISO 8601).
        reporting_year: Reporting period year.
        proration_factor: Computed proration factor (0.0-1.0).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "lease_start": lease_start,
        "lease_end": lease_end,
        "reporting_year": reporting_year,
        "proration_factor": str(proration_factor),
        "formula": "days_in_reporting_period / days_in_year",
    }
    return _compute_hash(data)


def hash_classification_result(
    asset_type: str,
    lease_type: str,
    accounting_standard: str,
) -> str:
    """
    Hash lease classification result.

    Args:
        asset_type: Asset type (building, vehicle, equipment, it_asset).
        lease_type: Classified lease type (operating, finance,
                    short_term, low_value).
        accounting_standard: Accounting standard used (IFRS_16, ASC_842).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "asset_type": asset_type,
        "lease_type": lease_type,
        "accounting_standard": accounting_standard,
    }
    return _compute_hash(data)


def hash_allocation_result(
    method: str,
    share: Union[Decimal, float],
    total: Union[Decimal, float],
    allocated: Union[Decimal, float],
) -> str:
    """
    Hash allocation calculation result.

    Args:
        method: Allocation method (e.g., "FLOOR_AREA", "HEADCOUNT",
                "FTE", "CUSTOM").
        share: This unit's share (e.g., 0.30 for 30%).
        total: Total emissions before allocation.
        allocated: Allocated emissions.

    Returns:
        SHA-256 hash (64-character hex string).
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
    by_asset_type: Dict[str, Any],
    by_building_type: Dict[str, Any],
    by_method: Dict[str, Any],
    total: Union[Decimal, float],
) -> str:
    """
    Hash aggregation result (parametric form).

    Args:
        by_asset_type: Emissions by asset type (building, vehicle, etc.).
        by_building_type: Emissions by building type.
        by_method: Emissions by calculation method.
        total: Total emissions (kgCO2e).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "by_asset_type": by_asset_type,
        "by_building_type": by_building_type,
        "by_method": by_method,
        "total": str(total),
    }
    return _compute_hash(data)


def hash_uncertainty_result(
    uncertainty_type: str,
    confidence_level: Union[Decimal, float],
    lower_bound: Union[Decimal, float],
    upper_bound: Union[Decimal, float],
) -> str:
    """
    Hash uncertainty quantification result.

    Args:
        uncertainty_type: Type of uncertainty (e.g., "parametric", "model",
                          "monte_carlo").
        confidence_level: Confidence level (e.g., 0.95 for 95%).
        lower_bound: Lower confidence bound.
        upper_bound: Upper confidence bound.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "uncertainty_type": uncertainty_type,
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
    Hash data quality indicator assessment.

    Args:
        dqi_scores: DQI scores by dimension (e.g., reliability, completeness,
                    temporal_correlation, geographical_correlation,
                    technological_correlation).
        composite: Composite DQI score.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "dqi_scores": {k: str(v) for k, v in dqi_scores.items()},
        "composite": str(composite),
    }
    return _compute_hash(data)


def hash_eui_benchmark(
    building_type: str,
    climate_zone: str,
    eui_kwh_per_sqm: Union[Decimal, float],
    source: str,
    year: int,
) -> str:
    """
    Hash EUI (Energy Use Intensity) benchmark lookup.

    Args:
        building_type: Building type (office, warehouse, retail, etc.).
        climate_zone: ASHRAE climate zone.
        eui_kwh_per_sqm: Energy Use Intensity value.
        source: Data source (ENERGY_STAR, CBECS, CIBSE).
        year: Benchmark data year.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "building_type": building_type,
        "climate_zone": climate_zone,
        "eui_kwh_per_sqm": str(eui_kwh_per_sqm),
        "source": source,
        "year": year,
    }
    return _compute_hash(data)


def hash_refrigerant_leakage(
    refrigerant_type: str,
    charge_kg: Union[Decimal, float],
    annual_leak_rate: Union[Decimal, float],
    gwp: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash building refrigerant leakage calculation.

    Args:
        refrigerant_type: Refrigerant type (e.g., R-410A, R-134a).
        charge_kg: Refrigerant charge in kg.
        annual_leak_rate: Annual leak rate (0.0-1.0).
        gwp: Global Warming Potential of refrigerant.
        co2e_kg: Fugitive emissions (kgCO2e).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "refrigerant_type": refrigerant_type,
        "charge_kg": str(charge_kg),
        "annual_leak_rate": str(annual_leak_rate),
        "gwp": str(gwp),
        "co2e_kg": str(co2e_kg),
        "formula": "charge_kg * annual_leak_rate * gwp",
    }
    return _compute_hash(data)


def hash_fleet_summary(
    vehicle_count: int,
    total_km: Union[Decimal, float],
    total_fuel_l: Union[Decimal, float],
    total_co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash fleet-level vehicle summary.

    Args:
        vehicle_count: Number of leased vehicles in fleet.
        total_km: Total annual kilometres across fleet.
        total_fuel_l: Total annual fuel consumption (litres).
        total_co2e_kg: Total fleet emissions (kgCO2e).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "vehicle_count": vehicle_count,
        "total_km": str(total_km),
        "total_fuel_l": str(total_fuel_l),
        "total_co2e_kg": str(total_co2e_kg),
    }
    return _compute_hash(data)


def hash_batch_result(batch_id: str, results: List[Dict[str, Any]]) -> str:
    """
    Hash batch calculation result.

    Args:
        batch_id: Batch identifier.
        results: List of individual asset results.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {"batch_id": batch_id, "results": results}
    return _compute_hash(data)


def hash_asset_input(asset: Dict[str, Any]) -> str:
    """
    Hash asset-level input data (generic).

    Args:
        asset: Leased asset data (type, characteristics, lease terms,
               energy consumption, etc.)

    Returns:
        SHA-256 hash (64-character hex string).
    """
    return _compute_hash(asset)


def hash_config(config: Dict[str, Any]) -> str:
    """
    Hash configuration data.

    Args:
        config: Configuration dictionary.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    return _compute_hash(config)


def hash_metadata(metadata: Dict[str, Any]) -> str:
    """
    Hash metadata.

    Args:
        metadata: Metadata dictionary.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    return _compute_hash(metadata)


def hash_arbitrary(data: Any) -> str:
    """
    Hash arbitrary data.

    Args:
        data: Any data to hash.

    Returns:
        SHA-256 hash (64-character hex string).
    """
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

    Thread-safe singleton pattern. Returns the same tracker instance
    across all callers in the process.

    Returns:
        ProvenanceTracker instance.

    Example:
        >>> tracker = get_provenance_tracker()
        >>> chain = tracker.start_chain("tenant-001")
    """
    global _tracker_instance
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = ProvenanceTracker()
        return _tracker_instance


def reset_provenance_tracker() -> None:
    """
    Reset singleton ProvenanceTracker instance.

    Useful for testing. Creates a fresh tracker on next access.
    Also resets the batch tracker since it depends on the main tracker.

    Example:
        >>> reset_provenance_tracker()
        >>> tracker = get_provenance_tracker()  # Fresh instance
    """
    global _tracker_instance, _batch_tracker_instance
    with _tracker_lock:
        _tracker_instance = None
    with _batch_tracker_lock:
        _batch_tracker_instance = None


def get_batch_provenance_tracker() -> BatchProvenanceTracker:
    """
    Get singleton BatchProvenanceTracker instance.

    Thread-safe singleton pattern. The batch tracker is initialized with
    the singleton ProvenanceTracker instance.

    Returns:
        BatchProvenanceTracker instance.

    Example:
        >>> batch_tracker = get_batch_provenance_tracker()
        >>> batch_id = batch_tracker.start_batch("tenant-001", 200)
    """
    global _batch_tracker_instance
    with _batch_tracker_lock:
        if _batch_tracker_instance is None:
            tracker = get_provenance_tracker()
            _batch_tracker_instance = BatchProvenanceTracker(tracker)
        return _batch_tracker_instance


def reset_batch_provenance_tracker() -> None:
    """
    Reset singleton BatchProvenanceTracker instance.

    Useful for testing. Creates a fresh batch tracker on next access.

    Example:
        >>> reset_batch_provenance_tracker()
        >>> batch_tracker = get_batch_provenance_tracker()
    """
    global _batch_tracker_instance
    with _batch_tracker_lock:
        _batch_tracker_instance = None


# ============================================================================
# CONVENIENCE FUNCTIONS (10 stage-specific recorders)
# ============================================================================


def create_chain(
    tenant_id: str = "default",
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[ProvenanceTracker, str]:
    """
    Create a new provenance chain using singleton tracker.

    Args:
        tenant_id: Tenant identifier (default: "default").
        metadata: Optional creation metadata.

    Returns:
        Tuple of (tracker, chain_id).

    Example:
        >>> tracker, chain_id = create_chain("tenant-001")
        >>> tracker.record_stage(chain_id, "VALIDATE", inp, out)
    """
    tracker = get_provenance_tracker()
    chain = tracker.start_chain(tenant_id, metadata)
    return tracker, chain.chain_id


def record_validate_stage(
    chain_id: str,
    input_data: Any,
    validated_data: Any,
    engine_id: str = "validation-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record VALIDATE stage.

    Records input validation and data quality checks for upstream
    leased assets data (floor area, energy, lease terms, etc.).

    Args:
        chain_id: Chain identifier.
        input_data: Raw input data (leased asset records).
        validated_data: Validated data (cleaned, type-checked records).
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata (e.g., rule_count, pass_rate, errors).
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.VALIDATE, input_data, validated_data,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_classify_stage(
    chain_id: str,
    input_data: Any,
    classified_data: Any,
    engine_id: str = "classification-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record CLASSIFY stage.

    Records lease classification results including asset type
    determination (building, vehicle, equipment, IT), lease type
    (operating vs finance per IFRS 16/ASC 842), and exemption
    checks (short-term, low-value).

    Args:
        chain_id: Chain identifier.
        input_data: Input data (validated asset records).
        classified_data: Classified data (asset_type, lease_type,
                         accounting_standard, exempt_status).
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata.
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.CLASSIFY, input_data, classified_data,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_normalize_stage(
    chain_id: str,
    input_data: Any,
    normalized_data: Any,
    engine_id: str = "normalization-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record NORMALIZE stage.

    Records unit normalization results: energy to kWh, currencies to
    USD, areas to sqm, fuel volumes to litres, and partial-year
    proration to annual equivalents.

    Args:
        chain_id: Chain identifier.
        input_data: Input data (classified records).
        normalized_data: Normalized data (standardized units).
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata.
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.NORMALIZE, input_data, normalized_data,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_resolve_efs_stage(
    chain_id: str,
    input_data: Any,
    resolved_efs: Any,
    engine_id: str = "ef-resolution-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record RESOLVE_EFS stage.

    Records emission factor resolution from DEFRA, EPA eGRID, IEA,
    Energy Star, or EEIO tables based on asset type, energy source,
    building type, region, and climate zone.

    Args:
        chain_id: Chain identifier.
        input_data: Input data (asset type, energy source, region, etc.).
        resolved_efs: Resolved emission factors with sources.
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata (e.g., ef_source, ef_year, hierarchy).
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.RESOLVE_EFS, input_data, resolved_efs,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_calculate_stage(
    chain_id: str,
    input_data: Any,
    calculation_results: Any,
    engine_id: str = "calculation-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record CALCULATE stage.

    Records leased asset emissions calculation applying asset-specific,
    average-data, spend-based, or lessor-specific methods to compute
    CO2e for each leased asset.

    Args:
        chain_id: Chain identifier.
        input_data: Calculation input (energy, EFs, allocation factors).
        calculation_results: Calculation results (CO2e per asset).
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata (e.g., method, formula, asset_type).
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.CALCULATE, input_data, calculation_results,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_allocate_stage(
    chain_id: str,
    input_data: Any,
    allocation_results: Any,
    engine_id: str = "allocation-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record ALLOCATE stage.

    Records allocation of shared-space or fleet emissions to the
    reporting entity's leased portion using floor-area, headcount,
    FTE, or custom allocation methods.

    Args:
        chain_id: Chain identifier.
        input_data: Pre-allocation data and allocation parameters.
        allocation_results: Allocated emissions by entity.
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata (e.g., allocation_method, shares,
                  common_area_factor).
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.ALLOCATE, input_data, allocation_results,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_aggregate_stage(
    chain_id: str,
    input_data: Any,
    aggregated_results: Any,
    engine_id: str = "aggregation-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record AGGREGATE stage.

    Records aggregation and summarization by asset type, building type,
    calculation method, region, and other dimensions.

    Args:
        chain_id: Chain identifier.
        input_data: Pre-aggregation data (per-asset results).
        aggregated_results: Aggregated results (totals by dimension).
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata (e.g., dimensions, asset_count).
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.AGGREGATE, input_data, aggregated_results,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_compliance_stage(
    chain_id: str,
    input_data: Any,
    compliance_results: Any,
    engine_id: str = "compliance-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record COMPLIANCE stage.

    Records regulatory compliance check results against GHG Protocol
    Scope 3, ISO 14064, CSRD, CDP, SBTi, SB 253, and GRI frameworks.

    Args:
        chain_id: Chain identifier.
        input_data: Data to check against compliance rules.
        compliance_results: Compliance check results per framework.
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata (e.g., frameworks_checked).
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.COMPLIANCE, input_data, compliance_results,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_provenance_stage(
    chain_id: str,
    input_data: Any,
    provenance_data: Any,
    engine_id: str = "provenance-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record PROVENANCE stage.

    Records provenance metadata assembly including DQI scores,
    uncertainty bounds, emission factor source citations, and
    methodology documentation for audit trail.

    Args:
        chain_id: Chain identifier.
        input_data: Calculation results with metadata.
        provenance_data: Assembled provenance record (DQI, citations,
                         methodology, uncertainty).
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata (e.g., dqi_composite, ef_sources).
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.PROVENANCE, input_data, provenance_data,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_seal_stage(
    chain_id: str,
    input_data: Any,
    seal_data: Any,
    engine_id: str = "seal-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record SEAL stage.

    Records the final sealing step before the chain is formally sealed.
    This stage captures the final state summary and any seal-specific
    metadata before the chain becomes immutable.

    Args:
        chain_id: Chain identifier.
        input_data: Pre-seal chain summary.
        seal_data: Seal verification data.
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata (e.g., seal_reason, auditor_id).
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.SEAL, input_data, seal_data,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def seal_and_verify(chain_id: str) -> Tuple[str, bool]:
    """
    Seal chain and verify integrity.

    Convenience function that seals the chain and immediately validates
    it, returning both the final hash and validation status.

    Args:
        chain_id: Chain identifier.

    Returns:
        Tuple of (final_hash, is_valid).

    Example:
        >>> final_hash, is_valid = seal_and_verify(chain_id)
        >>> assert is_valid, "Chain integrity check failed"
    """
    tracker = get_provenance_tracker()
    final_hash = tracker.seal_chain(chain_id)
    is_valid, _errors = tracker.validate_chain(chain_id)
    return final_hash, is_valid


def export_chain_json(chain_id: str) -> str:
    """
    Export chain as JSON.

    Args:
        chain_id: Chain identifier.

    Returns:
        JSON string representation of the complete chain.

    Example:
        >>> json_str = export_chain_json(chain_id)
        >>> chain_dict = json.loads(json_str)
    """
    tracker = get_provenance_tracker()
    chain_dict = tracker.export_chain(chain_id)
    return json.dumps(chain_dict, indent=2, sort_keys=True, default=str)


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
    # Singleton accessors
    "get_provenance_tracker",
    "reset_provenance_tracker",
    "get_batch_provenance_tracker",
    "reset_batch_provenance_tracker",
    # Hash utilities (internal but exported for testing)
    "_canonical_json",
    "_sha256",
    "_serialize",
    "_compute_hash",
    "_compute_chain_hash",
    "_merkle_hash",
    "_merkle_proof",
    "_verify_merkle_proof",
    # Standalone hash functions (20+)
    "hash_calculation_input",
    "hash_building_input",
    "hash_vehicle_input",
    "hash_equipment_input",
    "hash_it_asset_input",
    "hash_lease_classification",
    "hash_spend_input",
    "hash_average_data_input",
    "hash_lessor_specific_input",
    "hash_emission_factor",
    "hash_calculation_result",
    "hash_building_result",
    "hash_vehicle_result",
    "hash_equipment_result",
    "hash_it_asset_result",
    "hash_compliance_result",
    "hash_aggregation_result",
    "hash_allocation",
    "hash_batch_input",
    # Domain-specific parametric hash functions
    "hash_asset_specific_result",
    "hash_average_data_result",
    "hash_spend_result",
    "hash_lessor_specific_result",
    "hash_pue_calculation",
    "hash_wtt_result",
    "hash_partial_year_proration",
    "hash_classification_result",
    "hash_allocation_result",
    "hash_aggregation",
    "hash_uncertainty_result",
    "hash_dqi_result",
    "hash_eui_benchmark",
    "hash_refrigerant_leakage",
    "hash_fleet_summary",
    "hash_batch_result",
    "hash_asset_input",
    "hash_config",
    "hash_metadata",
    "hash_arbitrary",
    # Convenience functions (10 stage recorders + helpers)
    "create_chain",
    "record_validate_stage",
    "record_classify_stage",
    "record_normalize_stage",
    "record_resolve_efs_stage",
    "record_calculate_stage",
    "record_allocate_stage",
    "record_aggregate_stage",
    "record_compliance_stage",
    "record_provenance_stage",
    "record_seal_stage",
    "seal_and_verify",
    "export_chain_json",
]
