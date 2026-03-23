# -*- coding: utf-8 -*-
"""
Downstream Leased Assets Provenance Tracking - AGENT-MRV-026

SHA-256 chain-hashed provenance for zero-hallucination audit trails.
10 pipeline stages tracked with deterministic hashing.
Agent: GL-MRV-S3-013

This module implements a complete provenance tracking system for downstream
leased assets emissions calculations. Every step in the calculation pipeline
is recorded with SHA-256 hashes, creating an immutable audit trail that proves
no data was hallucinated or modified.

Pipeline Stages:
    1. VALIDATE - Input validation and data quality checks
    2. CLASSIFY - Asset classification (category, type, lease terms)
    3. NORMALIZE - Unit normalization (area, energy, currency, dates)
    4. RESOLVE_EFS - Emission factor resolution (DEFRA, EPA, IEA, CIBSE)
    5. CALCULATE - Core emissions calculation (deterministic formulas)
    6. ALLOCATE - Tenant/lessor allocation (floor area, headcount, FTE)
    7. AGGREGATE - Portfolio aggregation and summarization
    8. COMPLIANCE - Regulatory compliance checking
    9. PROVENANCE - Provenance chain finalization
    10. SEAL - Final sealing and verification

Example:
    >>> builder = get_provenance_builder()
    >>> chain_id = builder.start_chain()
    >>> builder.record_stage(chain_id, "VALIDATE", input_data, validated_data)
    >>> builder.record_stage(chain_id, "CALCULATE", calc_input, calc_output)
    >>> final_hash = builder.seal_chain(chain_id)
    >>> assert builder.validate_chain(chain_id)

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-013
"""

import hashlib
import json
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

AGENT_ID = "GL-MRV-S3-013"
AGENT_VERSION = "1.0.0"
HASH_ALGORITHM = "sha256"
ENCODING = "utf-8"


class ProvenanceStage(str, Enum):
    """Pipeline stages for provenance tracking."""

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
class StageRecord:
    """
    Single provenance record for one stage in the calculation pipeline.

    This is a frozen (immutable) record. The chain_hash links this record
    to the previous record, creating an immutable chain.

    Attributes:
        record_id: Unique identifier for this record (UUID)
        stage: Pipeline stage name
        timestamp: ISO 8601 UTC timestamp
        input_hash: SHA-256 hash of input data
        output_hash: SHA-256 hash of output data
        chain_hash: SHA-256 hash linking to previous record
        previous_hash: Chain hash of previous record (or empty for first)
        agent_id: Agent identifier (GL-MRV-S3-013)
        agent_version: Agent version (1.0.0)
        metadata: Additional context (optional)
    """

    record_id: str
    stage: str
    timestamp: str
    input_hash: str
    output_hash: str
    chain_hash: str
    previous_hash: str
    agent_id: str = AGENT_ID
    agent_version: str = AGENT_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert record to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass
class ProvenanceChain:
    """
    Complete provenance chain for a calculation.

    Tracks all records in order, with validation status and final hash
    when sealed.

    Attributes:
        chain_id: Unique identifier for this chain
        records: Ordered list of stage records
        started_at: ISO 8601 UTC timestamp of chain start
        sealed_at: ISO 8601 UTC timestamp when sealed (or None)
        final_hash: Final SHA-256 hash when sealed (or None)
        chain_valid: Whether chain validation passed
    """

    chain_id: str
    records: List[StageRecord] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sealed_at: Optional[str] = None
    final_hash: Optional[str] = None
    chain_valid: bool = True

    @property
    def is_valid(self) -> bool:
        """Whether the chain is valid."""
        return self.chain_valid

    @property
    def root_hash(self) -> str:
        """Root hash of the chain (first record's chain_hash or empty)."""
        if self.records:
            return self.records[0].chain_hash
        return ""

    @property
    def is_sealed(self) -> bool:
        """Whether the chain has been sealed."""
        return self.sealed_at is not None

    @property
    def record_count(self) -> int:
        """Number of records in the chain."""
        return len(self.records)

    @property
    def stages(self) -> List[str]:
        """List of stage names in order."""
        return [r.stage for r in self.records]

    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary."""
        return {
            "chain_id": self.chain_id,
            "records": [r.to_dict() for r in self.records],
            "started_at": self.started_at,
            "sealed_at": self.sealed_at,
            "final_hash": self.final_hash,
            "chain_valid": self.chain_valid,
            "is_valid": self.is_valid,
            "is_sealed": self.is_sealed,
            "root_hash": self.root_hash,
            "record_count": self.record_count,
            "stages": self.stages,
        }

    def to_json(self) -> str:
        """Convert chain to JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass
class BatchProvenance:
    """
    Provenance tracking for batch calculations.

    Tracks multiple individual chains plus batch-level Merkle aggregation.

    Attributes:
        batch_id: Unique identifier for this batch
        individual_chain_ids: List of chain IDs in this batch
        batch_started_at: ISO 8601 UTC timestamp
        batch_sealed_at: ISO 8601 UTC timestamp when sealed (or None)
        batch_hash: Merkle-style aggregate hash (or None)
        item_count: Number of items in batch
    """

    batch_id: str
    individual_chain_ids: List[str] = field(default_factory=list)
    batch_started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    batch_sealed_at: Optional[str] = None
    batch_hash: Optional[str] = None
    item_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch provenance to dictionary."""
        return {
            "batch_id": self.batch_id,
            "individual_chain_ids": self.individual_chain_ids,
            "batch_started_at": self.batch_started_at,
            "batch_sealed_at": self.batch_sealed_at,
            "batch_hash": self.batch_hash,
            "item_count": self.item_count,
            "chain_count": len(self.individual_chain_ids),
            "is_sealed": self.batch_sealed_at is not None,
        }


# ============================================================================
# HASH UTILITIES
# ============================================================================


def _serialize(obj: Any) -> str:
    """
    Serialize object to deterministic JSON string.

    Converts Decimal to string, datetime to ISO format, Enum to value,
    sorts keys, and handles nested dicts/lists and frozen models.

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
        Lowercase hex SHA-256 hash (64 characters)
    """
    serialized = _serialize(data)
    return hashlib.sha256(serialized.encode(ENCODING)).hexdigest()


def _compute_chain_hash(
    previous_hash: str, stage: str, input_hash: str, output_hash: str
) -> str:
    """
    Compute chain hash linking to previous record.

    Chain hash = SHA-256(previous_hash + stage + input_hash + output_hash)

    Args:
        previous_hash: Chain hash of previous record (empty string for first)
        stage: Current stage name
        input_hash: Hash of input data
        output_hash: Hash of output data

    Returns:
        SHA-256 chain hash (64 characters)
    """
    chain_data = f"{previous_hash}|{stage}|{input_hash}|{output_hash}"
    return hashlib.sha256(chain_data.encode(ENCODING)).hexdigest()


def _merkle_hash(hashes: List[str]) -> str:
    """
    Compute Merkle-style hash of list of hashes.

    Used for batch aggregation. Sorts hashes for determinism before
    combining and hashing.

    Args:
        hashes: List of SHA-256 hashes

    Returns:
        Aggregate SHA-256 hash (64 characters)
    """
    if not hashes:
        return hashlib.sha256(b"").hexdigest()
    if len(hashes) == 1:
        return hashes[0]

    # Sort for determinism
    sorted_hashes = sorted(hashes)
    combined = "|".join(sorted_hashes)
    return hashlib.sha256(combined.encode(ENCODING)).hexdigest()


# ============================================================================
# PROVENANCE CHAIN BUILDER
# ============================================================================


class ProvenanceChainBuilder:
    """
    Main provenance tracking system for downstream leased assets calculations.

    Manages multiple chains, records stages, validates integrity.
    Thread-safe with RLock.

    Example:
        >>> builder = ProvenanceChainBuilder()
        >>> chain_id = builder.start_chain()
        >>> builder.record_stage(chain_id, "VALIDATE", input_data, output_data)
        >>> builder.seal_chain(chain_id)
        >>> assert builder.validate_chain(chain_id)
    """

    def __init__(
        self, agent_id: str = AGENT_ID, agent_version: str = AGENT_VERSION
    ):
        """
        Initialize provenance chain builder.

        Args:
            agent_id: Agent identifier
            agent_version: Agent version
        """
        self.agent_id = agent_id
        self.agent_version = agent_version
        self._chains: Dict[str, ProvenanceChain] = {}
        self._lock = threading.RLock()

    def start_chain(self, chain_id: Optional[str] = None) -> str:
        """
        Start a new provenance chain.

        Args:
            chain_id: Optional chain ID (UUID generated if not provided)

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

            chain = ProvenanceChain(chain_id=chain_id)
            self._chains[chain_id] = chain
            return chain_id

    def record_stage(
        self,
        chain_id: str,
        stage: Union[str, ProvenanceStage],
        input_data: Any,
        output_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StageRecord:
        """
        Record a pipeline stage in the provenance chain.

        Args:
            chain_id: Chain identifier
            stage: Pipeline stage name
            input_data: Input data for this stage
            output_data: Output data from this stage
            metadata: Optional metadata

        Returns:
            Created StageRecord

        Raises:
            ValueError: If chain not found or already sealed
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            if chain.sealed_at is not None:
                raise ValueError(f"Chain {chain_id} already sealed")

            # Convert enum to string
            stage_str = stage.value if isinstance(stage, ProvenanceStage) else stage

            # Compute hashes
            input_hash = _compute_hash(input_data)
            output_hash = _compute_hash(output_data)

            # Get previous hash
            previous_hash = ""
            if chain.records:
                previous_hash = chain.records[-1].chain_hash

            # Compute chain hash
            chain_hash = _compute_chain_hash(
                previous_hash, stage_str, input_hash, output_hash
            )

            # Create record
            record = StageRecord(
                record_id=str(uuid.uuid4()),
                stage=stage_str,
                timestamp=datetime.now(timezone.utc).isoformat(),
                input_hash=input_hash,
                output_hash=output_hash,
                chain_hash=chain_hash,
                previous_hash=previous_hash,
                agent_id=self.agent_id,
                agent_version=self.agent_version,
                metadata=metadata or {},
            )

            # Add to chain
            chain.records.append(record)
            return record

    def validate_chain(self, chain_id: str) -> bool:
        """
        Validate integrity of provenance chain.

        Checks:
        1. Each record's chain_hash matches recomputed value
        2. Each record's previous_hash matches previous record's chain_hash
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

            if not chain.records:
                return True

            for i, record in enumerate(chain.records):
                # Check previous_hash
                if i == 0:
                    if record.previous_hash != "":
                        chain.chain_valid = False
                        return False
                else:
                    if record.previous_hash != chain.records[i - 1].chain_hash:
                        chain.chain_valid = False
                        return False

                # Recompute and verify chain_hash
                expected_hash = _compute_chain_hash(
                    record.previous_hash,
                    record.stage,
                    record.input_hash,
                    record.output_hash,
                )
                if record.chain_hash != expected_hash:
                    chain.chain_valid = False
                    return False

            chain.chain_valid = True
            return True

    def seal_chain(self, chain_id: str) -> str:
        """
        Seal the provenance chain.

        Records final timestamp and hash. No more records can be added.

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

            # Validate before sealing
            if not self.validate_chain(chain_id):
                raise ValueError(f"Chain {chain_id} failed validation")

            # Compute final hash
            if chain.records:
                final_hash = chain.records[-1].chain_hash
            else:
                final_hash = hashlib.sha256(chain_id.encode(ENCODING)).hexdigest()

            chain.sealed_at = datetime.now(timezone.utc).isoformat()
            chain.final_hash = final_hash
            return final_hash

    def build_chain(
        self,
        stages: List[Tuple[Union[str, ProvenanceStage], Any, Any]],
        chain_id: Optional[str] = None,
        seal: bool = True,
    ) -> Tuple[str, str]:
        """
        Build a complete provenance chain from a list of stages.

        Convenience method that starts a chain, records all stages,
        and optionally seals it.

        Args:
            stages: List of (stage, input_data, output_data) tuples
            chain_id: Optional chain ID
            seal: Whether to seal the chain after recording (default True)

        Returns:
            Tuple of (chain_id, final_hash)
        """
        cid = self.start_chain(chain_id)
        for stage, inp, out in stages:
            self.record_stage(cid, stage, inp, out)

        final_hash = ""
        if seal:
            final_hash = self.seal_chain(cid)
        else:
            chain = self._chains[cid]
            final_hash = chain.records[-1].chain_hash if chain.records else ""

        return cid, final_hash

    def export_chain(
        self, chain_id: str, format: str = "json"
    ) -> str:
        """
        Export provenance chain.

        Args:
            chain_id: Chain identifier
            format: Export format ("json")

        Returns:
            Serialized chain

        Raises:
            ValueError: If chain not found or format unsupported
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            if format == "json":
                return chain.to_json()
            else:
                raise ValueError(f"Unsupported format: {format}")

    def verify_record(self, record: StageRecord) -> bool:
        """
        Verify a single provenance record.

        Recomputes chain_hash and checks it matches.

        Args:
            record: StageRecord to verify

        Returns:
            True if record is valid, False otherwise
        """
        expected_hash = _compute_chain_hash(
            record.previous_hash,
            record.stage,
            record.input_hash,
            record.output_hash,
        )
        return record.chain_hash == expected_hash

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

        Raises:
            ValueError: If chain not found
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            stage_str = stage.value if isinstance(stage, ProvenanceStage) else stage
            for record in chain.records:
                if record.stage == stage_str:
                    return record.output_hash
            return None

    def compute_chain_hash(self, chain_id: str) -> str:
        """
        Compute the current chain hash (last record's chain_hash).

        Args:
            chain_id: Chain identifier

        Returns:
            Current chain hash

        Raises:
            ValueError: If chain not found or empty
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            if not chain.records:
                raise ValueError(f"Chain {chain_id} has no records")
            return chain.records[-1].chain_hash

    def compute_merkle_root(self, chain_ids: List[str]) -> str:
        """
        Compute Merkle root hash from multiple chain final hashes.

        Args:
            chain_ids: List of chain IDs

        Returns:
            Merkle root hash

        Raises:
            ValueError: If any chain not found
        """
        with self._lock:
            hashes: List[str] = []
            for cid in chain_ids:
                chain = self._chains.get(cid)
                if chain is None:
                    raise ValueError(f"Chain {cid} not found")
                if chain.final_hash:
                    hashes.append(chain.final_hash)
                elif chain.records:
                    hashes.append(chain.records[-1].chain_hash)
            return _merkle_hash(hashes)

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

    def get_chain_summary(self, chain_id: str) -> Dict[str, Any]:
        """
        Get summary of provenance chain.

        Args:
            chain_id: Chain identifier

        Returns:
            Summary dictionary with key metrics

        Raises:
            ValueError: If chain not found
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            return {
                "chain_id": chain.chain_id,
                "agent_id": self.agent_id,
                "agent_version": self.agent_version,
                "started_at": chain.started_at,
                "sealed_at": chain.sealed_at,
                "final_hash": chain.final_hash,
                "chain_valid": chain.chain_valid,
                "record_count": chain.record_count,
                "stages": chain.stages,
                "is_sealed": chain.is_sealed,
                "root_hash": chain.root_hash,
            }

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
        """
        Get list of all chain IDs.

        Returns:
            List of chain IDs
        """
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

    def reset(self) -> None:
        """
        Clear all chains tracked by this builder instance.

        Useful for testing. Removes all chain state.
        """
        with self._lock:
            self._chains.clear()


# ============================================================================
# BATCH PROVENANCE BUILDER
# ============================================================================


class BatchProvenanceBuilder:
    """
    Provenance tracking for batch calculations.

    Manages multiple individual chains plus batch-level Merkle aggregation.

    Example:
        >>> builder = ProvenanceChainBuilder()
        >>> batch_builder = BatchProvenanceBuilder(builder)
        >>> batch_id = batch_builder.start_batch(100)
        >>> for item in items:
        ...     chain_id = builder.start_chain()
        ...     # ... process item ...
        ...     batch_builder.add_chain(batch_id, chain_id)
        >>> batch_hash = batch_builder.seal_batch(batch_id)
    """

    def __init__(self, builder: ProvenanceChainBuilder):
        """
        Initialize batch provenance builder.

        Args:
            builder: ProvenanceChainBuilder instance for individual chains
        """
        self.builder = builder
        self._batches: Dict[str, BatchProvenance] = {}
        self._lock = threading.RLock()

    def start_batch(
        self, item_count: int = 0, batch_id: Optional[str] = None
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

    def seal_batch(self, batch_id: str) -> str:
        """
        Seal the batch.

        Computes Merkle-style aggregate hash of all chain hashes.

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

            # Collect final hashes from all chains
            chain_hashes: List[str] = []
            for chain_id in batch.individual_chain_ids:
                try:
                    chain = self.builder.get_chain(chain_id)
                    if chain.final_hash:
                        chain_hashes.append(chain.final_hash)
                    elif chain.records:
                        chain_hashes.append(chain.records[-1].chain_hash)
                except ValueError:
                    # Chain not found - skip
                    pass

            # Compute Merkle hash
            batch_hash = _merkle_hash(chain_hashes)
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

        Raises:
            ValueError: If batch not found
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
            }

    def get_batch(self, batch_id: str) -> BatchProvenance:
        """
        Get batch provenance.

        Args:
            batch_id: Batch identifier

        Returns:
            BatchProvenance

        Raises:
            ValueError: If batch not found
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            return batch


# ============================================================================
# STANDALONE HASH FUNCTIONS (15)
# All return 64-character hex SHA-256 strings.
# ============================================================================


def hash_input(data: Dict[str, Any]) -> str:
    """
    Hash complete calculation input data.

    Args:
        data: Input data dictionary

    Returns:
        64-char hex SHA-256 hash
    """
    return _compute_hash(data)


def hash_building_asset(building_data: Dict[str, Any]) -> str:
    """
    Hash building asset data.

    Args:
        building_data: Building asset data (type, area, EUI, climate zone,
                       vintage, occupancy, lease terms, etc.)

    Returns:
        64-char hex SHA-256 hash
    """
    return _compute_hash(building_data)


def hash_vehicle_asset(vehicle_data: Dict[str, Any]) -> str:
    """
    Hash vehicle asset data.

    Args:
        vehicle_data: Vehicle asset data (type, fuel type, distance,
                       fuel consumption, lease terms, etc.)

    Returns:
        64-char hex SHA-256 hash
    """
    return _compute_hash(vehicle_data)


def hash_equipment_asset(equipment_data: Dict[str, Any]) -> str:
    """
    Hash equipment asset data.

    Args:
        equipment_data: Equipment asset data (type, power rating,
                         load factor, operating hours, fuel, etc.)

    Returns:
        64-char hex SHA-256 hash
    """
    return _compute_hash(equipment_data)


def hash_it_asset(it_data: Dict[str, Any]) -> str:
    """
    Hash IT asset data.

    Args:
        it_data: IT asset data (type, power consumption, PUE,
                  utilization, cooling, etc.)

    Returns:
        64-char hex SHA-256 hash
    """
    return _compute_hash(it_data)


def hash_lease_info(lease_data: Dict[str, Any]) -> str:
    """
    Hash lease information.

    Args:
        lease_data: Lease data (start date, end date, lease type,
                     lessee, lessor share, etc.)

    Returns:
        64-char hex SHA-256 hash
    """
    return _compute_hash(lease_data)


def hash_allocation(allocation_data: Dict[str, Any]) -> str:
    """
    Hash allocation data.

    Args:
        allocation_data: Allocation data (method, lessor share,
                          lessee share, common area, vacancy, etc.)

    Returns:
        64-char hex SHA-256 hash
    """
    return _compute_hash(allocation_data)


def hash_asset_result(
    asset_id: str,
    asset_category: str,
    asset_type: str,
    co2e_kg: Union[Decimal, float],
    method: str,
) -> str:
    """
    Hash individual asset calculation result.

    Args:
        asset_id: Asset identifier
        asset_category: Category (building/vehicle/equipment/it_asset)
        asset_type: Specific type within category
        co2e_kg: Total emissions in kgCO2e
        method: Calculation method used

    Returns:
        64-char hex SHA-256 hash
    """
    data = {
        "asset_id": asset_id,
        "asset_category": asset_category,
        "asset_type": asset_type,
        "co2e_kg": str(co2e_kg),
        "method": method,
    }
    return _compute_hash(data)


def hash_portfolio_result(
    portfolio_id: str,
    total_co2e_kg: Union[Decimal, float],
    asset_count: int,
    method_breakdown: Dict[str, int],
) -> str:
    """
    Hash portfolio-level aggregation result.

    Args:
        portfolio_id: Portfolio identifier
        total_co2e_kg: Total portfolio emissions in kgCO2e
        asset_count: Number of assets in portfolio
        method_breakdown: Count of assets by calculation method

    Returns:
        64-char hex SHA-256 hash
    """
    data = {
        "portfolio_id": portfolio_id,
        "total_co2e_kg": str(total_co2e_kg),
        "asset_count": asset_count,
        "method_breakdown": method_breakdown,
    }
    return _compute_hash(data)


def hash_calculation(
    energy_value: Union[Decimal, float],
    emission_factor: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
    formula: str,
) -> str:
    """
    Hash a deterministic calculation step.

    Args:
        energy_value: Energy consumption or activity data
        emission_factor: Emission factor applied
        co2e_kg: Resulting emissions in kgCO2e
        formula: Formula string (e.g., "energy_value * emission_factor")

    Returns:
        64-char hex SHA-256 hash
    """
    data = {
        "energy_value": str(energy_value),
        "emission_factor": str(emission_factor),
        "co2e_kg": str(co2e_kg),
        "formula": formula,
    }
    return _compute_hash(data)


def hash_aggregation(
    category_totals: Dict[str, Union[Decimal, float]],
    grand_total: Union[Decimal, float],
    asset_count: int,
) -> str:
    """
    Hash aggregation result.

    Args:
        category_totals: Emissions totals per asset category
        grand_total: Grand total emissions in kgCO2e
        asset_count: Total number of assets aggregated

    Returns:
        64-char hex SHA-256 hash
    """
    data = {
        "category_totals": {k: str(v) for k, v in category_totals.items()},
        "grand_total": str(grand_total),
        "asset_count": asset_count,
    }
    return _compute_hash(data)


def hash_compliance(
    framework: str,
    status: str,
    findings: List[Dict[str, Any]],
) -> str:
    """
    Hash compliance check result.

    Args:
        framework: Compliance framework name
        status: Overall compliance status
        findings: List of compliance findings

    Returns:
        64-char hex SHA-256 hash
    """
    data = {
        "framework": framework,
        "status": status,
        "findings": findings,
    }
    return _compute_hash(data)


def hash_dqi(
    technological: Union[Decimal, float],
    temporal: Union[Decimal, float],
    geographical: Union[Decimal, float],
    completeness: Union[Decimal, float],
    reliability: Union[Decimal, float],
    weighted_score: Union[Decimal, float],
) -> str:
    """
    Hash Data Quality Indicator scores.

    Args:
        technological: Technological representativeness score (1-5)
        temporal: Temporal representativeness score (1-5)
        geographical: Geographical representativeness score (1-5)
        completeness: Completeness score (1-5)
        reliability: Reliability score (1-5)
        weighted_score: Weighted aggregate score

    Returns:
        64-char hex SHA-256 hash
    """
    data = {
        "technological": str(technological),
        "temporal": str(temporal),
        "geographical": str(geographical),
        "completeness": str(completeness),
        "reliability": str(reliability),
        "weighted_score": str(weighted_score),
    }
    return _compute_hash(data)


def hash_uncertainty(
    method: str,
    mean: Union[Decimal, float],
    lower_bound: Union[Decimal, float],
    upper_bound: Union[Decimal, float],
    confidence: Union[Decimal, float],
) -> str:
    """
    Hash uncertainty quantification result.

    Args:
        method: Uncertainty method (MONTE_CARLO, BOOTSTRAP, etc.)
        mean: Mean emissions estimate
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        confidence: Confidence level (e.g., 0.95)

    Returns:
        64-char hex SHA-256 hash
    """
    data = {
        "method": method,
        "mean": str(mean),
        "lower_bound": str(lower_bound),
        "upper_bound": str(upper_bound),
        "confidence": str(confidence),
    }
    return _compute_hash(data)


def hash_provenance_record(
    record_id: str,
    stage: str,
    input_hash: str,
    output_hash: str,
    chain_hash: str,
) -> str:
    """
    Hash a provenance record itself for external verification.

    Args:
        record_id: Record UUID
        stage: Pipeline stage name
        input_hash: Input data hash
        output_hash: Output data hash
        chain_hash: Chain hash

    Returns:
        64-char hex SHA-256 hash
    """
    data = {
        "record_id": record_id,
        "stage": stage,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "chain_hash": chain_hash,
    }
    return _compute_hash(data)


# ============================================================================
# SINGLETON PATTERN
# ============================================================================


_builder_instance: Optional[ProvenanceChainBuilder] = None
_builder_lock = threading.RLock()


def get_provenance_builder() -> ProvenanceChainBuilder:
    """
    Get the singleton ProvenanceChainBuilder instance.

    Thread-safe accessor for the global provenance builder.

    Returns:
        ProvenanceChainBuilder singleton instance

    Example:
        >>> builder = get_provenance_builder()
        >>> chain_id = builder.start_chain()
        >>> builder.record_stage(chain_id, "VALIDATE", input_data, output_data)
    """
    global _builder_instance

    if _builder_instance is None:
        with _builder_lock:
            if _builder_instance is None:
                _builder_instance = ProvenanceChainBuilder()

    return _builder_instance


def reset_provenance_builder() -> None:
    """
    Reset the singleton provenance builder for testing purposes.

    Clears all chains and resets the singleton instance.
    Should only be called in test teardown.
    """
    global _builder_instance

    with _builder_lock:
        if _builder_instance is not None:
            _builder_instance.reset()
        _builder_instance = None


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Constants
    "AGENT_ID",
    "AGENT_VERSION",
    "HASH_ALGORITHM",
    # Enum
    "ProvenanceStage",
    # Data models
    "StageRecord",
    "ProvenanceChain",
    "BatchProvenance",
    # Builder classes
    "ProvenanceChainBuilder",
    "BatchProvenanceBuilder",
    # Singleton accessors
    "get_provenance_builder",
    "reset_provenance_builder",
    # Standalone hash functions (15)
    "hash_input",
    "hash_building_asset",
    "hash_vehicle_asset",
    "hash_equipment_asset",
    "hash_it_asset",
    "hash_lease_info",
    "hash_allocation",
    "hash_asset_result",
    "hash_portfolio_result",
    "hash_calculation",
    "hash_aggregation",
    "hash_compliance",
    "hash_dqi",
    "hash_uncertainty",
    "hash_provenance_record",
]
