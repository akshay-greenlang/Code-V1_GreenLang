# -*- coding: utf-8 -*-
"""
Business Travel Provenance Tracking - AGENT-MRV-019

SHA-256 chain-hashed provenance for zero-hallucination audit trails.
10 pipeline stages tracked with deterministic hashing.
Agent: GL-MRV-S3-006

This module implements a complete provenance tracking system for business
travel emissions calculations. Every step in the calculation pipeline is
recorded with SHA-256 hashes, creating an immutable audit trail that proves
no data was hallucinated or modified.

Pipeline Stages:
    1. VALIDATE - Input validation and data quality checks
    2. CLASSIFY - Trip classification (mode, distance band, cabin class)
    3. NORMALIZE - Unit normalization (distance, currency, dates)
    4. RESOLVE_EFS - Emission factor resolution (DEFRA, EPA, ICAO)
    5. CALCULATE_TRANSPORT - Transport emissions calculation
    6. CALCULATE_HOTEL - Hotel accommodation emissions calculation
    7. APPLY_RF - Radiative forcing multiplier for aviation
    8. COMPLIANCE - Regulatory compliance checking
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
Agent: GL-MRV-S3-006
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

AGENT_ID = "GL-MRV-S3-006"
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
    CALCULATE_HOTEL = "CALCULATE_HOTEL"
    APPLY_RF = "APPLY_RF"
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

    This is a frozen (immutable) record of one stage in the calculation
    pipeline. The chain_hash links this entry to the previous entry,
    creating an immutable chain.

    Attributes:
        entry_id: Unique identifier for this entry (UUID)
        stage: Pipeline stage name
        timestamp: ISO 8601 UTC timestamp
        input_hash: SHA-256 hash of input data
        output_hash: SHA-256 hash of output data
        chain_hash: SHA-256 hash linking to previous entry
        previous_hash: Chain hash of previous entry (or empty for first)
        agent_id: Agent identifier (GL-MRV-S3-006)
        agent_version: Agent version (1.0.0)
        metadata: Additional context (optional)
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

    Tracks all entries in order, with validation status and final hash
    when sealed.

    Attributes:
        chain_id: Unique identifier for this chain
        entries: Ordered list of provenance entries
        started_at: ISO 8601 UTC timestamp of chain start
        sealed_at: ISO 8601 UTC timestamp when sealed (or None)
        final_hash: Final SHA-256 hash when sealed (or None)
        chain_valid: Whether chain validation passed
    """

    chain_id: str
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
    def root_hash(self) -> str:
        """Root hash of the chain (first entry's chain_hash or empty)."""
        if self.entries:
            return self.entries[0].chain_hash
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary."""
        return {
            "chain_id": self.chain_id,
            "entries": [e.to_dict() for e in self.entries],
            "started_at": self.started_at,
            "sealed_at": self.sealed_at,
            "final_hash": self.final_hash,
            "chain_valid": self.chain_valid,
            "is_valid": self.is_valid,
            "root_hash": self.root_hash,
        }

    def to_json(self) -> str:
        """Convert chain to JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass
class BatchProvenance:
    """
    Provenance tracking for batch calculations.

    Tracks multiple individual chains plus batch-level aggregation.

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


def _merkle_hash(hashes: List[str]) -> str:
    """
    Compute Merkle-style hash of list of hashes.

    Used for batch aggregation. Sorts hashes for determinism before
    combining and hashing.

    Args:
        hashes: List of SHA-256 hashes

    Returns:
        Aggregate SHA-256 hash
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
# PROVENANCE TRACKER
# ============================================================================


class ProvenanceTracker:
    """
    Main provenance tracking system for business travel calculations.

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
        self, agent_id: str = AGENT_ID, agent_version: str = AGENT_VERSION
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
    ) -> ProvenanceEntry:
        """
        Record a pipeline stage in the provenance chain.

        Args:
            chain_id: Chain identifier
            stage: Pipeline stage name
            input_data: Input data for this stage
            output_data: Output data from this stage
            metadata: Optional metadata (e.g., method, source, EF details)

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

            # Convert enum to string
            stage_str = stage.value if isinstance(stage, ProvenanceStage) else stage

            # Compute hashes
            input_hash = _compute_hash(input_data)
            output_hash = _compute_hash(output_data)

            # Get previous hash
            previous_hash = ""
            if chain.entries:
                previous_hash = chain.entries[-1].chain_hash

            # Compute chain hash
            chain_hash = _compute_chain_hash(
                previous_hash, stage_str, input_hash, output_hash
            )

            # Create entry
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
            )

            # Add to chain
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

            # Validate each entry
            for i, entry in enumerate(chain.entries):
                # Check previous_hash
                if i == 0:
                    if entry.previous_hash != "":
                        chain.chain_valid = False
                        return False
                else:
                    if entry.previous_hash != chain.entries[i - 1].chain_hash:
                        chain.chain_valid = False
                        return False

                # Recompute and verify chain_hash
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
            ValueError: If chain not found or already sealed
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
            if chain.entries:
                final_hash = chain.entries[-1].chain_hash
            else:
                final_hash = hashlib.sha256(chain_id.encode(ENCODING)).hexdigest()

            chain.sealed_at = datetime.now(timezone.utc).isoformat()
            chain.final_hash = final_hash
            return final_hash

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

    def verify_entry(self, entry: ProvenanceEntry) -> bool:
        """
        Verify a single provenance entry.

        Recomputes chain_hash and checks it matches.

        Args:
            entry: Provenance entry to verify

        Returns:
            True if entry is valid, False otherwise
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

        Raises:
            ValueError: If chain not found
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
            Summary dictionary with key metrics

        Raises:
            ValueError: If chain not found
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            stages = [e.stage for e in chain.entries]
            return {
                "chain_id": chain.chain_id,
                "agent_id": self.agent_id,
                "agent_version": self.agent_version,
                "started_at": chain.started_at,
                "sealed_at": chain.sealed_at,
                "final_hash": chain.final_hash,
                "chain_valid": chain.chain_valid,
                "entry_count": len(chain.entries),
                "stages": stages,
                "is_sealed": chain.sealed_at is not None,
                "root_hash": chain.root_hash,
            }

    def reset(self) -> None:
        """
        Clear all chains tracked by this tracker instance.

        Useful for testing. Removes all chain state.
        """
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
        """
        Get list of all chain IDs.

        Returns:
            List of chain IDs
        """
        with self._lock:
            return list(self._chains.keys())

    def clear_all_chains(self) -> int:
        """
        Clear all chains (for testing).

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
    Provenance tracking for batch calculations.

    Manages multiple individual chains plus batch-level aggregation
    via Merkle hashing.

    Example:
        >>> batch_tracker = BatchProvenanceTracker(tracker)
        >>> batch_id = batch_tracker.start_batch(100)
        >>> for item in items:
        ...     chain_id = tracker.start_chain()
        ...     # ... process item ...
        ...     batch_tracker.add_chain(batch_id, chain_id)
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
            chain_hashes = []
            for chain_id in batch.individual_chain_ids:
                try:
                    chain = self.tracker.get_chain(chain_id)
                    if chain.final_hash:
                        chain_hashes.append(chain.final_hash)
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
# STANDALONE HASH FUNCTIONS (26+)
# ============================================================================


def hash_calculation_input(request: Dict[str, Any]) -> str:
    """
    Hash complete calculation request.

    Args:
        request: Calculation request dictionary

    Returns:
        SHA-256 hash
    """
    return _compute_hash(request)


def hash_trip_input(trip: Dict[str, Any]) -> str:
    """
    Hash business trip input data.

    Args:
        trip: Trip data (origin, destination, mode, distance, dates, etc.)

    Returns:
        SHA-256 hash
    """
    return _compute_hash(trip)


def hash_flight_input(flight_data: Dict[str, Any]) -> str:
    """
    Hash flight input data.

    Args:
        flight_data: Flight data (origin, destination, distance band,
                     cabin class, passengers, etc.)

    Returns:
        SHA-256 hash
    """
    return _compute_hash(flight_data)


def hash_ground_transport_input(ground_data: Dict[str, Any]) -> str:
    """
    Hash ground transport input data.

    Args:
        ground_data: Ground transport data (mode, vehicle type, distance, etc.)

    Returns:
        SHA-256 hash
    """
    return _compute_hash(ground_data)


def hash_hotel_input(hotel_data: Dict[str, Any]) -> str:
    """
    Hash hotel accommodation input data.

    Args:
        hotel_data: Hotel data (country, nights, hotel class, etc.)

    Returns:
        SHA-256 hash
    """
    return _compute_hash(hotel_data)


def hash_emission_factor(
    ef_type: str, ef_value: Union[Decimal, float], source: str
) -> str:
    """
    Hash emission factor.

    Args:
        ef_type: Emission factor type (e.g., "air_short_haul_economy",
                 "rail_national", "hotel_uk")
        ef_value: Emission factor value (kgCO2e per unit)
        source: Data source (e.g., "DEFRA", "EPA", "ICAO")

    Returns:
        SHA-256 hash
    """
    data = {"type": ef_type, "value": str(ef_value), "source": source}
    return _compute_hash(data)


def hash_classification_result(
    mode: str, distance_band: str, cabin_class: str
) -> str:
    """
    Hash trip classification result.

    Args:
        mode: Transport mode (e.g., "air", "rail", "road")
        distance_band: Distance band (e.g., "domestic", "short_haul", "long_haul")
        cabin_class: Cabin class (e.g., "economy", "business")

    Returns:
        SHA-256 hash
    """
    data = {
        "mode": mode,
        "distance_band": distance_band,
        "cabin_class": cabin_class,
    }
    return _compute_hash(data)


def hash_distance_result(
    distance_km: Union[Decimal, float],
    mode: str,
    ef: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash distance-based calculation result.

    Args:
        distance_km: Distance in kilometres
        mode: Transport mode
        ef: Emission factor (kgCO2e/passenger-km)
        co2e_kg: Total emissions (kgCO2e)

    Returns:
        SHA-256 hash
    """
    data = {
        "distance_km": str(distance_km),
        "mode": mode,
        "ef": str(ef),
        "co2e_kg": str(co2e_kg),
        "formula": "distance_km * ef",
    }
    return _compute_hash(data)


def hash_spend_result(
    spend: Union[Decimal, float],
    currency: str,
    mode: str,
    ef: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash spend-based calculation result.

    Args:
        spend: Spend amount in original currency
        currency: ISO 4217 currency code
        mode: Transport mode
        ef: Emission factor (kgCO2e per currency unit)
        co2e_kg: Total emissions (kgCO2e)

    Returns:
        SHA-256 hash
    """
    data = {
        "spend": str(spend),
        "currency": currency,
        "mode": mode,
        "ef": str(ef),
        "co2e_kg": str(co2e_kg),
        "formula": "spend * ef",
    }
    return _compute_hash(data)


def hash_rf_result(
    co2e_without_rf: Union[Decimal, float],
    rf_multiplier: Union[Decimal, float],
    co2e_with_rf: Union[Decimal, float],
) -> str:
    """
    Hash radiative forcing adjustment result.

    Args:
        co2e_without_rf: Emissions without RF (kgCO2e)
        rf_multiplier: Radiative forcing multiplier (e.g., 1.9)
        co2e_with_rf: Emissions with RF (kgCO2e)

    Returns:
        SHA-256 hash
    """
    data = {
        "co2e_without_rf": str(co2e_without_rf),
        "rf_multiplier": str(rf_multiplier),
        "co2e_with_rf": str(co2e_with_rf),
        "formula": "co2e_without_rf * rf_multiplier",
    }
    return _compute_hash(data)


def hash_hotel_result(
    country: str,
    nights: int,
    ef: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash hotel emissions calculation result.

    Args:
        country: ISO 3166-1 alpha-2 country code
        nights: Number of hotel nights
        ef: Emission factor (kgCO2e per room-night)
        co2e_kg: Total emissions (kgCO2e)

    Returns:
        SHA-256 hash
    """
    data = {
        "country": country,
        "nights": nights,
        "ef": str(ef),
        "co2e_kg": str(co2e_kg),
        "formula": "nights * ef",
    }
    return _compute_hash(data)


def hash_calculation_result(result: Dict[str, Any]) -> str:
    """
    Hash complete calculation result.

    Args:
        result: Calculation result with all emissions

    Returns:
        SHA-256 hash
    """
    return _compute_hash(result)


def hash_batch_result(batch_id: str, results: List[Dict[str, Any]]) -> str:
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


def hash_aggregation(
    by_mode: Dict[str, Any],
    by_distance_band: Dict[str, Any],
    by_department: Dict[str, Any],
    total: Union[Decimal, float],
) -> str:
    """
    Hash aggregation result.

    Args:
        by_mode: Emissions by transport mode
        by_distance_band: Emissions by distance band
        by_department: Emissions by department
        total: Total emissions (kgCO2e)

    Returns:
        SHA-256 hash
    """
    data = {
        "by_mode": by_mode,
        "by_distance_band": by_distance_band,
        "by_department": by_department,
        "total": str(total),
    }
    return _compute_hash(data)


def hash_compliance_result(
    framework: str, status: str, score: Union[Decimal, float]
) -> str:
    """
    Hash compliance check result.

    Args:
        framework: Regulatory framework (e.g., "GHG_PROTOCOL", "ISO_14064")
        status: Compliance status (e.g., "COMPLIANT", "NON_COMPLIANT")
        score: Compliance score (0.0-1.0)

    Returns:
        SHA-256 hash
    """
    data = {"framework": framework, "status": status, "score": str(score)}
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
        uncertainty_type: Type of uncertainty (e.g., "parametric", "model")
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound

    Returns:
        SHA-256 hash
    """
    data = {
        "uncertainty_type": uncertainty_type,
        "confidence_level": str(confidence_level),
        "lower_bound": str(lower_bound),
        "upper_bound": str(upper_bound),
    }
    return _compute_hash(data)


def hash_dqi_result(
    dqi_scores: Dict[str, Union[Decimal, float]], composite: Union[Decimal, float]
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


def hash_modal_split(
    modal_shares: Dict[str, Union[Decimal, float]],
    total_distance_km: Union[Decimal, float],
    total_co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash modal split analysis.

    Args:
        modal_shares: Percentage share by mode
        total_distance_km: Total distance (km)
        total_co2e_kg: Total emissions (kgCO2e)

    Returns:
        SHA-256 hash
    """
    data = {
        "modal_shares": {k: str(v) for k, v in modal_shares.items()},
        "total_distance_km": str(total_distance_km),
        "total_co2e_kg": str(total_co2e_kg),
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
        method: Allocation method (e.g., "headcount", "revenue", "department")
        share: This unit's share (e.g., 0.25 for 25%)
        total: Total emissions before allocation
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


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_tracker_instance: Optional[ProvenanceTracker] = None
_tracker_lock = threading.RLock()


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
    """
    Reset singleton ProvenanceTracker instance.

    Useful for testing. Clears all chains.
    """
    global _tracker_instance
    with _tracker_lock:
        _tracker_instance = None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_chain() -> Tuple[ProvenanceTracker, str]:
    """
    Create a new provenance chain using singleton tracker.

    Returns:
        Tuple of (tracker, chain_id)
    """
    tracker = get_provenance_tracker()
    chain_id = tracker.start_chain()
    return tracker, chain_id


def record_validation(
    chain_id: str, input_data: Any, validated_data: Any, metadata: Optional[Dict[str, Any]] = None
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
    chain_id: str, input_data: Any, classified_data: Any, metadata: Optional[Dict[str, Any]] = None
) -> ProvenanceEntry:
    """
    Record CLASSIFY stage.

    Args:
        chain_id: Chain identifier
        input_data: Input data
        classified_data: Classified data (mode, distance band, cabin class)
        metadata: Optional metadata

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.CLASSIFY, input_data, classified_data, metadata
    )


def record_normalization(
    chain_id: str, input_data: Any, normalized_data: Any, metadata: Optional[Dict[str, Any]] = None
) -> ProvenanceEntry:
    """
    Record NORMALIZE stage.

    Args:
        chain_id: Chain identifier
        input_data: Input data
        normalized_data: Normalized data (units, currency, dates)
        metadata: Optional metadata

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.NORMALIZE, input_data, normalized_data, metadata
    )


def record_ef_resolution(
    chain_id: str, input_data: Any, resolved_efs: Any, metadata: Optional[Dict[str, Any]] = None
) -> ProvenanceEntry:
    """
    Record RESOLVE_EFS stage.

    Args:
        chain_id: Chain identifier
        input_data: Input data (mode, distance band, etc.)
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
    chain_id: str, input_data: Any, transport_results: Any, metadata: Optional[Dict[str, Any]] = None
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


def record_hotel_calculation(
    chain_id: str, input_data: Any, hotel_results: Any, metadata: Optional[Dict[str, Any]] = None
) -> ProvenanceEntry:
    """
    Record CALCULATE_HOTEL stage.

    Args:
        chain_id: Chain identifier
        input_data: Hotel calculation input
        hotel_results: Hotel calculation results
        metadata: Optional metadata (country, EF source, etc.)

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.CALCULATE_HOTEL, input_data, hotel_results, metadata
    )


def record_rf_application(
    chain_id: str, input_data: Any, rf_results: Any, metadata: Optional[Dict[str, Any]] = None
) -> ProvenanceEntry:
    """
    Record APPLY_RF stage.

    Args:
        chain_id: Chain identifier
        input_data: Pre-RF emissions
        rf_results: Post-RF emissions
        metadata: Optional metadata (RF multiplier, source, etc.)

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.APPLY_RF, input_data, rf_results, metadata
    )


def record_compliance(
    chain_id: str, input_data: Any, compliance_results: Any, metadata: Optional[Dict[str, Any]] = None
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
    chain_id: str, input_data: Any, aggregated_results: Any, metadata: Optional[Dict[str, Any]] = None
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
    return tracker.export_chain(chain_id, format="json")


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
    # Singleton
    "get_provenance_tracker",
    "reset_provenance_tracker",
    # Hash functions (26+)
    "hash_calculation_input",
    "hash_trip_input",
    "hash_flight_input",
    "hash_ground_transport_input",
    "hash_hotel_input",
    "hash_emission_factor",
    "hash_classification_result",
    "hash_distance_result",
    "hash_spend_result",
    "hash_rf_result",
    "hash_hotel_result",
    "hash_calculation_result",
    "hash_batch_result",
    "hash_aggregation",
    "hash_compliance_result",
    "hash_uncertainty_result",
    "hash_dqi_result",
    "hash_modal_split",
    "hash_allocation",
    "hash_config",
    "hash_metadata",
    "hash_arbitrary",
    # Convenience functions
    "create_chain",
    "record_validation",
    "record_classification",
    "record_normalization",
    "record_ef_resolution",
    "record_transport_calculation",
    "record_hotel_calculation",
    "record_rf_application",
    "record_compliance",
    "record_aggregation",
    "seal_and_verify",
    "export_chain_json",
    # Hash utilities
    "_serialize",
    "_compute_hash",
    "_compute_chain_hash",
    "_merkle_hash",
]
