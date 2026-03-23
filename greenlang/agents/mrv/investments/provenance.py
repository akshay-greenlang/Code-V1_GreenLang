# -*- coding: utf-8 -*-
"""
Investments Provenance Tracking - AGENT-MRV-028

SHA-256 chain-hashed provenance for zero-hallucination audit trails.
12 pipeline stages tracked with deterministic hashing.
Agent: GL-MRV-S3-015

This module implements a complete provenance tracking system for financed
emissions calculations. Every step in the calculation pipeline is recorded
with SHA-256 hashes, creating an immutable audit trail that proves no data
was hallucinated or modified.

Pipeline Stages:
    1. VALIDATE - Input validation and data quality checks
    2. CLASSIFY - Asset class and method classification
    3. RESOLVE_FINANCIAL - Financial data resolution (EVIC, balance sheet, GDP)
    4. RESOLVE_EMISSIONS - Emission factor and investee emission resolution
    5. CALCULATE_ATTRIBUTION - Attribution factor calculation
    6. CALCULATE_FINANCED - Financed emissions calculation
    7. SCORE_DATA_QUALITY - PCAF data quality scoring
    8. COMPLIANCE - Regulatory compliance checking
    9. AGGREGATE - Portfolio-level aggregation
    10. WACI - WACI and intensity metric calculation
    11. UNCERTAINTY - Uncertainty quantification (Monte Carlo)
    12. SEAL - Final sealing and verification

16 Hash Functions:
    1. hash_equity_input() - Hash equity investment inputs
    2. hash_debt_input() - Hash debt investment inputs
    3. hash_project_finance_input() - Hash project finance inputs
    4. hash_cre_input() - Hash CRE investment inputs
    5. hash_mortgage_input() - Hash mortgage inputs
    6. hash_motor_vehicle_input() - Hash auto loan inputs
    7. hash_sovereign_bond_input() - Hash sovereign bond inputs
    8. hash_portfolio_input() - Hash portfolio-level inputs
    9. hash_emission_factors() - Hash resolved EFs
    10. hash_attribution_factor() - Hash attribution calculation
    11. hash_calculation_result() - Hash per-investment result
    12. hash_portfolio_aggregation() - Hash portfolio aggregation
    13. hash_compliance_result() - Hash compliance check
    14. hash_data_quality_score() - Hash PCAF DQ scores
    15. build_merkle_tree() - Build Merkle tree from hash list
    16. verify_provenance_chain() - Verify chain integrity

Example:
    >>> tracker = get_provenance_tracker()
    >>> chain_id = tracker.start_chain()
    >>> tracker.record_stage(chain_id, "VALIDATE", input_data, validated_data)
    >>> tracker.record_stage(chain_id, "CALCULATE_FINANCED", calc_input, calc_output)
    >>> final_hash = tracker.seal_chain(chain_id)
    >>> assert tracker.validate_chain(chain_id)

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-015
"""

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

AGENT_ID = "GL-MRV-S3-015"
AGENT_VERSION = "1.0.0"
HASH_ALGORITHM = "sha256"
ENCODING = "utf-8"


class ProvenanceStage(str, Enum):
    """Pipeline stages for investments provenance tracking."""

    VALIDATE = "VALIDATE"
    CLASSIFY = "CLASSIFY"
    RESOLVE_FINANCIAL = "RESOLVE_FINANCIAL"
    RESOLVE_EMISSIONS = "RESOLVE_EMISSIONS"
    CALCULATE_ATTRIBUTION = "CALCULATE_ATTRIBUTION"
    CALCULATE_FINANCED = "CALCULATE_FINANCED"
    SCORE_DATA_QUALITY = "SCORE_DATA_QUALITY"
    COMPLIANCE = "COMPLIANCE"
    AGGREGATE = "AGGREGATE"
    WACI = "WACI"
    UNCERTAINTY = "UNCERTAINTY"
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
        previous_hash: Chain hash of previous entry (or empty for first)
        agent_id: Agent identifier (GL-MRV-S3-015)
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

    sorted_hashes = sorted(hashes)
    combined = "|".join(sorted_hashes)
    return hashlib.sha256(combined.encode(ENCODING)).hexdigest()


# ============================================================================
# PROVENANCE TRACKER
# ============================================================================


class ProvenanceTracker:
    """
    Main provenance tracking system for investments calculations.

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
            metadata: Optional metadata

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
            ValueError: If chain not found or already sealed
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
        """Clear all chains tracked by this tracker instance."""
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

            chain_hashes = []
            for chain_id in batch.individual_chain_ids:
                try:
                    chain = self.tracker.get_chain(chain_id)
                    if chain.final_hash:
                        chain_hashes.append(chain.final_hash)
                except ValueError:
                    pass

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
# STANDALONE HASH FUNCTIONS (16)
# ============================================================================


def hash_equity_input(equity_data: Dict[str, Any]) -> str:
    """
    Hash equity investment input data.

    Args:
        equity_data: Equity investment data (issuer, ISIN, EVIC,
                     outstanding amount, scope inclusion, etc.)

    Returns:
        SHA-256 hash
    """
    return _compute_hash(equity_data)


def hash_debt_input(debt_data: Dict[str, Any]) -> str:
    """
    Hash debt investment input data.

    Args:
        debt_data: Debt investment data (issuer, instrument type,
                   outstanding balance, total equity + debt, etc.)

    Returns:
        SHA-256 hash
    """
    return _compute_hash(debt_data)


def hash_project_finance_input(project_data: Dict[str, Any]) -> str:
    """
    Hash project finance input data.

    Args:
        project_data: Project finance data (project ID, total financing,
                      investor share, project lifetime, construction
                      vs operational, capacity factor, etc.)

    Returns:
        SHA-256 hash
    """
    return _compute_hash(project_data)


def hash_cre_input(cre_data: Dict[str, Any]) -> str:
    """
    Hash commercial real estate investment input data.

    Args:
        cre_data: CRE data (property ID, building type, floor area,
                  EUI, property value, ENERGY STAR score, GRESB, etc.)

    Returns:
        SHA-256 hash
    """
    return _compute_hash(cre_data)


def hash_mortgage_input(mortgage_data: Dict[str, Any]) -> str:
    """
    Hash mortgage investment input data.

    Args:
        mortgage_data: Mortgage data (property address, LTV ratio,
                       outstanding balance, property value, floor
                       area, EPC rating, grid factor, etc.)

    Returns:
        SHA-256 hash
    """
    return _compute_hash(mortgage_data)


def hash_motor_vehicle_input(vehicle_data: Dict[str, Any]) -> str:
    """
    Hash motor vehicle loan input data.

    Args:
        vehicle_data: Vehicle data (loan outstanding, vehicle value,
                      fuel type, vehicle class, annual distance,
                      tailpipe EF, etc.)

    Returns:
        SHA-256 hash
    """
    return _compute_hash(vehicle_data)


def hash_sovereign_bond_input(sovereign_data: Dict[str, Any]) -> str:
    """
    Hash sovereign bond investment input data.

    Args:
        sovereign_data: Sovereign data (country ISO, investment amount,
                        GDP, national emissions, PPP adjustment,
                        LULUCF inclusion, etc.)

    Returns:
        SHA-256 hash
    """
    return _compute_hash(sovereign_data)


def hash_portfolio_input(portfolio_data: Dict[str, Any]) -> str:
    """
    Hash portfolio-level input data.

    Args:
        portfolio_data: Portfolio data (portfolio ID, positions list,
                        total AUM, benchmark, reporting year, etc.)

    Returns:
        SHA-256 hash
    """
    return _compute_hash(portfolio_data)


def hash_emission_factors(
    asset_class: str,
    ef_source: str,
    ef_values: Dict[str, Union[Decimal, float, str]],
) -> str:
    """
    Hash resolved emission factors.

    Args:
        asset_class: PCAF asset class
        ef_source: Emission factor data source (e.g., "CDP", "DEFRA")
        ef_values: Dictionary of emission factor values by gas/scope

    Returns:
        SHA-256 hash
    """
    data = {
        "asset_class": asset_class,
        "ef_source": ef_source,
        "ef_values": {k: str(v) for k, v in ef_values.items()},
    }
    return _compute_hash(data)


def hash_attribution_factor(
    method: str,
    outstanding_amount: Union[Decimal, float],
    denominator_value: Union[Decimal, float],
    denominator_type: str,
    attribution_factor: Union[Decimal, float],
) -> str:
    """
    Hash attribution factor calculation.

    PCAF attribution = outstanding_amount / denominator_value

    Args:
        method: Attribution method (e.g., "EVIC", "EQUITY_PLUS_DEBT")
        outstanding_amount: Investor's outstanding amount
        denominator_value: Attribution denominator (EVIC, total assets, etc.)
        denominator_type: Type of denominator
        attribution_factor: Calculated attribution factor

    Returns:
        SHA-256 hash
    """
    data = {
        "method": method,
        "outstanding_amount": str(outstanding_amount),
        "denominator_value": str(denominator_value),
        "denominator_type": denominator_type,
        "attribution_factor": str(attribution_factor),
        "formula": "outstanding_amount / denominator_value",
    }
    return _compute_hash(data)


def hash_calculation_result(
    asset_class: str,
    attribution_factor: Union[Decimal, float],
    investee_emissions_tco2e: Union[Decimal, float],
    financed_emissions_tco2e: Union[Decimal, float],
) -> str:
    """
    Hash per-investment financed emissions result.

    Financed emissions = attribution_factor * investee_emissions

    Args:
        asset_class: PCAF asset class
        attribution_factor: Calculated attribution factor
        investee_emissions_tco2e: Investee emissions in tCO2e
        financed_emissions_tco2e: Financed emissions in tCO2e

    Returns:
        SHA-256 hash
    """
    data = {
        "asset_class": asset_class,
        "attribution_factor": str(attribution_factor),
        "investee_emissions_tco2e": str(investee_emissions_tco2e),
        "financed_emissions_tco2e": str(financed_emissions_tco2e),
        "formula": "attribution_factor * investee_emissions_tco2e",
    }
    return _compute_hash(data)


def hash_portfolio_aggregation(
    portfolio_id: str,
    total_financed_emissions_tco2e: Union[Decimal, float],
    by_asset_class: Dict[str, Union[Decimal, float]],
    by_sector: Dict[str, Union[Decimal, float]],
    waci: Optional[Union[Decimal, float]] = None,
) -> str:
    """
    Hash portfolio-level aggregation result.

    Args:
        portfolio_id: Portfolio identifier
        total_financed_emissions_tco2e: Total financed emissions
        by_asset_class: Emissions breakdown by PCAF asset class
        by_sector: Emissions breakdown by GICS sector
        waci: Weighted Average Carbon Intensity (optional)

    Returns:
        SHA-256 hash
    """
    data = {
        "portfolio_id": portfolio_id,
        "total_financed_emissions_tco2e": str(total_financed_emissions_tco2e),
        "by_asset_class": {k: str(v) for k, v in by_asset_class.items()},
        "by_sector": {k: str(v) for k, v in by_sector.items()},
    }
    if waci is not None:
        data["waci"] = str(waci)
    return _compute_hash(data)


def hash_compliance_result(
    framework: str, status: str, score: Union[Decimal, float]
) -> str:
    """
    Hash compliance check result.

    Args:
        framework: Regulatory framework (e.g., "GHG_PROTOCOL", "PCAF")
        status: Compliance status (e.g., "COMPLIANT", "NON_COMPLIANT")
        score: Compliance score (0.0-1.0)

    Returns:
        SHA-256 hash
    """
    data = {"framework": framework, "status": status, "score": str(score)}
    return _compute_hash(data)


def hash_data_quality_score(
    asset_class: str,
    pcaf_score: Union[Decimal, float, int],
    dimension_scores: Dict[str, Union[Decimal, float]],
    data_coverage_pct: Union[Decimal, float],
) -> str:
    """
    Hash PCAF data quality assessment.

    Args:
        asset_class: PCAF asset class
        pcaf_score: Overall PCAF data quality score (1-5)
        dimension_scores: Scores by quality dimension
        data_coverage_pct: Data coverage percentage

    Returns:
        SHA-256 hash
    """
    data = {
        "asset_class": asset_class,
        "pcaf_score": str(pcaf_score),
        "dimension_scores": {k: str(v) for k, v in dimension_scores.items()},
        "data_coverage_pct": str(data_coverage_pct),
    }
    return _compute_hash(data)


def build_merkle_tree(hashes: List[str]) -> str:
    """
    Build Merkle tree from hash list.

    Sorts hashes for determinism, then recursively combines pairs.

    Args:
        hashes: List of SHA-256 hashes

    Returns:
        Root hash of Merkle tree
    """
    if not hashes:
        return hashlib.sha256(b"").hexdigest()
    if len(hashes) == 1:
        return hashes[0]

    sorted_hashes = sorted(hashes)

    # Build tree bottom-up
    current_level = sorted_hashes
    while len(current_level) > 1:
        next_level: List[str] = []
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                combined = current_level[i] + current_level[i + 1]
            else:
                # Odd number: duplicate last hash
                combined = current_level[i] + current_level[i]
            next_level.append(
                hashlib.sha256(combined.encode(ENCODING)).hexdigest()
            )
        current_level = next_level

    return current_level[0]


def verify_provenance_chain(chain_data: Dict[str, Any]) -> bool:
    """
    Verify provenance chain integrity from exported data.

    Reconstructs chain from dictionary and validates all hashes.

    Args:
        chain_data: Exported chain data (from ProvenanceChain.to_dict())

    Returns:
        True if chain is valid, False otherwise
    """
    entries = chain_data.get("entries", [])
    if not entries:
        return True

    for i, entry_data in enumerate(entries):
        if i == 0:
            if entry_data.get("previous_hash", "") != "":
                return False
        else:
            prev_entry = entries[i - 1]
            if entry_data.get("previous_hash") != prev_entry.get("chain_hash"):
                return False

        expected_hash = _compute_chain_hash(
            entry_data.get("previous_hash", ""),
            entry_data.get("stage", ""),
            entry_data.get("input_hash", ""),
            entry_data.get("output_hash", ""),
        )
        if entry_data.get("chain_hash") != expected_hash:
            return False

    return True


# Additional standalone hash utilities

def hash_uncertainty_result(
    uncertainty_type: str,
    confidence_level: Union[Decimal, float],
    lower_bound: Union[Decimal, float],
    upper_bound: Union[Decimal, float],
) -> str:
    """
    Hash uncertainty quantification result.

    Args:
        uncertainty_type: Type of uncertainty (e.g., "monte_carlo", "parametric")
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


def hash_waci_result(
    portfolio_id: str,
    waci_tco2e_per_m_revenue: Union[Decimal, float],
    position_count: int,
    coverage_pct: Union[Decimal, float],
) -> str:
    """
    Hash WACI (Weighted Average Carbon Intensity) result.

    Args:
        portfolio_id: Portfolio identifier
        waci_tco2e_per_m_revenue: WACI value (tCO2e/$M revenue)
        position_count: Number of positions included
        coverage_pct: Portfolio coverage percentage

    Returns:
        SHA-256 hash
    """
    data = {
        "portfolio_id": portfolio_id,
        "waci_tco2e_per_m_revenue": str(waci_tco2e_per_m_revenue),
        "position_count": position_count,
        "coverage_pct": str(coverage_pct),
        "formula": "SUM(weight_i * emissions_i / revenue_i)",
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
    chain_id: str, input_data: Any, validated_data: Any,
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
    chain_id: str, input_data: Any, classified_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record CLASSIFY stage.

    Args:
        chain_id: Chain identifier
        input_data: Input data
        classified_data: Classified data (asset class, method)
        metadata: Optional metadata

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.CLASSIFY, input_data, classified_data, metadata
    )


def record_financial_resolution(
    chain_id: str, input_data: Any, resolved_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record RESOLVE_FINANCIAL stage.

    Args:
        chain_id: Chain identifier
        input_data: Input data (issuer, identifiers)
        resolved_data: Resolved financial data (EVIC, balance sheet, GDP)
        metadata: Optional metadata (source, date)

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.RESOLVE_FINANCIAL, input_data, resolved_data, metadata
    )


def record_emission_resolution(
    chain_id: str, input_data: Any, resolved_emissions: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record RESOLVE_EMISSIONS stage.

    Args:
        chain_id: Chain identifier
        input_data: Input data (issuer, asset class)
        resolved_emissions: Resolved investee emissions
        metadata: Optional metadata (source, year, scope)

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.RESOLVE_EMISSIONS, input_data, resolved_emissions, metadata
    )


def record_attribution_calculation(
    chain_id: str, input_data: Any, attribution_result: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record CALCULATE_ATTRIBUTION stage.

    Args:
        chain_id: Chain identifier
        input_data: Attribution inputs (outstanding, denominator)
        attribution_result: Calculated attribution factor
        metadata: Optional metadata (method, formula)

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.CALCULATE_ATTRIBUTION, input_data, attribution_result, metadata
    )


def record_financed_emissions(
    chain_id: str, input_data: Any, financed_result: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record CALCULATE_FINANCED stage.

    Args:
        chain_id: Chain identifier
        input_data: Attribution + investee emissions
        financed_result: Financed emissions result
        metadata: Optional metadata (method, asset class)

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.CALCULATE_FINANCED, input_data, financed_result, metadata
    )


def record_data_quality_scoring(
    chain_id: str, input_data: Any, quality_result: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record SCORE_DATA_QUALITY stage.

    Args:
        chain_id: Chain identifier
        input_data: Data quality inputs
        quality_result: PCAF quality scores
        metadata: Optional metadata

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.SCORE_DATA_QUALITY, input_data, quality_result, metadata
    )


def record_compliance(
    chain_id: str, input_data: Any, compliance_results: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record COMPLIANCE stage.

    Args:
        chain_id: Chain identifier
        input_data: Data to check
        compliance_results: Compliance check results
        metadata: Optional metadata (framework, rules)

    Returns:
        ProvenanceEntry
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.COMPLIANCE, input_data, compliance_results, metadata
    )


def record_aggregation(
    chain_id: str, input_data: Any, aggregated_results: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceEntry:
    """
    Record AGGREGATE stage.

    Args:
        chain_id: Chain identifier
        input_data: Pre-aggregation data
        aggregated_results: Aggregated results
        metadata: Optional metadata (dimensions, totals)

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
    # Hash functions (16 + extras)
    "hash_equity_input",
    "hash_debt_input",
    "hash_project_finance_input",
    "hash_cre_input",
    "hash_mortgage_input",
    "hash_motor_vehicle_input",
    "hash_sovereign_bond_input",
    "hash_portfolio_input",
    "hash_emission_factors",
    "hash_attribution_factor",
    "hash_calculation_result",
    "hash_portfolio_aggregation",
    "hash_compliance_result",
    "hash_data_quality_score",
    "build_merkle_tree",
    "verify_provenance_chain",
    # Additional hash utilities
    "hash_uncertainty_result",
    "hash_waci_result",
    "hash_config",
    "hash_metadata",
    "hash_arbitrary",
    # Convenience functions
    "create_chain",
    "record_validation",
    "record_classification",
    "record_financial_resolution",
    "record_emission_resolution",
    "record_attribution_calculation",
    "record_financed_emissions",
    "record_data_quality_scoring",
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
