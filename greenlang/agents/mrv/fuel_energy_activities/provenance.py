# -*- coding: utf-8 -*-
"""
Fuel & Energy Activities Agent - Provenance Tracking Module

This module provides SHA-256 provenance chain tracking for GL-MRV-S3-003 (AGENT-MRV-016).
Thread-safe singleton pattern ensures audit trail integrity across all fuel- and
energy-related activities emissions operations, including well-to-tank (WTT) upstream
fuel emissions, upstream electricity generation emissions, and transmission & distribution
(T&D) loss emissions.

Fuel- and energy-related activities (GHG Protocol Scope 3, Category 3) cover upstream
emissions associated with the production of fuels and energy purchased and consumed
by the reporting company that are not already included in Scope 1 (direct combustion)
or Scope 2 (purchased electricity, steam, heating, cooling). The provenance chain
tracks 10 stages from initial activity data validation through pipeline seal, ensuring
deterministic reproducibility of every calculation step.

The three sub-activities tracked are:
  - Activity 3a: Upstream emissions of purchased fuels (well-to-tank)
  - Activity 3b: Upstream emissions of purchased electricity (generation-related)
  - Activity 3c: Transmission and distribution (T&D) losses

Agent: GL-MRV-S3-003 (AGENT-MRV-016)
Purpose: Track fuel & energy activities emissions lineage across three sub-activities
Regulatory: GHG Protocol Scope 3, ISO 14064-1, CSRD E1, CDP Climate Change,
            SBTi, UK SECR, EU ETS

Example:
    >>> provenance = get_provenance()
    >>> chain_hash = provenance.start_chain({"facility_id": "FAC-001", "period": "2025"})
    >>> entry = provenance.record_stage(
    ...     stage="VALIDATE",
    ...     input_data={"fuel_records": 150, "electricity_records": 42},
    ...     output_data={"validated": True, "errors": 0},
    ...     parameters={"validation_mode": "strict"},
    ...     metadata={"source": "ERP", "timestamp": "2025-01-15T10:30:00Z"}
    ... )
    >>> seal_hash = provenance.seal_chain()

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-016 Fuel & Energy Activities (GL-MRV-S3-003)
Status: Production Ready
"""

import copy
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

#: Agent identifier for GL-MRV-S3-003 Fuel & Energy Activities
AGENT_ID: str = "GL-MRV-S3-003"

#: Module version
VERSION: str = "1.0.0"

#: SHA-256 digest size in hex characters
SHA256_HEX_LENGTH: int = 64

#: Genesis hash for initial chain entry - domain-specific anchor
GENESIS_HASH: str = hashlib.sha256(
    "GL-MRV-S3-003-FEA-GENESIS-2026".encode("utf-8")
).hexdigest()

#: Table prefix for fuel & energy activities provenance records
TABLE_PREFIX: str = "gl_fea_"

#: Genesis seed string for reference / verification
GENESIS_SEED: str = "GL-MRV-S3-003-FEA-GENESIS-2026"


# ============================================================================
# Pipeline Stage Enumeration
# ============================================================================


class PipelineStage(str, Enum):
    """
    10 provenance stages for fuel & energy activities emissions calculation.

    Each stage represents a discrete, auditable step in the GL-MRV-S3-003
    pipeline. Stages are ordered sequentially; the provenance chain enforces
    that each stage links cryptographically to the previous one.

    Stage ordering follows the GHG Protocol Technical Guidance for Scope 3
    Category 3 (Fuel- and Energy-Related Activities) calculation workflow.
    """

    # Stage 1: Validate incoming fuel consumption, electricity consumption,
    # and energy purchase records against schema and business rules
    VALIDATE = "validate"

    # Stage 2: Classify activity records into sub-activities (3a/3b/3c),
    # fuel types, energy types, and consumption categories
    CLASSIFY = "classify"

    # Stage 3: Normalize units (MJ, kWh, litres, tonnes, therms) to
    # canonical SI units with consistent precision
    NORMALIZE = "normalize"

    # Stage 4: Resolve emission factors from hierarchical source priority
    # (WTT factors, upstream electricity factors, T&D loss factors)
    RESOLVE_EFS = "resolve_efs"

    # Stage 5: Calculate Activity 3a - upstream emissions of purchased fuels
    # (well-to-tank emissions from extraction, refining, transportation)
    CALCULATE_3A = "calculate_3a"

    # Stage 6: Calculate Activity 3b - upstream emissions of purchased
    # electricity (generation-related emissions not in Scope 2)
    CALCULATE_3B = "calculate_3b"

    # Stage 7: Calculate Activity 3c - transmission and distribution losses
    # for purchased electricity, steam, heating, and cooling
    CALCULATE_3C = "calculate_3c"

    # Stage 8: Regulatory compliance check across supported frameworks
    # (GHG Protocol, ISO 14064, CSRD, CDP, SBTi, UK SECR, EU ETS)
    COMPLIANCE = "compliance"

    # Stage 9: Aggregate results across sub-activities, facilities, and
    # reporting periods with weighted DQI scoring
    AGGREGATE = "aggregate"

    # Stage 10: Final provenance chain seal (immutable after sealing)
    SEAL = "seal"


#: Ordered list of all stages for validation
STAGE_ORDER: List[PipelineStage] = list(PipelineStage)

#: Total number of provenance stages
STAGE_COUNT: int = len(STAGE_ORDER)


# ============================================================================
# Provenance Entry Dataclass
# ============================================================================


@dataclass
class ProvenanceEntry:
    """
    Single provenance chain entry representing one auditable pipeline stage.

    Each entry captures the cryptographic fingerprint of inputs, outputs, and
    parameters for a specific stage. The chain_hash links this entry to all
    previous entries via a Merkle-like chain, ensuring tamper detection.

    Attributes:
        stage: Pipeline stage name (one of 10 PipelineStage values).
        timestamp: ISO 8601 timestamp of when the stage was recorded.
        input_hash: SHA-256 hex digest of the input data for this stage.
        output_hash: SHA-256 hex digest of the output data produced.
        parameters_hash: SHA-256 hex digest of parameters/configuration used.
        chain_hash: SHA-256 combining previous chain_hash with current hashes.
        agent_id: Agent identifier, always "GL-MRV-S3-003" for fuel & energy.
        version: Module version string (SemVer).
        metadata: Additional key-value metadata for audit trail enrichment.

    Example:
        >>> entry = ProvenanceEntry(
        ...     stage="VALIDATE",
        ...     timestamp="2025-01-15T10:30:00Z",
        ...     input_hash="abc123...",
        ...     output_hash="def456...",
        ...     parameters_hash="ghi789...",
        ...     chain_hash="jkl012...",
        ... )
    """

    stage: str
    timestamp: str
    input_hash: str
    output_hash: str
    parameters_hash: str
    chain_hash: str
    agent_id: str = AGENT_ID
    version: str = VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize entry to a JSON-compatible dictionary.

        Returns:
            Dictionary representation with all fields including metadata.
        """
        return {
            "stage": self.stage,
            "timestamp": self.timestamp,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "parameters_hash": self.parameters_hash,
            "chain_hash": self.chain_hash,
            "agent_id": self.agent_id,
            "version": self.version,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceEntry":
        """
        Deserialize entry from a dictionary.

        Args:
            data: Dictionary with ProvenanceEntry fields.

        Returns:
            Reconstructed ProvenanceEntry instance.

        Raises:
            KeyError: If required fields are missing.
        """
        return cls(
            stage=data["stage"],
            timestamp=data["timestamp"],
            input_hash=data["input_hash"],
            output_hash=data["output_hash"],
            parameters_hash=data["parameters_hash"],
            chain_hash=data["chain_hash"],
            agent_id=data.get("agent_id", AGENT_ID),
            version=data.get("version", VERSION),
            metadata=data.get("metadata", {}),
        )

    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate entry field integrity.

        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        if not self.stage:
            return False, "stage is empty"
        if not self.timestamp:
            return False, "timestamp is empty"
        if not self.input_hash or len(self.input_hash) != SHA256_HEX_LENGTH:
            return False, (
                f"input_hash length invalid: "
                f"{len(self.input_hash) if self.input_hash else 0}"
            )
        if not self.output_hash or len(self.output_hash) != SHA256_HEX_LENGTH:
            return False, (
                f"output_hash length invalid: "
                f"{len(self.output_hash) if self.output_hash else 0}"
            )
        if not self.parameters_hash or len(self.parameters_hash) != SHA256_HEX_LENGTH:
            return False, (
                f"parameters_hash length invalid: "
                f"{len(self.parameters_hash) if self.parameters_hash else 0}"
            )
        if not self.chain_hash or len(self.chain_hash) != SHA256_HEX_LENGTH:
            return False, (
                f"chain_hash length invalid: "
                f"{len(self.chain_hash) if self.chain_hash else 0}"
            )
        return True, None


# ============================================================================
# Deterministic Serialization Helpers
# ============================================================================


def _serialize_for_hash(obj: Any) -> str:
    """
    Deterministic JSON serialization for SHA-256 hashing.

    Handles Decimal, datetime, date, Pydantic BaseModel, dataclass, set, tuple,
    bytes, Enum, and nested structures. All dictionary keys are sorted
    recursively for reproducible hashing.

    Args:
        obj: Any Python object to serialize deterministically.

    Returns:
        Deterministic JSON string representation.

    Raises:
        TypeError: If obj contains a type that cannot be serialized.

    Example:
        >>> _serialize_for_hash({"b": Decimal("1.5"), "a": 2})
        '{"a":2,"b":"1.5"}'
    """
    def _convert(o: Any) -> Any:
        """Recursively convert non-JSON-native types."""
        if o is None:
            return None
        if isinstance(o, bool):
            return o
        if isinstance(o, (int, float)):
            return o
        if isinstance(o, str):
            return o
        if isinstance(o, Decimal):
            # Normalize Decimal to remove trailing zeros for consistency
            return str(o.normalize())
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, date):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, bytes):
            return o.hex()
        if isinstance(o, set):
            return sorted([_convert(item) for item in o], key=str)
        if isinstance(o, (list, tuple)):
            return [_convert(item) for item in o]
        if isinstance(o, dict):
            return {
                str(k): _convert(v)
                for k, v in sorted(o.items(), key=lambda x: str(x[0]))
            }
        # Handle Pydantic BaseModel (v1 and v2)
        if hasattr(o, "model_dump"):
            return _convert(o.model_dump())
        if hasattr(o, "dict"):
            return _convert(o.dict())
        # Handle dataclasses
        if hasattr(o, "__dataclass_fields__"):
            return _convert(asdict(o))
        # Handle objects with __dict__
        if hasattr(o, "__dict__"):
            return _convert(vars(o))
        # Fallback to string representation
        return str(o)

    converted = _convert(obj)
    return json.dumps(converted, sort_keys=True, separators=(",", ":"))


def _hash_data(data: Any) -> str:
    """
    Compute SHA-256 hex digest of any data.

    Uses deterministic serialization to ensure identical data always
    produces identical hashes, regardless of insertion order or type
    representation.

    Args:
        data: Any Python object to hash.

    Returns:
        64-character lowercase hex SHA-256 digest string.

    Example:
        >>> _hash_data({"emissions": Decimal("1234.56"), "unit": "tCO2e"})
        'a1b2c3...'  # deterministic 64-char hex
    """
    serialized = _serialize_for_hash(data)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _compute_chain_hash(
    previous_hash: str,
    stage: str,
    input_hash: str,
    output_hash: str,
    parameters_hash: str,
) -> str:
    """
    Compute the chain hash for a provenance entry.

    Combines the previous chain hash with the current stage's input, output,
    and parameters hashes into a single SHA-256 digest. This creates a
    Merkle-like chain where tampering with any entry invalidates all
    subsequent chain hashes.

    Args:
        previous_hash: Chain hash from the preceding entry (or GENESIS_HASH).
        stage: Stage name string.
        input_hash: SHA-256 hex digest of stage input.
        output_hash: SHA-256 hex digest of stage output.
        parameters_hash: SHA-256 hex digest of stage parameters.

    Returns:
        64-character lowercase hex SHA-256 chain hash.

    Example:
        >>> _compute_chain_hash(GENESIS_HASH, "VALIDATE", "aaa...", "bbb...", "ccc...")
        'ddd...'
    """
    chain_input = {
        "previous_hash": previous_hash,
        "stage": stage,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "parameters_hash": parameters_hash,
    }
    serialized = json.dumps(chain_input, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ============================================================================
# Internal Hash Helper
# ============================================================================


def _hash_helper(data: Dict[str, Any]) -> str:
    """
    Internal helper for SHA-256 hashing of structured dictionaries.

    This is the canonical hashing function used by all standalone hash helpers.
    It performs deterministic JSON serialization with sorted keys and consistent
    type coercion before computing the SHA-256 digest.

    Args:
        data: Dictionary to hash. All values are coerced via the default=str
              fallback for non-JSON-native types.

    Returns:
        64-character lowercase hex SHA-256 digest string.
    """
    json_str = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


# ============================================================================
# Fuel & Energy Activities Provenance Tracker (Thread-Safe Singleton)
# ============================================================================


class FuelEnergyActivitiesProvenance:
    """
    Thread-safe singleton provenance tracker for fuel & energy activities emissions.

    Tracks 10-stage fuel and energy activities emissions lineage with SHA-256
    chain integrity. Supports three sub-activities: Activity 3a (upstream fuel
    WTT emissions), Activity 3b (upstream electricity generation), and
    Activity 3c (T&D losses).

    Thread Safety:
        - Singleton with double-checked locking pattern
        - Per-instance reentrant lock for chain operations
        - Atomic chain sealing prevents race conditions

    Chain Integrity:
        - Each stage entry includes input_hash, output_hash, parameters_hash
        - chain_hash links current entry to all previous entries
        - seal_chain() produces a final tamper-evident digest
        - verify_chain() re-derives all hashes to detect corruption

    Usage Patterns:
        1. Single calculation: start_chain -> record_stage (x10) -> seal_chain
        2. Batch processing: reset() between calculations
        3. Verification: verify_chain() after deserialization from storage

    Example:
        >>> provenance = FuelEnergyActivitiesProvenance.get_instance()
        >>> provenance.start_chain({"facility_id": "FAC-001", "period": "2025"})
        '4f3e2d...'
        >>> provenance.record_stage(
        ...     stage="VALIDATE",
        ...     input_data={"fuel_records": 150},
        ...     output_data={"validated": True},
        ... )
        ProvenanceEntry(stage='VALIDATE', ...)
        >>> provenance.seal_chain()
        'a1b2c3...'
        >>> provenance.verify_chain()
        True
    """

    _instance: Optional["FuelEnergyActivitiesProvenance"] = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """
        Initialize FuelEnergyActivitiesProvenance.

        Direct instantiation is permitted but the singleton pattern via
        get_instance() is strongly recommended for production use.
        """
        self._chain: List[ProvenanceEntry] = []
        self._current_chain_hash: str = GENESIS_HASH
        self._sealed: bool = False
        self._chain_id: Optional[str] = None
        self._organization_id: Optional[str] = None
        self._reporting_period: Optional[str] = None
        self._created_at: Optional[str] = None
        self._seal_hash: Optional[str] = None
        self._seal_timestamp: Optional[str] = None
        self._lock: threading.RLock = threading.RLock()
        logger.debug("FuelEnergyActivitiesProvenance instance initialized")

    @classmethod
    def get_instance(cls) -> "FuelEnergyActivitiesProvenance":
        """
        Get singleton instance with double-checked locking.

        Thread-safe lazy initialization ensures exactly one instance
        exists per process.

        Returns:
            The singleton FuelEnergyActivitiesProvenance instance.

        Example:
            >>> p1 = FuelEnergyActivitiesProvenance.get_instance()
            >>> p2 = FuelEnergyActivitiesProvenance.get_instance()
            >>> assert p1 is p2
        """
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.info(
                        "FuelEnergyActivitiesProvenance singleton created"
                    )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (primarily for testing).

        Thread-safe destruction of the singleton instance. After calling
        this method, the next call to get_instance() will create a fresh
        instance.
        """
        with cls._singleton_lock:
            cls._instance = None
            logger.debug("FuelEnergyActivitiesProvenance singleton reset")

    # ------------------------------------------------------------------
    # Core Chain Lifecycle Methods
    # ------------------------------------------------------------------

    def start_chain(
        self,
        initial_data: Any,
        chain_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        reporting_period: Optional[str] = None,
    ) -> str:
        """
        Initialize a new provenance chain.

        Clears any existing chain state and establishes the genesis entry
        by hashing the initial input data. The returned hash is the
        starting point for the chain.

        Args:
            initial_data: Initial input data to hash (e.g., raw activity records).
            chain_id: Optional unique calculation identifier.
            organization_id: Optional organization identifier.
            reporting_period: Optional reporting period (YYYY or YYYY-QN).

        Returns:
            SHA-256 hex digest of the initial data (genesis hash seed).

        Raises:
            ValueError: If initial_data is None.

        Example:
            >>> provenance = get_provenance()
            >>> initial_hash = provenance.start_chain(
            ...     initial_data={"fuels": [{"id": "F001", "litres": 5000}]},
            ...     chain_id="CALC-2025-001",
            ...     organization_id="ORG-123",
            ...     reporting_period="2025",
            ... )
        """
        if initial_data is None:
            raise ValueError("initial_data cannot be None")

        with self._lock:
            # Reset state for new chain
            self._chain = []
            self._current_chain_hash = GENESIS_HASH
            self._sealed = False
            self._chain_id = chain_id
            self._organization_id = organization_id
            self._reporting_period = reporting_period
            self._created_at = datetime.utcnow().isoformat() + "Z"
            self._seal_hash = None
            self._seal_timestamp = None

            initial_hash = _hash_data(initial_data)

            logger.info(
                "Provenance chain started: chain_id=%s, org=%s, period=%s, "
                "initial_hash=%s",
                chain_id,
                organization_id,
                reporting_period,
                initial_hash[:16] + "...",
            )

            return initial_hash

    def add_entry(
        self,
        stage: str,
        input_data: Any,
        output_data: Any,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEntry:
        """
        Add a provenance entry for a pipeline stage.

        Computes SHA-256 hashes of the input data, output data, and parameters,
        then chains them with the previous chain hash to produce a new chain hash.
        The entry is appended to the provenance chain.

        This is an alias for record_stage() provided for API consistency with
        the requirements specification.

        Args:
            stage: Stage name (should match a PipelineStage value).
            input_data: Input data consumed by this stage.
            output_data: Output data produced by this stage.
            parameters: Optional parameters/configuration used in this stage.
            metadata: Optional metadata for audit enrichment (source, timing).

        Returns:
            The ProvenanceEntry created for this stage.

        Raises:
            ValueError: If the chain is sealed or stage name is empty.
            RuntimeError: If chain has not been started via start_chain().

        Example:
            >>> entry = provenance.add_entry(
            ...     stage="VALIDATE",
            ...     input_data=raw_records,
            ...     output_data=validated_records,
            ...     parameters={"validation_mode": "strict"},
            ...     metadata={"source_system": "SAP", "record_count": 150},
            ... )
            >>> assert len(entry.chain_hash) == 64
        """
        return self.record_stage(
            stage=stage,
            input_data=input_data,
            output_data=output_data,
            parameters=parameters,
            metadata=metadata,
        )

    def record_stage(
        self,
        stage: str,
        input_data: Any,
        output_data: Any,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEntry:
        """
        Record a provenance entry for a pipeline stage.

        Computes SHA-256 hashes of the input data, output data, and parameters,
        then chains them with the previous chain hash to produce a new chain hash.
        The entry is appended to the provenance chain.

        Chain hash formula:
            chain_hash = SHA-256(prev_hash + stage + input_hash + output_hash)

        Args:
            stage: Stage name (should match a PipelineStage value).
            input_data: Input data consumed by this stage.
            output_data: Output data produced by this stage.
            parameters: Optional parameters/configuration used in this stage.
            metadata: Optional metadata for audit enrichment (source, timing).

        Returns:
            The ProvenanceEntry created for this stage.

        Raises:
            ValueError: If the chain is sealed or stage name is empty.
            RuntimeError: If chain has not been started via start_chain().

        Example:
            >>> entry = provenance.record_stage(
            ...     stage="CALCULATE_3A",
            ...     input_data=fuel_consumption_data,
            ...     output_data=wtt_emissions_result,
            ...     parameters={"wtt_source": "DEFRA", "gwp": "AR6"},
            ...     metadata={"fuel_types_processed": 8},
            ... )
            >>> assert len(entry.chain_hash) == 64
        """
        if not stage:
            raise ValueError("stage cannot be empty")

        with self._lock:
            if self._sealed:
                raise ValueError(
                    f"Cannot record stage '{stage}': chain is sealed"
                )

            if self._created_at is None:
                raise RuntimeError(
                    "Chain has not been started. Call start_chain() first."
                )

            # Compute individual hashes
            input_hash = _hash_data(input_data)
            output_hash = _hash_data(output_data)
            parameters_hash = _hash_data(
                parameters if parameters is not None else {}
            )

            # Compute chain hash linking to previous
            chain_hash = _compute_chain_hash(
                previous_hash=self._current_chain_hash,
                stage=stage,
                input_hash=input_hash,
                output_hash=output_hash,
                parameters_hash=parameters_hash,
            )

            # Build entry
            timestamp = datetime.utcnow().isoformat() + "Z"
            entry = ProvenanceEntry(
                stage=stage,
                timestamp=timestamp,
                input_hash=input_hash,
                output_hash=output_hash,
                parameters_hash=parameters_hash,
                chain_hash=chain_hash,
                agent_id=AGENT_ID,
                version=VERSION,
                metadata=copy.deepcopy(metadata) if metadata else {},
            )

            # Validate entry
            is_valid, error = entry.validate()
            if not is_valid:
                raise ValueError(f"Invalid provenance entry: {error}")

            # Append and advance chain
            self._chain.append(entry)
            self._current_chain_hash = chain_hash

            logger.debug(
                "Provenance stage recorded: stage=%s, chain_hash=%s",
                stage,
                chain_hash[:16] + "...",
            )

            return entry

    def seal_chain(self) -> str:
        """
        Finalize and seal the provenance chain.

        Computes a final seal hash over the entire chain, including all
        entry chain hashes, the chain metadata, and the stage count.
        After sealing, no further stages can be recorded.

        Returns:
            SHA-256 hex digest of the sealed chain (final provenance hash).

        Raises:
            ValueError: If chain is already sealed or has no entries.
            RuntimeError: If chain has not been started.

        Example:
            >>> seal_hash = provenance.seal_chain()
            >>> assert len(seal_hash) == 64
            >>> assert provenance.verify_chain()
        """
        with self._lock:
            if self._created_at is None:
                raise RuntimeError(
                    "Chain has not been started. Call start_chain() first."
                )

            if self._sealed:
                raise ValueError("Chain is already sealed")

            if not self._chain:
                raise ValueError("Cannot seal an empty chain")

            # Build seal data from full chain state
            seal_data = {
                "agent_id": AGENT_ID,
                "version": VERSION,
                "chain_id": self._chain_id,
                "organization_id": self._organization_id,
                "reporting_period": self._reporting_period,
                "created_at": self._created_at,
                "entry_count": len(self._chain),
                "entry_chain_hashes": [
                    entry.chain_hash for entry in self._chain
                ],
                "stages_completed": [entry.stage for entry in self._chain],
                "final_chain_hash": self._current_chain_hash,
            }

            seal_hash = _hash_data(seal_data)
            self._sealed = True
            self._seal_hash = seal_hash
            self._seal_timestamp = datetime.utcnow().isoformat() + "Z"

            logger.info(
                "Provenance chain sealed: chain_id=%s, entries=%d, "
                "seal_hash=%s",
                self._chain_id,
                len(self._chain),
                seal_hash[:16] + "...",
            )

            return seal_hash

    # ------------------------------------------------------------------
    # Chain Query Methods
    # ------------------------------------------------------------------

    def get_chain(self) -> List[ProvenanceEntry]:
        """
        Return the full provenance chain as a list of entries.

        Returns a deep copy to prevent external mutation of the chain.

        Returns:
            Ordered list of ProvenanceEntry objects.

        Example:
            >>> chain = provenance.get_chain()
            >>> for entry in chain:
            ...     print(f"{entry.stage}: {entry.chain_hash[:16]}...")
        """
        with self._lock:
            return [
                ProvenanceEntry(
                    stage=e.stage,
                    timestamp=e.timestamp,
                    input_hash=e.input_hash,
                    output_hash=e.output_hash,
                    parameters_hash=e.parameters_hash,
                    chain_hash=e.chain_hash,
                    agent_id=e.agent_id,
                    version=e.version,
                    metadata=copy.deepcopy(e.metadata),
                )
                for e in self._chain
            ]

    def get_entries(self) -> List[ProvenanceEntry]:
        """
        Get all entries in the provenance chain.

        Alias for get_chain() provided for API consistency.

        Returns:
            Ordered list of ProvenanceEntry objects (deep copy).
        """
        return self.get_chain()

    def get_stage(self, stage: str) -> Optional[ProvenanceEntry]:
        """
        Get the provenance entry for a specific stage.

        If the stage appears multiple times (e.g., after a retry),
        the last recorded entry for that stage is returned.

        Args:
            stage: Stage name to look up.

        Returns:
            ProvenanceEntry for the stage, or None if not found.

        Example:
            >>> entry = provenance.get_stage("CALCULATE_3A")
            >>> if entry:
            ...     print(f"WTT emissions hash: {entry.output_hash}")
        """
        with self._lock:
            # Search in reverse to get the latest entry for the stage
            for entry in reversed(self._chain):
                if entry.stage == stage:
                    return ProvenanceEntry(
                        stage=entry.stage,
                        timestamp=entry.timestamp,
                        input_hash=entry.input_hash,
                        output_hash=entry.output_hash,
                        parameters_hash=entry.parameters_hash,
                        chain_hash=entry.chain_hash,
                        agent_id=entry.agent_id,
                        version=entry.version,
                        metadata=copy.deepcopy(entry.metadata),
                    )
            return None

    def get_stages_completed(self) -> List[str]:
        """
        Get the list of stage names recorded so far.

        Returns:
            Ordered list of stage name strings.
        """
        with self._lock:
            return [entry.stage for entry in self._chain]

    def get_entry_count(self) -> int:
        """
        Get the number of entries in the chain.

        Returns:
            Integer count of recorded provenance entries.
        """
        with self._lock:
            return len(self._chain)

    def is_sealed(self) -> bool:
        """
        Check whether the chain has been sealed.

        Returns:
            True if the chain is sealed, False otherwise.
        """
        with self._lock:
            return self._sealed

    def get_seal_hash(self) -> Optional[str]:
        """
        Get the seal hash if the chain has been sealed.

        Returns:
            SHA-256 seal hash string, or None if not sealed.
        """
        with self._lock:
            return self._seal_hash

    def get_chain_hash(self) -> str:
        """
        Get the current running chain hash.

        Returns:
            The latest chain hash (GENESIS_HASH if no entries recorded).
        """
        with self._lock:
            return self._current_chain_hash

    def get_current_chain_hash(self) -> str:
        """
        Get the current running chain hash.

        Alias for get_chain_hash() for backward compatibility.

        Returns:
            The latest chain hash (GENESIS_HASH if no entries recorded).
        """
        return self.get_chain_hash()

    # ------------------------------------------------------------------
    # Chain Verification
    # ------------------------------------------------------------------

    def verify_chain(self) -> bool:
        """
        Verify integrity of the entire provenance chain.

        Re-derives every chain hash from scratch and compares against
        the recorded values. Also verifies the seal hash if the chain
        is sealed.

        Returns:
            True if the chain is intact (all hashes match), False if
            any tampering is detected.

        Example:
            >>> provenance.seal_chain()
            >>> assert provenance.verify_chain() is True
        """
        with self._lock:
            return self._verify_chain_internal()[0]

    def verify_chain_detailed(self) -> Tuple[bool, Optional[str]]:
        """
        Verify chain integrity with detailed error reporting.

        Returns:
            Tuple of (is_valid, error_message). error_message is None
            when the chain is valid.

        Example:
            >>> is_valid, error = provenance.verify_chain_detailed()
            >>> if not is_valid:
            ...     logger.error(f"Chain integrity failure: {error}")
        """
        with self._lock:
            return self._verify_chain_internal()

    def _verify_chain_internal(self) -> Tuple[bool, Optional[str]]:
        """
        Internal chain verification (must be called with lock held).

        Walks the chain from genesis, re-computing each chain_hash and
        comparing against the stored value. Detects insertion, deletion,
        modification, and reordering attacks.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not self._chain:
            return False, "Chain is empty - no entries to verify"

        running_hash = GENESIS_HASH

        for i, entry in enumerate(self._chain):
            # Validate entry structure
            is_valid, error = entry.validate()
            if not is_valid:
                return False, (
                    f"Entry {i} ({entry.stage}) structural validation "
                    f"failed: {error}"
                )

            # Re-derive chain hash
            expected_chain_hash = _compute_chain_hash(
                previous_hash=running_hash,
                stage=entry.stage,
                input_hash=entry.input_hash,
                output_hash=entry.output_hash,
                parameters_hash=entry.parameters_hash,
            )

            if entry.chain_hash != expected_chain_hash:
                return False, (
                    f"Entry {i} ({entry.stage}) chain_hash mismatch: "
                    f"recorded={entry.chain_hash[:16]}..., "
                    f"expected={expected_chain_hash[:16]}..."
                )

            running_hash = entry.chain_hash

        # Verify the running chain hash matches current state
        if running_hash != self._current_chain_hash:
            return False, (
                f"Final chain hash mismatch: "
                f"running={running_hash[:16]}..., "
                f"current={self._current_chain_hash[:16]}..."
            )

        # Verify seal if sealed
        if self._sealed and self._seal_hash:
            seal_data = {
                "agent_id": AGENT_ID,
                "version": VERSION,
                "chain_id": self._chain_id,
                "organization_id": self._organization_id,
                "reporting_period": self._reporting_period,
                "created_at": self._created_at,
                "entry_count": len(self._chain),
                "entry_chain_hashes": [
                    entry.chain_hash for entry in self._chain
                ],
                "stages_completed": [entry.stage for entry in self._chain],
                "final_chain_hash": self._current_chain_hash,
            }

            expected_seal_hash = _hash_data(seal_data)
            if self._seal_hash != expected_seal_hash:
                return False, (
                    f"Seal hash mismatch: "
                    f"recorded={self._seal_hash[:16]}..., "
                    f"expected={expected_seal_hash[:16]}..."
                )

        return True, None

    # ------------------------------------------------------------------
    # Serialization / Deserialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the full provenance chain to a JSON-compatible dictionary.

        Includes chain metadata, all entries, seal state, and verification
        status. The output can be stored in a database, exported to JSON,
        or transmitted via API.

        Returns:
            Complete dictionary representation of the provenance chain.

        Example:
            >>> chain_dict = provenance.to_dict()
            >>> json_str = json.dumps(chain_dict, indent=2)
        """
        with self._lock:
            is_valid, error = self._verify_chain_internal()

            return {
                "agent_id": AGENT_ID,
                "version": VERSION,
                "chain_id": self._chain_id,
                "organization_id": self._organization_id,
                "reporting_period": self._reporting_period,
                "created_at": self._created_at,
                "sealed": self._sealed,
                "seal_hash": self._seal_hash,
                "seal_timestamp": self._seal_timestamp,
                "current_chain_hash": self._current_chain_hash,
                "entry_count": len(self._chain),
                "integrity_verified": is_valid,
                "integrity_error": error,
                "entries": [entry.to_dict() for entry in self._chain],
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FuelEnergyActivitiesProvenance":
        """
        Deserialize a provenance chain from a dictionary.

        Creates a new FuelEnergyActivitiesProvenance instance (not the singleton)
        populated with the serialized chain data. Use this for loading
        provenance records from storage or API responses.

        Args:
            data: Dictionary containing serialized provenance chain.

        Returns:
            New FuelEnergyActivitiesProvenance instance with restored chain.

        Raises:
            KeyError: If required fields are missing from the data.
            ValueError: If deserialized chain fails integrity verification.

        Example:
            >>> stored_data = db.load_provenance("CALC-2025-001")
            >>> provenance = FuelEnergyActivitiesProvenance.from_dict(stored_data)
            >>> assert provenance.verify_chain()
        """
        instance = cls()

        instance._chain_id = data.get("chain_id")
        instance._organization_id = data.get("organization_id")
        instance._reporting_period = data.get("reporting_period")
        instance._created_at = data.get("created_at")
        instance._sealed = data.get("sealed", False)
        instance._seal_hash = data.get("seal_hash")
        instance._seal_timestamp = data.get("seal_timestamp")
        instance._current_chain_hash = data.get(
            "current_chain_hash", GENESIS_HASH
        )

        entries_data = data.get("entries", [])
        instance._chain = [
            ProvenanceEntry.from_dict(entry_data) for entry_data in entries_data
        ]

        # Verify chain integrity after deserialization
        is_valid, error = instance._verify_chain_internal()
        if not is_valid:
            logger.warning(
                "Deserialized provenance chain failed verification: %s", error
            )

        return instance

    # ------------------------------------------------------------------
    # Chain Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Clear the provenance chain for a new calculation.

        Resets all internal state including chain entries, hashes,
        seal state, and metadata. The instance remains usable for
        a new start_chain() call.

        Example:
            >>> provenance.reset()
            >>> provenance.start_chain(new_data)
        """
        with self._lock:
            self._chain = []
            self._current_chain_hash = GENESIS_HASH
            self._sealed = False
            self._chain_id = None
            self._organization_id = None
            self._reporting_period = None
            self._created_at = None
            self._seal_hash = None
            self._seal_timestamp = None

            logger.debug("Provenance chain reset")


# ============================================================================
# Batch Provenance Tracker
# ============================================================================


class FuelEnergyActivitiesBatchProvenance:
    """
    Track provenance for multi-period or multi-facility batch calculations.

    Manages per-calculation provenance chains with batch-level aggregation
    and cross-chain verification. Useful for processing multiple reporting
    periods or facility portfolios in a single pipeline run.

    Thread Safety:
        Uses a dedicated lock for batch-level operations. Individual chain
        operations are protected by FuelEnergyActivitiesProvenance instance locks.

    Example:
        >>> batch = FuelEnergyActivitiesBatchProvenance()
        >>> batch_id = batch.create_batch("BATCH-2025-Q1")
        >>> chain = FuelEnergyActivitiesProvenance()
        >>> chain.start_chain(activity_data)
        >>> # ... record stages ...
        >>> chain.seal_chain()
        >>> batch.add_chain(batch_id, "CALC-001", chain)
        >>> batch_hash = batch.seal_batch(batch_id)
    """

    def __init__(self) -> None:
        """Initialize batch provenance tracker."""
        self._batches: Dict[str, Dict[str, Any]] = {}
        self._batch_lock: threading.Lock = threading.Lock()
        logger.debug("FuelEnergyActivitiesBatchProvenance initialized")

    def create_batch(
        self,
        batch_id: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new batch provenance container.

        Args:
            batch_id: Unique batch identifier.
            description: Optional human-readable description.
            metadata: Optional batch-level metadata.

        Returns:
            The batch_id.

        Raises:
            ValueError: If batch already exists.
        """
        with self._batch_lock:
            if batch_id in self._batches:
                raise ValueError(f"Batch {batch_id} already exists")

            self._batches[batch_id] = {
                "batch_id": batch_id,
                "description": description,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat() + "Z",
                "chains": {},
                "sealed": False,
                "seal_hash": None,
                "seal_timestamp": None,
            }

            logger.info("Batch provenance created: batch_id=%s", batch_id)

        return batch_id

    def add_chain(
        self,
        batch_id: str,
        chain_id: str,
        provenance: "FuelEnergyActivitiesProvenance",
    ) -> None:
        """
        Add a completed provenance chain to the batch.

        Args:
            batch_id: Batch identifier.
            chain_id: Unique chain identifier within the batch.
            provenance: The FuelEnergyActivitiesProvenance instance to add.

        Raises:
            ValueError: If batch not found, batch is sealed, or chain_id
                        already exists in the batch.
        """
        with self._batch_lock:
            if batch_id not in self._batches:
                raise ValueError(f"Batch {batch_id} not found")

            batch = self._batches[batch_id]

            if batch["sealed"]:
                raise ValueError(f"Batch {batch_id} is sealed")

            if chain_id in batch["chains"]:
                raise ValueError(
                    f"Chain {chain_id} already exists in batch {batch_id}"
                )

            batch["chains"][chain_id] = provenance.to_dict()

            logger.debug(
                "Chain added to batch: batch_id=%s, chain_id=%s",
                batch_id,
                chain_id,
            )

    def seal_batch(self, batch_id: str) -> str:
        """
        Seal the batch provenance (immutable after sealing).

        Computes a batch-level seal hash over all contained chain seal
        hashes.

        Args:
            batch_id: Batch identifier.

        Returns:
            SHA-256 hex digest of the batch seal.

        Raises:
            ValueError: If batch not found, already sealed, or empty.
        """
        with self._batch_lock:
            if batch_id not in self._batches:
                raise ValueError(f"Batch {batch_id} not found")

            batch = self._batches[batch_id]

            if batch["sealed"]:
                raise ValueError(f"Batch {batch_id} already sealed")

            if not batch["chains"]:
                raise ValueError(f"Cannot seal empty batch {batch_id}")

            # Collect all chain seal hashes in deterministic order
            chain_seal_hashes = {}
            for cid in sorted(batch["chains"].keys()):
                chain_data = batch["chains"][cid]
                chain_seal_hashes[cid] = chain_data.get(
                    "seal_hash", "UNSEALED"
                )

            seal_data = {
                "batch_id": batch_id,
                "agent_id": AGENT_ID,
                "version": VERSION,
                "chain_count": len(batch["chains"]),
                "chain_seal_hashes": chain_seal_hashes,
                "created_at": batch["created_at"],
            }

            seal_hash = _hash_data(seal_data)
            batch["sealed"] = True
            batch["seal_hash"] = seal_hash
            batch["seal_timestamp"] = datetime.utcnow().isoformat() + "Z"

            logger.info(
                "Batch sealed: batch_id=%s, chains=%d, seal_hash=%s",
                batch_id,
                len(batch["chains"]),
                seal_hash[:16] + "...",
            )

            return seal_hash

    def verify_batch(self, batch_id: str) -> Tuple[bool, Optional[str]]:
        """
        Verify batch provenance integrity.

        Verifies each contained chain and the batch seal hash.

        Args:
            batch_id: Batch identifier.

        Returns:
            Tuple of (is_valid, error_message).
        """
        with self._batch_lock:
            if batch_id not in self._batches:
                return False, f"Batch {batch_id} not found"

            batch = self._batches[batch_id]

            if not batch["chains"]:
                return False, f"Batch {batch_id} has no chains"

            # Verify each chain
            for cid in sorted(batch["chains"].keys()):
                chain_data = batch["chains"][cid]
                chain = FuelEnergyActivitiesProvenance.from_dict(chain_data)
                is_valid, error = chain.verify_chain_detailed()
                if not is_valid:
                    return False, (
                        f"Chain {cid} in batch {batch_id}: {error}"
                    )

            # Verify batch seal
            if batch["sealed"] and batch["seal_hash"]:
                chain_seal_hashes = {}
                for cid in sorted(batch["chains"].keys()):
                    chain_data = batch["chains"][cid]
                    chain_seal_hashes[cid] = chain_data.get(
                        "seal_hash", "UNSEALED"
                    )

                seal_data = {
                    "batch_id": batch_id,
                    "agent_id": AGENT_ID,
                    "version": VERSION,
                    "chain_count": len(batch["chains"]),
                    "chain_seal_hashes": chain_seal_hashes,
                    "created_at": batch["created_at"],
                }

                expected_seal_hash = _hash_data(seal_data)
                if batch["seal_hash"] != expected_seal_hash:
                    return False, f"Batch {batch_id} seal hash mismatch"

        return True, None

    def get_batch_summary(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of the batch provenance state.

        Args:
            batch_id: Batch identifier.

        Returns:
            Summary dictionary or None if batch not found.
        """
        with self._batch_lock:
            if batch_id not in self._batches:
                return None

            batch = self._batches[batch_id]

            chain_summaries = []
            for cid in sorted(batch["chains"].keys()):
                chain_data = batch["chains"][cid]
                chain_summaries.append({
                    "chain_id": cid,
                    "entry_count": chain_data.get("entry_count", 0),
                    "sealed": chain_data.get("sealed", False),
                    "seal_hash": chain_data.get("seal_hash"),
                    "integrity_verified": chain_data.get(
                        "integrity_verified", False
                    ),
                })

            return {
                "batch_id": batch_id,
                "description": batch.get("description"),
                "created_at": batch["created_at"],
                "chain_count": len(batch["chains"]),
                "sealed": batch["sealed"],
                "seal_hash": batch["seal_hash"],
                "seal_timestamp": batch.get("seal_timestamp"),
                "chains": chain_summaries,
            }

    def export_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Export the full batch provenance to a JSON-serializable dictionary.

        Args:
            batch_id: Batch identifier.

        Returns:
            Complete batch dictionary or None if not found.
        """
        with self._batch_lock:
            if batch_id not in self._batches:
                return None
            return copy.deepcopy(self._batches[batch_id])


# ============================================================================
# Standalone Hash Helper Functions - Fuel Consumption Records
# ============================================================================


def hash_fuel_consumption(
    record_id: str,
    facility_id: str,
    fuel_type: str,
    fuel_category: str,
    quantity: Decimal,
    unit: str,
    energy_content_mj: Optional[Decimal] = None,
    reporting_period: Optional[str] = None,
    supplier_id: Optional[str] = None,
    country_code: Optional[str] = None,
    biofuel_blend_pct: Optional[Decimal] = None,
) -> str:
    """
    Hash a fuel consumption record for provenance tracking.

    Creates a deterministic SHA-256 digest of the fuel consumption data
    used for Activity 3a (upstream/WTT) emissions calculations. Used in
    the VALIDATE and CLASSIFY stages.

    Args:
        record_id: Unique consumption record identifier.
        facility_id: Facility/site identifier.
        fuel_type: Fuel type code (e.g., "natural_gas", "diesel", "lpg").
        fuel_category: Fuel category (gaseous/liquid/solid/biofuel).
        quantity: Fuel quantity consumed.
        unit: Consumption unit (litres, m3, kg, tonnes, therms, MWh).
        energy_content_mj: Optional energy content in megajoules.
        reporting_period: Optional reporting period (YYYY or YYYY-MM).
        supplier_id: Optional fuel supplier identifier.
        country_code: Optional ISO 3166-1 alpha-2 country code.
        biofuel_blend_pct: Optional biofuel blend percentage (0-100).

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_fuel_consumption(
        ...     record_id="FC-2025-001",
        ...     facility_id="FAC-001",
        ...     fuel_type="natural_gas",
        ...     fuel_category="gaseous",
        ...     quantity=Decimal("125000"),
        ...     unit="m3",
        ...     energy_content_mj=Decimal("4750000"),
        ...     reporting_period="2025",
        ... )
    """
    data = {
        "record_id": record_id,
        "facility_id": facility_id,
        "fuel_type": fuel_type,
        "fuel_category": fuel_category,
        "quantity": str(quantity),
        "unit": unit,
        "energy_content_mj": (
            str(energy_content_mj)
            if energy_content_mj is not None
            else None
        ),
        "reporting_period": reporting_period,
        "supplier_id": supplier_id,
        "country_code": country_code,
        "biofuel_blend_pct": (
            str(biofuel_blend_pct)
            if biofuel_blend_pct is not None
            else None
        ),
    }
    return _hash_helper(data)


def hash_electricity_consumption(
    record_id: str,
    facility_id: str,
    energy_type: str,
    quantity_kwh: Decimal,
    grid_region: str,
    tariff_type: Optional[str] = None,
    renewable_pct: Optional[Decimal] = None,
    supplier_id: Optional[str] = None,
    reporting_period: Optional[str] = None,
    country_code: Optional[str] = None,
    contract_type: Optional[str] = None,
    residual_mix_applied: bool = False,
) -> str:
    """
    Hash an electricity/energy consumption record for provenance tracking.

    Creates a deterministic SHA-256 digest of the electricity or energy
    consumption data used for Activity 3b (upstream electricity) and
    Activity 3c (T&D loss) emissions calculations. Used in the VALIDATE
    and CLASSIFY stages.

    Args:
        record_id: Unique consumption record identifier.
        facility_id: Facility/site identifier.
        energy_type: Energy type (electricity/steam/heating/cooling).
        quantity_kwh: Energy quantity in kilowatt-hours.
        grid_region: Grid region or balancing area identifier.
        tariff_type: Optional tariff type (standard/green/time_of_use).
        renewable_pct: Optional renewable energy percentage (0-100).
        supplier_id: Optional energy supplier identifier.
        reporting_period: Optional reporting period (YYYY or YYYY-MM).
        country_code: Optional ISO 3166-1 alpha-2 country code.
        contract_type: Optional contract type (grid/ppa/rec/onsite).
        residual_mix_applied: Whether residual mix factor was applied.

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_electricity_consumption(
        ...     record_id="EC-2025-001",
        ...     facility_id="FAC-001",
        ...     energy_type="electricity",
        ...     quantity_kwh=Decimal("500000"),
        ...     grid_region="US-WECC",
        ...     tariff_type="standard",
        ...     reporting_period="2025",
        ...     country_code="US",
        ... )
    """
    data = {
        "record_id": record_id,
        "facility_id": facility_id,
        "energy_type": energy_type,
        "quantity_kwh": str(quantity_kwh),
        "grid_region": grid_region,
        "tariff_type": tariff_type,
        "renewable_pct": (
            str(renewable_pct)
            if renewable_pct is not None
            else None
        ),
        "supplier_id": supplier_id,
        "reporting_period": reporting_period,
        "country_code": country_code,
        "contract_type": contract_type,
        "residual_mix_applied": residual_mix_applied,
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Emission Factors
# ============================================================================


def hash_wtt_factor(
    fuel_type: str,
    factor_value: Decimal,
    unit: str,
    source: str,
    source_version: str,
    region: str,
    base_year: int,
    ghg_species: Optional[str] = None,
    lifecycle_stage: Optional[str] = None,
    uncertainty_pct: Optional[Decimal] = None,
) -> str:
    """
    Hash a well-to-tank (WTT) emission factor for provenance tracking.

    Creates a deterministic SHA-256 digest of the WTT emission factor
    used in Activity 3a calculations. WTT factors cover upstream fuel
    emissions from extraction, processing, refining, and transportation
    to the point of sale.

    Args:
        fuel_type: Fuel type code.
        factor_value: WTT emission factor value.
        unit: Factor unit (e.g., "kgCO2e/litre", "kgCO2e/kWh", "kgCO2e/MJ").
        source: Data source (DEFRA/EPA/ECOINVENT/IEA/CUSTOM).
        source_version: Source version string (e.g., "DEFRA-2025").
        region: Geographic region (e.g., "UK", "US", "EU", "GLOBAL").
        base_year: Base year for the emission factor.
        ghg_species: Optional GHG species (CO2/CH4/N2O/total).
        lifecycle_stage: Optional lifecycle stage (extraction/refining/transport).
        uncertainty_pct: Optional uncertainty percentage.

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_wtt_factor(
        ...     fuel_type="diesel",
        ...     factor_value=Decimal("0.62567"),
        ...     unit="kgCO2e/litre",
        ...     source="DEFRA",
        ...     source_version="DEFRA-2025",
        ...     region="UK",
        ...     base_year=2025,
        ... )
    """
    data = {
        "fuel_type": fuel_type,
        "factor_value": str(factor_value),
        "unit": unit,
        "source": source,
        "source_version": source_version,
        "region": region,
        "base_year": base_year,
        "ghg_species": ghg_species,
        "lifecycle_stage": lifecycle_stage,
        "uncertainty_pct": (
            str(uncertainty_pct)
            if uncertainty_pct is not None
            else None
        ),
    }
    return _hash_helper(data)


def hash_upstream_factor(
    energy_type: str,
    grid_region: str,
    factor_value: Decimal,
    unit: str,
    source: str,
    source_version: str,
    base_year: int,
    generation_mix: Optional[Dict[str, Decimal]] = None,
    methodology: Optional[str] = None,
    scope_coverage: Optional[str] = None,
) -> str:
    """
    Hash an upstream electricity emission factor for provenance tracking.

    Creates a deterministic SHA-256 digest of the upstream electricity
    generation emission factor used in Activity 3b calculations. This
    covers emissions from fuel extraction, processing, and transportation
    used in electricity generation (excluding Scope 2 generation emissions).

    Args:
        energy_type: Energy type (electricity/steam/heating/cooling).
        grid_region: Grid region or balancing area identifier.
        factor_value: Upstream emission factor value.
        unit: Factor unit (e.g., "kgCO2e/kWh", "kgCO2e/MJ").
        source: Data source (IEA/EPA/DEFRA/ECOINVENT/NATIONAL).
        source_version: Source version string.
        base_year: Base year for the emission factor.
        generation_mix: Optional generation mix (fuel -> percentage).
        methodology: Optional methodology (attributional/consequential).
        scope_coverage: Optional scope coverage description.

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_upstream_factor(
        ...     energy_type="electricity",
        ...     grid_region="US-WECC",
        ...     factor_value=Decimal("0.04523"),
        ...     unit="kgCO2e/kWh",
        ...     source="EPA",
        ...     source_version="eGRID-2024",
        ...     base_year=2024,
        ... )
    """
    gen_mix_str = None
    if generation_mix is not None:
        gen_mix_str = {k: str(v) for k, v in generation_mix.items()}

    data = {
        "energy_type": energy_type,
        "grid_region": grid_region,
        "factor_value": str(factor_value),
        "unit": unit,
        "source": source,
        "source_version": source_version,
        "base_year": base_year,
        "generation_mix": gen_mix_str,
        "methodology": methodology,
        "scope_coverage": scope_coverage,
    }
    return _hash_helper(data)


def hash_td_loss_factor(
    energy_type: str,
    grid_region: str,
    loss_factor_pct: Decimal,
    source: str,
    source_version: str,
    base_year: int,
    voltage_level: Optional[str] = None,
    country_code: Optional[str] = None,
    grid_ef_kgco2e_kwh: Optional[Decimal] = None,
    td_emission_factor: Optional[Decimal] = None,
) -> str:
    """
    Hash a transmission and distribution (T&D) loss factor for provenance.

    Creates a deterministic SHA-256 digest of the T&D loss factor used
    in Activity 3c calculations. T&D losses represent the energy lost
    during transmission from generation to consumption point.

    Args:
        energy_type: Energy type (electricity/steam/heating/cooling).
        grid_region: Grid region or balancing area identifier.
        loss_factor_pct: T&D loss factor as percentage (e.g., 5.5 for 5.5%).
        source: Data source (IEA/EIA/NATIONAL_GRID/CUSTOM).
        source_version: Source version string.
        base_year: Base year for the loss factor.
        voltage_level: Optional voltage level (high/medium/low/distribution).
        country_code: Optional ISO 3166-1 alpha-2 country code.
        grid_ef_kgco2e_kwh: Optional grid emission factor for loss calc.
        td_emission_factor: Optional pre-calculated T&D emission factor.

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_td_loss_factor(
        ...     energy_type="electricity",
        ...     grid_region="US-WECC",
        ...     loss_factor_pct=Decimal("5.5"),
        ...     source="EIA",
        ...     source_version="EIA-2024",
        ...     base_year=2024,
        ...     voltage_level="distribution",
        ...     country_code="US",
        ... )
    """
    data = {
        "energy_type": energy_type,
        "grid_region": grid_region,
        "loss_factor_pct": str(loss_factor_pct),
        "source": source,
        "source_version": source_version,
        "base_year": base_year,
        "voltage_level": voltage_level,
        "country_code": country_code,
        "grid_ef_kgco2e_kwh": (
            str(grid_ef_kgco2e_kwh)
            if grid_ef_kgco2e_kwh is not None
            else None
        ),
        "td_emission_factor": (
            str(td_emission_factor)
            if td_emission_factor is not None
            else None
        ),
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Activity 3a Results (WTT)
# ============================================================================


def hash_activity_3a_result(
    record_id: str,
    facility_id: str,
    fuel_type: str,
    quantity: Decimal,
    quantity_unit: str,
    wtt_factor: Decimal,
    wtt_factor_unit: str,
    wtt_source: str,
    total_emissions_tco2e: Decimal,
    co2_tonnes: Decimal,
    ch4_tco2e: Decimal,
    n2o_tco2e: Decimal,
    biogenic_tco2e: Optional[Decimal] = None,
    energy_content_mj: Optional[Decimal] = None,
) -> str:
    """
    Hash an Activity 3a (upstream fuel WTT) calculation result.

    Creates a deterministic SHA-256 digest of the CALCULATE_3A stage
    output. Covers well-to-tank emissions from extraction, processing,
    refining, and transportation of purchased fuels.

    Args:
        record_id: Source consumption record identifier.
        facility_id: Facility identifier.
        fuel_type: Fuel type code.
        quantity: Fuel quantity consumed.
        quantity_unit: Fuel quantity unit.
        wtt_factor: WTT emission factor applied.
        wtt_factor_unit: WTT factor unit.
        wtt_source: WTT factor source database.
        total_emissions_tco2e: Total WTT emissions (tCO2e).
        co2_tonnes: CO2 component (tonnes).
        ch4_tco2e: CH4 component (tCO2e).
        n2o_tco2e: N2O component (tCO2e).
        biogenic_tco2e: Optional biogenic CO2 (tCO2e).
        energy_content_mj: Optional energy content (MJ).

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_activity_3a_result(
        ...     record_id="FC-2025-001",
        ...     facility_id="FAC-001",
        ...     fuel_type="diesel",
        ...     quantity=Decimal("50000"),
        ...     quantity_unit="litres",
        ...     wtt_factor=Decimal("0.62567"),
        ...     wtt_factor_unit="kgCO2e/litre",
        ...     wtt_source="DEFRA",
        ...     total_emissions_tco2e=Decimal("31.28"),
        ...     co2_tonnes=Decimal("28.15"),
        ...     ch4_tco2e=Decimal("2.01"),
        ...     n2o_tco2e=Decimal("1.12"),
        ... )
    """
    data = {
        "record_id": record_id,
        "facility_id": facility_id,
        "fuel_type": fuel_type,
        "quantity": str(quantity),
        "quantity_unit": quantity_unit,
        "wtt_factor": str(wtt_factor),
        "wtt_factor_unit": wtt_factor_unit,
        "wtt_source": wtt_source,
        "total_emissions_tco2e": str(total_emissions_tco2e),
        "co2_tonnes": str(co2_tonnes),
        "ch4_tco2e": str(ch4_tco2e),
        "n2o_tco2e": str(n2o_tco2e),
        "biogenic_tco2e": (
            str(biogenic_tco2e) if biogenic_tco2e is not None else None
        ),
        "energy_content_mj": (
            str(energy_content_mj) if energy_content_mj is not None else None
        ),
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Activity 3b Results (Upstream Electricity)
# ============================================================================


def hash_activity_3b_result(
    record_id: str,
    facility_id: str,
    energy_type: str,
    quantity_kwh: Decimal,
    grid_region: str,
    upstream_factor: Decimal,
    upstream_factor_unit: str,
    upstream_source: str,
    total_emissions_tco2e: Decimal,
    co2_tonnes: Decimal,
    ch4_tco2e: Decimal,
    n2o_tco2e: Decimal,
    renewable_adjustment_tco2e: Optional[Decimal] = None,
    residual_mix_applied: bool = False,
) -> str:
    """
    Hash an Activity 3b (upstream electricity) calculation result.

    Creates a deterministic SHA-256 digest of the CALCULATE_3B stage
    output. Covers upstream emissions from fuel extraction, processing,
    and transportation used in electricity generation (not in Scope 2).

    Args:
        record_id: Source consumption record identifier.
        facility_id: Facility identifier.
        energy_type: Energy type (electricity/steam/heating/cooling).
        quantity_kwh: Energy quantity (kWh).
        grid_region: Grid region or balancing area.
        upstream_factor: Upstream emission factor applied.
        upstream_factor_unit: Upstream factor unit.
        upstream_source: Upstream factor source database.
        total_emissions_tco2e: Total upstream emissions (tCO2e).
        co2_tonnes: CO2 component (tonnes).
        ch4_tco2e: CH4 component (tCO2e).
        n2o_tco2e: N2O component (tCO2e).
        renewable_adjustment_tco2e: Optional renewable energy adjustment.
        residual_mix_applied: Whether residual mix factor was applied.

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_activity_3b_result(
        ...     record_id="EC-2025-001",
        ...     facility_id="FAC-001",
        ...     energy_type="electricity",
        ...     quantity_kwh=Decimal("500000"),
        ...     grid_region="US-WECC",
        ...     upstream_factor=Decimal("0.04523"),
        ...     upstream_factor_unit="kgCO2e/kWh",
        ...     upstream_source="EPA",
        ...     total_emissions_tco2e=Decimal("22.62"),
        ...     co2_tonnes=Decimal("20.35"),
        ...     ch4_tco2e=Decimal("1.47"),
        ...     n2o_tco2e=Decimal("0.80"),
        ... )
    """
    data = {
        "record_id": record_id,
        "facility_id": facility_id,
        "energy_type": energy_type,
        "quantity_kwh": str(quantity_kwh),
        "grid_region": grid_region,
        "upstream_factor": str(upstream_factor),
        "upstream_factor_unit": upstream_factor_unit,
        "upstream_source": upstream_source,
        "total_emissions_tco2e": str(total_emissions_tco2e),
        "co2_tonnes": str(co2_tonnes),
        "ch4_tco2e": str(ch4_tco2e),
        "n2o_tco2e": str(n2o_tco2e),
        "renewable_adjustment_tco2e": (
            str(renewable_adjustment_tco2e)
            if renewable_adjustment_tco2e is not None
            else None
        ),
        "residual_mix_applied": residual_mix_applied,
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Activity 3c Results (T&D Losses)
# ============================================================================


def hash_activity_3c_result(
    record_id: str,
    facility_id: str,
    energy_type: str,
    quantity_kwh: Decimal,
    grid_region: str,
    td_loss_pct: Decimal,
    grid_ef_kgco2e_kwh: Decimal,
    td_source: str,
    total_emissions_tco2e: Decimal,
    co2_tonnes: Decimal,
    ch4_tco2e: Decimal,
    n2o_tco2e: Decimal,
    energy_lost_kwh: Optional[Decimal] = None,
    voltage_level: Optional[str] = None,
) -> str:
    """
    Hash an Activity 3c (T&D losses) calculation result.

    Creates a deterministic SHA-256 digest of the CALCULATE_3C stage
    output. Covers emissions from electricity lost during transmission
    and distribution from generation to the reporting company.

    T&D loss emissions = consumption_kwh * td_loss_pct/100 * grid_ef

    Args:
        record_id: Source consumption record identifier.
        facility_id: Facility identifier.
        energy_type: Energy type (electricity/steam/heating/cooling).
        quantity_kwh: Energy quantity consumed (kWh).
        grid_region: Grid region or balancing area.
        td_loss_pct: T&D loss percentage (e.g., 5.5 for 5.5%).
        grid_ef_kgco2e_kwh: Grid emission factor (kgCO2e/kWh).
        td_source: T&D loss factor source database.
        total_emissions_tco2e: Total T&D loss emissions (tCO2e).
        co2_tonnes: CO2 component (tonnes).
        ch4_tco2e: CH4 component (tCO2e).
        n2o_tco2e: N2O component (tCO2e).
        energy_lost_kwh: Optional energy lost in T&D (kWh).
        voltage_level: Optional voltage level (high/medium/low/distribution).

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_activity_3c_result(
        ...     record_id="EC-2025-001",
        ...     facility_id="FAC-001",
        ...     energy_type="electricity",
        ...     quantity_kwh=Decimal("500000"),
        ...     grid_region="US-WECC",
        ...     td_loss_pct=Decimal("5.5"),
        ...     grid_ef_kgco2e_kwh=Decimal("0.3526"),
        ...     td_source="EIA",
        ...     total_emissions_tco2e=Decimal("9.70"),
        ...     co2_tonnes=Decimal("8.73"),
        ...     ch4_tco2e=Decimal("0.63"),
        ...     n2o_tco2e=Decimal("0.34"),
        ...     energy_lost_kwh=Decimal("27500"),
        ... )
    """
    data = {
        "record_id": record_id,
        "facility_id": facility_id,
        "energy_type": energy_type,
        "quantity_kwh": str(quantity_kwh),
        "grid_region": grid_region,
        "td_loss_pct": str(td_loss_pct),
        "grid_ef_kgco2e_kwh": str(grid_ef_kgco2e_kwh),
        "td_source": td_source,
        "total_emissions_tco2e": str(total_emissions_tco2e),
        "co2_tonnes": str(co2_tonnes),
        "ch4_tco2e": str(ch4_tco2e),
        "n2o_tco2e": str(n2o_tco2e),
        "energy_lost_kwh": (
            str(energy_lost_kwh) if energy_lost_kwh is not None else None
        ),
        "voltage_level": voltage_level,
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Combined Calculation Results
# ============================================================================


def hash_calculation_result(
    calculation_id: str,
    facility_id: str,
    reporting_period: str,
    activity_3a_tco2e: Decimal,
    activity_3b_tco2e: Decimal,
    activity_3c_tco2e: Decimal,
    total_emissions_tco2e: Decimal,
    co2_tonnes: Decimal,
    ch4_tco2e: Decimal,
    n2o_tco2e: Decimal,
    biogenic_tco2e: Optional[Decimal] = None,
    fuel_record_count: int = 0,
    electricity_record_count: int = 0,
    method: Optional[str] = None,
) -> str:
    """
    Hash a full fuel & energy activities calculation result.

    Creates a deterministic SHA-256 digest of the combined per-facility
    emissions output across all three sub-activities. Tracks all GHG
    species separately.

    Args:
        calculation_id: Unique calculation run identifier.
        facility_id: Facility identifier.
        reporting_period: Reporting period (YYYY or YYYY-QN).
        activity_3a_tco2e: Activity 3a total (WTT upstream fuel) (tCO2e).
        activity_3b_tco2e: Activity 3b total (upstream electricity) (tCO2e).
        activity_3c_tco2e: Activity 3c total (T&D losses) (tCO2e).
        total_emissions_tco2e: Grand total emissions (tCO2e).
        co2_tonnes: CO2 component (tonnes).
        ch4_tco2e: CH4 component (tCO2e).
        n2o_tco2e: N2O component (tCO2e).
        biogenic_tco2e: Optional biogenic CO2 (tCO2e).
        fuel_record_count: Number of fuel records processed.
        electricity_record_count: Number of electricity records processed.
        method: Optional calculation method (HYBRID/WTT_ONLY/TD_ONLY/etc.).

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_calculation_result(
        ...     calculation_id="CALC-2025-001",
        ...     facility_id="FAC-001",
        ...     reporting_period="2025",
        ...     activity_3a_tco2e=Decimal("31.28"),
        ...     activity_3b_tco2e=Decimal("22.62"),
        ...     activity_3c_tco2e=Decimal("9.70"),
        ...     total_emissions_tco2e=Decimal("63.60"),
        ...     co2_tonnes=Decimal("57.23"),
        ...     ch4_tco2e=Decimal("4.11"),
        ...     n2o_tco2e=Decimal("2.26"),
        ... )
    """
    data = {
        "calculation_id": calculation_id,
        "facility_id": facility_id,
        "reporting_period": reporting_period,
        "activity_3a_tco2e": str(activity_3a_tco2e),
        "activity_3b_tco2e": str(activity_3b_tco2e),
        "activity_3c_tco2e": str(activity_3c_tco2e),
        "total_emissions_tco2e": str(total_emissions_tco2e),
        "co2_tonnes": str(co2_tonnes),
        "ch4_tco2e": str(ch4_tco2e),
        "n2o_tco2e": str(n2o_tco2e),
        "biogenic_tco2e": (
            str(biogenic_tco2e) if biogenic_tco2e is not None else None
        ),
        "fuel_record_count": fuel_record_count,
        "electricity_record_count": electricity_record_count,
        "method": method,
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Compliance Results
# ============================================================================


def hash_compliance_result(
    framework: str,
    status: str,
    requirements_met: int,
    requirements_total: int,
    findings: List[str],
    non_conformities: Optional[List[str]] = None,
    recommendations: Optional[List[str]] = None,
    scope3_cat3_specific: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Hash a regulatory compliance check result.

    Creates a deterministic SHA-256 digest of the compliance assessment
    for one of seven supported regulatory frameworks. Specific to
    Scope 3 Category 3 requirements.

    Args:
        framework: Framework name (GHG_PROTOCOL/ISO_14064/CSRD/CDP/
                   SBTi/UK_SECR/EU_ETS).
        status: Compliance status (COMPLIANT/NON_COMPLIANT/PARTIAL/
                NOT_APPLICABLE).
        requirements_met: Number of requirements satisfied.
        requirements_total: Total number of applicable requirements.
        findings: List of compliance findings/observations.
        non_conformities: Optional list of non-conformity descriptions.
        recommendations: Optional list of improvement recommendations.
        scope3_cat3_specific: Optional Scope 3 Cat 3 specific checks
                              (e.g., WTT coverage, T&D methodology).

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_compliance_result(
        ...     framework="GHG_PROTOCOL",
        ...     status="COMPLIANT",
        ...     requirements_met=15,
        ...     requirements_total=15,
        ...     findings=["All Category 3 requirements satisfied"],
        ... )
    """
    data = {
        "framework": framework,
        "status": status,
        "requirements_met": requirements_met,
        "requirements_total": requirements_total,
        "findings": findings,
        "non_conformities": non_conformities or [],
        "recommendations": recommendations or [],
        "scope3_cat3_specific": scope3_cat3_specific or {},
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Batch Results
# ============================================================================


def hash_batch_result(
    batch_id: str,
    total_facilities: int,
    processed_facilities: int,
    failed_facilities: int,
    skipped_facilities: int,
    total_emissions_tco2e: Decimal,
    activity_3a_total_tco2e: Decimal,
    activity_3b_total_tco2e: Decimal,
    activity_3c_total_tco2e: Decimal,
    fuel_records_processed: int = 0,
    electricity_records_processed: int = 0,
    weighted_dqi: Optional[Decimal] = None,
    duration_ms: int = 0,
) -> str:
    """
    Hash a batch calculation result summary.

    Creates a deterministic SHA-256 digest of the batch-level
    aggregation output for multi-facility processing.

    Args:
        batch_id: Batch identifier.
        total_facilities: Total number of facilities in batch.
        processed_facilities: Successfully processed count.
        failed_facilities: Failed processing count.
        skipped_facilities: Skipped count.
        total_emissions_tco2e: Total batch emissions (tCO2e).
        activity_3a_total_tco2e: Batch Activity 3a total (tCO2e).
        activity_3b_total_tco2e: Batch Activity 3b total (tCO2e).
        activity_3c_total_tco2e: Batch Activity 3c total (tCO2e).
        fuel_records_processed: Total fuel records processed.
        electricity_records_processed: Total electricity records processed.
        weighted_dqi: Optional weighted average DQI score.
        duration_ms: Batch processing duration (milliseconds).

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_batch_result(
        ...     batch_id="BATCH-2025-Q1",
        ...     total_facilities=10,
        ...     processed_facilities=10,
        ...     failed_facilities=0,
        ...     skipped_facilities=0,
        ...     total_emissions_tco2e=Decimal("636.0"),
        ...     activity_3a_total_tco2e=Decimal("312.8"),
        ...     activity_3b_total_tco2e=Decimal("226.2"),
        ...     activity_3c_total_tco2e=Decimal("97.0"),
        ...     fuel_records_processed=150,
        ...     electricity_records_processed=42,
        ...     duration_ms=5320,
        ... )
    """
    data = {
        "batch_id": batch_id,
        "total_facilities": total_facilities,
        "processed_facilities": processed_facilities,
        "failed_facilities": failed_facilities,
        "skipped_facilities": skipped_facilities,
        "total_emissions_tco2e": str(total_emissions_tco2e),
        "activity_3a_total_tco2e": str(activity_3a_total_tco2e),
        "activity_3b_total_tco2e": str(activity_3b_total_tco2e),
        "activity_3c_total_tco2e": str(activity_3c_total_tco2e),
        "fuel_records_processed": fuel_records_processed,
        "electricity_records_processed": electricity_records_processed,
        "weighted_dqi": (
            str(weighted_dqi) if weighted_dqi is not None else None
        ),
        "duration_ms": duration_ms,
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Emission Factor Resolution
# ============================================================================


def hash_ef_resolution(
    record_id: str,
    activity_type: str,
    resolution_strategy: str,
    source_priority: List[str],
    selected_source: str,
    selected_factor: Decimal,
    selected_unit: str,
    fallback_used: bool,
    fallback_reason: Optional[str] = None,
    candidate_count: int = 0,
) -> str:
    """
    Hash an emission factor resolution result.

    Creates a deterministic SHA-256 digest of the RESOLVE_EFS stage
    output. Tracks the hierarchical source selection strategy and any
    fallback behavior for WTT, upstream, and T&D factors.

    Args:
        record_id: Record identifier being resolved.
        activity_type: Activity type (3a/3b/3c).
        resolution_strategy: Strategy (hierarchical/best_available/manual).
        source_priority: Ordered list of EF sources in priority order.
        selected_source: The source that was ultimately selected.
        selected_factor: The emission factor value selected.
        selected_unit: Unit of the selected factor.
        fallback_used: Whether a fallback source was used.
        fallback_reason: Optional reason for fallback.
        candidate_count: Number of candidate factors evaluated.

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_ef_resolution(
        ...     record_id="FC-2025-001",
        ...     activity_type="3a",
        ...     resolution_strategy="hierarchical",
        ...     source_priority=["DEFRA", "EPA", "ECOINVENT"],
        ...     selected_source="DEFRA",
        ...     selected_factor=Decimal("0.62567"),
        ...     selected_unit="kgCO2e/litre",
        ...     fallback_used=False,
        ... )
    """
    data = {
        "record_id": record_id,
        "activity_type": activity_type,
        "resolution_strategy": resolution_strategy,
        "source_priority": source_priority,
        "selected_source": selected_source,
        "selected_factor": str(selected_factor),
        "selected_unit": selected_unit,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "candidate_count": candidate_count,
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Normalization
# ============================================================================


def hash_normalization_result(
    record_id: str,
    original_quantity: Decimal,
    original_unit: str,
    normalized_quantity: Decimal,
    normalized_unit: str,
    conversion_factor: Decimal,
    conversion_source: str,
    energy_content_mj: Optional[Decimal] = None,
    density_kg_per_litre: Optional[Decimal] = None,
    calorific_value_mj_per_kg: Optional[Decimal] = None,
) -> str:
    """
    Hash a unit normalization result for provenance tracking.

    Creates a deterministic SHA-256 digest of the NORMALIZE stage output
    for a single activity record. Tracks the unit conversion chain from
    original units to canonical SI units.

    Args:
        record_id: Record identifier.
        original_quantity: Quantity before normalization.
        original_unit: Original unit of measurement.
        normalized_quantity: Quantity after normalization.
        normalized_unit: Canonical SI unit.
        conversion_factor: Conversion factor applied.
        conversion_source: Source of conversion factor.
        energy_content_mj: Optional energy content (MJ).
        density_kg_per_litre: Optional fuel density (kg/litre).
        calorific_value_mj_per_kg: Optional calorific value (MJ/kg).

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_normalization_result(
        ...     record_id="FC-2025-001",
        ...     original_quantity=Decimal("50000"),
        ...     original_unit="litres",
        ...     normalized_quantity=Decimal("42000"),
        ...     normalized_unit="kg",
        ...     conversion_factor=Decimal("0.84"),
        ...     conversion_source="DEFRA-densities-2025",
        ... )
    """
    data = {
        "record_id": record_id,
        "original_quantity": str(original_quantity),
        "original_unit": original_unit,
        "normalized_quantity": str(normalized_quantity),
        "normalized_unit": normalized_unit,
        "conversion_factor": str(conversion_factor),
        "conversion_source": conversion_source,
        "energy_content_mj": (
            str(energy_content_mj)
            if energy_content_mj is not None
            else None
        ),
        "density_kg_per_litre": (
            str(density_kg_per_litre)
            if density_kg_per_litre is not None
            else None
        ),
        "calorific_value_mj_per_kg": (
            str(calorific_value_mj_per_kg)
            if calorific_value_mj_per_kg is not None
            else None
        ),
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Classification
# ============================================================================


def hash_classification_result(
    record_id: str,
    record_type: str,
    activity_type: str,
    fuel_type: Optional[str] = None,
    fuel_category: Optional[str] = None,
    energy_type: Optional[str] = None,
    energy_category: Optional[str] = None,
    classification_method: str = "rule_based",
    confidence_score: Optional[Decimal] = None,
) -> str:
    """
    Hash a record classification result for provenance tracking.

    Creates a deterministic SHA-256 digest of the CLASSIFY stage output
    for a single activity record. Tracks how each record was classified
    into sub-activity type and fuel/energy category.

    Args:
        record_id: Record identifier.
        record_type: Record type (fuel_consumption/electricity_consumption).
        activity_type: Assigned activity (3a/3b/3c).
        fuel_type: Optional fuel type (for fuel records).
        fuel_category: Optional fuel category (gaseous/liquid/solid/biofuel).
        energy_type: Optional energy type (for electricity records).
        energy_category: Optional energy category (grid/onsite/purchased).
        classification_method: Method used (rule_based/ml_classifier/manual).
        confidence_score: Optional classification confidence (0-1).

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_classification_result(
        ...     record_id="FC-2025-001",
        ...     record_type="fuel_consumption",
        ...     activity_type="3a",
        ...     fuel_type="diesel",
        ...     fuel_category="liquid",
        ...     classification_method="rule_based",
        ...     confidence_score=Decimal("0.95"),
        ... )
    """
    data = {
        "record_id": record_id,
        "record_type": record_type,
        "activity_type": activity_type,
        "fuel_type": fuel_type,
        "fuel_category": fuel_category,
        "energy_type": energy_type,
        "energy_category": energy_category,
        "classification_method": classification_method,
        "confidence_score": (
            str(confidence_score)
            if confidence_score is not None
            else None
        ),
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Aggregation
# ============================================================================


def hash_aggregation_result(
    aggregation_id: str,
    aggregation_level: str,
    entity_id: str,
    reporting_period: str,
    activity_3a_tco2e: Decimal,
    activity_3b_tco2e: Decimal,
    activity_3c_tco2e: Decimal,
    total_emissions_tco2e: Decimal,
    facility_count: int,
    record_count: int,
    coverage_pct: Decimal,
    weighted_dqi_score: Optional[Decimal] = None,
) -> str:
    """
    Hash an aggregation result across sub-activities and facilities.

    Creates a deterministic SHA-256 digest of the AGGREGATE stage output.
    Tracks emissions roll-up at organization, division, or facility level.

    Args:
        aggregation_id: Aggregation run identifier.
        aggregation_level: Level (organization/division/facility/region).
        entity_id: Entity being aggregated (org/div/facility ID).
        reporting_period: Reporting period (YYYY or YYYY-QN).
        activity_3a_tco2e: Aggregated Activity 3a total (tCO2e).
        activity_3b_tco2e: Aggregated Activity 3b total (tCO2e).
        activity_3c_tco2e: Aggregated Activity 3c total (tCO2e).
        total_emissions_tco2e: Aggregated grand total (tCO2e).
        facility_count: Number of facilities included.
        record_count: Total activity records included.
        coverage_pct: Data coverage percentage (0-100).
        weighted_dqi_score: Optional weighted DQI score (1-5).

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_aggregation_result(
        ...     aggregation_id="AGG-2025-001",
        ...     aggregation_level="organization",
        ...     entity_id="ORG-123",
        ...     reporting_period="2025",
        ...     activity_3a_tco2e=Decimal("312.8"),
        ...     activity_3b_tco2e=Decimal("226.2"),
        ...     activity_3c_tco2e=Decimal("97.0"),
        ...     total_emissions_tco2e=Decimal("636.0"),
        ...     facility_count=10,
        ...     record_count=192,
        ...     coverage_pct=Decimal("98.5"),
        ... )
    """
    data = {
        "aggregation_id": aggregation_id,
        "aggregation_level": aggregation_level,
        "entity_id": entity_id,
        "reporting_period": reporting_period,
        "activity_3a_tco2e": str(activity_3a_tco2e),
        "activity_3b_tco2e": str(activity_3b_tco2e),
        "activity_3c_tco2e": str(activity_3c_tco2e),
        "total_emissions_tco2e": str(total_emissions_tco2e),
        "facility_count": facility_count,
        "record_count": record_count,
        "coverage_pct": str(coverage_pct),
        "weighted_dqi_score": (
            str(weighted_dqi_score)
            if weighted_dqi_score is not None
            else None
        ),
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - DQI Assessment
# ============================================================================


def hash_dqi_assessment(
    record_id: str,
    activity_type: str,
    technology_score: int,
    temporal_score: int,
    geography_score: int,
    completeness_score: int,
    reliability_score: int,
    composite_score: Decimal,
    assessment_notes: Optional[str] = None,
) -> str:
    """
    Hash a data quality indicator (DQI) assessment.

    Creates a deterministic SHA-256 digest of the five-dimension DQI
    scoring per GHG Protocol Technical Guidance.

    Args:
        record_id: Record identifier.
        activity_type: Activity type assessed (3a/3b/3c).
        technology_score: Technology representativeness (1=best, 5=worst).
        temporal_score: Temporal representativeness (1=best, 5=worst).
        geography_score: Geographic representativeness (1=best, 5=worst).
        completeness_score: Data completeness (1=best, 5=worst).
        reliability_score: Data reliability (1=best, 5=worst).
        composite_score: Composite DQI score (1-5, lower is better).
        assessment_notes: Optional assessor notes.

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_dqi_assessment(
        ...     record_id="FC-2025-001",
        ...     activity_type="3a",
        ...     technology_score=2,
        ...     temporal_score=1,
        ...     geography_score=2,
        ...     completeness_score=1,
        ...     reliability_score=1,
        ...     composite_score=Decimal("1.4"),
        ... )
    """
    data = {
        "record_id": record_id,
        "activity_type": activity_type,
        "technology_score": technology_score,
        "temporal_score": temporal_score,
        "geography_score": geography_score,
        "completeness_score": completeness_score,
        "reliability_score": reliability_score,
        "composite_score": str(composite_score),
        "assessment_notes": assessment_notes,
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Uncertainty Quantification
# ============================================================================


def hash_uncertainty_quantification(
    facility_id: str,
    activity_type: str,
    method: str,
    emissions_mean: Decimal,
    emissions_std_dev: Decimal,
    ci_lower_95: Decimal,
    ci_upper_95: Decimal,
    cv_percent: Decimal,
    monte_carlo_iterations: Optional[int] = None,
    uncertainty_sources: Optional[List[str]] = None,
    sensitivity_ranking: Optional[List[str]] = None,
) -> str:
    """
    Hash an uncertainty quantification result.

    Creates a deterministic SHA-256 digest of uncertainty analysis output,
    supporting both Monte Carlo and analytical methods.

    Args:
        facility_id: Facility identifier.
        activity_type: Activity type (3a/3b/3c/combined).
        method: Uncertainty method (monte_carlo/analytical/error_propagation).
        emissions_mean: Mean emissions estimate (tCO2e).
        emissions_std_dev: Standard deviation (tCO2e).
        ci_lower_95: 95% confidence interval lower bound (tCO2e).
        ci_upper_95: 95% confidence interval upper bound (tCO2e).
        cv_percent: Coefficient of variation (%).
        monte_carlo_iterations: Optional number of MC iterations.
        uncertainty_sources: Optional ordered list of uncertainty sources.
        sensitivity_ranking: Optional sensitivity ranking of parameters.

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_uncertainty_quantification(
        ...     facility_id="FAC-001",
        ...     activity_type="combined",
        ...     method="monte_carlo",
        ...     emissions_mean=Decimal("63.60"),
        ...     emissions_std_dev=Decimal("8.50"),
        ...     ci_lower_95=Decimal("49.12"),
        ...     ci_upper_95=Decimal("81.44"),
        ...     cv_percent=Decimal("13.4"),
        ...     monte_carlo_iterations=10000,
        ... )
    """
    data = {
        "facility_id": facility_id,
        "activity_type": activity_type,
        "method": method,
        "emissions_mean": str(emissions_mean),
        "emissions_std_dev": str(emissions_std_dev),
        "ci_lower_95": str(ci_lower_95),
        "ci_upper_95": str(ci_upper_95),
        "cv_percent": str(cv_percent),
        "monte_carlo_iterations": monte_carlo_iterations,
        "uncertainty_sources": uncertainty_sources or [],
        "sensitivity_ranking": sensitivity_ranking or [],
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Validation
# ============================================================================


def hash_validation_result(
    record_id: str,
    record_type: str,
    validation_status: str,
    rules_passed: int,
    rules_failed: int,
    rules_total: int,
    errors: List[str],
    warnings: List[str],
    validation_mode: str = "strict",
) -> str:
    """
    Hash a validation result for a single activity record.

    Creates a deterministic SHA-256 digest of the VALIDATE stage output
    for a single record, tracking which rules passed/failed.

    Args:
        record_id: Record identifier.
        record_type: Record type (fuel_consumption/electricity_consumption).
        validation_status: Status (PASS/FAIL/WARN).
        rules_passed: Number of validation rules passed.
        rules_failed: Number of validation rules failed.
        rules_total: Total validation rules applied.
        errors: List of error descriptions.
        warnings: List of warning descriptions.
        validation_mode: Validation mode (strict/lenient).

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_validation_result(
        ...     record_id="FC-2025-001",
        ...     record_type="fuel_consumption",
        ...     validation_status="PASS",
        ...     rules_passed=12,
        ...     rules_failed=0,
        ...     rules_total=12,
        ...     errors=[],
        ...     warnings=[],
        ... )
    """
    data = {
        "record_id": record_id,
        "record_type": record_type,
        "validation_status": validation_status,
        "rules_passed": rules_passed,
        "rules_failed": rules_failed,
        "rules_total": rules_total,
        "errors": errors,
        "warnings": warnings,
        "validation_mode": validation_mode,
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Pipeline Context
# ============================================================================


def hash_pipeline_context(
    calculation_id: str,
    organization_id: str,
    reporting_period: str,
    pipeline_version: str,
    started_at: str,
    completed_at: str,
    duration_ms: int,
    facility_count: int,
    fuel_record_count: int,
    electricity_record_count: int,
    total_emissions_tco2e: Decimal,
    stages_completed: int,
    stages_total: int,
) -> str:
    """
    Hash the pipeline execution context for the SEAL stage.

    Creates a deterministic SHA-256 digest of the complete pipeline
    run metadata, providing an immutable record of the calculation
    execution environment.

    Args:
        calculation_id: Unique calculation run identifier.
        organization_id: Organization identifier.
        reporting_period: Reporting period (YYYY or YYYY-QN).
        pipeline_version: Pipeline code version.
        started_at: Pipeline start timestamp (ISO 8601).
        completed_at: Pipeline completion timestamp (ISO 8601).
        duration_ms: Total pipeline duration in milliseconds.
        facility_count: Number of facilities processed.
        fuel_record_count: Number of fuel records processed.
        electricity_record_count: Number of electricity records processed.
        total_emissions_tco2e: Total emissions calculated (tCO2e).
        stages_completed: Number of stages completed.
        stages_total: Total number of stages expected.

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_pipeline_context(
        ...     calculation_id="CALC-2025-001",
        ...     organization_id="ORG-123",
        ...     reporting_period="2025",
        ...     pipeline_version="1.0.0",
        ...     started_at="2025-03-15T10:00:00Z",
        ...     completed_at="2025-03-15T10:02:15Z",
        ...     duration_ms=135000,
        ...     facility_count=10,
        ...     fuel_record_count=150,
        ...     electricity_record_count=42,
        ...     total_emissions_tco2e=Decimal("636.0"),
        ...     stages_completed=10,
        ...     stages_total=10,
        ... )
    """
    data = {
        "calculation_id": calculation_id,
        "organization_id": organization_id,
        "reporting_period": reporting_period,
        "pipeline_version": pipeline_version,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": duration_ms,
        "facility_count": facility_count,
        "fuel_record_count": fuel_record_count,
        "electricity_record_count": electricity_record_count,
        "total_emissions_tco2e": str(total_emissions_tco2e),
        "stages_completed": stages_completed,
        "stages_total": stages_total,
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Double Counting Check
# ============================================================================


def hash_double_counting_check(
    facility_id: str,
    scope1_overlap_detected: bool,
    scope2_overlap_detected: bool,
    overlap_amount_tco2e: Decimal,
    adjustment_applied_tco2e: Decimal,
    check_method: str,
    overlapping_items: List[str],
    activity_3a_adjusted: bool = False,
    activity_3b_adjusted: bool = False,
    activity_3c_adjusted: bool = False,
) -> str:
    """
    Hash a double-counting check result.

    Creates a deterministic SHA-256 digest ensuring fuel & energy
    activities emissions do not overlap with Scope 1 (direct combustion)
    or Scope 2 (purchased electricity) reported emissions.

    Args:
        facility_id: Facility identifier.
        scope1_overlap_detected: Whether overlap with Scope 1 was found.
        scope2_overlap_detected: Whether overlap with Scope 2 was found.
        overlap_amount_tco2e: Total identified overlap (tCO2e).
        adjustment_applied_tco2e: Deduction applied (tCO2e).
        check_method: Verification method (id_matching/source_matching).
        overlapping_items: List of overlapping item identifiers.
        activity_3a_adjusted: Whether Activity 3a was adjusted.
        activity_3b_adjusted: Whether Activity 3b was adjusted.
        activity_3c_adjusted: Whether Activity 3c was adjusted.

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "facility_id": facility_id,
        "scope1_overlap_detected": scope1_overlap_detected,
        "scope2_overlap_detected": scope2_overlap_detected,
        "overlap_amount_tco2e": str(overlap_amount_tco2e),
        "adjustment_applied_tco2e": str(adjustment_applied_tco2e),
        "check_method": check_method,
        "overlapping_items": overlapping_items,
        "activity_3a_adjusted": activity_3a_adjusted,
        "activity_3b_adjusted": activity_3b_adjusted,
        "activity_3c_adjusted": activity_3c_adjusted,
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Energy Content
# ============================================================================


def hash_energy_content(
    fuel_type: str,
    net_calorific_value: Decimal,
    gross_calorific_value: Optional[Decimal],
    unit: str,
    source: str,
    source_version: str,
    region: Optional[str] = None,
    moisture_content_pct: Optional[Decimal] = None,
) -> str:
    """
    Hash a fuel energy content record for provenance tracking.

    Creates a deterministic SHA-256 digest of the calorific value
    data used for energy content normalization in the NORMALIZE stage.

    Args:
        fuel_type: Fuel type code.
        net_calorific_value: Net calorific value (NCV/LHV).
        gross_calorific_value: Optional gross calorific value (GCV/HHV).
        unit: Calorific value unit (MJ/kg, MJ/litre, MJ/m3).
        source: Data source (IPCC/DEFRA/EPA/national).
        source_version: Source version.
        region: Optional geographic region.
        moisture_content_pct: Optional moisture content percentage.

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "fuel_type": fuel_type,
        "net_calorific_value": str(net_calorific_value),
        "gross_calorific_value": (
            str(gross_calorific_value)
            if gross_calorific_value is not None
            else None
        ),
        "unit": unit,
        "source": source,
        "source_version": source_version,
        "region": region,
        "moisture_content_pct": (
            str(moisture_content_pct)
            if moisture_content_pct is not None
            else None
        ),
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Grid Mix
# ============================================================================


def hash_grid_mix(
    grid_region: str,
    base_year: int,
    source: str,
    source_version: str,
    generation_mix: Dict[str, Decimal],
    total_generation_gwh: Optional[Decimal] = None,
    grid_ef_kgco2e_kwh: Optional[Decimal] = None,
    country_code: Optional[str] = None,
) -> str:
    """
    Hash a grid generation mix record for provenance tracking.

    Creates a deterministic SHA-256 digest of the electricity grid
    generation mix used for upstream electricity (3b) and T&D loss (3c)
    factor derivation.

    Args:
        grid_region: Grid region or balancing area identifier.
        base_year: Base year for the generation mix data.
        source: Data source (IEA/EPA/EIA/national_grid).
        source_version: Source version.
        generation_mix: Fuel source to percentage mapping.
        total_generation_gwh: Optional total generation (GWh).
        grid_ef_kgco2e_kwh: Optional derived grid EF.
        country_code: Optional ISO country code.

    Returns:
        64-character SHA-256 hex digest.
    """
    gen_mix_str = {k: str(v) for k, v in generation_mix.items()}

    data = {
        "grid_region": grid_region,
        "base_year": base_year,
        "source": source,
        "source_version": source_version,
        "generation_mix": gen_mix_str,
        "total_generation_gwh": (
            str(total_generation_gwh)
            if total_generation_gwh is not None
            else None
        ),
        "grid_ef_kgco2e_kwh": (
            str(grid_ef_kgco2e_kwh)
            if grid_ef_kgco2e_kwh is not None
            else None
        ),
        "country_code": country_code,
    }
    return _hash_helper(data)


# ============================================================================
# Provenance Query Utilities
# ============================================================================


def get_stage_hash(
    provenance: FuelEnergyActivitiesProvenance,
    stage: str,
) -> Optional[str]:
    """
    Get the chain hash for a specific stage.

    Args:
        provenance: FuelEnergyActivitiesProvenance instance.
        stage: Stage name to look up.

    Returns:
        Chain hash for the stage, or None if not found.
    """
    entry = provenance.get_stage(stage)
    return entry.chain_hash if entry else None


def get_stage_metadata(
    provenance: FuelEnergyActivitiesProvenance,
    stage: str,
) -> Optional[Dict[str, Any]]:
    """
    Get the metadata for a specific stage.

    Args:
        provenance: FuelEnergyActivitiesProvenance instance.
        stage: Stage name to look up.

    Returns:
        Metadata dictionary for the stage, or None if not found.
    """
    entry = provenance.get_stage(stage)
    return entry.metadata if entry else None


def get_chain_summary(
    provenance: FuelEnergyActivitiesProvenance,
) -> Dict[str, Any]:
    """
    Get a summary of the provenance chain state.

    Args:
        provenance: FuelEnergyActivitiesProvenance instance.

    Returns:
        Summary dictionary with chain metadata and stage progression.
    """
    chain_dict = provenance.to_dict()
    return {
        "agent_id": chain_dict.get("agent_id"),
        "version": chain_dict.get("version"),
        "chain_id": chain_dict.get("chain_id"),
        "organization_id": chain_dict.get("organization_id"),
        "reporting_period": chain_dict.get("reporting_period"),
        "created_at": chain_dict.get("created_at"),
        "entry_count": chain_dict.get("entry_count"),
        "stages_completed": provenance.get_stages_completed(),
        "sealed": chain_dict.get("sealed"),
        "seal_hash": chain_dict.get("seal_hash"),
        "seal_timestamp": chain_dict.get("seal_timestamp"),
        "current_chain_hash": chain_dict.get("current_chain_hash"),
        "integrity_verified": chain_dict.get("integrity_verified"),
    }


def verify_upstream_hash(
    provenance: FuelEnergyActivitiesProvenance,
    stage: str,
    expected_hash: str,
) -> bool:
    """
    Verify that a specific stage's chain hash matches an expected value.

    Useful for cross-agent provenance verification where a downstream
    agent needs to confirm the upstream hash before proceeding.

    Args:
        provenance: FuelEnergyActivitiesProvenance instance.
        stage: Stage name to verify.
        expected_hash: Expected chain hash value.

    Returns:
        True if the hashes match, False otherwise.
    """
    actual_hash = get_stage_hash(provenance, stage)
    if actual_hash is None:
        return False
    return actual_hash == expected_hash


def compare_chains(
    chain_a: FuelEnergyActivitiesProvenance,
    chain_b: FuelEnergyActivitiesProvenance,
) -> Dict[str, Any]:
    """
    Compare two provenance chains and report differences.

    Useful for regression testing, reproducibility verification,
    and audit comparison between calculation runs.

    Args:
        chain_a: First provenance chain.
        chain_b: Second provenance chain.

    Returns:
        Dictionary with comparison results including:
        - matching_stages: Stages with identical hashes
        - differing_stages: Stages with different hashes
        - only_in_a: Stages only in chain A
        - only_in_b: Stages only in chain B
        - chains_identical: Whether chains are fully identical
    """
    stages_a = {e.stage: e for e in chain_a.get_chain()}
    stages_b = {e.stage: e for e in chain_b.get_chain()}

    all_stages = set(stages_a.keys()) | set(stages_b.keys())

    matching: List[str] = []
    differing: List[Dict[str, str]] = []
    only_in_a: List[str] = []
    only_in_b: List[str] = []

    for stage in sorted(all_stages):
        if stage in stages_a and stage in stages_b:
            if stages_a[stage].chain_hash == stages_b[stage].chain_hash:
                matching.append(stage)
            else:
                differing.append({
                    "stage": stage,
                    "chain_a_hash": stages_a[stage].chain_hash,
                    "chain_b_hash": stages_b[stage].chain_hash,
                })
        elif stage in stages_a:
            only_in_a.append(stage)
        else:
            only_in_b.append(stage)

    return {
        "matching_stages": matching,
        "differing_stages": differing,
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "chains_identical": (
            len(differing) == 0
            and len(only_in_a) == 0
            and len(only_in_b) == 0
        ),
    }


# ============================================================================
# Convenience: Module-Level Access
# ============================================================================


def get_provenance() -> FuelEnergyActivitiesProvenance:
    """
    Get the singleton FuelEnergyActivitiesProvenance instance.

    Convenience function providing a clean module-level entry point
    for provenance tracking. Equivalent to calling
    FuelEnergyActivitiesProvenance.get_instance().

    Returns:
        The singleton FuelEnergyActivitiesProvenance instance.

    Example:
        >>> provenance = get_provenance()
        >>> provenance.start_chain(initial_data)
        >>> provenance.record_stage("VALIDATE", input_data, output_data)
        >>> provenance.seal_chain()
        >>> assert provenance.verify_chain()
    """
    return FuelEnergyActivitiesProvenance.get_instance()


# ============================================================================
# Backward Compatibility Alias
# ============================================================================

#: Alias used by the pipeline engine, setup.py, and __init__.py.
FuelEnergyActivitiesProvenanceTracker = FuelEnergyActivitiesProvenance
