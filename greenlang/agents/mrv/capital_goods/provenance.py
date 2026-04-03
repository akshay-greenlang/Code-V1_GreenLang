"""
Capital Goods Agent - Provenance Tracking Module

This module provides SHA-256 provenance chain tracking for GL-MRV-S3-002 (AGENT-MRV-015).
Thread-safe singleton pattern ensures audit trail integrity across all capital goods
emissions operations, including spend-based EEIO, average-data physical, and
supplier-specific EPD/PCF calculation methods.

Capital goods (GHG Protocol Scope 3, Category 2) differ from purchased goods and services
(Category 1) in that they represent assets that are capitalized on the balance sheet, with
emissions allocated to the reporting year of acquisition rather than amortized over the
asset's useful life. The provenance chain tracks 17 stages from initial asset intake
through pipeline seal, ensuring deterministic reproducibility of every calculation step.

Agent: GL-MRV-S3-002 (AGENT-MRV-015)
Purpose: Track capital asset emissions lineage across three calculation methods
Regulatory: GHG Protocol Scope 3, ISO 14064-1, CSRD E1, CDP Supply Chain, SBTi, PCAF, EU Taxonomy

Example:
    >>> provenance = get_provenance()
    >>> chain_hash = provenance.start_chain({"asset_id": "ASSET-001", "capex": 500000})
    >>> entry = provenance.record_stage(
    ...     stage="ASSET_INTAKE",
    ...     input_data={"asset_id": "ASSET-001"},
    ...     output_data={"validated": True, "asset_type": "machinery"},
    ...     parameters={"validation_mode": "strict"},
    ...     metadata={"source": "ERP", "timestamp": "2024-01-15T10:30:00Z"}
    ... )
    >>> seal_hash = provenance.seal_chain()
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

#: Agent identifier for GL-MRV-S3-002 Capital Goods
AGENT_ID: str = "GL-MRV-S3-002"

#: Module version
VERSION: str = "1.0.0"

#: SHA-256 digest size in hex characters
SHA256_HEX_LENGTH: int = 64

#: Genesis hash for initial chain entry (all zeros)
GENESIS_HASH: str = "0" * SHA256_HEX_LENGTH

#: Table prefix for capital goods provenance records
TABLE_PREFIX: str = "gl_cg_"


# ============================================================================
# Provenance Stage Enumeration
# ============================================================================


class ProvenanceStage(str, Enum):
    """
    17 provenance stages for capital goods emissions calculation.

    Each stage represents a discrete, auditable step in the GL-MRV-S3-002
    pipeline. Stages are ordered sequentially; the provenance chain enforces
    that each stage links cryptographically to the previous one.

    Stage ordering follows the GHG Protocol Technical Guidance for Scope 3
    Category 2 (Capital Goods) calculation workflow.
    """

    # Stage 1: Initial capital asset data ingestion from ERP/CMMS/FAR
    ASSET_INTAKE = "asset_intake"

    # Stage 2: Asset category/subcategory classification (PP&E taxonomy)
    ASSET_CLASSIFICATION = "asset_classification"

    # Stage 3: Verify asset meets capitalization threshold per GAAP/IFRS
    CAPITALIZATION_CHECK = "capitalization_check"

    # Stage 4: NAICS/NACE/UNSPSC sector classification for EEIO mapping
    SPEND_CLASSIFICATION = "spend_classification"

    # Stage 5: Foreign currency to reporting currency (USD) conversion
    CURRENCY_CONVERSION = "currency_conversion"

    # Stage 6: CPI-based year deflation to EEIO base year
    INFLATION_DEFLATION = "inflation_deflation"

    # Stage 7: Purchaser-to-producer price adjustment (margin removal)
    MARGIN_REMOVAL = "margin_removal"

    # Stage 8: Emission factor selection from hierarchical source priority
    EF_RESOLUTION = "ef_resolution"

    # Stage 9: Spend-based EEIO calculation (USEEIO/EXIOBASE/DEFRA)
    SPEND_BASED_CALC = "spend_based_calc"

    # Stage 10: Average-data physical quantity calculation (LCA databases)
    AVERAGE_DATA_CALC = "average_data_calc"

    # Stage 11: Supplier-specific EPD/PCF calculation (product-level data)
    SUPPLIER_SPECIFIC_CALC = "supplier_specific_calc"

    # Stage 12: Multi-method hybrid aggregation with priority weighting
    HYBRID_AGGREGATION = "hybrid_aggregation"

    # Stage 13: Verify no overlap with Category 1 / Scope 1 / Scope 2
    DOUBLE_COUNTING_CHECK = "double_counting_check"

    # Stage 14: Data quality indicator (DQI) assessment per GHG Protocol
    DQI_SCORING = "dqi_scoring"

    # Stage 15: Monte Carlo / analytical uncertainty quantification
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"

    # Stage 16: Seven-framework regulatory compliance check
    COMPLIANCE_CHECK = "compliance_check"

    # Stage 17: Final provenance chain seal (immutable after sealing)
    PIPELINE_SEAL = "pipeline_seal"


#: Ordered list of all stages for validation
STAGE_ORDER: List[ProvenanceStage] = list(ProvenanceStage)

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
        stage: Pipeline stage name (one of 17 ProvenanceStage values).
        timestamp: ISO 8601 timestamp of when the stage was recorded.
        input_hash: SHA-256 hex digest of the input data for this stage.
        output_hash: SHA-256 hex digest of the output data produced.
        parameters_hash: SHA-256 hex digest of parameters/configuration used.
        chain_hash: SHA-256 combining previous chain_hash with current hashes.
        agent_id: Agent identifier, always "GL-MRV-S3-002" for capital goods.
        version: Module version string (SemVer).
        metadata: Additional key-value metadata for audit trail enrichment.

    Example:
        >>> entry = ProvenanceEntry(
        ...     stage="ASSET_INTAKE",
        ...     timestamp="2024-01-15T10:30:00Z",
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
            return False, f"input_hash length invalid: {len(self.input_hash) if self.input_hash else 0}"
        if not self.output_hash or len(self.output_hash) != SHA256_HEX_LENGTH:
            return False, f"output_hash length invalid: {len(self.output_hash) if self.output_hash else 0}"
        if not self.parameters_hash or len(self.parameters_hash) != SHA256_HEX_LENGTH:
            return False, f"parameters_hash length invalid: {len(self.parameters_hash) if self.parameters_hash else 0}"
        if not self.chain_hash or len(self.chain_hash) != SHA256_HEX_LENGTH:
            return False, f"chain_hash length invalid: {len(self.chain_hash) if self.chain_hash else 0}"
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
        '{"a": 2, "b": "1.5"}'
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
            return {str(k): _convert(v) for k, v in sorted(o.items(), key=lambda x: str(x[0]))}
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
        >>> _compute_chain_hash(GENESIS_HASH, "ASSET_INTAKE", "aaa...", "bbb...", "ccc...")
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
# Capital Goods Provenance Tracker (Thread-Safe Singleton)
# ============================================================================


class CapitalGoodsProvenance:
    """
    Thread-safe singleton provenance tracker for capital goods emissions.

    Tracks 17-stage capital asset emissions lineage with SHA-256 chain integrity.
    Supports three calculation methods: spend-based EEIO, average-data physical,
    and supplier-specific EPD/PCF.

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
        1. Single calculation: start_chain -> record_stage (x17) -> seal_chain
        2. Batch processing: reset() between calculations
        3. Verification: verify_chain() after deserialization from storage

    Example:
        >>> provenance = CapitalGoodsProvenance.get_instance()
        >>> provenance.start_chain({"asset_id": "ASSET-001", "capex": 500000})
        '4f3e2d...'
        >>> provenance.record_stage(
        ...     stage="ASSET_INTAKE",
        ...     input_data={"asset_id": "ASSET-001"},
        ...     output_data={"validated": True},
        ... )
        ProvenanceEntry(stage='ASSET_INTAKE', ...)
        >>> provenance.seal_chain()
        'a1b2c3...'
        >>> provenance.verify_chain()
        True
    """

    _instance: Optional["CapitalGoodsProvenance"] = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """
        Initialize CapitalGoodsProvenance.

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
        logger.debug("CapitalGoodsProvenance instance initialized")

    @classmethod
    def get_instance(cls) -> "CapitalGoodsProvenance":
        """
        Get singleton instance with double-checked locking.

        Thread-safe lazy initialization ensures exactly one instance
        exists per process.

        Returns:
            The singleton CapitalGoodsProvenance instance.

        Example:
            >>> p1 = CapitalGoodsProvenance.get_instance()
            >>> p2 = CapitalGoodsProvenance.get_instance()
            >>> assert p1 is p2
        """
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.info("CapitalGoodsProvenance singleton created")
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
            logger.debug("CapitalGoodsProvenance singleton reset")

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
            initial_data: Initial input data to hash (e.g., raw asset records).
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
            ...     initial_data={"assets": [{"id": "A001", "capex": 100000}]},
            ...     chain_id="CALC-2024-001",
            ...     organization_id="ORG-123",
            ...     reporting_period="2024",
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
                "Provenance chain started: chain_id=%s, org=%s, period=%s, initial_hash=%s",
                chain_id,
                organization_id,
                reporting_period,
                initial_hash[:16] + "...",
            )

            return initial_hash

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

        Args:
            stage: Stage name (should match a ProvenanceStage value).
            input_data: Input data consumed by this stage.
            output_data: Output data produced by this stage.
            parameters: Optional parameters/configuration used in this stage.
            metadata: Optional metadata for audit enrichment (source, timing, etc.).

        Returns:
            The ProvenanceEntry created for this stage.

        Raises:
            ValueError: If the chain is sealed or stage name is empty.
            RuntimeError: If chain has not been started via start_chain().

        Example:
            >>> entry = provenance.record_stage(
            ...     stage="ASSET_INTAKE",
            ...     input_data=raw_records,
            ...     output_data=validated_records,
            ...     parameters={"validation_mode": "strict", "threshold": 5000},
            ...     metadata={"source_system": "SAP", "record_count": 150},
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
            parameters_hash = _hash_data(parameters if parameters is not None else {})

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
                "entry_chain_hashes": [entry.chain_hash for entry in self._chain],
                "stages_completed": [entry.stage for entry in self._chain],
                "final_chain_hash": self._current_chain_hash,
            }

            seal_hash = _hash_data(seal_data)
            self._sealed = True
            self._seal_hash = seal_hash
            self._seal_timestamp = datetime.utcnow().isoformat() + "Z"

            logger.info(
                "Provenance chain sealed: chain_id=%s, entries=%d, seal_hash=%s",
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
            >>> entry = provenance.get_stage("SPEND_BASED_CALC")
            >>> if entry:
            ...     print(f"Emissions hash: {entry.output_hash}")
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

    def get_current_chain_hash(self) -> str:
        """
        Get the current running chain hash.

        Returns:
            The latest chain hash (GENESIS_HASH if no entries recorded).
        """
        with self._lock:
            return self._current_chain_hash

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
            logger.error("Chain integrity failure: %s", error)
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
                return False, f"Entry {i} ({entry.stage}) structural validation failed: {error}"

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
                "entry_chain_hashes": [entry.chain_hash for entry in self._chain],
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
    def from_dict(cls, data: Dict[str, Any]) -> "CapitalGoodsProvenance":
        """
        Deserialize a provenance chain from a dictionary.

        Creates a new CapitalGoodsProvenance instance (not the singleton)
        populated with the serialized chain data. Use this for loading
        provenance records from storage or API responses.

        Args:
            data: Dictionary containing serialized provenance chain.

        Returns:
            New CapitalGoodsProvenance instance with restored chain state.

        Raises:
            KeyError: If required fields are missing from the data.
            ValueError: If deserialized chain fails integrity verification.

        Example:
            >>> stored_data = db.load_provenance("CALC-2024-001")
            >>> provenance = CapitalGoodsProvenance.from_dict(stored_data)
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
        instance._current_chain_hash = data.get("current_chain_hash", GENESIS_HASH)

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


class CapitalGoodsBatchProvenance:
    """
    Track provenance for multi-period or multi-asset batch calculations.

    Manages per-calculation provenance chains with batch-level aggregation
    and cross-chain verification. Useful for processing multiple reporting
    periods or asset portfolios in a single pipeline run.

    Thread Safety:
        Uses a dedicated lock for batch-level operations. Individual chain
        operations are protected by the CapitalGoodsProvenance instance locks.

    Example:
        >>> batch = CapitalGoodsBatchProvenance()
        >>> batch_id = batch.create_batch("BATCH-2024-Q1")
        >>> chain = CapitalGoodsProvenance()
        >>> chain.start_chain(asset_data)
        >>> # ... record stages ...
        >>> chain.seal_chain()
        >>> batch.add_chain(batch_id, "CALC-001", chain)
        >>> batch_hash = batch.seal_batch(batch_id)
    """

    def __init__(self) -> None:
        """Initialize batch provenance tracker."""
        self._batches: Dict[str, Dict[str, Any]] = {}
        self._batch_lock: threading.Lock = threading.Lock()
        logger.debug("CapitalGoodsBatchProvenance initialized")

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
        provenance: "CapitalGoodsProvenance",
    ) -> None:
        """
        Add a completed provenance chain to the batch.

        Args:
            batch_id: Batch identifier.
            chain_id: Unique chain identifier within the batch.
            provenance: The CapitalGoodsProvenance instance to add.

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
                chain_seal_hashes[cid] = chain_data.get("seal_hash", "UNSEALED")

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
                chain = CapitalGoodsProvenance.from_dict(chain_data)
                is_valid, error = chain.verify_chain_detailed()
                if not is_valid:
                    return False, f"Chain {cid} in batch {batch_id}: {error}"

            # Verify batch seal
            if batch["sealed"] and batch["seal_hash"]:
                chain_seal_hashes = {}
                for cid in sorted(batch["chains"].keys()):
                    chain_data = batch["chains"][cid]
                    chain_seal_hashes[cid] = chain_data.get("seal_hash", "UNSEALED")

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
                    "integrity_verified": chain_data.get("integrity_verified", False),
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
# Standalone Hash Helper Functions - Asset Records
# ============================================================================


def hash_asset_record(
    asset_id: str,
    asset_name: str,
    asset_type: str,
    acquisition_date: str,
    acquisition_cost: Decimal,
    currency: str,
    useful_life_years: Optional[int] = None,
    supplier_id: Optional[str] = None,
    facility_id: Optional[str] = None,
) -> str:
    """
    Hash a capital asset record for provenance tracking.

    Creates a deterministic SHA-256 digest of the asset's identifying
    attributes and financial data. Used in the ASSET_INTAKE stage.

    Args:
        asset_id: Unique asset identifier from fixed asset register.
        asset_name: Human-readable asset name/description.
        asset_type: Asset type classification (e.g., "machinery", "vehicle").
        acquisition_date: Acquisition date (ISO 8601, YYYY-MM-DD).
        acquisition_cost: Total acquisition cost (CapEx) in original currency.
        currency: Currency code (ISO 4217, e.g., "USD", "EUR").
        useful_life_years: Optional expected useful life in years.
        supplier_id: Optional supplier/vendor identifier.
        facility_id: Optional facility/site identifier.

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_asset_record(
        ...     asset_id="FA-2024-001",
        ...     asset_name="CNC Milling Machine",
        ...     asset_type="machinery",
        ...     acquisition_date="2024-03-15",
        ...     acquisition_cost=Decimal("250000.00"),
        ...     currency="USD",
        ...     useful_life_years=10,
        ... )
    """
    data = {
        "asset_id": asset_id,
        "asset_name": asset_name,
        "asset_type": asset_type,
        "acquisition_date": acquisition_date,
        "acquisition_cost": str(acquisition_cost),
        "currency": currency,
        "useful_life_years": useful_life_years,
        "supplier_id": supplier_id,
        "facility_id": facility_id,
    }
    return _hash_helper(data)


def hash_capex_spend(
    asset_id: str,
    spend_amount: Decimal,
    currency: str,
    spend_category: str,
    fiscal_year: int,
    capitalization_flag: bool,
    depreciation_method: Optional[str] = None,
    salvage_value: Optional[Decimal] = None,
) -> str:
    """
    Hash a capital expenditure spend record.

    Creates a deterministic SHA-256 digest of the CapEx spend data
    used for spend-based EEIO calculations. Used in the
    CAPITALIZATION_CHECK and SPEND_CLASSIFICATION stages.

    Args:
        asset_id: Asset identifier.
        spend_amount: Capital expenditure amount.
        currency: Currency code (ISO 4217).
        spend_category: CapEx spend category.
        fiscal_year: Fiscal year of expenditure.
        capitalization_flag: Whether the spend meets capitalization threshold.
        depreciation_method: Optional depreciation method (straight-line/declining).
        salvage_value: Optional estimated salvage value.

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "asset_id": asset_id,
        "spend_amount": str(spend_amount),
        "currency": currency,
        "spend_category": spend_category,
        "fiscal_year": fiscal_year,
        "capitalization_flag": capitalization_flag,
        "depreciation_method": depreciation_method,
        "salvage_value": str(salvage_value) if salvage_value is not None else None,
    }
    return _hash_helper(data)


def hash_physical_record(
    asset_id: str,
    material_type: str,
    quantity: Decimal,
    unit: str,
    weight_kg: Optional[Decimal] = None,
    product_code: Optional[str] = None,
    region: Optional[str] = None,
) -> str:
    """
    Hash a physical activity data record for average-data calculation.

    Creates a deterministic SHA-256 digest of the physical attributes
    used for average-data calculation with LCA emission factors.

    Args:
        asset_id: Asset identifier.
        material_type: Material/product type classification.
        quantity: Physical quantity (e.g., number of units, mass).
        unit: Physical unit (e.g., "kg", "tonne", "unit", "m2").
        weight_kg: Optional weight in kilograms.
        product_code: Optional product classification code.
        region: Optional geographic region of manufacture.

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_physical_record(
        ...     asset_id="FA-2024-001",
        ...     material_type="steel_machinery",
        ...     quantity=Decimal("15000"),
        ...     unit="kg",
        ...     weight_kg=Decimal("15000"),
        ...     product_code="PRD-MACH-001",
        ...     region="EU",
        ... )
    """
    data = {
        "asset_id": asset_id,
        "material_type": material_type,
        "quantity": str(quantity),
        "unit": unit,
        "weight_kg": str(weight_kg) if weight_kg is not None else None,
        "product_code": product_code,
        "region": region,
    }
    return _hash_helper(data)


def hash_supplier_record(
    supplier_id: str,
    supplier_name: str,
    asset_id: str,
    total_emissions_tco2e: Decimal,
    allocation_method: str,
    allocation_factor: Decimal,
    data_source: str,
    data_year: int,
    epd_reference: Optional[str] = None,
    pcf_reference: Optional[str] = None,
) -> str:
    """
    Hash a supplier-specific emissions record.

    Creates a deterministic SHA-256 digest of supplier-provided
    emissions data including EPD/PCF references. Used in the
    SUPPLIER_SPECIFIC_CALC stage.

    Args:
        supplier_id: Supplier/vendor identifier.
        supplier_name: Supplier legal name.
        asset_id: Asset identifier this data relates to.
        total_emissions_tco2e: Total supplier-reported emissions (tCO2e).
        allocation_method: Allocation method (revenue/mass/economic/physical).
        allocation_factor: Allocation factor (0-1).
        data_source: Data source type (EPD/PCF/CDP/direct_report).
        data_year: Year of the supplier data.
        epd_reference: Optional Environmental Product Declaration reference.
        pcf_reference: Optional Product Carbon Footprint reference.

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "supplier_id": supplier_id,
        "supplier_name": supplier_name,
        "asset_id": asset_id,
        "total_emissions_tco2e": str(total_emissions_tco2e),
        "allocation_method": allocation_method,
        "allocation_factor": str(allocation_factor),
        "data_source": data_source,
        "data_year": data_year,
        "epd_reference": epd_reference,
        "pcf_reference": pcf_reference,
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Emission Factors
# ============================================================================


def hash_eeio_factor(
    sector_code: str,
    classification_system: str,
    emission_factor: Decimal,
    unit: str,
    source: str,
    source_version: str,
    region: str,
    base_year: int,
    ghg_scope: Optional[str] = None,
) -> str:
    """
    Hash an EEIO (Environmentally Extended Input-Output) emission factor.

    Creates a deterministic SHA-256 digest of the EEIO factor used in
    spend-based calculations. Supports USEEIO, EXIOBASE, and DEFRA sources.

    Args:
        sector_code: Sector classification code (NAICS/NACE/UNSPSC).
        classification_system: Classification system name.
        emission_factor: Emission factor value (kgCO2e per currency unit).
        unit: Factor unit (e.g., "kgCO2e/USD", "kgCO2e/EUR").
        source: Data source (USEEIO/EXIOBASE/DEFRA).
        source_version: Source version string (e.g., "v2.0.1").
        region: Geographic region (e.g., "US", "EU", "UK").
        base_year: Base year for the EEIO model.
        ghg_scope: Optional GHG scope coverage (e.g., "upstream", "cradle-to-gate").

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_eeio_factor(
        ...     sector_code="333249",
        ...     classification_system="NAICS",
        ...     emission_factor=Decimal("0.452"),
        ...     unit="kgCO2e/USD",
        ...     source="USEEIO",
        ...     source_version="v2.0.1",
        ...     region="US",
        ...     base_year=2019,
        ... )
    """
    data = {
        "sector_code": sector_code,
        "classification_system": classification_system,
        "emission_factor": str(emission_factor),
        "unit": unit,
        "source": source,
        "source_version": source_version,
        "region": region,
        "base_year": base_year,
        "ghg_scope": ghg_scope,
    }
    return _hash_helper(data)


def hash_physical_ef(
    product_code: str,
    emission_factor: Decimal,
    unit: str,
    scope_coverage: str,
    source: str,
    source_version: str,
    region: str,
    lca_methodology: Optional[str] = None,
    system_boundary: Optional[str] = None,
) -> str:
    """
    Hash a physical (average-data) emission factor.

    Creates a deterministic SHA-256 digest of an LCA-based emission factor
    used for average-data calculations. Supports GaBi, Ecoinvent, DEFRA,
    and other LCA databases.

    Args:
        product_code: Product classification code.
        emission_factor: Emission factor (kgCO2e per physical unit).
        unit: Physical unit (e.g., "kgCO2e/kg", "kgCO2e/unit").
        scope_coverage: Scope coverage (upstream/cradle-to-gate/cradle-to-grave).
        source: Data source (GaBi/Ecoinvent/DEFRA/EPA).
        source_version: Source version string.
        region: Geographic region.
        lca_methodology: Optional LCA methodology (attributional/consequential).
        system_boundary: Optional system boundary description.

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "product_code": product_code,
        "emission_factor": str(emission_factor),
        "unit": unit,
        "scope_coverage": scope_coverage,
        "source": source,
        "source_version": source_version,
        "region": region,
        "lca_methodology": lca_methodology,
        "system_boundary": system_boundary,
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Calculation Results
# ============================================================================


def hash_calculation_result(
    calculation_id: str,
    asset_id: str,
    method: str,
    total_emissions_tco2e: Decimal,
    co2_tonnes: Decimal,
    ch4_tco2e: Decimal,
    n2o_tco2e: Decimal,
    hfc_tco2e: Optional[Decimal] = None,
    pfc_tco2e: Optional[Decimal] = None,
    sf6_tco2e: Optional[Decimal] = None,
    biogenic_tco2e: Optional[Decimal] = None,
) -> str:
    """
    Hash a capital goods emissions calculation result.

    Creates a deterministic SHA-256 digest of the final per-asset
    emissions calculation output. Tracks all GHG species separately.

    Args:
        calculation_id: Unique calculation run identifier.
        asset_id: Asset identifier.
        method: Calculation method used (spend_based/average_data/supplier_specific).
        total_emissions_tco2e: Total emissions in tonnes CO2 equivalent.
        co2_tonnes: CO2 emissions in tonnes.
        ch4_tco2e: CH4 emissions in tonnes CO2 equivalent.
        n2o_tco2e: N2O emissions in tonnes CO2 equivalent.
        hfc_tco2e: Optional HFC emissions in tonnes CO2 equivalent.
        pfc_tco2e: Optional PFC emissions in tonnes CO2 equivalent.
        sf6_tco2e: Optional SF6 emissions in tonnes CO2 equivalent.
        biogenic_tco2e: Optional biogenic CO2 in tonnes CO2 equivalent.

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_calculation_result(
        ...     calculation_id="CALC-2024-001",
        ...     asset_id="FA-2024-001",
        ...     method="spend_based",
        ...     total_emissions_tco2e=Decimal("125.7"),
        ...     co2_tonnes=Decimal("120.3"),
        ...     ch4_tco2e=Decimal("3.2"),
        ...     n2o_tco2e=Decimal("2.2"),
        ... )
    """
    data = {
        "calculation_id": calculation_id,
        "asset_id": asset_id,
        "method": method,
        "total_emissions_tco2e": str(total_emissions_tco2e),
        "co2_tonnes": str(co2_tonnes),
        "ch4_tco2e": str(ch4_tco2e),
        "n2o_tco2e": str(n2o_tco2e),
        "hfc_tco2e": str(hfc_tco2e) if hfc_tco2e is not None else None,
        "pfc_tco2e": str(pfc_tco2e) if pfc_tco2e is not None else None,
        "sf6_tco2e": str(sf6_tco2e) if sf6_tco2e is not None else None,
        "biogenic_tco2e": str(biogenic_tco2e) if biogenic_tco2e is not None else None,
    }
    return _hash_helper(data)


def hash_compliance_result(
    framework: str,
    status: str,
    requirements_met: int,
    requirements_total: int,
    findings: List[str],
    non_conformities: Optional[List[str]] = None,
    recommendations: Optional[List[str]] = None,
) -> str:
    """
    Hash a regulatory compliance check result.

    Creates a deterministic SHA-256 digest of the compliance assessment
    for one of seven supported regulatory frameworks.

    Args:
        framework: Framework name (GHG_PROTOCOL/ISO_14064/CSRD/CDP/SBTi/PCAF/EU_TAXONOMY).
        status: Compliance status (COMPLIANT/NON_COMPLIANT/PARTIAL/NOT_APPLICABLE).
        requirements_met: Number of requirements satisfied.
        requirements_total: Total number of applicable requirements.
        findings: List of compliance findings/observations.
        non_conformities: Optional list of non-conformity descriptions.
        recommendations: Optional list of improvement recommendations.

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_compliance_result(
        ...     framework="GHG_PROTOCOL",
        ...     status="COMPLIANT",
        ...     requirements_met=12,
        ...     requirements_total=12,
        ...     findings=["All Category 2 requirements satisfied"],
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
    }
    return _hash_helper(data)


def hash_aggregation(
    aggregation_id: str,
    method: str,
    asset_count: int,
    total_emissions_tco2e: Decimal,
    spend_based_emissions: Decimal,
    average_data_emissions: Decimal,
    supplier_specific_emissions: Decimal,
    coverage_percentage: Decimal,
    weighted_dqi_score: Decimal,
) -> str:
    """
    Hash a hybrid aggregation result across multiple calculation methods.

    Creates a deterministic SHA-256 digest of the multi-method aggregation
    output from the HYBRID_AGGREGATION stage.

    Args:
        aggregation_id: Aggregation run identifier.
        method: Aggregation method (priority_hierarchy/weighted_average/highest_dqi).
        asset_count: Number of assets included in aggregation.
        total_emissions_tco2e: Total aggregated emissions (tCO2e).
        spend_based_emissions: Emissions from spend-based method (tCO2e).
        average_data_emissions: Emissions from average-data method (tCO2e).
        supplier_specific_emissions: Emissions from supplier-specific method (tCO2e).
        coverage_percentage: Percentage of total CapEx covered (0-100).
        weighted_dqi_score: Weighted average data quality score (1-5).

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "aggregation_id": aggregation_id,
        "method": method,
        "asset_count": asset_count,
        "total_emissions_tco2e": str(total_emissions_tco2e),
        "spend_based_emissions": str(spend_based_emissions),
        "average_data_emissions": str(average_data_emissions),
        "supplier_specific_emissions": str(supplier_specific_emissions),
        "coverage_percentage": str(coverage_percentage),
        "weighted_dqi_score": str(weighted_dqi_score),
    }
    return _hash_helper(data)


def hash_dqi_assessment(
    asset_id: str,
    method: str,
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
    scoring per GHG Protocol Technical Guidance. Used in the DQI_SCORING
    stage.

    Args:
        asset_id: Asset identifier.
        method: Calculation method assessed.
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
        ...     asset_id="FA-2024-001",
        ...     method="supplier_specific",
        ...     technology_score=2,
        ...     temporal_score=1,
        ...     geography_score=2,
        ...     completeness_score=1,
        ...     reliability_score=1,
        ...     composite_score=Decimal("1.4"),
        ... )
    """
    data = {
        "asset_id": asset_id,
        "method": method,
        "technology_score": technology_score,
        "temporal_score": temporal_score,
        "geography_score": geography_score,
        "completeness_score": completeness_score,
        "reliability_score": reliability_score,
        "composite_score": str(composite_score),
        "assessment_notes": assessment_notes,
    }
    return _hash_helper(data)


def hash_depreciation_context(
    asset_id: str,
    acquisition_cost: Decimal,
    salvage_value: Decimal,
    useful_life_years: int,
    depreciation_method: str,
    accumulated_depreciation: Decimal,
    net_book_value: Decimal,
    reporting_year: int,
    capitalization_threshold: Decimal,
) -> str:
    """
    Hash the depreciation context for a capital asset.

    Creates a deterministic SHA-256 digest of the asset's financial
    depreciation attributes. Note: per GHG Protocol, capital goods
    emissions are allocated to the year of acquisition, not amortized.
    This hash captures the financial context for audit completeness.

    Args:
        asset_id: Asset identifier.
        acquisition_cost: Original acquisition cost.
        salvage_value: Estimated salvage/residual value.
        useful_life_years: Useful life in years.
        depreciation_method: Method (straight_line/declining_balance/units_of_production).
        accumulated_depreciation: Accumulated depreciation to date.
        net_book_value: Net book value (acquisition_cost - accumulated_depreciation).
        reporting_year: Current reporting year.
        capitalization_threshold: Organization's capitalization threshold.

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_depreciation_context(
        ...     asset_id="FA-2024-001",
        ...     acquisition_cost=Decimal("250000"),
        ...     salvage_value=Decimal("25000"),
        ...     useful_life_years=10,
        ...     depreciation_method="straight_line",
        ...     accumulated_depreciation=Decimal("0"),
        ...     net_book_value=Decimal("250000"),
        ...     reporting_year=2024,
        ...     capitalization_threshold=Decimal("5000"),
        ... )
    """
    data = {
        "asset_id": asset_id,
        "acquisition_cost": str(acquisition_cost),
        "salvage_value": str(salvage_value),
        "useful_life_years": useful_life_years,
        "depreciation_method": depreciation_method,
        "accumulated_depreciation": str(accumulated_depreciation),
        "net_book_value": str(net_book_value),
        "reporting_year": reporting_year,
        "capitalization_threshold": str(capitalization_threshold),
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Classification & Conversion
# ============================================================================


def hash_asset_classification(
    asset_id: str,
    asset_category: str,
    asset_subcategory: str,
    ppe_class: str,
    classification_method: str,
    confidence_score: Decimal,
) -> str:
    """
    Hash an asset classification result.

    Creates a deterministic SHA-256 digest of the PP&E (Property, Plant &
    Equipment) classification output from the ASSET_CLASSIFICATION stage.

    Args:
        asset_id: Asset identifier.
        asset_category: Primary category (e.g., "buildings", "machinery", "vehicles").
        asset_subcategory: Subcategory (e.g., "production_machinery", "office_equipment").
        ppe_class: PP&E class per GAAP/IFRS taxonomy.
        classification_method: Method used (rule_based/ml_classifier/manual).
        confidence_score: Classification confidence (0-1).

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "asset_id": asset_id,
        "asset_category": asset_category,
        "asset_subcategory": asset_subcategory,
        "ppe_class": ppe_class,
        "classification_method": classification_method,
        "confidence_score": str(confidence_score),
    }
    return _hash_helper(data)


def hash_spend_classification(
    asset_id: str,
    naics_code: Optional[str],
    nace_code: Optional[str],
    unspsc_code: Optional[str],
    primary_system: str,
    classification_method: str,
    confidence_score: Decimal,
) -> str:
    """
    Hash a spend classification (NAICS/NACE/UNSPSC) result.

    Creates a deterministic SHA-256 digest of the sector classification
    output from the SPEND_CLASSIFICATION stage.

    Args:
        asset_id: Asset identifier.
        naics_code: NAICS code (6-digit, US/Canada).
        nace_code: NACE code (EU statistical classification).
        unspsc_code: UNSPSC code (UN commodity classification).
        primary_system: Primary classification system used for EF lookup.
        classification_method: Method used (rule_based/ml_classifier/lookup).
        confidence_score: Classification confidence (0-1).

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "asset_id": asset_id,
        "naics_code": naics_code,
        "nace_code": nace_code,
        "unspsc_code": unspsc_code,
        "primary_system": primary_system,
        "classification_method": classification_method,
        "confidence_score": str(confidence_score),
    }
    return _hash_helper(data)


def hash_currency_conversion(
    asset_id: str,
    original_amount: Decimal,
    original_currency: str,
    converted_amount: Decimal,
    target_currency: str,
    exchange_rate: Decimal,
    rate_date: str,
    rate_source: str,
) -> str:
    """
    Hash a currency conversion record.

    Creates a deterministic SHA-256 digest of the foreign currency to
    reporting currency conversion from the CURRENCY_CONVERSION stage.

    Args:
        asset_id: Asset identifier.
        original_amount: Amount in original currency.
        original_currency: Original currency code (ISO 4217).
        converted_amount: Amount in target currency.
        target_currency: Target currency code (ISO 4217, typically "USD").
        exchange_rate: Exchange rate applied.
        rate_date: Date of the exchange rate (YYYY-MM-DD).
        rate_source: Rate source (ECB/FederalReserve/Bloomberg).

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "asset_id": asset_id,
        "original_amount": str(original_amount),
        "original_currency": original_currency,
        "converted_amount": str(converted_amount),
        "target_currency": target_currency,
        "exchange_rate": str(exchange_rate),
        "rate_date": rate_date,
        "rate_source": rate_source,
    }
    return _hash_helper(data)


def hash_inflation_deflation(
    asset_id: str,
    nominal_amount: Decimal,
    deflated_amount: Decimal,
    data_year: int,
    base_year: int,
    cpi_data_year: Decimal,
    cpi_base_year: Decimal,
    deflation_factor: Decimal,
    cpi_source: str,
) -> str:
    """
    Hash a CPI-based inflation/deflation adjustment.

    Creates a deterministic SHA-256 digest of the year deflation from
    the INFLATION_DEFLATION stage.

    Args:
        asset_id: Asset identifier.
        nominal_amount: Nominal (current-year) spend amount.
        deflated_amount: Deflated (base-year) spend amount.
        data_year: Year of the spend data.
        base_year: EEIO model base year.
        cpi_data_year: CPI index for data year.
        cpi_base_year: CPI index for base year.
        deflation_factor: Deflation factor (cpi_base / cpi_data).
        cpi_source: CPI data source (BLS/OECD/Eurostat).

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "asset_id": asset_id,
        "nominal_amount": str(nominal_amount),
        "deflated_amount": str(deflated_amount),
        "data_year": data_year,
        "base_year": base_year,
        "cpi_data_year": str(cpi_data_year),
        "cpi_base_year": str(cpi_base_year),
        "deflation_factor": str(deflation_factor),
        "cpi_source": cpi_source,
    }
    return _hash_helper(data)


def hash_margin_removal(
    asset_id: str,
    purchaser_price: Decimal,
    producer_price: Decimal,
    margin_percentage: Decimal,
    margin_factor: Decimal,
    sector_code: str,
    source: str,
    source_year: int,
) -> str:
    """
    Hash a purchaser-to-producer price margin removal.

    Creates a deterministic SHA-256 digest of the margin removal
    adjustment from the MARGIN_REMOVAL stage.

    Args:
        asset_id: Asset identifier.
        purchaser_price: Original purchaser price.
        producer_price: Adjusted producer price (net of margin).
        margin_percentage: Margin percentage (0-100).
        margin_factor: Margin factor applied (1 - margin_percentage/100).
        sector_code: Sector code for margin lookup.
        source: Margin data source (BEA/OECD/national_stats).
        source_year: Year of the margin data.

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "asset_id": asset_id,
        "purchaser_price": str(purchaser_price),
        "producer_price": str(producer_price),
        "margin_percentage": str(margin_percentage),
        "margin_factor": str(margin_factor),
        "sector_code": sector_code,
        "source": source,
        "source_year": source_year,
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - Double Counting & Uncertainty
# ============================================================================


def hash_double_counting_check(
    asset_id: str,
    cat1_overlap_detected: bool,
    scope1_overlap_detected: bool,
    scope2_overlap_detected: bool,
    overlap_amount_tco2e: Decimal,
    adjustment_applied: Decimal,
    check_method: str,
    overlapping_items: List[str],
) -> str:
    """
    Hash a double-counting check result.

    Creates a deterministic SHA-256 digest of the DOUBLE_COUNTING_CHECK
    stage output. Ensures capital goods emissions do not overlap with
    Category 1 (Purchased Goods & Services) or Scope 1/2 reported
    emissions.

    Args:
        asset_id: Asset identifier.
        cat1_overlap_detected: Whether overlap with Category 1 was found.
        scope1_overlap_detected: Whether overlap with Scope 1 was found.
        scope2_overlap_detected: Whether overlap with Scope 2 was found.
        overlap_amount_tco2e: Total identified overlap (tCO2e).
        adjustment_applied: Deduction applied to avoid double counting (tCO2e).
        check_method: Verification method (id_matching/spend_matching/manual).
        overlapping_items: List of overlapping item identifiers.

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "asset_id": asset_id,
        "cat1_overlap_detected": cat1_overlap_detected,
        "scope1_overlap_detected": scope1_overlap_detected,
        "scope2_overlap_detected": scope2_overlap_detected,
        "overlap_amount_tco2e": str(overlap_amount_tco2e),
        "adjustment_applied": str(adjustment_applied),
        "check_method": check_method,
        "overlapping_items": overlapping_items,
    }
    return _hash_helper(data)


def hash_uncertainty_quantification(
    asset_id: str,
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

    Creates a deterministic SHA-256 digest of the UNCERTAINTY_QUANTIFICATION
    stage output, supporting both Monte Carlo and analytical methods.

    Args:
        asset_id: Asset identifier.
        method: Uncertainty method (monte_carlo/analytical/error_propagation).
        emissions_mean: Mean emissions estimate (tCO2e).
        emissions_std_dev: Standard deviation (tCO2e).
        ci_lower_95: 95% confidence interval lower bound (tCO2e).
        ci_upper_95: 95% confidence interval upper bound (tCO2e).
        cv_percent: Coefficient of variation (%).
        monte_carlo_iterations: Optional number of MC iterations.
        uncertainty_sources: Optional ordered list of uncertainty sources.
        sensitivity_ranking: Optional sensitivity ranking of input parameters.

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_uncertainty_quantification(
        ...     asset_id="FA-2024-001",
        ...     method="monte_carlo",
        ...     emissions_mean=Decimal("125.7"),
        ...     emissions_std_dev=Decimal("18.5"),
        ...     ci_lower_95=Decimal("95.2"),
        ...     ci_upper_95=Decimal("162.1"),
        ...     cv_percent=Decimal("14.7"),
        ...     monte_carlo_iterations=10000,
        ...     uncertainty_sources=["emission_factor", "spend_data", "classification"],
        ... )
    """
    data = {
        "asset_id": asset_id,
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
# Standalone Hash Helper Functions - Spend-Based Calculation
# ============================================================================


def hash_spend_based_result(
    asset_id: str,
    adjusted_spend: Decimal,
    adjusted_currency: str,
    sector_code: str,
    eeio_source: str,
    eeio_factor: Decimal,
    eeio_unit: str,
    total_emissions_tco2e: Decimal,
    co2_tonnes: Decimal,
    ch4_tco2e: Decimal,
    n2o_tco2e: Decimal,
) -> str:
    """
    Hash a spend-based EEIO calculation result.

    Creates a deterministic SHA-256 digest of the SPEND_BASED_CALC
    stage output.

    Args:
        asset_id: Asset identifier.
        adjusted_spend: Spend after FX, CPI deflation, and margin removal.
        adjusted_currency: Currency of adjusted spend (base currency).
        sector_code: EEIO sector code applied.
        eeio_source: EEIO database source.
        eeio_factor: EEIO emission factor applied.
        eeio_unit: Unit of the EEIO factor.
        total_emissions_tco2e: Total calculated emissions (tCO2e).
        co2_tonnes: CO2 component (tonnes).
        ch4_tco2e: CH4 component (tCO2e).
        n2o_tco2e: N2O component (tCO2e).

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "asset_id": asset_id,
        "adjusted_spend": str(adjusted_spend),
        "adjusted_currency": adjusted_currency,
        "sector_code": sector_code,
        "eeio_source": eeio_source,
        "eeio_factor": str(eeio_factor),
        "eeio_unit": eeio_unit,
        "total_emissions_tco2e": str(total_emissions_tco2e),
        "co2_tonnes": str(co2_tonnes),
        "ch4_tco2e": str(ch4_tco2e),
        "n2o_tco2e": str(n2o_tco2e),
    }
    return _hash_helper(data)


def hash_average_data_result(
    asset_id: str,
    material_type: str,
    quantity: Decimal,
    unit: str,
    ef_source: str,
    ef_value: Decimal,
    ef_unit: str,
    total_emissions_tco2e: Decimal,
    co2_tonnes: Decimal,
    ch4_tco2e: Decimal,
    n2o_tco2e: Decimal,
) -> str:
    """
    Hash an average-data physical calculation result.

    Creates a deterministic SHA-256 digest of the AVERAGE_DATA_CALC
    stage output.

    Args:
        asset_id: Asset identifier.
        material_type: Material/product type used for EF lookup.
        quantity: Physical quantity.
        unit: Physical unit.
        ef_source: Emission factor source database.
        ef_value: Physical emission factor value.
        ef_unit: Emission factor unit.
        total_emissions_tco2e: Total calculated emissions (tCO2e).
        co2_tonnes: CO2 component (tonnes).
        ch4_tco2e: CH4 component (tCO2e).
        n2o_tco2e: N2O component (tCO2e).

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "asset_id": asset_id,
        "material_type": material_type,
        "quantity": str(quantity),
        "unit": unit,
        "ef_source": ef_source,
        "ef_value": str(ef_value),
        "ef_unit": ef_unit,
        "total_emissions_tco2e": str(total_emissions_tco2e),
        "co2_tonnes": str(co2_tonnes),
        "ch4_tco2e": str(ch4_tco2e),
        "n2o_tco2e": str(n2o_tco2e),
    }
    return _hash_helper(data)


def hash_supplier_specific_result(
    asset_id: str,
    supplier_id: str,
    data_source: str,
    reported_emissions_tco2e: Decimal,
    allocation_method: str,
    allocation_factor: Decimal,
    allocated_emissions_tco2e: Decimal,
    co2_tonnes: Decimal,
    ch4_tco2e: Decimal,
    n2o_tco2e: Decimal,
    epd_number: Optional[str] = None,
    pcf_reference: Optional[str] = None,
    verification_status: Optional[str] = None,
) -> str:
    """
    Hash a supplier-specific EPD/PCF calculation result.

    Creates a deterministic SHA-256 digest of the SUPPLIER_SPECIFIC_CALC
    stage output. Includes EPD/PCF provenance references.

    Args:
        asset_id: Asset identifier.
        supplier_id: Supplier identifier.
        data_source: Data source type (EPD/PCF/CDP/direct_measurement).
        reported_emissions_tco2e: Total reported supplier emissions (tCO2e).
        allocation_method: Allocation method (revenue/physical/economic).
        allocation_factor: Allocation factor (0-1).
        allocated_emissions_tco2e: Allocated emissions to this asset (tCO2e).
        co2_tonnes: CO2 component (tonnes).
        ch4_tco2e: CH4 component (tCO2e).
        n2o_tco2e: N2O component (tCO2e).
        epd_number: Optional EPD registration number.
        pcf_reference: Optional PCF study reference.
        verification_status: Optional third-party verification status.

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "asset_id": asset_id,
        "supplier_id": supplier_id,
        "data_source": data_source,
        "reported_emissions_tco2e": str(reported_emissions_tco2e),
        "allocation_method": allocation_method,
        "allocation_factor": str(allocation_factor),
        "allocated_emissions_tco2e": str(allocated_emissions_tco2e),
        "co2_tonnes": str(co2_tonnes),
        "ch4_tco2e": str(ch4_tco2e),
        "n2o_tco2e": str(n2o_tco2e),
        "epd_number": epd_number,
        "pcf_reference": pcf_reference,
        "verification_status": verification_status,
    }
    return _hash_helper(data)


# ============================================================================
# Standalone Hash Helper Functions - EF Resolution
# ============================================================================


def hash_ef_resolution(
    asset_id: str,
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

    Creates a deterministic SHA-256 digest of the EF_RESOLUTION stage
    output. Tracks the hierarchical source selection strategy and any
    fallback behavior.

    Args:
        asset_id: Asset identifier.
        resolution_strategy: Strategy name (hierarchical/best_available/manual).
        source_priority: Ordered list of EF sources in priority order.
        selected_source: The source that was ultimately selected.
        selected_factor: The emission factor value selected.
        selected_unit: Unit of the selected factor.
        fallback_used: Whether a fallback source was used.
        fallback_reason: Optional reason for fallback (e.g., "no_sector_match").
        candidate_count: Number of candidate factors evaluated.

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "asset_id": asset_id,
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
    asset_count: int,
    total_emissions_tco2e: Decimal,
    stages_completed: int,
    stages_total: int,
) -> str:
    """
    Hash the pipeline execution context for the PIPELINE_SEAL stage.

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
        asset_count: Number of assets processed.
        total_emissions_tco2e: Total emissions calculated (tCO2e).
        stages_completed: Number of stages completed.
        stages_total: Total number of stages expected.

    Returns:
        64-character SHA-256 hex digest.

    Example:
        >>> h = hash_pipeline_context(
        ...     calculation_id="CALC-2024-001",
        ...     organization_id="ORG-123",
        ...     reporting_period="2024",
        ...     pipeline_version="1.0.0",
        ...     started_at="2024-03-15T10:00:00Z",
        ...     completed_at="2024-03-15T10:05:32Z",
        ...     duration_ms=332000,
        ...     asset_count=150,
        ...     total_emissions_tco2e=Decimal("12543.7"),
        ...     stages_completed=17,
        ...     stages_total=17,
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
        "asset_count": asset_count,
        "total_emissions_tco2e": str(total_emissions_tco2e),
        "stages_completed": stages_completed,
        "stages_total": stages_total,
    }
    return _hash_helper(data)


def hash_capitalization_check(
    asset_id: str,
    acquisition_cost: Decimal,
    capitalization_threshold: Decimal,
    meets_threshold: bool,
    gaap_standard: str,
    asset_life_years: Optional[int] = None,
    is_lease: bool = False,
    lease_classification: Optional[str] = None,
) -> str:
    """
    Hash a capitalization threshold check result.

    Creates a deterministic SHA-256 digest of the CAPITALIZATION_CHECK
    stage output. Verifies whether an asset meets the organization's
    capitalization policy per GAAP/IFRS.

    Args:
        asset_id: Asset identifier.
        acquisition_cost: Total acquisition cost.
        capitalization_threshold: Organization's capitalization threshold.
        meets_threshold: Whether the asset meets the threshold.
        gaap_standard: Accounting standard applied (US_GAAP/IFRS).
        asset_life_years: Optional expected useful life.
        is_lease: Whether the asset is a lease.
        lease_classification: Optional lease classification (finance/operating).

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "asset_id": asset_id,
        "acquisition_cost": str(acquisition_cost),
        "capitalization_threshold": str(capitalization_threshold),
        "meets_threshold": meets_threshold,
        "gaap_standard": gaap_standard,
        "asset_life_years": asset_life_years,
        "is_lease": is_lease,
        "lease_classification": lease_classification,
    }
    return _hash_helper(data)


def hash_method_selection(
    asset_id: str,
    available_methods: List[str],
    selected_method: str,
    selection_criteria: str,
    dqi_by_method: Dict[str, Decimal],
    data_availability: Dict[str, bool],
) -> str:
    """
    Hash a hybrid method selection decision.

    Creates a deterministic SHA-256 digest of the method selection
    logic used in the HYBRID_AGGREGATION stage. Records which methods
    were available and why the selected method was chosen.

    Args:
        asset_id: Asset identifier.
        available_methods: List of methods with data available.
        selected_method: Method ultimately selected.
        selection_criteria: Criteria used (highest_dqi/supplier_priority/cost_coverage).
        dqi_by_method: DQI scores by method (lower is better).
        data_availability: Data availability flags by method.

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "asset_id": asset_id,
        "available_methods": available_methods,
        "selected_method": selected_method,
        "selection_criteria": selection_criteria,
        "dqi_by_method": {k: str(v) for k, v in dqi_by_method.items()},
        "data_availability": data_availability,
    }
    return _hash_helper(data)


def hash_batch_result(
    batch_id: str,
    total_assets: int,
    processed_assets: int,
    failed_assets: int,
    skipped_assets: int,
    total_emissions_tco2e: Decimal,
    total_capex: Decimal,
    weighted_dqi: Decimal,
    duration_ms: int,
) -> str:
    """
    Hash a batch calculation result summary.

    Creates a deterministic SHA-256 digest of the batch-level
    aggregation output for multi-asset processing.

    Args:
        batch_id: Batch identifier.
        total_assets: Total number of assets in batch.
        processed_assets: Successfully processed count.
        failed_assets: Failed processing count.
        skipped_assets: Skipped (e.g., below threshold) count.
        total_emissions_tco2e: Total batch emissions (tCO2e).
        total_capex: Total CapEx value in batch.
        weighted_dqi: Weighted average DQI score.
        duration_ms: Batch processing duration (milliseconds).

    Returns:
        64-character SHA-256 hex digest.
    """
    data = {
        "batch_id": batch_id,
        "total_assets": total_assets,
        "processed_assets": processed_assets,
        "failed_assets": failed_assets,
        "skipped_assets": skipped_assets,
        "total_emissions_tco2e": str(total_emissions_tco2e),
        "total_capex": str(total_capex),
        "weighted_dqi": str(weighted_dqi),
        "duration_ms": duration_ms,
    }
    return _hash_helper(data)


# ============================================================================
# Provenance Query Utilities
# ============================================================================


def get_stage_hash(
    provenance: CapitalGoodsProvenance,
    stage: str,
) -> Optional[str]:
    """
    Get the chain hash for a specific stage.

    Args:
        provenance: CapitalGoodsProvenance instance.
        stage: Stage name to look up.

    Returns:
        Chain hash for the stage, or None if not found.
    """
    entry = provenance.get_stage(stage)
    return entry.chain_hash if entry else None


def get_stage_metadata(
    provenance: CapitalGoodsProvenance,
    stage: str,
) -> Optional[Dict[str, Any]]:
    """
    Get the metadata for a specific stage.

    Args:
        provenance: CapitalGoodsProvenance instance.
        stage: Stage name to look up.

    Returns:
        Metadata dictionary for the stage, or None if not found.
    """
    entry = provenance.get_stage(stage)
    return entry.metadata if entry else None


def get_chain_summary(provenance: CapitalGoodsProvenance) -> Dict[str, Any]:
    """
    Get a summary of the provenance chain state.

    Args:
        provenance: CapitalGoodsProvenance instance.

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
    provenance: CapitalGoodsProvenance,
    stage: str,
    expected_hash: str,
) -> bool:
    """
    Verify that a specific stage's chain hash matches an expected value.

    Useful for cross-agent provenance verification where a downstream
    agent needs to confirm the upstream hash before proceeding.

    Args:
        provenance: CapitalGoodsProvenance instance.
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
    chain_a: CapitalGoodsProvenance,
    chain_b: CapitalGoodsProvenance,
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
        "chains_identical": len(differing) == 0 and len(only_in_a) == 0 and len(only_in_b) == 0,
    }


# ============================================================================
# Convenience: Module-Level Access
# ============================================================================


def get_provenance() -> CapitalGoodsProvenance:
    """
    Get the singleton CapitalGoodsProvenance instance.

    Convenience function providing a clean module-level entry point
    for provenance tracking. Equivalent to calling
    CapitalGoodsProvenance.get_instance().

    Returns:
        The singleton CapitalGoodsProvenance instance.

    Example:
        >>> provenance = get_provenance()
        >>> provenance.start_chain(initial_data)
        >>> provenance.record_stage("ASSET_INTAKE", input_data, output_data)
        >>> provenance.seal_chain()
        >>> assert provenance.verify_chain()
    """
    return CapitalGoodsProvenance.get_instance()


# ============================================================================
# Backward Compatibility Alias
# ============================================================================

#: Alias used by the pipeline engine, setup.py, and __init__.py.
CapitalGoodsProvenanceTracker = CapitalGoodsProvenance
