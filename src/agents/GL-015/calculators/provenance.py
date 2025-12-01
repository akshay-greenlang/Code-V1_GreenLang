"""
GL-015 INSULSCAN Provenance Tracking Module

This module provides comprehensive audit trail tracking for insulation
thermal analysis calculations. All calculation inputs, outputs, and
intermediate values are tracked with SHA-256 hashes for complete
reproducibility and regulatory compliance.

Key Features:
- SHA-256 cryptographic hashing for data integrity
- Complete calculation lineage tracking
- Timestamp-based audit trails
- Configurable hash chain validation
- JSON serialization for audit records
- Database-ready provenance records

Usage:
    >>> from provenance import ProvenanceTracker, CalculationRecord
    >>> tracker = ProvenanceTracker()
    >>> record = tracker.create_record(
    ...     calculation_type="heat_loss",
    ...     inputs={"temp_hot": 150, "temp_ambient": 25},
    ...     outputs={"q_loss": 125.5}
    ... )
    >>> print(record.provenance_hash)

Compliance:
    - ISO 14064 GHG verification requirements
    - SOX Section 404 data integrity controls
    - CSRD audit trail requirements
    - GHG Protocol calculation transparency

Author: GreenLang Engineering Team
Version: 1.0.0
License: Apache 2.0
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CalculationType(Enum):
    """
    Types of calculations tracked by the provenance system.

    Each type has specific input/output schemas and validation rules.
    """

    # Core thermal calculations
    HEAT_LOSS_BARE = "heat_loss_bare"
    HEAT_LOSS_INSULATED = "heat_loss_insulated"
    SURFACE_TEMPERATURE = "surface_temperature"
    THICKNESS_OPTIMIZATION = "thickness_optimization"

    # Convection calculations
    NATURAL_CONVECTION = "natural_convection"
    FORCED_CONVECTION = "forced_convection"
    COMBINED_CONVECTION = "combined_convection"

    # Radiation calculations
    RADIATION_HEAT_TRANSFER = "radiation_heat_transfer"
    VIEW_FACTOR = "view_factor"

    # Economic calculations
    ENERGY_SAVINGS = "energy_savings"
    COST_ANALYSIS = "cost_analysis"
    ROI_CALCULATION = "roi_calculation"
    PAYBACK_PERIOD = "payback_period"
    NPV_CALCULATION = "npv_calculation"

    # Emissions calculations
    CARBON_EMISSIONS = "carbon_emissions"
    EMISSIONS_REDUCTION = "emissions_reduction"
    SCOPE_1_EMISSIONS = "scope_1_emissions"
    SCOPE_2_EMISSIONS = "scope_2_emissions"

    # Safety calculations
    PERSONNEL_PROTECTION = "personnel_protection"
    CONDENSATION_ANALYSIS = "condensation_analysis"
    FIRE_HAZARD = "fire_hazard"

    # System calculations
    PIPE_NETWORK = "pipe_network"
    EQUIPMENT_SURVEY = "equipment_survey"
    FACILITY_ASSESSMENT = "facility_assessment"

    # Material selection
    MATERIAL_COMPARISON = "material_comparison"
    THICKNESS_DETERMINATION = "thickness_determination"

    # Aggregations
    ANNUAL_SUMMARY = "annual_summary"
    FACILITY_TOTAL = "facility_total"


class HashAlgorithm(Enum):
    """
    Supported hash algorithms for provenance tracking.

    SHA-256 is the default and recommended algorithm.
    """
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"


class RecordStatus(Enum):
    """
    Status of a provenance record.
    """
    CREATED = "created"
    VALIDATED = "validated"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"
    ERROR = "error"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CalculationMetadata:
    """
    Metadata about a calculation for provenance tracking.

    Attributes:
        agent_id: Identifier of the calculating agent (e.g., "GL-015")
        agent_version: Version of the agent
        calculator_module: Python module that performed calculation
        calculator_function: Specific function called
        calculation_method: Algorithm or method used
        standards_reference: Relevant standards (ASTM, ISO, etc.)
        assumptions: List of assumptions made
        limitations: Known limitations
        confidence_level: Confidence in result (0-1)
    """
    agent_id: str = "GL-015"
    agent_version: str = "1.0.0"
    calculator_module: str = ""
    calculator_function: str = ""
    calculation_method: str = ""
    standards_reference: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    confidence_level: float = 1.0


@dataclass
class CalculationRecord:
    """
    Complete record of a calculation for audit trail.

    This is the primary record type stored in the provenance system.
    Each record is immutable once created and has a unique hash.

    Attributes:
        record_id: Unique identifier (UUID)
        calculation_type: Type of calculation performed
        inputs: Dictionary of input parameters
        outputs: Dictionary of output values
        intermediate_values: Optional intermediate calculation steps
        metadata: Calculation metadata
        provenance_hash: SHA-256 hash of record contents
        parent_hash: Hash of parent record (for chain)
        timestamp: UTC timestamp of creation
        status: Record status
    """
    record_id: str
    calculation_type: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    intermediate_values: Dict[str, Any]
    metadata: CalculationMetadata
    provenance_hash: str
    parent_hash: Optional[str]
    timestamp: str
    status: str = RecordStatus.CREATED.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "calculation_type": self.calculation_type,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "intermediate_values": self.intermediate_values,
            "metadata": asdict(self.metadata),
            "provenance_hash": self.provenance_hash,
            "parent_hash": self.parent_hash,
            "timestamp": self.timestamp,
            "status": self.status
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert record to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalculationRecord":
        """Create record from dictionary."""
        metadata = CalculationMetadata(**data.get("metadata", {}))
        return cls(
            record_id=data["record_id"],
            calculation_type=data["calculation_type"],
            inputs=data["inputs"],
            outputs=data["outputs"],
            intermediate_values=data.get("intermediate_values", {}),
            metadata=metadata,
            provenance_hash=data["provenance_hash"],
            parent_hash=data.get("parent_hash"),
            timestamp=data["timestamp"],
            status=data.get("status", RecordStatus.CREATED.value)
        )


@dataclass
class HashChainEntry:
    """
    Entry in a hash chain for linked calculations.

    Enables tracking of calculation sequences and dependencies.
    """
    sequence_number: int
    record_hash: str
    previous_hash: str
    timestamp: str
    calculation_type: str


# =============================================================================
# CANONICAL JSON SERIALIZER
# =============================================================================

class CanonicalJsonSerializer:
    """
    Deterministic JSON serialization for consistent hashing.

    Ensures that the same data always produces the same JSON string,
    regardless of dictionary ordering or floating-point formatting.

    Key Features:
    - Sorted dictionary keys
    - Consistent number formatting
    - Deterministic datetime handling
    - Unicode normalization
    """

    @staticmethod
    def serialize(data: Any) -> str:
        """
        Serialize data to canonical JSON string.

        Args:
            data: Data to serialize (dict, list, primitives)

        Returns:
            Deterministic JSON string

        Example:
            >>> CanonicalJsonSerializer.serialize({"b": 2, "a": 1})
            '{"a":1,"b":2}'
        """
        return json.dumps(
            data,
            sort_keys=True,
            separators=(',', ':'),
            default=CanonicalJsonSerializer._default_handler
        )

    @staticmethod
    def _default_handler(obj: Any) -> Any:
        """Handle non-JSON-serializable types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)


# =============================================================================
# PROVENANCE TRACKER
# =============================================================================

class ProvenanceTracker:
    """
    Main provenance tracking class for calculation audit trails.

    Provides methods to create, validate, and query provenance records.
    Supports hash chaining for linked calculations.

    Usage:
        >>> tracker = ProvenanceTracker()
        >>> record = tracker.create_record(
        ...     calculation_type="heat_loss_insulated",
        ...     inputs={"temp_hot": 150.0, "temp_ambient": 25.0},
        ...     outputs={"q_loss": 45.3}
        ... )
        >>> is_valid = tracker.validate_record(record)

    Attributes:
        algorithm: Hash algorithm to use
        records: In-memory record store
        hash_chain: Ordered hash chain for this session
    """

    def __init__(
        self,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        enable_chain: bool = True
    ):
        """
        Initialize provenance tracker.

        Args:
            algorithm: Hash algorithm to use (default SHA256)
            enable_chain: Enable hash chain tracking
        """
        self.algorithm = algorithm
        self.enable_chain = enable_chain
        self.records: Dict[str, CalculationRecord] = {}
        self.hash_chain: List[HashChainEntry] = []
        self._last_hash: Optional[str] = None
        self._session_id = str(uuid.uuid4())

        logger.info(
            f"ProvenanceTracker initialized - Session: {self._session_id}, "
            f"Algorithm: {algorithm.value}"
        )

    def create_record(
        self,
        calculation_type: Union[str, CalculationType],
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        intermediate_values: Optional[Dict[str, Any]] = None,
        metadata: Optional[CalculationMetadata] = None,
        parent_hash: Optional[str] = None
    ) -> CalculationRecord:
        """
        Create a new provenance record.

        Args:
            calculation_type: Type of calculation
            inputs: Input parameters dictionary
            outputs: Output values dictionary
            intermediate_values: Optional intermediate calculation values
            metadata: Optional calculation metadata
            parent_hash: Optional hash of parent record

        Returns:
            New CalculationRecord with computed hash

        Example:
            >>> record = tracker.create_record(
            ...     calculation_type="heat_loss_insulated",
            ...     inputs={"pipe_od": 0.1143, "insulation_thickness": 0.05},
            ...     outputs={"heat_loss_wm": 45.3, "surface_temp_c": 38.2}
            ... )
        """
        # Normalize calculation type
        if isinstance(calculation_type, CalculationType):
            calc_type_str = calculation_type.value
        else:
            calc_type_str = str(calculation_type)

        # Generate record ID and timestamp
        record_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Use default metadata if not provided
        if metadata is None:
            metadata = CalculationMetadata()

        # Initialize intermediate values
        if intermediate_values is None:
            intermediate_values = {}

        # Determine parent hash for chain
        if parent_hash is None and self.enable_chain and self._last_hash:
            parent_hash = self._last_hash

        # Create hash content
        hash_content = self._create_hash_content(
            record_id=record_id,
            calculation_type=calc_type_str,
            inputs=inputs,
            outputs=outputs,
            intermediate_values=intermediate_values,
            metadata=metadata,
            parent_hash=parent_hash,
            timestamp=timestamp
        )

        # Compute provenance hash
        provenance_hash = self._compute_hash(hash_content)

        # Create record
        record = CalculationRecord(
            record_id=record_id,
            calculation_type=calc_type_str,
            inputs=inputs,
            outputs=outputs,
            intermediate_values=intermediate_values,
            metadata=metadata,
            provenance_hash=provenance_hash,
            parent_hash=parent_hash,
            timestamp=timestamp,
            status=RecordStatus.CREATED.value
        )

        # Store record
        self.records[record_id] = record
        self.records[provenance_hash] = record  # Index by hash too

        # Update hash chain
        if self.enable_chain:
            self._add_to_chain(record)

        logger.debug(
            f"Created provenance record: {record_id[:8]}... "
            f"Hash: {provenance_hash[:16]}..."
        )

        return record

    def _create_hash_content(
        self,
        record_id: str,
        calculation_type: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        intermediate_values: Dict[str, Any],
        metadata: CalculationMetadata,
        parent_hash: Optional[str],
        timestamp: str
    ) -> str:
        """
        Create canonical content string for hashing.

        The content is deterministically serialized to ensure
        reproducible hash values.
        """
        content = {
            "record_id": record_id,
            "calculation_type": calculation_type,
            "inputs": inputs,
            "outputs": outputs,
            "intermediate_values": intermediate_values,
            "metadata": asdict(metadata),
            "parent_hash": parent_hash,
            "timestamp": timestamp
        }
        return CanonicalJsonSerializer.serialize(content)

    def _compute_hash(self, content: str) -> str:
        """
        Compute cryptographic hash of content.

        Args:
            content: String content to hash

        Returns:
            Hexadecimal hash string
        """
        if self.algorithm == HashAlgorithm.SHA256:
            hasher = hashlib.sha256()
        elif self.algorithm == HashAlgorithm.SHA384:
            hasher = hashlib.sha384()
        elif self.algorithm == HashAlgorithm.SHA512:
            hasher = hashlib.sha512()
        else:
            hasher = hashlib.sha256()

        hasher.update(content.encode('utf-8'))
        return hasher.hexdigest()

    def _add_to_chain(self, record: CalculationRecord) -> None:
        """Add record to hash chain."""
        entry = HashChainEntry(
            sequence_number=len(self.hash_chain),
            record_hash=record.provenance_hash,
            previous_hash=self._last_hash or "genesis",
            timestamp=record.timestamp,
            calculation_type=record.calculation_type
        )
        self.hash_chain.append(entry)
        self._last_hash = record.provenance_hash

    def validate_record(self, record: CalculationRecord) -> bool:
        """
        Validate a provenance record by recomputing its hash.

        Args:
            record: Record to validate

        Returns:
            True if hash matches, False otherwise
        """
        # Recompute hash
        hash_content = self._create_hash_content(
            record_id=record.record_id,
            calculation_type=record.calculation_type,
            inputs=record.inputs,
            outputs=record.outputs,
            intermediate_values=record.intermediate_values,
            metadata=record.metadata,
            parent_hash=record.parent_hash,
            timestamp=record.timestamp
        )
        computed_hash = self._compute_hash(hash_content)

        is_valid = computed_hash == record.provenance_hash

        if not is_valid:
            logger.warning(
                f"Record validation failed: {record.record_id}. "
                f"Expected: {record.provenance_hash[:16]}..., "
                f"Got: {computed_hash[:16]}..."
            )

        return is_valid

    def validate_chain(self) -> Tuple[bool, List[str]]:
        """
        Validate the entire hash chain.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if not self.hash_chain:
            return True, []

        for i, entry in enumerate(self.hash_chain):
            # Check sequence number
            if entry.sequence_number != i:
                errors.append(f"Sequence mismatch at position {i}")

            # Check chain linkage
            if i == 0:
                if entry.previous_hash != "genesis":
                    errors.append("First entry should have 'genesis' as previous")
            else:
                expected_prev = self.hash_chain[i - 1].record_hash
                if entry.previous_hash != expected_prev:
                    errors.append(f"Chain broken at position {i}")

            # Validate record if available
            if entry.record_hash in self.records:
                record = self.records[entry.record_hash]
                if not self.validate_record(record):
                    errors.append(f"Invalid record at position {i}")

        return len(errors) == 0, errors

    def get_record(
        self,
        identifier: str
    ) -> Optional[CalculationRecord]:
        """
        Get record by ID or hash.

        Args:
            identifier: Record ID or provenance hash

        Returns:
            CalculationRecord or None if not found
        """
        return self.records.get(identifier)

    def get_records_by_type(
        self,
        calculation_type: Union[str, CalculationType]
    ) -> List[CalculationRecord]:
        """
        Get all records of a specific calculation type.

        Args:
            calculation_type: Type to filter by

        Returns:
            List of matching records
        """
        if isinstance(calculation_type, CalculationType):
            type_str = calculation_type.value
        else:
            type_str = str(calculation_type)

        return [
            record for record in self.records.values()
            if record.calculation_type == type_str
            and hasattr(record, 'record_id')  # Filter out hash-indexed duplicates
        ]

    def get_chain_summary(self) -> Dict[str, Any]:
        """
        Get summary of the hash chain.

        Returns:
            Dictionary with chain statistics
        """
        return {
            "session_id": self._session_id,
            "algorithm": self.algorithm.value,
            "chain_length": len(self.hash_chain),
            "first_hash": self.hash_chain[0].record_hash if self.hash_chain else None,
            "last_hash": self._last_hash,
            "calculation_types": list(set(
                entry.calculation_type for entry in self.hash_chain
            )),
            "timestamp_range": {
                "start": self.hash_chain[0].timestamp if self.hash_chain else None,
                "end": self.hash_chain[-1].timestamp if self.hash_chain else None
            }
        }

    def export_audit_trail(
        self,
        include_records: bool = True
    ) -> Dict[str, Any]:
        """
        Export complete audit trail for external storage.

        Args:
            include_records: Include full record details

        Returns:
            Dictionary with complete audit trail
        """
        trail = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self._session_id,
            "algorithm": self.algorithm.value,
            "chain_summary": self.get_chain_summary(),
            "hash_chain": [
                {
                    "sequence": entry.sequence_number,
                    "hash": entry.record_hash,
                    "previous": entry.previous_hash,
                    "type": entry.calculation_type,
                    "timestamp": entry.timestamp
                }
                for entry in self.hash_chain
            ]
        }

        if include_records:
            trail["records"] = [
                record.to_dict()
                for record in self.records.values()
                if hasattr(record, 'record_id')
            ]

        return trail

    def export_to_json(
        self,
        filepath: Optional[str] = None,
        include_records: bool = True
    ) -> str:
        """
        Export audit trail to JSON.

        Args:
            filepath: Optional file path to write
            include_records: Include full record details

        Returns:
            JSON string
        """
        trail = self.export_audit_trail(include_records)
        json_str = json.dumps(trail, indent=2, default=str)

        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"Audit trail exported to {filepath}")

        return json_str


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_hash(data: Any, algorithm: str = "sha256") -> str:
    """
    Compute hash of arbitrary data.

    Args:
        data: Data to hash (will be JSON serialized)
        algorithm: Hash algorithm name

    Returns:
        Hexadecimal hash string
    """
    content = CanonicalJsonSerializer.serialize(data)

    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "sha384":
        hasher = hashlib.sha384()
    elif algorithm == "sha512":
        hasher = hashlib.sha512()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    hasher.update(content.encode('utf-8'))
    return hasher.hexdigest()


def create_input_hash(inputs: Dict[str, Any]) -> str:
    """
    Create hash of calculation inputs.

    Useful for caching and duplicate detection.

    Args:
        inputs: Input parameter dictionary

    Returns:
        SHA-256 hash of inputs
    """
    return compute_hash(inputs, "sha256")


def create_output_hash(outputs: Dict[str, Any]) -> str:
    """
    Create hash of calculation outputs.

    Args:
        outputs: Output values dictionary

    Returns:
        SHA-256 hash of outputs
    """
    return compute_hash(outputs, "sha256")


def verify_provenance(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    expected_hash: str,
    record_template: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Verify that inputs/outputs match expected provenance hash.

    Args:
        inputs: Calculation inputs
        outputs: Calculation outputs
        expected_hash: Expected provenance hash
        record_template: Optional additional record fields

    Returns:
        True if hash matches
    """
    content = {
        "inputs": inputs,
        "outputs": outputs
    }
    if record_template:
        content.update(record_template)

    computed = compute_hash(content)
    return computed == expected_hash


# =============================================================================
# BATCH PROVENANCE OPERATIONS
# =============================================================================

class BatchProvenanceTracker:
    """
    Batch provenance tracking for multiple calculations.

    Optimized for processing large datasets with single summary hash.

    Usage:
        >>> batch = BatchProvenanceTracker()
        >>> batch.add_calculation("heat_loss", inputs1, outputs1)
        >>> batch.add_calculation("heat_loss", inputs2, outputs2)
        >>> summary = batch.finalize()
    """

    def __init__(self, batch_id: Optional[str] = None):
        """
        Initialize batch tracker.

        Args:
            batch_id: Optional batch identifier
        """
        self.batch_id = batch_id or str(uuid.uuid4())
        self.calculations: List[Dict[str, Any]] = []
        self.start_time = datetime.now(timezone.utc)
        self._finalized = False

    def add_calculation(
        self,
        calculation_type: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add calculation to batch.

        Args:
            calculation_type: Type of calculation
            inputs: Input parameters
            outputs: Output values
            metadata: Optional metadata

        Returns:
            Index of calculation in batch
        """
        if self._finalized:
            raise RuntimeError("Batch already finalized")

        calculation = {
            "index": len(self.calculations),
            "type": calculation_type,
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Compute individual hash
        calculation["hash"] = compute_hash(calculation)

        self.calculations.append(calculation)
        return calculation["index"]

    def finalize(self) -> Dict[str, Any]:
        """
        Finalize batch and compute summary hash.

        Returns:
            Batch summary with provenance hash
        """
        if self._finalized:
            raise RuntimeError("Batch already finalized")

        self._finalized = True
        end_time = datetime.now(timezone.utc)

        # Compute batch hash from all calculation hashes
        all_hashes = [calc["hash"] for calc in self.calculations]
        batch_hash = compute_hash(all_hashes)

        summary = {
            "batch_id": self.batch_id,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "calculation_count": len(self.calculations),
            "calculation_types": list(set(
                calc["type"] for calc in self.calculations
            )),
            "batch_provenance_hash": batch_hash,
            "individual_hashes": all_hashes
        }

        return summary

    def get_batch_data(self) -> Dict[str, Any]:
        """
        Get complete batch data for export.

        Returns:
            Dictionary with all batch data
        """
        return {
            "batch_id": self.batch_id,
            "finalized": self._finalized,
            "calculation_count": len(self.calculations),
            "calculations": self.calculations
        }


# =============================================================================
# PROVENANCE VALIDATOR
# =============================================================================

class ProvenanceValidator:
    """
    Utility class for validating provenance records and chains.
    """

    @staticmethod
    def validate_record_schema(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate record has required fields.

        Args:
            record: Record dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        required_fields = [
            "record_id", "calculation_type", "inputs",
            "outputs", "provenance_hash", "timestamp"
        ]

        errors = []
        for field in required_fields:
            if field not in record:
                errors.append(f"Missing required field: {field}")

        # Validate types
        if "inputs" in record and not isinstance(record["inputs"], dict):
            errors.append("'inputs' must be a dictionary")

        if "outputs" in record and not isinstance(record["outputs"], dict):
            errors.append("'outputs' must be a dictionary")

        return len(errors) == 0, errors

    @staticmethod
    def validate_hash_format(hash_str: str, algorithm: str = "sha256") -> bool:
        """
        Validate hash string format.

        Args:
            hash_str: Hash string to validate
            algorithm: Expected algorithm

        Returns:
            True if valid format
        """
        expected_lengths = {
            "sha256": 64,
            "sha384": 96,
            "sha512": 128
        }

        if algorithm not in expected_lengths:
            return False

        if len(hash_str) != expected_lengths[algorithm]:
            return False

        # Check if valid hex
        try:
            int(hash_str, 16)
            return True
        except ValueError:
            return False


# =============================================================================
# VERSION AND METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "GreenLang Engineering Team"
__license__ = "Apache 2.0"

__all__ = [
    # Enums
    "CalculationType",
    "HashAlgorithm",
    "RecordStatus",

    # Data classes
    "CalculationMetadata",
    "CalculationRecord",
    "HashChainEntry",

    # Main classes
    "ProvenanceTracker",
    "BatchProvenanceTracker",
    "ProvenanceValidator",

    # Utilities
    "CanonicalJsonSerializer",
    "compute_hash",
    "create_input_hash",
    "create_output_hash",
    "verify_provenance",
]
