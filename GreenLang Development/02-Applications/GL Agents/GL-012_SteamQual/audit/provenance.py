"""
Provenance Tracker for GL-012 SteamQual SteamQualityController

This module implements SHA-256 provenance tracking for all calculations
in the steam quality controller. It ensures deterministic hashing for
reproducibility, maintains chain of custody tracking, and supports
model version tracking for ML-based estimators.

Key Features:
    - SHA-256 hashing for all inputs and outputs
    - Deterministic hashing for reproducibility
    - Chain of custody tracking for quality calculations
    - Model version tracking for dryness/superheat estimators
    - Data lineage graph from sensors to control outputs
    - Calculation lineage tracking
    - Formula version tracking
    - Verification of provenance records

Data Governance Compliance:
    - All sensor readings are traceable to source
    - All calculations are reproducible with stored hashes
    - Model predictions include version and confidence tracking
    - Control actions are linked to triggering calculations

Example:
    >>> tracker = ProvenanceTracker()
    >>> input_hash = tracker.compute_input_hash({"pressure_psi": 150.5})
    >>> output_hash = tracker.compute_output_hash({"dryness_fraction": 0.98})
    >>> record = tracker.create_provenance_record(
    ...     calculation_id="calc-001",
    ...     input_hash=input_hash,
    ...     output_hash=output_hash,
    ...     formula_version="STEAM_DRYNESS_V2"
    ... )
    >>> verification = tracker.verify_provenance(record)
    >>> assert verification.is_valid

Author: GreenLang Steam Quality Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class HashAlgorithm(str, Enum):
    """Supported cryptographic hash algorithms."""

    SHA256 = "SHA256"
    SHA384 = "SHA384"
    SHA512 = "SHA512"


class ModelVersionRecord(BaseModel):
    """
    Version record for ML model used in steam quality estimation.

    Tracks model identity, training metadata, and performance metrics
    for complete audit trail of ML-based calculations.
    """

    model_id: str = Field(..., description="Unique model identifier")
    model_name: str = Field(..., description="Human-readable model name")
    model_type: str = Field(
        ..., description="Model type (DRYNESS_ESTIMATOR, SUPERHEAT_ESTIMATOR, etc.)"
    )
    version: str = Field(..., description="Semantic version")

    # Training metadata
    training_date: Optional[datetime] = Field(
        None, description="Date model was trained"
    )
    training_data_hash: Optional[str] = Field(
        None, description="SHA-256 hash of training data"
    )
    training_samples: int = Field(0, ge=0, description="Number of training samples")

    # Performance metrics
    validation_mae: Optional[float] = Field(
        None, ge=0, description="Mean Absolute Error on validation set"
    )
    validation_r2: Optional[float] = Field(
        None, ge=-1, le=1, description="R-squared on validation set"
    )

    # Model artifact
    model_weights_hash: str = Field(..., description="SHA-256 hash of model weights")
    model_config_hash: str = Field(..., description="SHA-256 hash of model config")

    # Deployment info
    deployed_at: Optional[datetime] = Field(None, description="Deployment timestamp")
    deployed_by: Optional[str] = Field(None, description="Deployer ID")

    # Validity
    is_active: bool = Field(True, description="Whether model is currently active")
    superseded_by: Optional[str] = Field(
        None, description="ID of model that superseded this one"
    )

    class Config:
        frozen = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            UUID: lambda v: str(v),
        }

    @property
    def combined_hash(self) -> str:
        """Calculate combined hash of model weights and config."""
        combined = f"{self.model_weights_hash}{self.model_config_hash}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class LineageNode(BaseModel):
    """
    A node in the calculation lineage graph.

    Represents a single calculation, sensor reading, or data transformation
    with links to its inputs and outputs. Used to trace data flow from
    sensors through calculations to control actions.
    """

    node_id: UUID = Field(default_factory=uuid4, description="Unique node identifier")
    node_type: str = Field(
        ...,
        description="Type of node (SENSOR, CALCULATION, MODEL_PREDICTION, CONTROL_ACTION)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Node creation timestamp"
    )

    # Data hashes
    data_hash: str = Field(..., description="SHA-256 hash of node data")
    data_size_bytes: int = Field(0, ge=0, description="Size of data in bytes")

    # Relationships
    parent_node_ids: List[str] = Field(
        default_factory=list, description="Parent node IDs"
    )
    child_node_ids: List[str] = Field(
        default_factory=list, description="Child node IDs"
    )

    # Context
    calculation_id: Optional[str] = Field(None, description="Associated calculation ID")
    formula_id: Optional[str] = Field(None, description="Formula used if applicable")
    model_id: Optional[str] = Field(None, description="ML model used if applicable")
    sensor_tag: Optional[str] = Field(None, description="Sensor tag if sensor node")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        frozen = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    @property
    def node_hash(self) -> str:
        """Calculate SHA-256 hash of node for chain linking."""
        node_data = self.dict(exclude={"node_hash"})
        json_str = json.dumps(node_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class DataLineageGraph(BaseModel):
    """
    Complete data lineage graph for a steam quality calculation.

    Represents the full data flow from sensor readings through
    calculations and model predictions to control outputs.
    """

    graph_id: UUID = Field(default_factory=uuid4, description="Unique graph identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Graph creation timestamp"
    )

    # Graph structure
    nodes: Dict[str, LineageNode] = Field(
        default_factory=dict, description="All nodes by ID"
    )
    root_node_ids: List[str] = Field(
        default_factory=list, description="Root nodes (typically sensors)"
    )
    leaf_node_ids: List[str] = Field(
        default_factory=list, description="Leaf nodes (typically control actions)"
    )

    # Summary
    total_nodes: int = Field(0, ge=0, description="Total nodes in graph")
    sensor_nodes: int = Field(0, ge=0, description="Number of sensor nodes")
    calculation_nodes: int = Field(0, ge=0, description="Number of calculation nodes")
    model_nodes: int = Field(0, ge=0, description="Number of model prediction nodes")
    control_nodes: int = Field(0, ge=0, description="Number of control action nodes")

    # Hash for integrity
    graph_hash: Optional[str] = Field(None, description="SHA-256 hash of entire graph")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of graph structure."""
        node_hashes = sorted([n.node_hash for n in self.nodes.values()])
        combined = "".join(node_hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class ProvenanceRecord(BaseModel):
    """
    Immutable provenance record for a steam quality calculation.

    Links input data hash to output data hash with formula versioning
    and optional ML model versioning for complete audit trail.
    """

    record_id: UUID = Field(default_factory=uuid4, description="Unique record identifier")
    calculation_id: str = Field(..., description="Calculation identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record creation timestamp"
    )

    # Hash chain
    input_hash: str = Field(..., description="SHA-256 hash of inputs")
    output_hash: str = Field(..., description="SHA-256 hash of outputs")
    combined_hash: str = Field(..., description="SHA-256 hash of input+output")
    previous_record_hash: Optional[str] = Field(
        None, description="Hash of previous record in chain"
    )

    # Formula tracking
    formula_version: str = Field(..., description="Formula version identifier")
    formula_hash: Optional[str] = Field(
        None, description="Hash of formula definition"
    )

    # Model tracking (for ML-based calculations)
    model_version: Optional[str] = Field(
        None, description="ML model version if used"
    )
    model_hash: Optional[str] = Field(
        None, description="Hash of ML model weights"
    )
    model_confidence: Optional[float] = Field(
        None, ge=0, le=1, description="Model prediction confidence"
    )

    # Context
    correlation_id: Optional[str] = Field(
        None, description="Correlation ID for distributed tracing"
    )
    agent_id: str = Field(default="GL-012", description="Agent identifier")
    calculation_type: str = Field(
        default="STEAM_QUALITY",
        description="Type (DRYNESS, SUPERHEAT, ENTHALPY, CONTROL)"
    )

    # Verification
    sequence_number: int = Field(0, ge=0, description="Sequence in provenance chain")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        frozen = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    @property
    def record_hash(self) -> str:
        """Calculate SHA-256 hash of entire record."""
        record_data = self.dict(exclude={"record_hash"})
        json_str = json.dumps(record_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class VerificationResult(BaseModel):
    """Result of provenance verification."""

    is_valid: bool = Field(..., description="Whether verification passed")
    record_id: str = Field(..., description="Verified record ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Verification timestamp"
    )

    # Verification details
    input_hash_valid: bool = Field(True, description="Input hash verified")
    output_hash_valid: bool = Field(True, description="Output hash verified")
    combined_hash_valid: bool = Field(True, description="Combined hash verified")
    chain_link_valid: bool = Field(True, description="Chain linkage verified")
    formula_version_valid: bool = Field(True, description="Formula version verified")
    model_hash_valid: bool = Field(True, description="Model hash verified (if applicable)")

    # Error details
    errors: List[str] = Field(default_factory=list, description="Verification errors")
    warnings: List[str] = Field(default_factory=list, description="Verification warnings")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ProvenanceChain(BaseModel):
    """
    Complete provenance chain for a steam quality calculation output.

    Contains all records from sensor inputs to final output,
    forming a complete audit trail.
    """

    chain_id: UUID = Field(default_factory=uuid4, description="Unique chain identifier")
    output_id: str = Field(..., description="Final output identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Chain creation timestamp"
    )

    # Records in order
    records: List[ProvenanceRecord] = Field(
        default_factory=list, description="Records in chronological order"
    )

    # Summary
    total_records: int = Field(0, ge=0, description="Total records in chain")
    root_input_hash: Optional[str] = Field(None, description="Hash of original inputs")
    final_output_hash: Optional[str] = Field(None, description="Hash of final output")

    # Model versions used
    models_used: List[str] = Field(
        default_factory=list, description="Model versions used in chain"
    )

    # Merkle tree root for efficient verification
    merkle_root: Optional[str] = Field(
        None, description="Merkle tree root hash"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    @property
    def chain_hash(self) -> str:
        """Calculate hash of entire chain."""
        record_hashes = sorted([r.record_hash for r in self.records])
        combined = "".join(record_hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class ProvenanceTracker:
    """
    SHA-256 provenance tracker for steam quality calculations.

    Provides deterministic hashing of inputs and outputs,
    creates provenance records with chain linking,
    and supports verification, lineage tracking, and model version tracking.

    Attributes:
        hash_algorithm: Hash algorithm to use (default SHA256)
        records: In-memory record storage
        nodes: Lineage graph nodes
        models: Registered model versions

    Example:
        >>> tracker = ProvenanceTracker()
        >>> input_hash = tracker.compute_input_hash({"pressure": 150})
        >>> output_hash = tracker.compute_output_hash({"dryness": 0.98})
        >>> record = tracker.create_provenance_record(
        ...     calculation_id="calc-001",
        ...     input_hash=input_hash,
        ...     output_hash=output_hash,
        ...     formula_version="V1.0"
        ... )
    """

    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    ):
        """
        Initialize provenance tracker.

        Args:
            hash_algorithm: Hash algorithm to use (default SHA256)
        """
        self.hash_algorithm = hash_algorithm

        # In-memory storage (replace with persistent in production)
        self._records: Dict[str, ProvenanceRecord] = {}
        self._nodes: Dict[str, LineageNode] = {}
        self._models: Dict[str, ModelVersionRecord] = {}
        self._chain: List[str] = []  # Ordered record hashes
        self._sequence = 0

        logger.info(
            "ProvenanceTracker initialized",
            extra={"hash_algorithm": hash_algorithm.value}
        )

    def _compute_hash(self, data: Any) -> str:
        """
        Compute cryptographic hash of data.

        Args:
            data: Data to hash (will be JSON serialized)

        Returns:
            Hex-encoded hash string
        """
        if isinstance(data, bytes):
            data_bytes = data
        elif isinstance(data, str):
            data_bytes = data.encode("utf-8")
        else:
            # Deterministic JSON serialization
            data_bytes = json.dumps(data, sort_keys=True, default=str).encode("utf-8")

        if self.hash_algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(data_bytes).hexdigest()
        elif self.hash_algorithm == HashAlgorithm.SHA384:
            return hashlib.sha384(data_bytes).hexdigest()
        elif self.hash_algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512(data_bytes).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")

    def compute_input_hash(self, inputs: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash of input data.

        Uses deterministic JSON serialization with sorted keys
        to ensure reproducibility.

        Args:
            inputs: Input data dictionary

        Returns:
            Hex-encoded SHA-256 hash

        Example:
            >>> hash = tracker.compute_input_hash({"pressure_psi": 150.5})
            >>> print(hash)  # 64-character hex string
        """
        return self._compute_hash(inputs)

    def compute_output_hash(self, outputs: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash of output data.

        Uses deterministic JSON serialization with sorted keys
        to ensure reproducibility.

        Args:
            outputs: Output data dictionary

        Returns:
            Hex-encoded SHA-256 hash

        Example:
            >>> hash = tracker.compute_output_hash({"dryness_fraction": 0.98})
            >>> print(hash)  # 64-character hex string
        """
        return self._compute_hash(outputs)

    def _get_previous_record_hash(self) -> Optional[str]:
        """Get hash of previous record in chain."""
        if not self._chain:
            return None
        return self._chain[-1]

    def register_model_version(self, model: ModelVersionRecord) -> None:
        """
        Register a model version for tracking.

        Args:
            model: ModelVersionRecord to register
        """
        self._models[model.model_id] = model
        logger.info(
            f"Model version registered: {model.model_id} v{model.version}",
            extra={"model_type": model.model_type}
        )

    def get_model_version(self, model_id: str) -> Optional[ModelVersionRecord]:
        """Get a registered model version by ID."""
        return self._models.get(model_id)

    def create_provenance_record(
        self,
        calculation_id: str,
        input_hash: str,
        output_hash: str,
        formula_version: str,
        formula_hash: Optional[str] = None,
        model_version: Optional[str] = None,
        model_hash: Optional[str] = None,
        model_confidence: Optional[float] = None,
        calculation_type: str = "STEAM_QUALITY",
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceRecord:
        """
        Create a provenance record linking inputs to outputs.

        Args:
            calculation_id: Unique calculation identifier
            input_hash: SHA-256 hash of inputs
            output_hash: SHA-256 hash of outputs
            formula_version: Version of formula used
            formula_hash: Optional hash of formula definition
            model_version: Optional ML model version
            model_hash: Optional hash of ML model weights
            model_confidence: Optional model prediction confidence
            calculation_type: Type of calculation (DRYNESS, SUPERHEAT, etc.)
            correlation_id: Optional correlation ID for tracing
            metadata: Optional additional metadata

        Returns:
            Created ProvenanceRecord

        Example:
            >>> record = tracker.create_provenance_record(
            ...     calculation_id="calc-001",
            ...     input_hash="abc123...",
            ...     output_hash="def456...",
            ...     formula_version="STEAM_DRYNESS_V2"
            ... )
        """
        # Compute combined hash
        combined_input = f"{input_hash}{output_hash}"
        combined_hash = self._compute_hash(combined_input)

        # Get previous record hash for chain linking
        previous_hash = self._get_previous_record_hash()

        record = ProvenanceRecord(
            calculation_id=calculation_id,
            input_hash=input_hash,
            output_hash=output_hash,
            combined_hash=combined_hash,
            previous_record_hash=previous_hash,
            formula_version=formula_version,
            formula_hash=formula_hash,
            model_version=model_version,
            model_hash=model_hash,
            model_confidence=model_confidence,
            calculation_type=calculation_type,
            correlation_id=correlation_id,
            sequence_number=self._sequence,
            metadata=metadata or {},
        )

        # Store record
        record_id = str(record.record_id)
        self._records[record_id] = record
        self._chain.append(record.record_hash)
        self._sequence += 1

        logger.info(
            f"Provenance record created: {calculation_id}",
            extra={
                "record_id": record_id,
                "sequence": record.sequence_number,
                "calculation_type": calculation_type,
                "input_hash": input_hash[:16] + "...",
                "output_hash": output_hash[:16] + "...",
            }
        )

        return record

    def verify_provenance(self, record: ProvenanceRecord) -> VerificationResult:
        """
        Verify integrity of a provenance record.

        Checks:
        - Combined hash matches input_hash + output_hash
        - Chain linkage is valid
        - Record hash is in the chain
        - Model hash is valid (if applicable)

        Args:
            record: ProvenanceRecord to verify

        Returns:
            VerificationResult with verification details

        Example:
            >>> result = tracker.verify_provenance(record)
            >>> if not result.is_valid:
            ...     print(result.errors)
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Verify combined hash
        expected_combined = self._compute_hash(f"{record.input_hash}{record.output_hash}")
        combined_valid = record.combined_hash == expected_combined
        if not combined_valid:
            errors.append(
                f"Combined hash mismatch: expected {expected_combined}, got {record.combined_hash}"
            )

        # Verify chain linkage
        chain_valid = True
        if record.previous_record_hash:
            if record.previous_record_hash not in self._chain:
                chain_valid = False
                errors.append(
                    f"Previous record hash not in chain: {record.previous_record_hash}"
                )

        # Verify record is in chain
        record_in_chain = record.record_hash in self._chain
        if not record_in_chain:
            warnings.append("Record hash not found in current chain (may be from different session)")

        # Check formula version is present
        formula_valid = bool(record.formula_version)
        if not formula_valid:
            errors.append("Formula version is missing")

        # Verify model hash if present
        model_valid = True
        if record.model_version:
            model = self._models.get(record.model_version)
            if model and record.model_hash:
                if record.model_hash != model.model_weights_hash:
                    model_valid = False
                    errors.append(
                        f"Model hash mismatch for {record.model_version}"
                    )
            elif record.model_hash:
                warnings.append(f"Model {record.model_version} not registered, cannot verify hash")

        is_valid = combined_valid and chain_valid and formula_valid and model_valid

        result = VerificationResult(
            is_valid=is_valid,
            record_id=str(record.record_id),
            input_hash_valid=True,  # Assumed valid if provided
            output_hash_valid=True,  # Assumed valid if provided
            combined_hash_valid=combined_valid,
            chain_link_valid=chain_valid,
            formula_version_valid=formula_valid,
            model_hash_valid=model_valid,
            errors=errors,
            warnings=warnings,
        )

        logger.info(
            f"Provenance verification: {'PASS' if is_valid else 'FAIL'}",
            extra={
                "record_id": str(record.record_id),
                "errors": len(errors),
                "warnings": len(warnings),
            }
        )

        return result

    def get_calculation_lineage(self, output_id: str) -> List[ProvenanceRecord]:
        """
        Get complete lineage of provenance records for an output.

        Traces back through the chain to find all records
        that contributed to the specified output.

        Args:
            output_id: Output identifier (calculation_id of final output)

        Returns:
            List of ProvenanceRecords in chronological order

        Example:
            >>> lineage = tracker.get_calculation_lineage("output-001")
            >>> for record in lineage:
            ...     print(record.calculation_id)
        """
        # Find records related to this output
        lineage: List[ProvenanceRecord] = []
        visited: set = set()

        def find_related(calc_id: str) -> None:
            """Recursively find related records."""
            for record in self._records.values():
                if record.calculation_id == calc_id and str(record.record_id) not in visited:
                    visited.add(str(record.record_id))
                    lineage.append(record)

                    # Find records that produced inputs for this calculation
                    if "source_calculations" in record.metadata:
                        for source_id in record.metadata["source_calculations"]:
                            find_related(source_id)

        find_related(output_id)

        # Sort by sequence number
        lineage.sort(key=lambda r: r.sequence_number)

        logger.info(
            f"Retrieved lineage for {output_id}",
            extra={"records_found": len(lineage)}
        )

        return lineage

    def build_provenance_chain(self, output_id: str) -> ProvenanceChain:
        """
        Build complete provenance chain for an output.

        Creates a ProvenanceChain object with all records
        and Merkle tree root for efficient verification.

        Args:
            output_id: Output identifier

        Returns:
            ProvenanceChain object

        Example:
            >>> chain = tracker.build_provenance_chain("output-001")
            >>> print(chain.merkle_root)
        """
        records = self.get_calculation_lineage(output_id)

        # Calculate Merkle root
        merkle_root = self._calculate_merkle_root([r.record_hash for r in records])

        # Get root input and final output hashes
        root_input = records[0].input_hash if records else None
        final_output = records[-1].output_hash if records else None

        # Get unique model versions used
        models_used = list(set(
            r.model_version for r in records
            if r.model_version is not None
        ))

        chain = ProvenanceChain(
            output_id=output_id,
            records=records,
            total_records=len(records),
            root_input_hash=root_input,
            final_output_hash=final_output,
            models_used=models_used,
            merkle_root=merkle_root,
        )

        return chain

    def _calculate_merkle_root(self, hashes: List[str]) -> str:
        """
        Calculate Merkle tree root hash.

        Args:
            hashes: List of hash strings

        Returns:
            Merkle root hash
        """
        if not hashes:
            return self._compute_hash("")

        # Sort for determinism
        current_level = sorted(hashes)

        while len(current_level) > 1:
            # Pad if odd
            if len(current_level) % 2 == 1:
                current_level.append(current_level[-1])

            # Combine pairs
            next_level = []
            for i in range(0, len(current_level), 2):
                combined = current_level[i] + current_level[i + 1]
                next_level.append(self._compute_hash(combined))
            current_level = next_level

        return current_level[0]

    def create_lineage_node(
        self,
        node_type: str,
        data: Any,
        parent_node_ids: Optional[List[str]] = None,
        calculation_id: Optional[str] = None,
        formula_id: Optional[str] = None,
        model_id: Optional[str] = None,
        sensor_tag: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageNode:
        """
        Create a lineage graph node.

        Args:
            node_type: Type of node (SENSOR, CALCULATION, MODEL_PREDICTION, CONTROL_ACTION)
            data: Data content for hashing
            parent_node_ids: IDs of parent nodes
            calculation_id: Associated calculation ID
            formula_id: Formula used if applicable
            model_id: ML model used if applicable
            sensor_tag: Sensor tag if sensor node
            correlation_id: Correlation ID for tracing
            metadata: Additional metadata

        Returns:
            Created LineageNode
        """
        data_hash = self._compute_hash(data)

        if isinstance(data, bytes):
            data_size = len(data)
        elif isinstance(data, str):
            data_size = len(data.encode("utf-8"))
        else:
            data_size = len(json.dumps(data, default=str).encode("utf-8"))

        node = LineageNode(
            node_type=node_type,
            data_hash=data_hash,
            data_size_bytes=data_size,
            parent_node_ids=parent_node_ids or [],
            calculation_id=calculation_id,
            formula_id=formula_id,
            model_id=model_id,
            sensor_tag=sensor_tag,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )

        # Store node
        node_id = str(node.node_id)
        self._nodes[node_id] = node

        logger.debug(
            f"Lineage node created: {node_type}",
            extra={"node_id": node_id, "data_hash": data_hash[:16] + "..."}
        )

        return node

    def build_lineage_graph(
        self,
        root_node_ids: List[str],
    ) -> DataLineageGraph:
        """
        Build a complete data lineage graph from root nodes.

        Args:
            root_node_ids: IDs of root nodes (typically sensors)

        Returns:
            DataLineageGraph with all connected nodes
        """
        visited: Dict[str, LineageNode] = {}
        leaf_nodes: List[str] = []

        def traverse(node_id: str) -> None:
            """Recursively traverse and collect nodes."""
            if node_id in visited:
                return

            node = self._nodes.get(node_id)
            if node is None:
                return

            visited[node_id] = node

            if not node.child_node_ids:
                leaf_nodes.append(node_id)

            for child_id in node.child_node_ids:
                traverse(child_id)

        for root_id in root_node_ids:
            traverse(root_id)

        # Count node types
        sensor_count = sum(1 for n in visited.values() if n.node_type == "SENSOR")
        calc_count = sum(1 for n in visited.values() if n.node_type == "CALCULATION")
        model_count = sum(1 for n in visited.values() if n.node_type == "MODEL_PREDICTION")
        control_count = sum(1 for n in visited.values() if n.node_type == "CONTROL_ACTION")

        graph = DataLineageGraph(
            nodes=visited,
            root_node_ids=root_node_ids,
            leaf_node_ids=leaf_nodes,
            total_nodes=len(visited),
            sensor_nodes=sensor_count,
            calculation_nodes=calc_count,
            model_nodes=model_count,
            control_nodes=control_count,
        )

        # Calculate and set hash
        graph_dict = graph.dict()
        graph_dict["graph_hash"] = graph.calculate_hash()

        return DataLineageGraph(**graph_dict)

    def get_node(self, node_id: str) -> Optional[LineageNode]:
        """Get a lineage node by ID."""
        return self._nodes.get(node_id)

    def get_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Get a provenance record by ID."""
        return self._records.get(record_id)

    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """
        Verify integrity of the entire provenance chain.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(self._chain) < 2:
            return True, None

        # Get records in order
        records = sorted(self._records.values(), key=lambda r: r.sequence_number)

        prev_hash = None
        for record in records:
            # Verify previous hash linkage
            if prev_hash is not None and record.previous_record_hash != prev_hash:
                error = f"Chain broken at sequence {record.sequence_number}"
                logger.error(error)
                return False, error

            # Verify combined hash
            expected_combined = self._compute_hash(f"{record.input_hash}{record.output_hash}")
            if record.combined_hash != expected_combined:
                error = f"Combined hash invalid at sequence {record.sequence_number}"
                logger.error(error)
                return False, error

            prev_hash = record.record_hash

        logger.info("Provenance chain verified successfully")
        return True, None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get provenance tracking statistics.

        Returns:
            Dictionary of statistics
        """
        # Count records by calculation type
        type_counts: Dict[str, int] = {}
        for record in self._records.values():
            calc_type = record.calculation_type
            type_counts[calc_type] = type_counts.get(calc_type, 0) + 1

        # Count records by formula version
        formula_counts: Dict[str, int] = {}
        for record in self._records.values():
            formula = record.formula_version
            formula_counts[formula] = formula_counts.get(formula, 0) + 1

        # Count model usage
        model_counts: Dict[str, int] = {}
        for record in self._records.values():
            if record.model_version:
                model_counts[record.model_version] = model_counts.get(record.model_version, 0) + 1

        return {
            "total_records": len(self._records),
            "total_nodes": len(self._nodes),
            "registered_models": len(self._models),
            "chain_length": len(self._chain),
            "current_sequence": self._sequence,
            "records_by_type": type_counts,
            "records_by_formula": formula_counts,
            "records_by_model": model_counts,
            "hash_algorithm": self.hash_algorithm.value,
        }

    def export_chain(self) -> List[Dict[str, Any]]:
        """
        Export the provenance chain for external verification.

        Returns:
            List of record dictionaries with hashes
        """
        records = sorted(self._records.values(), key=lambda r: r.sequence_number)
        return [
            {
                "record_id": str(r.record_id),
                "calculation_id": r.calculation_id,
                "calculation_type": r.calculation_type,
                "sequence_number": r.sequence_number,
                "input_hash": r.input_hash,
                "output_hash": r.output_hash,
                "combined_hash": r.combined_hash,
                "record_hash": r.record_hash,
                "previous_hash": r.previous_record_hash,
                "formula_version": r.formula_version,
                "model_version": r.model_version,
                "model_confidence": r.model_confidence,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in records
        ]

    def clear(self) -> None:
        """Clear all tracked provenance data."""
        self._records.clear()
        self._nodes.clear()
        self._chain.clear()
        self._sequence = 0
        # Keep registered models
        logger.info("Provenance tracker cleared")
