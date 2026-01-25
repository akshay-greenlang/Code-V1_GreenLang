# -*- coding: utf-8 -*-
"""
GL-015 Insulscan - Chain of Custody

Complete chain of custody tracking for insulation assessment data from source
to recommendation. Tracks data movement through the entire pipeline:

    Thermal Camera -> Data Validation -> Analysis -> Prediction -> Recommendation -> CMMS

Each step is versioned and hash-linked for complete traceability, tamper detection,
and regulatory audit support. Optional blockchain integration for immutable records.

Features:
    - Track asset data through processing pipeline
    - Immutable audit log with SHA-256 hash chains
    - Integration with blockchain (optional)
    - Tamper detection with Merkle tree verification
    - Data provenance visualization (DOT graph export)
    - Complete version tracking at each step

Standards:
    - ISO 50001:2018 (Energy Management Systems)
    - ASHRAE Standards (Building Insulation)
    - 21 CFR Part 11 (Electronic Records)
    - ISO 27001 (Information Security)

Example:
    >>> from audit.chain_of_custody import InsulationChainOfCustody
    >>> custody = InsulationChainOfCustody()
    >>> custody.start_chain(
    ...     source_type=DataSource.THERMAL_CAMERA,
    ...     source_id="CAM-001",
    ...     asset_id="INSUL-001"
    ... )
    >>> custody.add_scan_step(scan_data, camera_model="FLIR-T640")
    >>> custody.add_analysis_step(analysis_type, inputs, outputs)
    >>> custody.add_recommendation_step(recommendation)
    >>> chain = custody.finalize()
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .schemas import (
    ChainOfCustodyEntry,
    ProvenanceChain,
    ComputationType,
    DataSource,
    CustodyType,
    ChainVerificationStatus,
    compute_sha256,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CUSTODY STEP TYPES
# =============================================================================

class CustodyStepType(str, Enum):
    """Types of steps in the chain of custody for insulation assessment."""

    # Data source steps
    THERMAL_CAMERA_SCAN = "thermal_camera_scan"
    TEMPERATURE_SENSOR_READ = "temperature_sensor_read"
    OPC_UA_INGESTION = "opc_ua_ingestion"
    HISTORIAN_QUERY = "historian_query"
    MANUAL_ENTRY = "manual_entry"
    WEATHER_DATA_FETCH = "weather_data_fetch"
    BMS_DATA_RECEIVE = "bms_data_receive"

    # Data processing steps
    SCHEMA_VALIDATION = "schema_validation"
    DATA_TRANSFORMATION = "data_transformation"
    DATA_ENRICHMENT = "data_enrichment"
    IMAGE_PROCESSING = "image_processing"
    CALIBRATION_CORRECTION = "calibration_correction"
    OUTLIER_DETECTION = "outlier_detection"

    # Analysis steps
    HEAT_LOSS_ANALYSIS = "heat_loss_analysis"
    R_VALUE_CALCULATION = "r_value_calculation"
    U_VALUE_CALCULATION = "u_value_calculation"
    THERMAL_BRIDGE_DETECTION = "thermal_bridge_detection"
    HOTSPOT_DETECTION = "hotspot_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    CONDITION_ASSESSMENT = "condition_assessment"

    # Prediction steps
    MODEL_INFERENCE = "model_inference"
    DEGRADATION_PREDICTION = "degradation_prediction"
    RUL_PREDICTION = "rul_prediction"
    ENERGY_SAVINGS_PREDICTION = "energy_savings_prediction"
    MAINTENANCE_PREDICTION = "maintenance_prediction"

    # Recommendation steps
    RECOMMENDATION_GENERATION = "recommendation_generation"
    PRIORITY_ASSIGNMENT = "priority_assignment"
    THRESHOLD_EVALUATION = "threshold_evaluation"
    ROI_CALCULATION = "roi_calculation"

    # Output steps
    CMMS_WORK_ORDER = "cmms_work_order"
    ERP_NOTIFICATION = "erp_notification"
    REPORT_GENERATION = "report_generation"
    ALERT_DISPATCH = "alert_dispatch"
    BLOCKCHAIN_REGISTRATION = "blockchain_registration"


# =============================================================================
# DETAILED CUSTODY STEP
# =============================================================================

@dataclass
class DetailedCustodyStep:
    """
    Detailed custody step with full version tracking.

    Extends basic custody entry with additional metadata for complete
    traceability through the insulation assessment pipeline.
    """

    step_id: str = ""
    step_number: int = 0
    step_type: CustodyStepType = CustodyStepType.THERMAL_CAMERA_SCAN

    # System identification
    source_system: str = ""
    target_system: str = ""
    source_component: str = ""
    target_component: str = ""

    # Data integrity
    input_data_hash: str = ""
    output_data_hash: str = ""
    input_record_count: int = 0
    output_record_count: int = 0

    # Version tracking
    schema_version: str = "1.0.0"
    transformation_version: str = "1.0.0"
    algorithm_version: str = "1.0.0"
    model_version: Optional[str] = None
    config_version: str = "1.0.0"

    # Transformation details
    transformation_applied: str = ""
    transformation_params: Dict[str, Any] = field(default_factory=dict)
    formula_reference: Optional[str] = None

    # Timing
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    forwarded_at: Optional[datetime] = None
    processing_duration_ms: float = 0.0

    # Quality tracking
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    data_quality_score: float = 1.0
    completeness_score: float = 1.0

    # Provenance
    parent_step_id: Optional[str] = None
    provenance_hash: str = ""

    # Metadata
    operator_id: Optional[str] = None
    correlation_id: str = ""
    notes: str = ""

    # Blockchain integration
    blockchain_registered: bool = False
    blockchain_tx_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize computed fields."""
        if not self.step_id:
            self.step_id = f"STEP-{uuid.uuid4().hex[:12].upper()}"

        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance_hash()

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 provenance hash for this step."""
        hash_data = {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "step_type": self.step_type.value if isinstance(self.step_type, CustodyStepType) else str(self.step_type),
            "source_system": self.source_system,
            "target_system": self.target_system,
            "input_data_hash": self.input_data_hash,
            "output_data_hash": self.output_data_hash,
            "schema_version": self.schema_version,
            "transformation_version": self.transformation_version,
            "algorithm_version": self.algorithm_version,
            "model_version": self.model_version,
            "received_at": self.received_at.isoformat() if isinstance(self.received_at, datetime) else str(self.received_at),
            "parent_step_id": self.parent_step_id,
        }
        return compute_sha256(hash_data)

    def mark_processing_start(self) -> None:
        """Mark the start of processing."""
        self.processing_started_at = datetime.now(timezone.utc)

    def mark_processing_complete(self, output_hash: str) -> None:
        """Mark completion of processing."""
        self.processing_completed_at = datetime.now(timezone.utc)
        self.output_data_hash = output_hash

        if self.processing_started_at:
            delta = self.processing_completed_at - self.processing_started_at
            self.processing_duration_ms = delta.total_seconds() * 1000

        # Recalculate provenance hash
        self.provenance_hash = self._calculate_provenance_hash()

    def add_validation_error(self, error: str) -> None:
        """Add a validation error."""
        self.validation_errors.append(error)
        self.validation_passed = False

    def add_validation_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.validation_warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "step_type": self.step_type.value if isinstance(self.step_type, CustodyStepType) else str(self.step_type),
            "source_system": self.source_system,
            "target_system": self.target_system,
            "source_component": self.source_component,
            "target_component": self.target_component,
            "input_data_hash": self.input_data_hash,
            "output_data_hash": self.output_data_hash,
            "input_record_count": self.input_record_count,
            "output_record_count": self.output_record_count,
            "schema_version": self.schema_version,
            "transformation_version": self.transformation_version,
            "algorithm_version": self.algorithm_version,
            "model_version": self.model_version,
            "config_version": self.config_version,
            "transformation_applied": self.transformation_applied,
            "transformation_params": self.transformation_params,
            "formula_reference": self.formula_reference,
            "received_at": self.received_at.isoformat() if isinstance(self.received_at, datetime) else str(self.received_at),
            "processing_started_at": self.processing_started_at.isoformat() if self.processing_started_at else None,
            "processing_completed_at": self.processing_completed_at.isoformat() if self.processing_completed_at else None,
            "forwarded_at": self.forwarded_at.isoformat() if self.forwarded_at else None,
            "processing_duration_ms": self.processing_duration_ms,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "data_quality_score": self.data_quality_score,
            "completeness_score": self.completeness_score,
            "parent_step_id": self.parent_step_id,
            "provenance_hash": self.provenance_hash,
            "operator_id": self.operator_id,
            "correlation_id": self.correlation_id,
            "notes": self.notes,
            "blockchain_registered": self.blockchain_registered,
            "blockchain_tx_id": self.blockchain_tx_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetailedCustodyStep":
        """Create from dictionary."""
        if isinstance(data.get("step_type"), str):
            data["step_type"] = CustodyStepType(data["step_type"])

        for dt_field in ["received_at", "processing_started_at", "processing_completed_at", "forwarded_at"]:
            if isinstance(data.get(dt_field), str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])

        return cls(**data)


# =============================================================================
# CUSTODY CHAIN
# =============================================================================

@dataclass
class CustodyChain:
    """
    Complete chain of custody from source to destination.

    Represents the full journey of insulation assessment data through
    the pipeline with cryptographic verification at each step.
    """

    chain_id: str = ""
    asset_id: str = ""
    correlation_id: str = ""

    # Source information
    source_type: DataSource = DataSource.THERMAL_CAMERA
    source_id: str = ""
    source_model: str = ""
    source_timestamp: Optional[datetime] = None

    # Destination information
    destination_type: str = ""
    destination_id: str = ""
    destination_timestamp: Optional[datetime] = None

    # Chain of steps
    steps: List[DetailedCustodyStep] = field(default_factory=list)
    hash_links: List[str] = field(default_factory=list)

    # Chain status
    is_complete: bool = False
    is_verified: bool = False
    verification_status: ChainVerificationStatus = ChainVerificationStatus.PENDING

    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    total_duration_ms: float = 0.0

    # Provenance
    merkle_root: str = ""
    chain_hash: str = ""

    # Blockchain integration
    blockchain_anchored: bool = False
    blockchain_anchor_tx: Optional[str] = None
    blockchain_anchor_time: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Initialize chain ID if not provided."""
        if not self.chain_id:
            self.chain_id = f"COC-{uuid.uuid4().hex[:12].upper()}"
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())

    def add_step(self, step: DetailedCustodyStep) -> None:
        """
        Add a step to the custody chain.

        Args:
            step: Custody step to add

        Raises:
            ValueError: If chain is already complete
        """
        if self.is_complete:
            raise ValueError("Cannot add steps to a completed chain")

        # Set step number and parent
        step.step_number = len(self.steps)
        if self.steps:
            step.parent_step_id = self.steps[-1].step_id

        step.correlation_id = self.correlation_id

        self.steps.append(step)

        # Compute hash link
        prev_hash = self.hash_links[-1] if self.hash_links else "genesis"
        link_hash = compute_sha256(f"{prev_hash}:{step.provenance_hash}")
        self.hash_links.append(link_hash)

    def complete(self, destination_type: str, destination_id: str) -> str:
        """
        Mark the chain as complete.

        Args:
            destination_type: Type of final destination (e.g., "cmms", "report")
            destination_id: Identifier of destination (e.g., work order ID)

        Returns:
            Chain hash

        Raises:
            ValueError: If chain has no steps
        """
        if not self.steps:
            raise ValueError("Cannot complete empty chain")

        self.destination_type = destination_type
        self.destination_id = destination_id
        self.destination_timestamp = datetime.now(timezone.utc)
        self.completed_at = datetime.now(timezone.utc)
        self.is_complete = True

        # Calculate total duration
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            self.total_duration_ms = delta.total_seconds() * 1000

        # Compute merkle root and chain hash
        self.merkle_root = self._compute_merkle_root()
        self.chain_hash = self._compute_chain_hash()

        logger.info(
            f"Custody chain {self.chain_id} completed: {len(self.steps)} steps, "
            f"{self.total_duration_ms:.2f}ms total"
        )

        return self.chain_hash

    def verify(self) -> ChainVerificationStatus:
        """
        Verify the integrity of the custody chain.

        Returns:
            ChainVerificationStatus indicating verification result
        """
        if not self.steps:
            self.verification_status = ChainVerificationStatus.INCOMPLETE
            return self.verification_status

        # Verify each step's provenance hash
        prev_hash = "genesis"
        for i, step in enumerate(self.steps):
            # Recalculate step's provenance hash
            expected_hash = step._calculate_provenance_hash()
            if step.provenance_hash != expected_hash:
                logger.error(f"Step {step.step_id} provenance hash mismatch")
                self.verification_status = ChainVerificationStatus.INVALID
                return self.verification_status

            # Verify hash link
            expected_link = compute_sha256(f"{prev_hash}:{step.provenance_hash}")
            if i < len(self.hash_links) and self.hash_links[i] != expected_link:
                logger.error(f"Hash link {i} verification failed")
                self.verification_status = ChainVerificationStatus.BROKEN
                return self.verification_status

            prev_hash = self.hash_links[i] if i < len(self.hash_links) else expected_link

        # Verify merkle root if chain is complete
        if self.is_complete:
            expected_merkle = self._compute_merkle_root()
            if self.merkle_root != expected_merkle:
                logger.error("Merkle root verification failed")
                self.verification_status = ChainVerificationStatus.INVALID
                return self.verification_status

        self.is_verified = True
        self.verification_status = ChainVerificationStatus.VALID
        return self.verification_status

    def detect_tampering(self) -> List[str]:
        """
        Detect any tampering in the chain.

        Returns:
            List of step IDs where tampering was detected
        """
        tampered_steps = []

        prev_hash = "genesis"
        for i, step in enumerate(self.steps):
            # Check provenance hash
            expected_hash = step._calculate_provenance_hash()
            if step.provenance_hash != expected_hash:
                tampered_steps.append(step.step_id)
                logger.warning(f"Tampering detected at step {step.step_id}: provenance hash mismatch")

            # Check hash link
            expected_link = compute_sha256(f"{prev_hash}:{step.provenance_hash}")
            if i < len(self.hash_links) and self.hash_links[i] != expected_link:
                tampered_steps.append(step.step_id)
                logger.warning(f"Tampering detected at step {step.step_id}: hash link broken")

            prev_hash = self.hash_links[i] if i < len(self.hash_links) else expected_link

        return tampered_steps

    def _compute_merkle_root(self) -> str:
        """Compute Merkle tree root of all step hashes."""
        if not self.steps:
            return compute_sha256(b"")

        leaves = [step.provenance_hash for step in self.steps]

        while len(leaves) > 1:
            if len(leaves) % 2:
                leaves.append(leaves[-1])

            leaves = [
                compute_sha256(leaves[i] + leaves[i + 1])
                for i in range(0, len(leaves), 2)
            ]

        return leaves[0]

    def _compute_chain_hash(self) -> str:
        """Compute hash of the complete chain."""
        chain_data = {
            "chain_id": self.chain_id,
            "asset_id": self.asset_id,
            "source_type": self.source_type.value if isinstance(self.source_type, DataSource) else str(self.source_type),
            "source_id": self.source_id,
            "destination_type": self.destination_type,
            "destination_id": self.destination_id,
            "step_count": len(self.steps),
            "hash_links": self.hash_links,
            "merkle_root": self.merkle_root,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
        return compute_sha256(chain_data)

    def get_step_by_type(self, step_type: CustodyStepType) -> Optional[DetailedCustodyStep]:
        """Get the first step of a given type."""
        for step in self.steps:
            if step.step_type == step_type:
                return step
        return None

    def get_all_steps_by_type(self, step_type: CustodyStepType) -> List[DetailedCustodyStep]:
        """Get all steps of a given type."""
        return [step for step in self.steps if step.step_type == step_type]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chain_id": self.chain_id,
            "asset_id": self.asset_id,
            "correlation_id": self.correlation_id,
            "source_type": self.source_type.value if isinstance(self.source_type, DataSource) else str(self.source_type),
            "source_id": self.source_id,
            "source_model": self.source_model,
            "source_timestamp": self.source_timestamp.isoformat() if self.source_timestamp else None,
            "destination_type": self.destination_type,
            "destination_id": self.destination_id,
            "destination_timestamp": self.destination_timestamp.isoformat() if self.destination_timestamp else None,
            "steps": [step.to_dict() for step in self.steps],
            "hash_links": self.hash_links,
            "is_complete": self.is_complete,
            "is_verified": self.is_verified,
            "verification_status": self.verification_status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_ms": self.total_duration_ms,
            "merkle_root": self.merkle_root,
            "chain_hash": self.chain_hash,
            "blockchain_anchored": self.blockchain_anchored,
            "blockchain_anchor_tx": self.blockchain_anchor_tx,
            "blockchain_anchor_time": self.blockchain_anchor_time.isoformat() if self.blockchain_anchor_time else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustodyChain":
        """Create from dictionary."""
        steps = [DetailedCustodyStep.from_dict(s) for s in data.get("steps", [])]

        chain = cls(
            chain_id=data.get("chain_id", ""),
            asset_id=data.get("asset_id", ""),
            correlation_id=data.get("correlation_id", ""),
            source_id=data.get("source_id", ""),
            source_model=data.get("source_model", ""),
            destination_type=data.get("destination_type", ""),
            destination_id=data.get("destination_id", ""),
            steps=steps,
            hash_links=data.get("hash_links", []),
            is_complete=data.get("is_complete", False),
            is_verified=data.get("is_verified", False),
            merkle_root=data.get("merkle_root", ""),
            chain_hash=data.get("chain_hash", ""),
            total_duration_ms=data.get("total_duration_ms", 0.0),
            blockchain_anchored=data.get("blockchain_anchored", False),
            blockchain_anchor_tx=data.get("blockchain_anchor_tx"),
        )

        if isinstance(data.get("source_type"), str):
            chain.source_type = DataSource(data["source_type"])
        if isinstance(data.get("verification_status"), str):
            chain.verification_status = ChainVerificationStatus(data["verification_status"])

        for dt_field in ["source_timestamp", "destination_timestamp", "started_at",
                         "completed_at", "blockchain_anchor_time"]:
            if isinstance(data.get(dt_field), str):
                setattr(chain, dt_field, datetime.fromisoformat(data[dt_field]))

        return chain

    def to_dot(self) -> str:
        """
        Generate DOT graph representation of the custody chain.

        Returns:
            DOT language string for visualization
        """
        lines = ["digraph CustodyChain {"]
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box, style=filled];")

        # Source node
        source_label = f"Source\\n{self.source_type.value}\\n{self.source_id[:30]}..."
        lines.append(f'    source [label="{source_label}", fillcolor=lightblue];')

        # Step nodes
        prev_node = "source"
        for step in self.steps:
            node_id = f"step_{step.step_number}"
            step_type = step.step_type.value if isinstance(step.step_type, CustodyStepType) else str(step.step_type)

            # Color based on validation
            if not step.validation_passed:
                color = "lightcoral"
            elif step.validation_warnings:
                color = "lightyellow"
            else:
                color = "lightgreen"

            label = f"Step {step.step_number}\\n{step_type}\\nv{step.schema_version}"
            if step.model_version:
                label += f"\\nmodel: {step.model_version}"

            lines.append(f'    {node_id} [label="{label}", fillcolor={color}];')
            lines.append(f"    {prev_node} -> {node_id};")
            prev_node = node_id

        # Destination node
        if self.is_complete:
            dest_label = f"Destination\\n{self.destination_type}\\n{self.destination_id[:30]}..."
            lines.append(f'    destination [label="{dest_label}", fillcolor=lightblue];')
            lines.append(f"    {prev_node} -> destination;")

        lines.append("}")
        return "\n".join(lines)


# =============================================================================
# BLOCKCHAIN INTEGRATION (OPTIONAL)
# =============================================================================

class BlockchainAnchor:
    """
    Optional blockchain integration for immutable custody records.

    This is a placeholder for actual blockchain integration.
    Real implementation would connect to Ethereum, Hyperledger, etc.
    """

    def __init__(self, endpoint: Optional[str] = None, enabled: bool = False):
        """
        Initialize blockchain anchor.

        Args:
            endpoint: Blockchain RPC endpoint
            enabled: Whether blockchain integration is enabled
        """
        self.endpoint = endpoint
        self.enabled = enabled

        if enabled:
            logger.info(f"BlockchainAnchor initialized: endpoint={endpoint}")
        else:
            logger.info("BlockchainAnchor disabled (simulation mode)")

    def anchor_chain(self, chain: CustodyChain) -> Optional[str]:
        """
        Anchor a custody chain to the blockchain.

        Args:
            chain: Custody chain to anchor

        Returns:
            Transaction ID if successful, None otherwise
        """
        if not self.enabled:
            # Simulation mode - generate fake tx ID
            tx_id = f"SIM-TX-{uuid.uuid4().hex[:16].upper()}"
            logger.info(f"[SIMULATED] Anchored chain {chain.chain_id} with tx {tx_id}")
            return tx_id

        # Real blockchain integration would go here
        # This is a placeholder for actual implementation
        logger.warning("Real blockchain integration not implemented")
        return None

    def verify_anchor(self, chain: CustodyChain) -> bool:
        """
        Verify a chain's blockchain anchor.

        Args:
            chain: Custody chain to verify

        Returns:
            True if anchor is valid
        """
        if not chain.blockchain_anchored or not chain.blockchain_anchor_tx:
            return False

        if not self.enabled:
            # Simulation mode - always return True
            logger.info(f"[SIMULATED] Verified anchor for chain {chain.chain_id}")
            return True

        # Real verification would go here
        return False


# =============================================================================
# CHAIN OF CUSTODY MANAGER
# =============================================================================

class InsulationChainOfCustody:
    """
    Chain of Custody Manager for GL-015 Insulscan.

    Provides high-level API for tracking data custody through the
    entire insulation assessment pipeline from thermal scan to recommendation.

    Usage:
        >>> custody = InsulationChainOfCustody()
        >>> custody.start_chain(
        ...     source_type=DataSource.THERMAL_CAMERA,
        ...     source_id="CAM-FLIR-001",
        ...     asset_id="INSUL-001"
        ... )
        >>> custody.add_scan_step(scan_data, camera_model="FLIR-T640")
        >>> custody.add_analysis_step(ComputationType.HEAT_LOSS, inputs, outputs)
        >>> custody.add_recommendation_step(recommendation)
        >>> custody.add_cmms_step(work_order_id)
        >>> chain = custody.finalize()
    """

    VERSION = "1.0.0"

    def __init__(self, blockchain_enabled: bool = False, blockchain_endpoint: Optional[str] = None):
        """
        Initialize chain of custody manager.

        Args:
            blockchain_enabled: Enable blockchain integration
            blockchain_endpoint: Blockchain RPC endpoint
        """
        self._current_chain: Optional[CustodyChain] = None
        self._chains: Dict[str, CustodyChain] = {}
        self._blockchain = BlockchainAnchor(endpoint=blockchain_endpoint, enabled=blockchain_enabled)

        logger.info("InsulationChainOfCustody manager initialized")

    def start_chain(
        self,
        source_type: DataSource,
        source_id: str,
        asset_id: str,
        source_model: str = "",
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Start a new custody chain.

        Args:
            source_type: Type of data source
            source_id: Identifier of source
            asset_id: Insulation asset identifier
            source_model: Model of source device (e.g., camera model)
            correlation_id: Optional correlation ID

        Returns:
            Chain ID
        """
        self._current_chain = CustodyChain(
            source_type=source_type,
            source_id=source_id,
            source_model=source_model,
            asset_id=asset_id,
            correlation_id=correlation_id or str(uuid.uuid4()),
            source_timestamp=datetime.now(timezone.utc),
        )

        self._chains[self._current_chain.chain_id] = self._current_chain

        logger.info(
            f"Started custody chain {self._current_chain.chain_id} for asset {asset_id}"
        )

        return self._current_chain.chain_id

    def add_scan_step(
        self,
        scan_data: Any,
        camera_model: str = "",
        schema_version: str = "1.0.0",
        record_count: int = 1,
        validation_passed: bool = True,
        validation_errors: Optional[List[str]] = None,
        image_hash: str = "",
    ) -> DetailedCustodyStep:
        """
        Add a thermal scan step to the current chain.

        Args:
            scan_data: Thermal scan data
            camera_model: Thermal camera model
            schema_version: Version of data schema
            record_count: Number of scan records
            validation_passed: Whether validation passed
            validation_errors: List of validation errors
            image_hash: Hash of thermal image

        Returns:
            The created custody step
        """
        self._ensure_chain_active()

        step = DetailedCustodyStep(
            step_type=CustodyStepType.THERMAL_CAMERA_SCAN,
            source_system=self._current_chain.source_type.value,
            target_system="scan_processor",
            source_component=camera_model,
            input_data_hash=compute_sha256(scan_data),
            output_data_hash=image_hash or compute_sha256(scan_data),
            input_record_count=record_count,
            output_record_count=record_count,
            schema_version=schema_version,
            transformation_applied="capture_thermal_scan",
            transformation_params={"camera_model": camera_model},
            validation_passed=validation_passed,
            validation_errors=validation_errors or [],
        )

        self._current_chain.add_step(step)
        return step

    def add_validation_step(
        self,
        input_data: Any,
        output_data: Any,
        schema_version: str = "1.0.0",
        validation_rules_version: str = "1.0.0",
        validation_passed: bool = True,
        validation_errors: Optional[List[str]] = None,
        validation_warnings: Optional[List[str]] = None,
        data_quality_score: float = 1.0,
    ) -> DetailedCustodyStep:
        """
        Add a data validation step.

        Args:
            input_data: Data before validation
            output_data: Data after validation
            schema_version: Schema version used
            validation_rules_version: Version of validation rules
            validation_passed: Whether validation passed
            validation_errors: Validation errors found
            validation_warnings: Validation warnings found
            data_quality_score: Data quality score 0-1

        Returns:
            The created custody step
        """
        self._ensure_chain_active()

        step = DetailedCustodyStep(
            step_type=CustodyStepType.SCHEMA_VALIDATION,
            source_system="scan_processor",
            target_system="validated_data",
            input_data_hash=compute_sha256(input_data),
            output_data_hash=compute_sha256(output_data),
            schema_version=schema_version,
            transformation_version=validation_rules_version,
            transformation_applied="validate_schema",
            validation_passed=validation_passed,
            validation_errors=validation_errors or [],
            validation_warnings=validation_warnings or [],
            data_quality_score=data_quality_score,
        )

        self._current_chain.add_step(step)
        return step

    def add_analysis_step(
        self,
        analysis_type: ComputationType,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        algorithm_version: str = "1.0.0",
        formula_reference: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> DetailedCustodyStep:
        """
        Add an analysis step (heat loss, R-value, etc.).

        Args:
            analysis_type: Type of analysis
            inputs: Analysis inputs
            outputs: Analysis outputs
            algorithm_version: Version of algorithm
            formula_reference: Reference to formula used
            duration_ms: Analysis duration

        Returns:
            The created custody step
        """
        self._ensure_chain_active()

        # Map analysis type to step type
        step_type_map = {
            ComputationType.HEAT_LOSS: CustodyStepType.HEAT_LOSS_ANALYSIS,
            ComputationType.R_VALUE_CALCULATION: CustodyStepType.R_VALUE_CALCULATION,
            ComputationType.U_VALUE_CALCULATION: CustodyStepType.U_VALUE_CALCULATION,
            ComputationType.THERMAL_BRIDGE_ANALYSIS: CustodyStepType.THERMAL_BRIDGE_DETECTION,
            ComputationType.HOTSPOT_DETECTION: CustodyStepType.HOTSPOT_DETECTION,
            ComputationType.ANOMALY_DETECTION: CustodyStepType.ANOMALY_DETECTION,
            ComputationType.DEGRADATION_ASSESSMENT: CustodyStepType.CONDITION_ASSESSMENT,
        }

        step_type = step_type_map.get(analysis_type, CustodyStepType.HEAT_LOSS_ANALYSIS)

        step = DetailedCustodyStep(
            step_type=step_type,
            source_system="validated_data",
            target_system="analysis_engine",
            input_data_hash=compute_sha256(inputs),
            output_data_hash=compute_sha256(outputs),
            algorithm_version=algorithm_version,
            transformation_applied=analysis_type.value,
            formula_reference=formula_reference,
            processing_duration_ms=duration_ms,
        )

        step.processing_started_at = datetime.now(timezone.utc)
        step.processing_completed_at = datetime.now(timezone.utc)

        self._current_chain.add_step(step)
        return step

    def add_prediction_step(
        self,
        prediction_type: str,
        prediction_value: Any,
        model_id: str,
        model_version: str,
        inputs: Dict[str, Any],
        confidence: float = 0.0,
        training_data_hash: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> DetailedCustodyStep:
        """
        Add a prediction step.

        Args:
            prediction_type: Type of prediction
            prediction_value: Prediction output
            model_id: Model identifier
            model_version: Model version
            inputs: Model input features
            confidence: Prediction confidence
            training_data_hash: Hash of training data
            duration_ms: Inference duration

        Returns:
            The created custody step
        """
        self._ensure_chain_active()

        # Map prediction type to step type
        step_type_map = {
            "degradation": CustodyStepType.DEGRADATION_PREDICTION,
            "rul": CustodyStepType.RUL_PREDICTION,
            "energy_savings": CustodyStepType.ENERGY_SAVINGS_PREDICTION,
            "maintenance": CustodyStepType.MAINTENANCE_PREDICTION,
        }

        step_type = step_type_map.get(prediction_type.lower(), CustodyStepType.MODEL_INFERENCE)

        outputs = {
            "prediction": prediction_value,
            "confidence": confidence,
        }

        step = DetailedCustodyStep(
            step_type=step_type,
            source_system="analysis_engine",
            target_system="prediction_engine",
            input_data_hash=compute_sha256(inputs),
            output_data_hash=compute_sha256(outputs),
            model_version=model_version,
            transformation_applied=f"predict_{prediction_type}",
            transformation_params={
                "model_id": model_id,
                "training_data_hash": training_data_hash,
                "confidence": confidence,
            },
            processing_duration_ms=duration_ms,
        )

        step.processing_started_at = datetime.now(timezone.utc)
        step.processing_completed_at = datetime.now(timezone.utc)

        self._current_chain.add_step(step)
        return step

    def add_recommendation_step(
        self,
        recommendation_type: str,
        recommendation: str,
        priority: str,
        supporting_data_hash: str,
        threshold_values: Optional[Dict[str, float]] = None,
        roi_estimate: Optional[float] = None,
    ) -> DetailedCustodyStep:
        """
        Add a recommendation generation step.

        Args:
            recommendation_type: Type of recommendation
            recommendation: Recommendation text
            priority: Priority level
            supporting_data_hash: Hash of supporting data
            threshold_values: Threshold values that triggered recommendation
            roi_estimate: ROI estimate if calculated

        Returns:
            The created custody step
        """
        self._ensure_chain_active()

        outputs = {
            "recommendation_type": recommendation_type,
            "recommendation": recommendation,
            "priority": priority,
            "roi_estimate": roi_estimate,
        }

        step = DetailedCustodyStep(
            step_type=CustodyStepType.RECOMMENDATION_GENERATION,
            source_system="prediction_engine",
            target_system="recommendation_engine",
            input_data_hash=supporting_data_hash,
            output_data_hash=compute_sha256(outputs),
            transformation_applied=f"generate_{recommendation_type}_recommendation",
            transformation_params={
                "priority": priority,
                "threshold_values": threshold_values or {},
                "roi_estimate": roi_estimate,
            },
        )

        self._current_chain.add_step(step)
        return step

    def add_cmms_step(
        self,
        work_order_id: str,
        cmms_system: str,
        work_order_type: str,
        recommendation_id: str,
    ) -> DetailedCustodyStep:
        """
        Add a CMMS work order creation step.

        Args:
            work_order_id: Work order identifier
            cmms_system: CMMS system name
            work_order_type: Type of work order
            recommendation_id: Source recommendation ID

        Returns:
            The created custody step
        """
        self._ensure_chain_active()

        outputs = {
            "work_order_id": work_order_id,
            "cmms_system": cmms_system,
            "work_order_type": work_order_type,
        }

        step = DetailedCustodyStep(
            step_type=CustodyStepType.CMMS_WORK_ORDER,
            source_system="recommendation_engine",
            target_system=cmms_system,
            source_component="recommendation_dispatcher",
            target_component="work_order_api",
            input_data_hash=compute_sha256({"recommendation_id": recommendation_id}),
            output_data_hash=compute_sha256(outputs),
            transformation_applied="create_work_order",
            transformation_params={
                "recommendation_id": recommendation_id,
                "work_order_type": work_order_type,
            },
        )

        self._current_chain.add_step(step)
        return step

    def finalize(self, anchor_to_blockchain: bool = False) -> CustodyChain:
        """
        Finalize the current custody chain.

        Args:
            anchor_to_blockchain: Whether to anchor to blockchain

        Returns:
            The completed custody chain

        Raises:
            ValueError: If no active chain
        """
        self._ensure_chain_active()

        # Determine destination from last step
        last_step = self._current_chain.steps[-1]
        destination_type = last_step.target_system

        # Get destination ID based on step type
        if last_step.step_type == CustodyStepType.CMMS_WORK_ORDER:
            destination_id = last_step.transformation_params.get("recommendation_id", "")
        else:
            destination_id = last_step.output_data_hash[:16]

        self._current_chain.complete(destination_type, destination_id)

        # Verify the chain
        self._current_chain.verify()

        # Optionally anchor to blockchain
        if anchor_to_blockchain:
            tx_id = self._blockchain.anchor_chain(self._current_chain)
            if tx_id:
                self._current_chain.blockchain_anchored = True
                self._current_chain.blockchain_anchor_tx = tx_id
                self._current_chain.blockchain_anchor_time = datetime.now(timezone.utc)

        completed_chain = self._current_chain
        self._current_chain = None

        return completed_chain

    def get_chain(self, chain_id: str) -> Optional[CustodyChain]:
        """Get a chain by ID."""
        return self._chains.get(chain_id)

    def get_all_chains_for_asset(self, asset_id: str) -> List[CustodyChain]:
        """Get all chains for a specific asset."""
        return [
            chain for chain in self._chains.values()
            if chain.asset_id == asset_id
        ]

    def verify_chain(self, chain_id: str) -> ChainVerificationStatus:
        """
        Verify a chain by ID.

        Args:
            chain_id: Chain identifier

        Returns:
            Verification status

        Raises:
            ValueError: If chain not found
        """
        chain = self._chains.get(chain_id)
        if chain is None:
            raise ValueError(f"Chain {chain_id} not found")

        return chain.verify()

    def detect_tampering(self, chain_id: str) -> List[str]:
        """
        Detect tampering in a chain.

        Args:
            chain_id: Chain identifier

        Returns:
            List of tampered step IDs

        Raises:
            ValueError: If chain not found
        """
        chain = self._chains.get(chain_id)
        if chain is None:
            raise ValueError(f"Chain {chain_id} not found")

        return chain.detect_tampering()

    def _ensure_chain_active(self) -> None:
        """Ensure there is an active chain."""
        if self._current_chain is None:
            raise ValueError("No active custody chain. Call start_chain() first.")

    def export_chain_lineage(
        self,
        chain_id: str,
        format: str = "json",
    ) -> str:
        """
        Export chain lineage for visualization or reporting.

        Args:
            chain_id: Chain identifier
            format: Export format ("json" or "dot")

        Returns:
            Exported lineage data

        Raises:
            ValueError: If chain not found
        """
        chain = self._chains.get(chain_id)
        if chain is None:
            raise ValueError(f"Chain {chain_id} not found")

        if format == "dot":
            return chain.to_dot()
        else:
            return json.dumps(chain.to_dict(), indent=2, default=str)

    def get_provenance_visualization(self, chain_id: str) -> str:
        """
        Get DOT graph visualization of data provenance.

        Args:
            chain_id: Chain identifier

        Returns:
            DOT language string

        Raises:
            ValueError: If chain not found
        """
        return self.export_chain_lineage(chain_id, format="dot")
