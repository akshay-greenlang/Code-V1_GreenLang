# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Audit Schemas

This module defines all data models for the audit and lineage tracking subsystem,
supporting complete traceability from raw sensor data to final recommendations
with immutable audit trails and regulatory compliance for heat exchanger operations.

Standards Compliance:
    - TEMA (Tubular Exchanger Manufacturers Association) standards
    - ASME PTC 12.5 (Single Phase Heat Exchangers)
    - ISO 50001:2018 (Energy Management Systems)
    - EPA 40 CFR Part 98 (Greenhouse Gas Reporting)
    - 21 CFR Part 11 (Electronic Records and Signatures)

Zero-Hallucination Principle:
    - Complete provenance tracking via SHA-256 hash chains
    - Deterministic calculation recording with formula references
    - Immutable audit entries with tamper detection
    - Full traceability from OPC-UA tags to CMMS work orders

Example:
    >>> from audit.schemas import AuditRecord, ProvenanceChain
    >>> record = AuditRecord(
    ...     record_id="AUD-001",
    ...     computation_type=ComputationType.HEAT_TRANSFER,
    ...     inputs_hash="abc123...",
    ...     outputs_hash="def456..."
    ... )
    >>> chain = ProvenanceChain(chain_id="CHN-001")
    >>> chain.add_record(record)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, ConfigDict


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ComputationType(str, Enum):
    """Types of computations performed by Exchangerpro."""

    # Heat transfer calculations
    HEAT_TRANSFER = "heat_transfer"
    LMTD_CALCULATION = "lmtd_calculation"
    NTU_EFFECTIVENESS = "ntu_effectiveness"
    OVERALL_HEAT_TRANSFER_COEFF = "overall_htc"
    FOULING_RESISTANCE = "fouling_resistance"

    # Thermal-hydraulic analysis
    PRESSURE_DROP = "pressure_drop"
    FLOW_DISTRIBUTION = "flow_distribution"
    TEMPERATURE_PROFILE = "temperature_profile"
    DUTY_CALCULATION = "duty_calculation"

    # Performance analysis
    EFFECTIVENESS_CALCULATION = "effectiveness"
    APPROACH_TEMPERATURE = "approach_temperature"
    THERMAL_EFFICIENCY = "thermal_efficiency"
    EXERGY_ANALYSIS = "exergy_analysis"

    # Predictive analytics
    FOULING_PREDICTION = "fouling_prediction"
    CLEANING_OPTIMIZATION = "cleaning_optimization"
    REMAINING_USEFUL_LIFE = "remaining_useful_life"
    PERFORMANCE_DEGRADATION = "performance_degradation"

    # Optimization
    OPERATING_POINT_OPTIMIZATION = "operating_point_optimization"
    CLEANING_SCHEDULE_OPTIMIZATION = "cleaning_schedule_optimization"
    ENERGY_OPTIMIZATION = "energy_optimization"
    COST_OPTIMIZATION = "cost_optimization"

    # Safety and compliance
    SAFETY_VALIDATION = "safety_validation"
    TEMA_COMPLIANCE_CHECK = "tema_compliance"
    REGULATORY_CALCULATION = "regulatory_calculation"

    # Explainability
    SHAP_EXPLANATION = "shap_explanation"
    LIME_EXPLANATION = "lime_explanation"
    FEATURE_IMPORTANCE = "feature_importance"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"


class AuditEventType(str, Enum):
    """Types of audit events for heat exchanger operations."""

    DATA_INGESTION = "data_ingestion"
    DATA_VALIDATION = "data_validation"
    DATA_TRANSFORMATION = "data_transformation"
    CALCULATION_EXECUTED = "calculation_executed"
    PREDICTION_GENERATED = "prediction_generated"
    RECOMMENDATION_ISSUED = "recommendation_issued"
    WORK_ORDER_CREATED = "work_order_created"
    SAFETY_CHECK_PERFORMED = "safety_check"
    COMPLIANCE_VERIFIED = "compliance_verified"
    CONFIGURATION_CHANGED = "config_change"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    MODEL_INFERENCE = "model_inference"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    CLEANING_TRIGGERED = "cleaning_triggered"


class DataSource(str, Enum):
    """Sources of data in the heat exchanger system."""

    OPC_UA_TAG = "opc_ua_tag"
    HISTORIAN = "historian"
    DCS = "dcs"
    SCADA = "scada"
    MANUAL_ENTRY = "manual_entry"
    LAB_ANALYSIS = "lab_analysis"
    CMMS = "cmms"
    ERP = "erp"
    CALCULATED = "calculated"
    INFERRED = "inferred"
    MODEL_PREDICTION = "model_prediction"


class ApprovalStatus(str, Enum):
    """Status of change approvals."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"
    ESCALATED = "escalated"
    EXPIRED = "expired"


class ChainVerificationStatus(str, Enum):
    """Status of provenance chain verification."""

    VALID = "valid"
    INVALID = "invalid"
    BROKEN = "broken"
    INCOMPLETE = "incomplete"
    PENDING = "pending"


class SeverityLevel(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ActorType(str, Enum):
    """Types of actors that can perform auditable actions."""

    USER = "user"
    SYSTEM = "system"
    AGENT = "agent"
    SCHEDULER = "scheduler"
    API = "api"
    INTEGRATION = "integration"
    MODEL = "model"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _serialize_value(value: Any) -> Any:
    """
    Serialize value for consistent hashing.

    Ensures deterministic JSON serialization for SHA-256 hash computation.

    Args:
        value: Any value to serialize

    Returns:
        Serialized value suitable for JSON encoding
    """
    if value is None:
        return None
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in sorted(value.items())}
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    elif isinstance(value, Decimal):
        return str(value)
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, Enum):
        return value.value
    elif isinstance(value, bytes):
        return value.hex()
    return value


def compute_sha256(data: Any) -> str:
    """
    Compute SHA-256 hash of data.

    Args:
        data: Data to hash (dict, list, or primitive)

    Returns:
        Hexadecimal SHA-256 hash string
    """
    if isinstance(data, (dict, list)):
        json_str = json.dumps(_serialize_value(data), sort_keys=True, default=str)
    elif isinstance(data, bytes):
        json_str = data.decode('utf-8', errors='replace')
    else:
        json_str = str(data)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


# =============================================================================
# CORE AUDIT SCHEMAS
# =============================================================================

@dataclass(frozen=True)
class AuditRecord:
    """
    Immutable audit record for a single computation or action.

    This is the fundamental unit of audit tracking in Exchangerpro.
    Each record captures a complete snapshot of an operation with
    cryptographic integrity verification.

    Attributes:
        record_id: Unique identifier for this audit record
        computation_type: Type of computation performed
        timestamp: UTC timestamp with microsecond precision
        inputs_hash: SHA-256 hash of all input data
        outputs_hash: SHA-256 hash of all output data
        algorithm_version: Version of algorithm/formula used
        actor_id: Who/what initiated this computation
        actor_type: Type of actor (user, system, agent)
        correlation_id: Links related operations together
        parent_record_id: Previous record in chain (if any)
        provenance_hash: Hash linking to complete provenance

    Example:
        >>> record = AuditRecord(
        ...     record_id="AUD-20241227-001",
        ...     computation_type=ComputationType.LMTD_CALCULATION,
        ...     inputs_hash=compute_sha256(inputs),
        ...     outputs_hash=compute_sha256(outputs),
        ...     algorithm_version="LMTD_v1.2.0"
        ... )
    """

    record_id: str
    computation_type: ComputationType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    inputs_hash: str = ""
    outputs_hash: str = ""
    algorithm_version: str = "1.0.0"

    # Actor information
    actor_id: str = "system"
    actor_type: ActorType = ActorType.SYSTEM

    # Traceability
    correlation_id: str = ""
    parent_record_id: Optional[str] = None

    # Computation details (immutable snapshot)
    input_summary: Dict[str, Any] = field(default_factory=dict)
    output_summary: Dict[str, Any] = field(default_factory=dict)
    formula_reference: str = ""
    calculation_trace: List[str] = field(default_factory=list)

    # Provenance
    provenance_hash: str = ""

    # Metadata
    session_id: str = ""
    request_id: str = ""
    exchanger_id: str = ""
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        """Compute provenance hash after initialization."""
        if not self.provenance_hash:
            # Use object.__setattr__ for frozen dataclass
            object.__setattr__(self, 'provenance_hash', self._calculate_provenance_hash())

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 provenance hash for this record."""
        hash_data = {
            "record_id": self.record_id,
            "computation_type": self.computation_type.value if isinstance(self.computation_type, ComputationType) else str(self.computation_type),
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "inputs_hash": self.inputs_hash,
            "outputs_hash": self.outputs_hash,
            "algorithm_version": self.algorithm_version,
            "actor_id": self.actor_id,
            "parent_record_id": self.parent_record_id,
            "formula_reference": self.formula_reference,
        }
        return compute_sha256(hash_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "computation_type": self.computation_type.value if isinstance(self.computation_type, ComputationType) else str(self.computation_type),
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "inputs_hash": self.inputs_hash,
            "outputs_hash": self.outputs_hash,
            "algorithm_version": self.algorithm_version,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type.value if isinstance(self.actor_type, ActorType) else str(self.actor_type),
            "correlation_id": self.correlation_id,
            "parent_record_id": self.parent_record_id,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "formula_reference": self.formula_reference,
            "calculation_trace": self.calculation_trace,
            "provenance_hash": self.provenance_hash,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "exchanger_id": self.exchanger_id,
            "duration_ms": self.duration_ms,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditRecord":
        """Create AuditRecord from dictionary."""
        # Handle enum conversion
        if isinstance(data.get("computation_type"), str):
            data["computation_type"] = ComputationType(data["computation_type"])
        if isinstance(data.get("actor_type"), str):
            data["actor_type"] = ActorType(data["actor_type"])
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class ProvenanceChain:
    """
    Chain of computation records for complete traceability.

    Maintains an ordered, hash-linked chain of AuditRecords that
    enables verification of the entire computation history from
    raw input to final output.

    Attributes:
        chain_id: Unique identifier for this chain
        records: Ordered list of audit records
        merkle_root: Merkle tree root hash for efficient verification
        chain_hash: Hash of the complete chain
        created_at: When the chain was started
        finalized_at: When the chain was sealed

    Example:
        >>> chain = ProvenanceChain(chain_id="CHN-001")
        >>> chain.add_record(ingestion_record)
        >>> chain.add_record(validation_record)
        >>> chain.add_record(calculation_record)
        >>> chain.finalize()
        >>> assert chain.verify() == ChainVerificationStatus.VALID
    """

    chain_id: str = ""
    records: List[AuditRecord] = field(default_factory=list)
    hash_links: List[str] = field(default_factory=list)
    merkle_root: str = ""
    chain_hash: str = ""

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finalized_at: Optional[datetime] = None
    is_finalized: bool = False

    # Source tracking
    source_type: str = ""
    source_id: str = ""
    exchanger_id: str = ""

    # Final output reference
    final_output_id: str = ""
    final_output_type: str = ""

    def __post_init__(self) -> None:
        """Initialize chain ID if not provided."""
        if not self.chain_id:
            self.chain_id = f"CHN-{uuid.uuid4().hex[:12].upper()}"

    def add_record(self, record: AuditRecord) -> None:
        """
        Add a record to the provenance chain.

        Args:
            record: AuditRecord to add

        Raises:
            ValueError: If chain is already finalized
        """
        if self.is_finalized:
            raise ValueError("Cannot add records to a finalized chain")

        self.records.append(record)

        # Compute hash link
        prev_hash = self.hash_links[-1] if self.hash_links else "genesis"
        link_hash = compute_sha256(f"{prev_hash}:{record.provenance_hash}")
        self.hash_links.append(link_hash)

    def finalize(self) -> str:
        """
        Finalize the chain and compute final hashes.

        Returns:
            The chain hash

        Raises:
            ValueError: If chain has no records
        """
        if not self.records:
            raise ValueError("Cannot finalize empty chain")

        self.is_finalized = True
        self.finalized_at = datetime.now(timezone.utc)
        self.merkle_root = self._compute_merkle_root()
        self.chain_hash = self._compute_chain_hash()

        return self.chain_hash

    def verify(self) -> ChainVerificationStatus:
        """
        Verify the integrity of the entire chain.

        Returns:
            ChainVerificationStatus indicating chain validity
        """
        if not self.records:
            return ChainVerificationStatus.INCOMPLETE

        # Verify each link in the chain
        prev_hash = "genesis"
        for i, record in enumerate(self.records):
            # Verify record's own hash
            expected_hash = record._calculate_provenance_hash()
            if record.provenance_hash != expected_hash:
                return ChainVerificationStatus.INVALID

            # Verify link hash
            expected_link = compute_sha256(f"{prev_hash}:{record.provenance_hash}")
            if i < len(self.hash_links) and self.hash_links[i] != expected_link:
                return ChainVerificationStatus.BROKEN

            prev_hash = self.hash_links[i] if i < len(self.hash_links) else expected_link

        # Verify merkle root if finalized
        if self.is_finalized:
            expected_merkle = self._compute_merkle_root()
            if self.merkle_root != expected_merkle:
                return ChainVerificationStatus.INVALID

        return ChainVerificationStatus.VALID

    def _compute_merkle_root(self) -> str:
        """Compute Merkle tree root hash of all records."""
        if not self.records:
            return compute_sha256(b"")

        leaves = [r.provenance_hash for r in self.records]

        while len(leaves) > 1:
            if len(leaves) % 2:
                leaves.append(leaves[-1])  # Duplicate last if odd

            leaves = [
                compute_sha256(leaves[i] + leaves[i + 1])
                for i in range(0, len(leaves), 2)
            ]

        return leaves[0]

    def _compute_chain_hash(self) -> str:
        """Compute hash of the complete chain."""
        chain_data = {
            "chain_id": self.chain_id,
            "record_count": len(self.records),
            "hash_links": self.hash_links,
            "merkle_root": self.merkle_root,
            "created_at": self.created_at.isoformat(),
            "finalized_at": self.finalized_at.isoformat() if self.finalized_at else None,
        }
        return compute_sha256(chain_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chain_id": self.chain_id,
            "records": [r.to_dict() for r in self.records],
            "hash_links": self.hash_links,
            "merkle_root": self.merkle_root,
            "chain_hash": self.chain_hash,
            "created_at": self.created_at.isoformat(),
            "finalized_at": self.finalized_at.isoformat() if self.finalized_at else None,
            "is_finalized": self.is_finalized,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "exchanger_id": self.exchanger_id,
            "final_output_id": self.final_output_id,
            "final_output_type": self.final_output_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceChain":
        """Create ProvenanceChain from dictionary."""
        records = [AuditRecord.from_dict(r) for r in data.get("records", [])]

        chain = cls(
            chain_id=data.get("chain_id", ""),
            records=records,
            hash_links=data.get("hash_links", []),
            merkle_root=data.get("merkle_root", ""),
            chain_hash=data.get("chain_hash", ""),
            source_type=data.get("source_type", ""),
            source_id=data.get("source_id", ""),
            exchanger_id=data.get("exchanger_id", ""),
            final_output_id=data.get("final_output_id", ""),
            final_output_type=data.get("final_output_type", ""),
        )

        if data.get("created_at"):
            chain.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("finalized_at"):
            chain.finalized_at = datetime.fromisoformat(data["finalized_at"])
        chain.is_finalized = data.get("is_finalized", False)

        return chain


@dataclass
class ChangeRecord:
    """
    Record of a change requiring approval or tracking.

    Used to track configuration changes, threshold updates, and
    any modifications that require audit trail or approval workflow.

    Attributes:
        change_id: Unique identifier for this change
        change_type: Category of change
        what_changed: Description of what was modified
        old_value: Previous value (before change)
        new_value: New value (after change)
        changed_by: User/system that made the change
        approved_by: User who approved (if applicable)
        approval_status: Current approval status

    Example:
        >>> change = ChangeRecord(
        ...     change_id="CHG-001",
        ...     change_type="threshold_update",
        ...     what_changed="fouling_alert_threshold",
        ...     old_value=0.0005,
        ...     new_value=0.0004,
        ...     changed_by="engineer@company.com",
        ...     justification="Reduced threshold based on new TEMA guidelines"
        ... )
    """

    change_id: str = ""
    change_type: str = ""
    what_changed: str = ""

    # Values
    old_value: Any = None
    new_value: Any = None
    old_value_hash: str = ""
    new_value_hash: str = ""

    # Actor tracking
    changed_by: str = ""
    changed_by_type: ActorType = ActorType.USER
    approved_by: Optional[str] = None
    approved_by_type: Optional[ActorType] = None

    # Timestamps
    changed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_at: Optional[datetime] = None
    effective_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Approval workflow
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    approval_required: bool = True
    auto_approval_rule: Optional[str] = None

    # Context
    justification: str = ""
    impact_assessment: str = ""
    rollback_plan: str = ""

    # Traceability
    correlation_id: str = ""
    related_changes: List[str] = field(default_factory=list)
    affected_exchangers: List[str] = field(default_factory=list)

    # Provenance
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Initialize computed fields."""
        if not self.change_id:
            self.change_id = f"CHG-{uuid.uuid4().hex[:12].upper()}"

        if not self.old_value_hash and self.old_value is not None:
            self.old_value_hash = compute_sha256(self.old_value)

        if not self.new_value_hash and self.new_value is not None:
            self.new_value_hash = compute_sha256(self.new_value)

        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance_hash()

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 provenance hash for this change."""
        hash_data = {
            "change_id": self.change_id,
            "change_type": self.change_type,
            "what_changed": self.what_changed,
            "old_value_hash": self.old_value_hash,
            "new_value_hash": self.new_value_hash,
            "changed_by": self.changed_by,
            "changed_at": self.changed_at.isoformat() if isinstance(self.changed_at, datetime) else str(self.changed_at),
            "justification": self.justification,
        }
        return compute_sha256(hash_data)

    def approve(self, approved_by: str, approved_by_type: ActorType = ActorType.USER) -> None:
        """
        Approve this change.

        Args:
            approved_by: Identifier of approver
            approved_by_type: Type of approver
        """
        self.approved_by = approved_by
        self.approved_by_type = approved_by_type
        self.approved_at = datetime.now(timezone.utc)
        self.approval_status = ApprovalStatus.APPROVED

        if self.effective_at is None:
            self.effective_at = self.approved_at

    def reject(self, rejected_by: str, reason: str = "") -> None:
        """
        Reject this change.

        Args:
            rejected_by: Identifier of rejector
            reason: Reason for rejection
        """
        self.approved_by = rejected_by
        self.approved_at = datetime.now(timezone.utc)
        self.approval_status = ApprovalStatus.REJECTED
        if reason:
            self.justification = f"{self.justification} | REJECTED: {reason}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "change_id": self.change_id,
            "change_type": self.change_type,
            "what_changed": self.what_changed,
            "old_value": _serialize_value(self.old_value),
            "new_value": _serialize_value(self.new_value),
            "old_value_hash": self.old_value_hash,
            "new_value_hash": self.new_value_hash,
            "changed_by": self.changed_by,
            "changed_by_type": self.changed_by_type.value if isinstance(self.changed_by_type, ActorType) else str(self.changed_by_type),
            "approved_by": self.approved_by,
            "approved_by_type": self.approved_by_type.value if self.approved_by_type else None,
            "changed_at": self.changed_at.isoformat() if isinstance(self.changed_at, datetime) else str(self.changed_at),
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "effective_at": self.effective_at.isoformat() if self.effective_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "approval_status": self.approval_status.value if isinstance(self.approval_status, ApprovalStatus) else str(self.approval_status),
            "approval_required": self.approval_required,
            "auto_approval_rule": self.auto_approval_rule,
            "justification": self.justification,
            "impact_assessment": self.impact_assessment,
            "rollback_plan": self.rollback_plan,
            "correlation_id": self.correlation_id,
            "related_changes": self.related_changes,
            "affected_exchangers": self.affected_exchangers,
            "provenance_hash": self.provenance_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChangeRecord":
        """Create ChangeRecord from dictionary."""
        # Handle enum conversion
        if isinstance(data.get("changed_by_type"), str):
            data["changed_by_type"] = ActorType(data["changed_by_type"])
        if isinstance(data.get("approved_by_type"), str):
            data["approved_by_type"] = ActorType(data["approved_by_type"])
        if isinstance(data.get("approval_status"), str):
            data["approval_status"] = ApprovalStatus(data["approval_status"])

        # Handle datetime conversion
        for dt_field in ["changed_at", "approved_at", "effective_at", "expires_at"]:
            if isinstance(data.get(dt_field), str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])

        return cls(**data)


# =============================================================================
# CUSTODY STEP SCHEMAS
# =============================================================================

@dataclass
class CustodyStep:
    """
    Single step in the chain of custody.

    Tracks data movement from one stage to another in the pipeline:
    OPC-UA -> Canonical Schema -> Computation -> Prediction -> Recommendation -> CMMS

    Attributes:
        step_id: Unique identifier for this step
        step_type: Type of processing step
        source_system: Origin system
        target_system: Destination system
        data_hash: Hash of data at this step
        transformation_applied: What transformation was applied
        version_info: Version information for schemas/models used
    """

    step_id: str = ""
    step_number: int = 0
    step_type: str = ""

    # System tracking
    source_system: str = ""
    target_system: str = ""

    # Data integrity
    input_data_hash: str = ""
    output_data_hash: str = ""

    # Transformation details
    transformation_applied: str = ""
    transformation_version: str = "1.0.0"
    schema_version: str = "1.0.0"
    model_version: Optional[str] = None

    # Timing
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: Optional[datetime] = None
    forwarded_at: Optional[datetime] = None

    # Quality
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)
    data_quality_score: float = 1.0

    # Provenance
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Initialize computed fields."""
        if not self.step_id:
            self.step_id = f"STEP-{uuid.uuid4().hex[:8].upper()}"

        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance_hash()

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 provenance hash for this step."""
        hash_data = {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "step_type": self.step_type,
            "source_system": self.source_system,
            "target_system": self.target_system,
            "input_data_hash": self.input_data_hash,
            "output_data_hash": self.output_data_hash,
            "transformation_applied": self.transformation_applied,
            "transformation_version": self.transformation_version,
            "received_at": self.received_at.isoformat() if isinstance(self.received_at, datetime) else str(self.received_at),
        }
        return compute_sha256(hash_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "step_type": self.step_type,
            "source_system": self.source_system,
            "target_system": self.target_system,
            "input_data_hash": self.input_data_hash,
            "output_data_hash": self.output_data_hash,
            "transformation_applied": self.transformation_applied,
            "transformation_version": self.transformation_version,
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "received_at": self.received_at.isoformat() if isinstance(self.received_at, datetime) else str(self.received_at),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "forwarded_at": self.forwarded_at.isoformat() if self.forwarded_at else None,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "data_quality_score": self.data_quality_score,
            "provenance_hash": self.provenance_hash,
        }


# =============================================================================
# PYDANTIC MODELS FOR API/VALIDATION
# =============================================================================

class AuditRecordModel(BaseModel):
    """Pydantic model for AuditRecord validation."""

    model_config = ConfigDict(use_enum_values=True)

    record_id: str = Field(..., description="Unique identifier for this audit record")
    computation_type: ComputationType = Field(..., description="Type of computation performed")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    inputs_hash: str = Field("", description="SHA-256 hash of all input data")
    outputs_hash: str = Field("", description="SHA-256 hash of all output data")
    algorithm_version: str = Field("1.0.0", description="Version of algorithm/formula used")
    actor_id: str = Field("system", description="Who/what initiated this computation")
    actor_type: ActorType = Field(ActorType.SYSTEM, description="Type of actor")
    correlation_id: str = Field("", description="Links related operations")
    parent_record_id: Optional[str] = Field(None, description="Previous record in chain")
    exchanger_id: str = Field("", description="Heat exchanger identifier")
    duration_ms: float = Field(0.0, ge=0, description="Computation duration in milliseconds")


class ChangeRecordModel(BaseModel):
    """Pydantic model for ChangeRecord validation."""

    model_config = ConfigDict(use_enum_values=True)

    change_id: str = Field(..., description="Unique identifier for this change")
    change_type: str = Field(..., description="Category of change")
    what_changed: str = Field(..., description="Description of what was modified")
    old_value: Any = Field(None, description="Previous value")
    new_value: Any = Field(None, description="New value")
    changed_by: str = Field(..., description="User/system that made the change")
    justification: str = Field("", description="Reason for the change")
    approval_required: bool = Field(True, description="Whether approval is needed")
    affected_exchangers: List[str] = Field(default_factory=list)

    @field_validator("what_changed")
    @classmethod
    def validate_what_changed(cls, v: str) -> str:
        """Ensure what_changed is not empty."""
        if not v.strip():
            raise ValueError("what_changed cannot be empty")
        return v


class ProvenanceChainModel(BaseModel):
    """Pydantic model for ProvenanceChain summary."""

    model_config = ConfigDict(use_enum_values=True)

    chain_id: str = Field(..., description="Unique identifier for this chain")
    record_count: int = Field(0, ge=0, description="Number of records in chain")
    merkle_root: str = Field("", description="Merkle tree root hash")
    chain_hash: str = Field("", description="Hash of the complete chain")
    is_finalized: bool = Field(False, description="Whether chain is sealed")
    verification_status: ChainVerificationStatus = Field(
        ChainVerificationStatus.PENDING,
        description="Current verification status"
    )
    exchanger_id: str = Field("", description="Associated heat exchanger")
    source_type: str = Field("", description="Original data source type")


# =============================================================================
# AUDIT STATISTICS AND REPORTING
# =============================================================================

class AuditStatistics(BaseModel):
    """Summary statistics for audit reporting."""

    total_records: int = Field(0, description="Total number of audit records")
    records_by_type: Dict[str, int] = Field(default_factory=dict)
    records_by_exchanger: Dict[str, int] = Field(default_factory=dict)
    records_by_actor: Dict[str, int] = Field(default_factory=dict)
    records_by_day: Dict[str, int] = Field(default_factory=dict)

    first_record_time: Optional[datetime] = Field(None)
    last_record_time: Optional[datetime] = Field(None)

    chain_count: int = Field(0, description="Number of provenance chains")
    verified_chains: int = Field(0, description="Number of verified chains")
    broken_chains: int = Field(0, description="Number of broken chains")

    change_count: int = Field(0, description="Number of change records")
    pending_approvals: int = Field(0, description="Number of pending approvals")

    avg_computation_time_ms: float = Field(0.0)
    total_computation_time_ms: float = Field(0.0)


class ComplianceStatus(BaseModel):
    """Compliance status for regulatory reporting."""

    model_config = ConfigDict(use_enum_values=True)

    framework: str = Field(..., description="Regulatory framework (TEMA, EPA, etc.)")
    is_compliant: bool = Field(True, description="Overall compliance status")
    compliance_score: float = Field(1.0, ge=0, le=1, description="Compliance score 0-1")

    checks_performed: int = Field(0, description="Number of checks performed")
    checks_passed: int = Field(0, description="Number of checks passed")
    checks_failed: int = Field(0, description="Number of checks failed")

    violations: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

    last_check_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    next_check_due: Optional[datetime] = Field(None)

    evidence_pack_id: Optional[str] = Field(None, description="Associated evidence pack")
