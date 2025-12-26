# -*- coding: utf-8 -*-
"""
GL-015 Insulscan - Audit Schemas

This module defines all Pydantic v2 data models for the audit and lineage tracking
subsystem, supporting complete traceability from thermal scan data to insulation
assessment recommendations with immutable audit trails and regulatory compliance.

Standards Compliance:
    - ISO 50001:2018 (Energy Management Systems)
    - ASHRAE Standards (Building Insulation)
    - ASTM C1060 (Thermographic Inspection of Insulation)
    - EPA 40 CFR Part 98 (Greenhouse Gas Reporting)
    - 21 CFR Part 11 (Electronic Records and Signatures)

Zero-Hallucination Principle:
    - Complete provenance tracking via SHA-256 hash chains
    - Deterministic calculation recording with formula references
    - Immutable audit entries with tamper detection
    - Full traceability from thermal scans to maintenance work orders

Example:
    >>> from audit.schemas import AuditEvent, EvidenceRecord, ChainOfCustodyEntry
    >>> event = AuditEvent(
    ...     event_type=AuditEventType.CALCULATION_EXECUTED,
    ...     actor="system",
    ...     resource_id="INSUL-001",
    ...     action="heat_loss_calculation"
    ... )
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
    """Types of computations performed by Insulscan."""

    # Heat loss calculations
    HEAT_LOSS = "heat_loss"
    SURFACE_HEAT_TRANSFER = "surface_heat_transfer"
    CONDUCTION_LOSS = "conduction_loss"
    CONVECTION_LOSS = "convection_loss"
    RADIATION_LOSS = "radiation_loss"

    # Insulation analysis
    R_VALUE_CALCULATION = "r_value_calculation"
    U_VALUE_CALCULATION = "u_value_calculation"
    THERMAL_CONDUCTIVITY = "thermal_conductivity"
    INSULATION_THICKNESS = "insulation_thickness"
    THERMAL_BRIDGE_ANALYSIS = "thermal_bridge_analysis"

    # Performance assessment
    EFFICIENCY_CALCULATION = "efficiency_calculation"
    ENERGY_SAVINGS_ESTIMATE = "energy_savings_estimate"
    DEGRADATION_ASSESSMENT = "degradation_assessment"
    REMAINING_USEFUL_LIFE = "remaining_useful_life"

    # Thermal imaging analysis
    THERMAL_IMAGE_PROCESSING = "thermal_image_processing"
    HOTSPOT_DETECTION = "hotspot_detection"
    TEMPERATURE_MAPPING = "temperature_mapping"
    ANOMALY_DETECTION = "anomaly_detection"

    # Compliance calculations
    ASHRAE_COMPLIANCE_CHECK = "ashrae_compliance_check"
    ISO_50001_COMPLIANCE = "iso_50001_compliance"
    REGULATORY_CALCULATION = "regulatory_calculation"

    # Optimization
    INSULATION_OPTIMIZATION = "insulation_optimization"
    ROI_CALCULATION = "roi_calculation"
    COST_BENEFIT_ANALYSIS = "cost_benefit_analysis"

    # Safety validation
    SAFETY_VALIDATION = "safety_validation"
    TEMPERATURE_LIMIT_CHECK = "temperature_limit_check"

    # Explainability
    SHAP_EXPLANATION = "shap_explanation"
    LIME_EXPLANATION = "lime_explanation"
    FEATURE_IMPORTANCE = "feature_importance"


class AuditEventType(str, Enum):
    """Types of audit events for insulation scanning operations."""

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
    THERMAL_SCAN_COMPLETED = "thermal_scan_completed"
    INSULATION_ASSESSMENT = "insulation_assessment"
    MAINTENANCE_TRIGGERED = "maintenance_triggered"


class DataSource(str, Enum):
    """Sources of data in the insulation scanning system."""

    THERMAL_CAMERA = "thermal_camera"
    TEMPERATURE_SENSOR = "temperature_sensor"
    OPC_UA_TAG = "opc_ua_tag"
    HISTORIAN = "historian"
    DCS = "dcs"
    SCADA = "scada"
    MANUAL_ENTRY = "manual_entry"
    BUILDING_MANAGEMENT_SYSTEM = "building_management_system"
    WEATHER_API = "weather_api"
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
    ALERT = "alert"
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


class CustodyType(str, Enum):
    """Types of custody transfers in data processing pipeline."""

    INGESTION = "ingestion"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    CALCULATION = "calculation"
    PREDICTION = "prediction"
    RECOMMENDATION = "recommendation"
    ARCHIVAL = "archival"
    EXPORT = "export"


class ComplianceStandard(str, Enum):
    """Supported compliance standards for insulation assessment."""

    ISO_50001 = "iso_50001"
    ASHRAE_90_1 = "ashrae_90_1"
    ASHRAE_189_1 = "ashrae_189_1"
    ASTM_C1060 = "astm_c1060"
    ASTM_C680 = "astm_c680"
    EPA_40_CFR_98 = "epa_40_cfr_98"
    IECC = "iecc"


class ComplianceRequirementStatus(str, Enum):
    """Status of compliance requirement."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    PENDING_REVIEW = "pending_review"


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
# CORE AUDIT SCHEMAS - PYDANTIC V2 MODELS
# =============================================================================

class AuditEvent(BaseModel):
    """
    Pydantic v2 model for audit events.

    Captures all details of an auditable operation with cryptographic
    verification for insulation scanning and assessment operations.

    Attributes:
        event_id: Unique identifier for this audit event
        timestamp: UTC timestamp with microsecond precision
        event_type: Type of audit event
        actor: Who/what performed the action
        resource_id: Identifier of affected resource (asset, scan, etc.)
        action: Action performed
        outcome: Result of the action (success/failure)
    """

    model_config = ConfigDict(
        use_enum_values=False,
        validate_assignment=True,
        frozen=True,
    )

    event_id: str = Field(
        default_factory=lambda: f"EVT-{uuid.uuid4().hex[:12].upper()}",
        description="Unique identifier for this audit event"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp with microsecond precision"
    )
    event_type: AuditEventType = Field(
        ...,
        description="Type of audit event"
    )
    actor: str = Field(
        ...,
        min_length=1,
        description="Actor who performed the action"
    )
    actor_type: ActorType = Field(
        default=ActorType.SYSTEM,
        description="Type of actor"
    )
    resource_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of affected resource"
    )
    resource_type: str = Field(
        default="insulation_asset",
        description="Type of resource affected"
    )
    action: str = Field(
        ...,
        min_length=1,
        description="Action performed"
    )
    outcome: str = Field(
        default="success",
        description="Result of the action"
    )
    severity: SeverityLevel = Field(
        default=SeverityLevel.INFO,
        description="Severity level of the event"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event details"
    )
    correlation_id: str = Field(
        default="",
        description="Correlation ID for tracing related events"
    )
    session_id: str = Field(
        default="",
        description="Session identifier"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash after initialization."""
        if not self.provenance_hash:
            hash_value = self._calculate_provenance_hash()
            object.__setattr__(self, 'provenance_hash', hash_value)

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 provenance hash for this event."""
        hash_data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value if isinstance(self.event_type, AuditEventType) else str(self.event_type),
            "actor": self.actor,
            "resource_id": self.resource_id,
            "action": self.action,
            "outcome": self.outcome,
        }
        return compute_sha256(hash_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value if isinstance(self.event_type, AuditEventType) else str(self.event_type),
            "actor": self.actor,
            "actor_type": self.actor_type.value if isinstance(self.actor_type, ActorType) else str(self.actor_type),
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "action": self.action,
            "outcome": self.outcome,
            "severity": self.severity.value if isinstance(self.severity, SeverityLevel) else str(self.severity),
            "details": self.details,
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "provenance_hash": self.provenance_hash,
        }


class EvidenceRecord(BaseModel):
    """
    Pydantic v2 model for evidence records.

    Captures calculation evidence with cryptographic hashing for
    audit trail verification in insulation assessments.

    Attributes:
        evidence_id: Unique identifier for this evidence record
        calculation_type: Type of calculation performed
        inputs_hash: SHA-256 hash of all input data
        outputs_hash: SHA-256 hash of all output data
        formula_version: Version of formula/algorithm used
    """

    model_config = ConfigDict(
        use_enum_values=False,
        validate_assignment=True,
    )

    evidence_id: str = Field(
        default_factory=lambda: f"EVID-{uuid.uuid4().hex[:12].upper()}",
        description="Unique identifier for this evidence record"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the evidence was captured"
    )
    calculation_type: ComputationType = Field(
        ...,
        description="Type of calculation performed"
    )
    asset_id: str = Field(
        default="",
        description="Insulation asset identifier"
    )
    inputs_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of all input data"
    )
    outputs_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of all output data"
    )
    formula_version: str = Field(
        default="1.0.0",
        description="Version of formula/algorithm used"
    )
    formula_reference: str = Field(
        default="",
        description="Reference to formula specification"
    )
    inputs_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of input values"
    )
    outputs_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of output values"
    )
    calculation_trace: List[str] = Field(
        default_factory=list,
        description="Step-by-step calculation trace"
    )
    model_id: Optional[str] = Field(
        default=None,
        description="ML model identifier if prediction"
    )
    model_version: Optional[str] = Field(
        default=None,
        description="ML model version if prediction"
    )
    training_data_hash: Optional[str] = Field(
        default=None,
        description="Hash of training data if ML model"
    )
    reproducibility_verified: bool = Field(
        default=False,
        description="Whether calculation was verified reproducible"
    )
    verification_timestamp: Optional[datetime] = Field(
        default=None,
        description="When reproducibility was verified"
    )
    provenance_chain: List[str] = Field(
        default_factory=list,
        description="Chain of hashes linking to source data"
    )
    regulatory_format: Optional[str] = Field(
        default=None,
        description="Export format for regulatory submission"
    )
    retention_years: int = Field(
        default=7,
        ge=1,
        le=100,
        description="Retention period in years"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for complete provenance"
    )

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance_hash()

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 provenance hash for this record."""
        hash_data = {
            "evidence_id": self.evidence_id,
            "timestamp": self.timestamp.isoformat(),
            "calculation_type": self.calculation_type.value if isinstance(self.calculation_type, ComputationType) else str(self.calculation_type),
            "inputs_hash": self.inputs_hash,
            "outputs_hash": self.outputs_hash,
            "formula_version": self.formula_version,
        }
        return compute_sha256(hash_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "evidence_id": self.evidence_id,
            "timestamp": self.timestamp.isoformat(),
            "calculation_type": self.calculation_type.value if isinstance(self.calculation_type, ComputationType) else str(self.calculation_type),
            "asset_id": self.asset_id,
            "inputs_hash": self.inputs_hash,
            "outputs_hash": self.outputs_hash,
            "formula_version": self.formula_version,
            "formula_reference": self.formula_reference,
            "inputs_summary": self.inputs_summary,
            "outputs_summary": self.outputs_summary,
            "calculation_trace": self.calculation_trace,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "training_data_hash": self.training_data_hash,
            "reproducibility_verified": self.reproducibility_verified,
            "verification_timestamp": self.verification_timestamp.isoformat() if self.verification_timestamp else None,
            "provenance_chain": self.provenance_chain,
            "regulatory_format": self.regulatory_format,
            "retention_years": self.retention_years,
            "provenance_hash": self.provenance_hash,
        }


class ChainOfCustodyEntry(BaseModel):
    """
    Pydantic v2 model for chain of custody entries.

    Tracks data movement through the processing pipeline with
    immutable audit trail and tamper detection.

    Attributes:
        entry_id: Unique identifier for this entry
        asset_id: Identifier of the asset being tracked
        custody_type: Type of custody transfer
        transferred_from: Source of the transfer
        transferred_to: Destination of the transfer
        timestamp: When the transfer occurred
    """

    model_config = ConfigDict(
        use_enum_values=False,
        validate_assignment=True,
    )

    entry_id: str = Field(
        default_factory=lambda: f"COC-{uuid.uuid4().hex[:12].upper()}",
        description="Unique identifier for this entry"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the transfer occurred"
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the asset being tracked"
    )
    custody_type: CustodyType = Field(
        ...,
        description="Type of custody transfer"
    )
    transferred_from: str = Field(
        ...,
        min_length=1,
        description="Source of the transfer"
    )
    transferred_to: str = Field(
        ...,
        min_length=1,
        description="Destination of the transfer"
    )
    data_hash: str = Field(
        default="",
        description="SHA-256 hash of data at this point"
    )
    previous_entry_id: Optional[str] = Field(
        default=None,
        description="Link to previous entry in chain"
    )
    previous_hash: Optional[str] = Field(
        default=None,
        description="Hash of previous entry for chain verification"
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Version of data schema"
    )
    transformation_applied: str = Field(
        default="",
        description="Transformation applied during transfer"
    )
    validation_passed: bool = Field(
        default=True,
        description="Whether validation passed"
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Validation errors if any"
    )
    operator_id: Optional[str] = Field(
        default=None,
        description="Operator who performed transfer"
    )
    correlation_id: str = Field(
        default="",
        description="Correlation ID for tracing"
    )
    blockchain_tx_id: Optional[str] = Field(
        default=None,
        description="Blockchain transaction ID if registered"
    )
    immutable_sealed: bool = Field(
        default=False,
        description="Whether entry has been sealed as immutable"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for provenance"
    )

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance_hash()

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 provenance hash for this entry."""
        hash_data = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "asset_id": self.asset_id,
            "custody_type": self.custody_type.value if isinstance(self.custody_type, CustodyType) else str(self.custody_type),
            "transferred_from": self.transferred_from,
            "transferred_to": self.transferred_to,
            "data_hash": self.data_hash,
            "previous_hash": self.previous_hash,
        }
        return compute_sha256(hash_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "asset_id": self.asset_id,
            "custody_type": self.custody_type.value if isinstance(self.custody_type, CustodyType) else str(self.custody_type),
            "transferred_from": self.transferred_from,
            "transferred_to": self.transferred_to,
            "data_hash": self.data_hash,
            "previous_entry_id": self.previous_entry_id,
            "previous_hash": self.previous_hash,
            "schema_version": self.schema_version,
            "transformation_applied": self.transformation_applied,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "operator_id": self.operator_id,
            "correlation_id": self.correlation_id,
            "blockchain_tx_id": self.blockchain_tx_id,
            "immutable_sealed": self.immutable_sealed,
            "provenance_hash": self.provenance_hash,
        }


class ComplianceRecord(BaseModel):
    """
    Pydantic v2 model for compliance records.

    Tracks compliance with ISO 50001, ASHRAE, and other standards
    for insulation assessment operations.

    Attributes:
        record_id: Unique identifier for this record
        standard: Compliance standard (ISO 50001, ASHRAE, etc.)
        requirement: Specific requirement being assessed
        status: Compliance status
        evidence_refs: References to supporting evidence
    """

    model_config = ConfigDict(
        use_enum_values=False,
        validate_assignment=True,
    )

    record_id: str = Field(
        default_factory=lambda: f"COMP-{uuid.uuid4().hex[:12].upper()}",
        description="Unique identifier for this record"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When compliance was assessed"
    )
    standard: ComplianceStandard = Field(
        ...,
        description="Compliance standard (ISO 50001, ASHRAE, etc.)"
    )
    standard_version: str = Field(
        default="latest",
        description="Version of the standard"
    )
    requirement: str = Field(
        ...,
        min_length=1,
        description="Specific requirement being assessed"
    )
    requirement_id: str = Field(
        default="",
        description="Formal identifier of the requirement"
    )
    section: str = Field(
        default="",
        description="Section of the standard"
    )
    status: ComplianceRequirementStatus = Field(
        ...,
        description="Compliance status"
    )
    compliance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Compliance score 0-1"
    )
    asset_id: str = Field(
        default="",
        description="Asset being assessed"
    )
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="References to supporting evidence"
    )
    assessment_method: str = Field(
        default="automated",
        description="How compliance was assessed"
    )
    assessor_id: Optional[str] = Field(
        default=None,
        description="Who performed the assessment"
    )
    findings: str = Field(
        default="",
        description="Detailed findings"
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Identified compliance gaps"
    )
    remediation_required: bool = Field(
        default=False,
        description="Whether remediation is required"
    )
    remediation_deadline: Optional[datetime] = Field(
        default=None,
        description="Deadline for remediation"
    )
    remediation_status: str = Field(
        default="",
        description="Current status of remediation"
    )
    next_review_date: Optional[datetime] = Field(
        default=None,
        description="When next review is due"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for provenance"
    )

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance_hash()

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 provenance hash for this record."""
        hash_data = {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "standard": self.standard.value if isinstance(self.standard, ComplianceStandard) else str(self.standard),
            "requirement": self.requirement,
            "status": self.status.value if isinstance(self.status, ComplianceRequirementStatus) else str(self.status),
            "evidence_refs": self.evidence_refs,
        }
        return compute_sha256(hash_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "standard": self.standard.value if isinstance(self.standard, ComplianceStandard) else str(self.standard),
            "standard_version": self.standard_version,
            "requirement": self.requirement,
            "requirement_id": self.requirement_id,
            "section": self.section,
            "status": self.status.value if isinstance(self.status, ComplianceRequirementStatus) else str(self.status),
            "compliance_score": self.compliance_score,
            "asset_id": self.asset_id,
            "evidence_refs": self.evidence_refs,
            "assessment_method": self.assessment_method,
            "assessor_id": self.assessor_id,
            "findings": self.findings,
            "gaps": self.gaps,
            "remediation_required": self.remediation_required,
            "remediation_deadline": self.remediation_deadline.isoformat() if self.remediation_deadline else None,
            "remediation_status": self.remediation_status,
            "next_review_date": self.next_review_date.isoformat() if self.next_review_date else None,
            "provenance_hash": self.provenance_hash,
        }


# =============================================================================
# DATACLASS-BASED MODELS FOR INTERNAL USE
# =============================================================================

@dataclass
class ProvenanceChain:
    """
    Chain of computation records for complete traceability.

    Maintains an ordered, hash-linked chain of records that enables
    verification of the entire computation history from raw input
    to final output.
    """

    chain_id: str = ""
    asset_id: str = ""
    records: List[EvidenceRecord] = field(default_factory=list)
    hash_links: List[str] = field(default_factory=list)
    merkle_root: str = ""
    chain_hash: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finalized_at: Optional[datetime] = None
    is_finalized: bool = False
    verification_status: ChainVerificationStatus = ChainVerificationStatus.PENDING

    def __post_init__(self) -> None:
        """Initialize chain ID if not provided."""
        if not self.chain_id:
            self.chain_id = f"PROV-{uuid.uuid4().hex[:12].upper()}"

    def add_record(self, record: EvidenceRecord) -> None:
        """Add a record to the provenance chain."""
        if self.is_finalized:
            raise ValueError("Cannot add records to a finalized chain")

        self.records.append(record)

        prev_hash = self.hash_links[-1] if self.hash_links else "genesis"
        link_hash = compute_sha256(f"{prev_hash}:{record.provenance_hash}")
        self.hash_links.append(link_hash)

    def finalize(self) -> str:
        """Finalize the chain and compute final hashes."""
        if not self.records:
            raise ValueError("Cannot finalize empty chain")

        self.is_finalized = True
        self.finalized_at = datetime.now(timezone.utc)
        self.merkle_root = self._compute_merkle_root()
        self.chain_hash = self._compute_chain_hash()
        self.verification_status = ChainVerificationStatus.VALID

        return self.chain_hash

    def verify(self) -> ChainVerificationStatus:
        """Verify the integrity of the entire chain."""
        if not self.records:
            return ChainVerificationStatus.INCOMPLETE

        prev_hash = "genesis"
        for i, record in enumerate(self.records):
            expected_link = compute_sha256(f"{prev_hash}:{record.provenance_hash}")
            if i < len(self.hash_links) and self.hash_links[i] != expected_link:
                return ChainVerificationStatus.BROKEN
            prev_hash = self.hash_links[i] if i < len(self.hash_links) else expected_link

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
            "asset_id": self.asset_id,
            "records": [r.to_dict() for r in self.records],
            "hash_links": self.hash_links,
            "merkle_root": self.merkle_root,
            "chain_hash": self.chain_hash,
            "created_at": self.created_at.isoformat(),
            "finalized_at": self.finalized_at.isoformat() if self.finalized_at else None,
            "is_finalized": self.is_finalized,
            "verification_status": self.verification_status.value,
        }


@dataclass
class AuditStatistics:
    """Summary statistics for audit reporting."""

    total_records: int = 0
    records_by_type: Dict[str, int] = field(default_factory=dict)
    records_by_asset: Dict[str, int] = field(default_factory=dict)
    records_by_actor: Dict[str, int] = field(default_factory=dict)
    records_by_day: Dict[str, int] = field(default_factory=dict)
    first_record_time: Optional[datetime] = None
    last_record_time: Optional[datetime] = None
    chain_count: int = 0
    verified_chains: int = 0
    broken_chains: int = 0
    compliance_records: int = 0
    compliant_count: int = 0
    non_compliant_count: int = 0
    avg_computation_time_ms: float = 0.0
    total_computation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_records": self.total_records,
            "records_by_type": self.records_by_type,
            "records_by_asset": self.records_by_asset,
            "records_by_actor": self.records_by_actor,
            "records_by_day": self.records_by_day,
            "first_record_time": self.first_record_time.isoformat() if self.first_record_time else None,
            "last_record_time": self.last_record_time.isoformat() if self.last_record_time else None,
            "chain_count": self.chain_count,
            "verified_chains": self.verified_chains,
            "broken_chains": self.broken_chains,
            "compliance_records": self.compliance_records,
            "compliant_count": self.compliant_count,
            "non_compliant_count": self.non_compliant_count,
            "avg_computation_time_ms": self.avg_computation_time_ms,
            "total_computation_time_ms": self.total_computation_time_ms,
        }


# =============================================================================
# INSULATION-SPECIFIC SCHEMAS
# =============================================================================

class InsulationAssetRecord(BaseModel):
    """Record of an insulation asset for audit tracking."""

    model_config = ConfigDict(use_enum_values=False)

    asset_id: str = Field(..., description="Unique asset identifier")
    asset_name: str = Field(default="", description="Asset name/description")
    asset_type: str = Field(default="pipe_insulation", description="Type of insulation")
    location: str = Field(default="", description="Physical location")
    installation_date: Optional[datetime] = Field(default=None)
    insulation_material: str = Field(default="", description="Material type")
    nominal_thickness_mm: float = Field(default=0.0, ge=0)
    nominal_r_value: float = Field(default=0.0, ge=0)
    design_temperature_c: float = Field(default=0.0)
    operating_temperature_c: float = Field(default=0.0)
    ambient_temperature_c: float = Field(default=20.0)
    surface_area_m2: float = Field(default=0.0, ge=0)
    last_inspection_date: Optional[datetime] = Field(default=None)
    condition_score: float = Field(default=1.0, ge=0, le=1)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            self.provenance_hash = compute_sha256({
                "asset_id": self.asset_id,
                "asset_type": self.asset_type,
                "insulation_material": self.insulation_material,
                "nominal_thickness_mm": self.nominal_thickness_mm,
            })


class ThermalScanRecord(BaseModel):
    """Record of a thermal scan for audit tracking."""

    model_config = ConfigDict(use_enum_values=False)

    scan_id: str = Field(
        default_factory=lambda: f"SCAN-{uuid.uuid4().hex[:12].upper()}",
        description="Unique scan identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When scan was performed"
    )
    asset_id: str = Field(..., description="Asset scanned")
    operator_id: str = Field(default="system", description="Who performed scan")
    camera_model: str = Field(default="", description="Thermal camera model")
    camera_serial: str = Field(default="", description="Camera serial number")
    calibration_date: Optional[datetime] = Field(default=None)
    ambient_temperature_c: float = Field(default=20.0)
    humidity_percent: float = Field(default=50.0, ge=0, le=100)
    wind_speed_mps: float = Field(default=0.0, ge=0)
    emissivity_setting: float = Field(default=0.95, ge=0, le=1)
    min_temperature_c: float = Field(default=0.0)
    max_temperature_c: float = Field(default=0.0)
    avg_temperature_c: float = Field(default=0.0)
    hotspot_count: int = Field(default=0, ge=0)
    anomaly_detected: bool = Field(default=False)
    image_hash: str = Field(default="", description="SHA-256 of thermal image")
    raw_data_hash: str = Field(default="", description="SHA-256 of raw data")
    quality_score: float = Field(default=1.0, ge=0, le=1)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            self.provenance_hash = compute_sha256({
                "scan_id": self.scan_id,
                "timestamp": self.timestamp.isoformat(),
                "asset_id": self.asset_id,
                "image_hash": self.image_hash,
                "raw_data_hash": self.raw_data_hash,
            })
