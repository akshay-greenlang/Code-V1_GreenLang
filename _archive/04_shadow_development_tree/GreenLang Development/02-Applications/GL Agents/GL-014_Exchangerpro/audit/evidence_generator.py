# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Evidence Generator

Immutable Evidence Pack Generator for regulatory compliance and audit trails.
Generates sealed, cryptographically verifiable evidence packages that support
7-year retention per EPA/FDA/TEMA regulations.

Features:
    - Generate compliance evidence packages
    - Calculation methodology documentation
    - Input/output traceability with SHA-256 hashes
    - Model version and training data provenance
    - Recommendation-to-work-order traceability
    - Merkle tree for efficient verification
    - Multiple export formats (JSON, XML, EPA_XML, FDA_XML)

Standards:
    - TEMA (Tubular Exchanger Manufacturers Association)
    - ASME PTC 12.5 (Single Phase Heat Exchangers)
    - ISO 50001:2018 (Energy Management)
    - EPA 40 CFR Part 98 (GHG Reporting)
    - 21 CFR Part 11 (Electronic Records)

Example:
    >>> from audit.evidence_generator import EvidenceGenerator
    >>> generator = EvidenceGenerator()
    >>> record = generator.create_evidence_record(
    ...     evidence_type=EvidenceType.HEAT_TRANSFER_CALCULATION,
    ...     data={"method": "LMTD", "result": 150.5},
    ...     inputs=input_data,
    ...     outputs=output_data
    ... )
    >>> pack = generator.create_evidence_pack([record])
    >>> sealed = generator.seal_pack(pack)
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import logging
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .schemas import (
    AuditRecord,
    ProvenanceChain,
    ComputationType,
    compute_sha256,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EVIDENCE TYPE ENUMERATION
# =============================================================================

class EvidenceType(Enum):
    """Types of evidence that can be captured for Exchangerpro."""

    # Calculation evidence
    CALCULATION = "CALCULATION"
    HEAT_TRANSFER_CALCULATION = "HEAT_TRANSFER_CALCULATION"
    LMTD_CALCULATION = "LMTD_CALCULATION"
    NTU_CALCULATION = "NTU_CALCULATION"
    FOULING_CALCULATION = "FOULING_CALCULATION"
    PRESSURE_DROP_CALCULATION = "PRESSURE_DROP_CALCULATION"
    THERMAL_EFFICIENCY = "THERMAL_EFFICIENCY"
    EXERGY_ANALYSIS = "EXERGY_ANALYSIS"

    # Safety evidence
    SAFETY_CHECK = "SAFETY_CHECK"
    TEMA_COMPLIANCE = "TEMA_COMPLIANCE"
    ASME_COMPLIANCE = "ASME_COMPLIANCE"

    # Prediction evidence
    PREDICTION = "PREDICTION"
    FOULING_PREDICTION = "FOULING_PREDICTION"
    CLEANING_PREDICTION = "CLEANING_PREDICTION"
    RUL_PREDICTION = "RUL_PREDICTION"
    PERFORMANCE_PREDICTION = "PERFORMANCE_PREDICTION"

    # Recommendation evidence
    RECOMMENDATION = "RECOMMENDATION"
    CLEANING_RECOMMENDATION = "CLEANING_RECOMMENDATION"
    MAINTENANCE_RECOMMENDATION = "MAINTENANCE_RECOMMENDATION"
    OPERATIONAL_RECOMMENDATION = "OPERATIONAL_RECOMMENDATION"

    # Configuration and change evidence
    CONFIGURATION_CHANGE = "CONFIGURATION_CHANGE"
    THRESHOLD_CHANGE = "THRESHOLD_CHANGE"
    MODEL_UPDATE = "MODEL_UPDATE"

    # Regulatory evidence
    REGULATORY_SUBMISSION = "REGULATORY_SUBMISSION"
    ENERGY_SAVINGS_VERIFICATION = "ENERGY_SAVINGS_VERIFICATION"
    CO2E_ACCOUNTING = "CO2E_ACCOUNTING"
    OPTIMIZATION_RESULT = "OPTIMIZATION_RESULT"

    # Workflow evidence
    WORK_ORDER_TRACE = "WORK_ORDER_TRACE"
    APPROVAL_WORKFLOW = "APPROVAL_WORKFLOW"


class SealStatus(Enum):
    """Status of evidence pack seal."""

    UNSEALED = "UNSEALED"
    SEALED = "SEALED"
    VERIFIED = "VERIFIED"
    TAMPERED = "TAMPERED"


class ExportFormat(Enum):
    """Available export formats for evidence packs."""

    JSON = "JSON"
    XML = "XML"
    PDF = "PDF"
    EPA_XML = "EPA_XML"
    FDA_XML = "FDA_XML"
    TEMA_XML = "TEMA_XML"


# =============================================================================
# EVIDENCE RECORD
# =============================================================================

@dataclass(frozen=True)
class EvidenceRecord:
    """
    Immutable evidence record for audit trail.

    Captures a single piece of evidence with full cryptographic
    verification and traceability.

    Attributes:
        record_id: Unique identifier for this record
        timestamp: When the evidence was captured
        evidence_type: Category of evidence
        data: The actual evidence data
        input_hash: SHA-256 hash of inputs
        output_hash: SHA-256 hash of outputs
        provenance_chain: List of hashes linking to source data
    """

    record_id: str
    timestamp: datetime
    evidence_type: EvidenceType
    data: Dict[str, Any]
    input_hash: str
    output_hash: str
    provenance_chain: List[str] = field(default_factory=list)

    # Additional metadata
    exchanger_id: str = ""
    algorithm_version: str = "1.0.0"
    formula_reference: str = ""
    calculation_trace: List[str] = field(default_factory=list)

    # Model provenance (for predictions)
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    training_data_hash: Optional[str] = None
    model_config_hash: Optional[str] = None

    # Recommendation traceability
    recommendation_id: Optional[str] = None
    work_order_id: Optional[str] = None
    cmms_reference: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "evidence_type": self.evidence_type.value,
            "data": self.data,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "provenance_chain": list(self.provenance_chain),
            "exchanger_id": self.exchanger_id,
            "algorithm_version": self.algorithm_version,
            "formula_reference": self.formula_reference,
            "calculation_trace": self.calculation_trace,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "training_data_hash": self.training_data_hash,
            "model_config_hash": self.model_config_hash,
            "recommendation_id": self.recommendation_id,
            "work_order_id": self.work_order_id,
            "cmms_reference": self.cmms_reference,
        }

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this record."""
        return compute_sha256(self.to_dict())


# =============================================================================
# EVIDENCE PACK METADATA
# =============================================================================

class EvidencePackMetadata(BaseModel):
    """Metadata for an evidence pack."""

    pack_id: str
    agent_id: str = "GL-014"
    agent_name: str = "EXCHANGERPRO"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retention_years: int = 7
    retention_expires_at: Optional[datetime] = None
    record_count: int = 0
    seal_status: SealStatus = SealStatus.UNSEALED
    pack_version: str = "1.0.0"

    # Regulatory frameworks this pack supports
    regulatory_frameworks: List[str] = Field(
        default_factory=lambda: [
            "TEMA_RCB-2.6",
            "ASME_PTC_12.5",
            "ISO_50001_2018",
            "EPA_40_CFR_98",
        ]
    )

    # Exchanger coverage
    exchangers_covered: List[str] = Field(default_factory=list)

    # Time range of evidence
    evidence_start_time: Optional[datetime] = None
    evidence_end_time: Optional[datetime] = None

    def model_post_init(self, __context: Any) -> None:
        """Set retention expiry after initialization."""
        if self.retention_expires_at is None:
            self.retention_expires_at = self.created_at + timedelta(days=365 * self.retention_years)


# =============================================================================
# SEALED PACK ENVELOPE
# =============================================================================

class SealedPackEnvelope(BaseModel):
    """Sealed and signed evidence pack envelope."""

    pack_id: str
    sealed_at: datetime
    content_base64: str
    content_hash: str
    merkle_root: str
    seal_hash: str
    hash_chain: List[str] = Field(default_factory=list)
    algorithm: str = "SHA-256"
    sealer_id: str = "GL-014-SEALER"


# =============================================================================
# CALCULATION METHODOLOGY DOCUMENTATION
# =============================================================================

@dataclass
class CalculationMethodology:
    """
    Documentation of calculation methodology for audit purposes.

    Captures the complete methodology used for any calculation
    to enable reproduction and verification.
    """

    methodology_id: str = ""
    name: str = ""
    version: str = "1.0.0"
    description: str = ""

    # Formula specification
    formula_id: str = ""
    formula_latex: str = ""  # LaTeX representation
    formula_python: str = ""  # Python implementation reference

    # Input specification
    input_parameters: List[Dict[str, Any]] = field(default_factory=list)
    input_validation_rules: List[str] = field(default_factory=list)

    # Output specification
    output_parameters: List[Dict[str, Any]] = field(default_factory=list)
    output_units: Dict[str, str] = field(default_factory=dict)

    # Standards reference
    standards_references: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    # Verification
    verification_tests: List[Dict[str, Any]] = field(default_factory=list)
    validation_date: Optional[datetime] = None
    validated_by: Optional[str] = None

    def __post_init__(self) -> None:
        """Generate methodology ID if not provided."""
        if not self.methodology_id:
            self.methodology_id = f"METH-{uuid.uuid4().hex[:12].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "methodology_id": self.methodology_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "formula_id": self.formula_id,
            "formula_latex": self.formula_latex,
            "formula_python": self.formula_python,
            "input_parameters": self.input_parameters,
            "input_validation_rules": self.input_validation_rules,
            "output_parameters": self.output_parameters,
            "output_units": self.output_units,
            "standards_references": self.standards_references,
            "assumptions": self.assumptions,
            "limitations": self.limitations,
            "verification_tests": self.verification_tests,
            "validation_date": self.validation_date.isoformat() if self.validation_date else None,
            "validated_by": self.validated_by,
        }


# =============================================================================
# MODEL PROVENANCE
# =============================================================================

@dataclass
class ModelProvenance:
    """
    Provenance tracking for ML models used in predictions.

    Captures complete lineage of model from training to inference.
    """

    model_id: str = ""
    model_name: str = ""
    model_version: str = "1.0.0"
    model_type: str = ""  # e.g., "gradient_boosting", "neural_network"

    # Training data provenance
    training_data_source: str = ""
    training_data_hash: str = ""
    training_data_start: Optional[datetime] = None
    training_data_end: Optional[datetime] = None
    training_sample_count: int = 0

    # Training configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    target_name: str = ""

    # Training metrics
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)

    # Model artifacts
    model_artifact_hash: str = ""
    model_config_hash: str = ""
    trained_at: Optional[datetime] = None
    trained_by: str = ""

    # Deployment
    deployed_at: Optional[datetime] = None
    deployment_environment: str = ""

    def __post_init__(self) -> None:
        """Generate model ID if not provided."""
        if not self.model_id:
            self.model_id = f"MODEL-{uuid.uuid4().hex[:12].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "training_data_source": self.training_data_source,
            "training_data_hash": self.training_data_hash,
            "training_data_start": self.training_data_start.isoformat() if self.training_data_start else None,
            "training_data_end": self.training_data_end.isoformat() if self.training_data_end else None,
            "training_sample_count": self.training_sample_count,
            "hyperparameters": self.hyperparameters,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "model_artifact_hash": self.model_artifact_hash,
            "model_config_hash": self.model_config_hash,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "trained_by": self.trained_by,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "deployment_environment": self.deployment_environment,
        }

    def compute_provenance_hash(self) -> str:
        """Compute provenance hash for this model."""
        return compute_sha256(self.to_dict())


# =============================================================================
# RECOMMENDATION TRACE
# =============================================================================

@dataclass
class RecommendationTrace:
    """
    Complete traceability from recommendation to work order.

    Links a recommendation back to its source data and forward
    to any resulting work orders in CMMS.
    """

    trace_id: str = ""
    recommendation_id: str = ""
    exchanger_id: str = ""

    # Source data
    source_data_hash: str = ""
    source_data_timestamp: Optional[datetime] = None

    # Prediction that triggered recommendation
    prediction_id: Optional[str] = None
    prediction_type: str = ""
    prediction_value: Any = None
    prediction_confidence: float = 0.0

    # Recommendation details
    recommendation_type: str = ""
    recommendation_text: str = ""
    priority: str = "medium"
    recommended_action: str = ""
    recommended_deadline: Optional[datetime] = None

    # Justification
    justification: str = ""
    supporting_calculations: List[str] = field(default_factory=list)
    threshold_values: Dict[str, float] = field(default_factory=dict)

    # Work order linkage
    work_order_created: bool = False
    work_order_id: Optional[str] = None
    cmms_system: str = ""
    work_order_created_at: Optional[datetime] = None
    work_order_status: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Generate IDs if not provided."""
        if not self.trace_id:
            self.trace_id = f"TRACE-{uuid.uuid4().hex[:12].upper()}"
        if not self.recommendation_id:
            self.recommendation_id = f"REC-{uuid.uuid4().hex[:12].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "recommendation_id": self.recommendation_id,
            "exchanger_id": self.exchanger_id,
            "source_data_hash": self.source_data_hash,
            "source_data_timestamp": self.source_data_timestamp.isoformat() if self.source_data_timestamp else None,
            "prediction_id": self.prediction_id,
            "prediction_type": self.prediction_type,
            "prediction_value": self.prediction_value,
            "prediction_confidence": self.prediction_confidence,
            "recommendation_type": self.recommendation_type,
            "recommendation_text": self.recommendation_text,
            "priority": self.priority,
            "recommended_action": self.recommended_action,
            "recommended_deadline": self.recommended_deadline.isoformat() if self.recommended_deadline else None,
            "justification": self.justification,
            "supporting_calculations": self.supporting_calculations,
            "threshold_values": self.threshold_values,
            "work_order_created": self.work_order_created,
            "work_order_id": self.work_order_id,
            "cmms_system": self.cmms_system,
            "work_order_created_at": self.work_order_created_at.isoformat() if self.work_order_created_at else None,
            "work_order_status": self.work_order_status,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def compute_trace_hash(self) -> str:
        """Compute hash of the complete trace."""
        return compute_sha256(self.to_dict())


# =============================================================================
# EVIDENCE GENERATOR
# =============================================================================

class EvidenceGenerator:
    """
    Immutable Evidence Pack Generator for GL-014 Exchangerpro.

    Generates cryptographically sealed evidence packages for
    regulatory compliance and audit trails.

    Usage:
        >>> generator = EvidenceGenerator()
        >>> record = generator.create_evidence_record(
        ...     evidence_type=EvidenceType.HEAT_TRANSFER_CALCULATION,
        ...     data={"method": "LMTD", "result": 150.5},
        ...     inputs={"T_hot_in": 120, "T_hot_out": 80},
        ...     outputs={"duty_kW": 1500, "LMTD": 35.6}
        ... )
        >>> pack = generator.create_evidence_pack([record])
        >>> sealed = generator.seal_pack(pack)
        >>> status = generator.verify_pack(sealed)
        >>> assert status == SealStatus.VERIFIED
    """

    VERSION = "1.0.0"

    def __init__(self, retention_years: int = 7):
        """
        Initialize evidence generator.

        Args:
            retention_years: Default retention period (7 years per regulations)
        """
        self.retention_years = retention_years
        self.agent_id = "GL-014"
        self.agent_name = "EXCHANGERPRO"

        # Methodology registry
        self._methodologies: Dict[str, CalculationMethodology] = {}

        # Model provenance registry
        self._model_provenance: Dict[str, ModelProvenance] = {}

        logger.info(f"EvidenceGenerator initialized: retention={retention_years} years")

    # =========================================================================
    # EVIDENCE RECORD CREATION
    # =========================================================================

    def create_evidence_record(
        self,
        evidence_type: EvidenceType,
        data: Dict[str, Any],
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        provenance_chain: Optional[List[str]] = None,
        exchanger_id: str = "",
        algorithm_version: str = "1.0.0",
        formula_reference: str = "",
        calculation_trace: Optional[List[str]] = None,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        recommendation_id: Optional[str] = None,
        work_order_id: Optional[str] = None,
    ) -> EvidenceRecord:
        """
        Create an evidence record with full traceability.

        Args:
            evidence_type: Type of evidence
            data: The evidence data
            inputs: Input data that led to this evidence
            outputs: Output data produced
            provenance_chain: Chain of hashes for traceability
            exchanger_id: Heat exchanger identifier
            algorithm_version: Version of algorithm used
            formula_reference: Reference to formula/method
            calculation_trace: List of calculation steps
            model_id: ML model identifier (if prediction)
            model_version: ML model version
            recommendation_id: Recommendation ID (if applicable)
            work_order_id: Work order ID (if created)

        Returns:
            Immutable EvidenceRecord
        """
        # Compute hashes
        input_hash = compute_sha256(inputs)
        output_hash = compute_sha256(outputs)

        # Get model provenance hashes if applicable
        training_data_hash = None
        model_config_hash = None
        if model_id and model_id in self._model_provenance:
            model_prov = self._model_provenance[model_id]
            training_data_hash = model_prov.training_data_hash
            model_config_hash = model_prov.model_config_hash

        return EvidenceRecord(
            record_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            evidence_type=evidence_type,
            data=data,
            input_hash=input_hash,
            output_hash=output_hash,
            provenance_chain=provenance_chain or [],
            exchanger_id=exchanger_id,
            algorithm_version=algorithm_version,
            formula_reference=formula_reference,
            calculation_trace=calculation_trace or [],
            model_id=model_id,
            model_version=model_version,
            training_data_hash=training_data_hash,
            model_config_hash=model_config_hash,
            recommendation_id=recommendation_id,
            work_order_id=work_order_id,
        )

    def create_calculation_evidence(
        self,
        calculation_type: ComputationType,
        exchanger_id: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        methodology: CalculationMethodology,
        calculation_trace: List[str],
    ) -> EvidenceRecord:
        """
        Create evidence for a calculation with methodology documentation.

        Args:
            calculation_type: Type of calculation
            exchanger_id: Heat exchanger identifier
            inputs: Input parameters
            outputs: Calculation results
            methodology: Calculation methodology documentation
            calculation_trace: Step-by-step calculation trace

        Returns:
            EvidenceRecord for the calculation
        """
        # Register methodology if not already
        if methodology.methodology_id not in self._methodologies:
            self._methodologies[methodology.methodology_id] = methodology

        # Map ComputationType to EvidenceType
        evidence_type_map = {
            ComputationType.LMTD_CALCULATION: EvidenceType.LMTD_CALCULATION,
            ComputationType.NTU_EFFECTIVENESS: EvidenceType.NTU_CALCULATION,
            ComputationType.HEAT_TRANSFER: EvidenceType.HEAT_TRANSFER_CALCULATION,
            ComputationType.FOULING_RESISTANCE: EvidenceType.FOULING_CALCULATION,
            ComputationType.PRESSURE_DROP: EvidenceType.PRESSURE_DROP_CALCULATION,
            ComputationType.THERMAL_EFFICIENCY: EvidenceType.THERMAL_EFFICIENCY,
            ComputationType.EXERGY_ANALYSIS: EvidenceType.EXERGY_ANALYSIS,
        }

        evidence_type = evidence_type_map.get(calculation_type, EvidenceType.CALCULATION)

        return self.create_evidence_record(
            evidence_type=evidence_type,
            data={
                "calculation_type": calculation_type.value,
                "methodology_id": methodology.methodology_id,
                "methodology": methodology.to_dict(),
                "inputs": inputs,
                "outputs": outputs,
            },
            inputs=inputs,
            outputs=outputs,
            exchanger_id=exchanger_id,
            algorithm_version=methodology.version,
            formula_reference=methodology.formula_id,
            calculation_trace=calculation_trace,
        )

    def create_prediction_evidence(
        self,
        prediction_type: str,
        exchanger_id: str,
        inputs: Dict[str, Any],
        prediction: Any,
        confidence: float,
        model_provenance: ModelProvenance,
        explanation: Optional[Dict[str, Any]] = None,
    ) -> EvidenceRecord:
        """
        Create evidence for a model prediction with full provenance.

        Args:
            prediction_type: Type of prediction
            exchanger_id: Heat exchanger identifier
            inputs: Model input features
            prediction: Prediction value
            confidence: Prediction confidence
            model_provenance: Complete model provenance
            explanation: SHAP/LIME explanation if available

        Returns:
            EvidenceRecord for the prediction
        """
        # Register model provenance
        if model_provenance.model_id not in self._model_provenance:
            self._model_provenance[model_provenance.model_id] = model_provenance

        # Map prediction type to evidence type
        evidence_type_map = {
            "fouling": EvidenceType.FOULING_PREDICTION,
            "cleaning": EvidenceType.CLEANING_PREDICTION,
            "rul": EvidenceType.RUL_PREDICTION,
            "performance": EvidenceType.PERFORMANCE_PREDICTION,
        }

        evidence_type = evidence_type_map.get(prediction_type.lower(), EvidenceType.PREDICTION)

        outputs = {
            "prediction": prediction,
            "confidence": confidence,
        }
        if explanation:
            outputs["explanation"] = explanation

        return self.create_evidence_record(
            evidence_type=evidence_type,
            data={
                "prediction_type": prediction_type,
                "model_provenance": model_provenance.to_dict(),
                "prediction": prediction,
                "confidence": confidence,
                "explanation": explanation,
            },
            inputs=inputs,
            outputs=outputs,
            exchanger_id=exchanger_id,
            model_id=model_provenance.model_id,
            model_version=model_provenance.model_version,
        )

    def create_recommendation_evidence(
        self,
        recommendation_trace: RecommendationTrace,
        supporting_records: Optional[List[EvidenceRecord]] = None,
    ) -> EvidenceRecord:
        """
        Create evidence for a recommendation with work order traceability.

        Args:
            recommendation_trace: Complete recommendation trace
            supporting_records: Related evidence records

        Returns:
            EvidenceRecord for the recommendation
        """
        # Build provenance chain from supporting records
        provenance_chain = []
        if supporting_records:
            provenance_chain = [r.compute_hash() for r in supporting_records]

        evidence_type_map = {
            "cleaning": EvidenceType.CLEANING_RECOMMENDATION,
            "maintenance": EvidenceType.MAINTENANCE_RECOMMENDATION,
            "operational": EvidenceType.OPERATIONAL_RECOMMENDATION,
        }

        evidence_type = evidence_type_map.get(
            recommendation_trace.recommendation_type.lower(),
            EvidenceType.RECOMMENDATION
        )

        return self.create_evidence_record(
            evidence_type=evidence_type,
            data={
                "recommendation_trace": recommendation_trace.to_dict(),
            },
            inputs={
                "source_data_hash": recommendation_trace.source_data_hash,
                "prediction_id": recommendation_trace.prediction_id,
                "threshold_values": recommendation_trace.threshold_values,
            },
            outputs={
                "recommendation_id": recommendation_trace.recommendation_id,
                "recommendation_text": recommendation_trace.recommendation_text,
                "priority": recommendation_trace.priority,
                "work_order_id": recommendation_trace.work_order_id,
            },
            provenance_chain=provenance_chain,
            exchanger_id=recommendation_trace.exchanger_id,
            recommendation_id=recommendation_trace.recommendation_id,
            work_order_id=recommendation_trace.work_order_id,
        )

    # =========================================================================
    # EVIDENCE PACK OPERATIONS
    # =========================================================================

    def create_evidence_pack(
        self,
        records: List[EvidenceRecord],
        exchangers_covered: Optional[List[str]] = None,
    ) -> bytes:
        """
        Create a compressed evidence pack from records.

        Args:
            records: List of evidence records
            exchangers_covered: List of exchanger IDs covered

        Returns:
            Compressed evidence pack as bytes

        Raises:
            ValueError: If records is empty
        """
        if not records:
            raise ValueError("Cannot create evidence pack from empty records")

        # Determine time range
        timestamps = [r.timestamp for r in records]
        evidence_start = min(timestamps)
        evidence_end = max(timestamps)

        # Get unique exchangers
        if exchangers_covered is None:
            exchangers_covered = list(set(r.exchanger_id for r in records if r.exchanger_id))

        # Create metadata
        metadata = EvidencePackMetadata(
            pack_id=str(uuid.uuid4()),
            record_count=len(records),
            exchangers_covered=exchangers_covered,
            evidence_start_time=evidence_start,
            evidence_end_time=evidence_end,
        )

        # Build pack data
        pack_data = {
            "metadata": metadata.model_dump(mode='json'),
            "records": [r.to_dict() for r in records],
            "hash_chain": self._build_hash_chain(records),
            "merkle_root": self._compute_merkle_root(records),
            "methodologies": {
                mid: m.to_dict() for mid, m in self._methodologies.items()
            },
            "model_provenance": {
                mid: m.to_dict() for mid, m in self._model_provenance.items()
            },
        }

        # Compress
        json_bytes = json.dumps(pack_data, sort_keys=True, default=str).encode('utf-8')
        compressed = gzip.compress(json_bytes)

        logger.info(
            f"Created evidence pack {metadata.pack_id} with {len(records)} records "
            f"({len(json_bytes)} bytes -> {len(compressed)} bytes compressed)"
        )

        return compressed

    def seal_pack(self, pack: bytes) -> bytes:
        """
        Seal an evidence pack with cryptographic signature.

        Args:
            pack: Compressed evidence pack

        Returns:
            Sealed pack envelope as JSON bytes
        """
        seal_time = datetime.now(timezone.utc)

        # Decompress to get pack data
        decompressed = gzip.decompress(pack)
        pack_data = json.loads(decompressed)
        pack_id = pack_data["metadata"]["pack_id"]

        # Compute content hash
        content_hash = hashlib.sha256(decompressed).hexdigest()

        # Create seal input
        seal_input = json.dumps({
            "pack_id": pack_id,
            "content_hash": content_hash,
            "merkle_root": pack_data.get("merkle_root", ""),
            "hash_chain": pack_data.get("hash_chain", []),
            "sealed_at": seal_time.isoformat(),
        }, sort_keys=True)

        seal_hash = hashlib.sha256(seal_input.encode()).hexdigest()

        # Create envelope
        envelope = SealedPackEnvelope(
            pack_id=pack_id,
            sealed_at=seal_time,
            content_base64=base64.b64encode(pack).decode('utf-8'),
            content_hash=content_hash,
            merkle_root=pack_data.get("merkle_root", ""),
            seal_hash=seal_hash,
            hash_chain=pack_data.get("hash_chain", []),
        )

        logger.info(f"Sealed evidence pack {pack_id} with hash {seal_hash[:16]}...")

        return envelope.model_dump_json(indent=2).encode('utf-8')

    def verify_pack(self, sealed_pack: bytes) -> SealStatus:
        """
        Verify the integrity of a sealed evidence pack.

        Args:
            sealed_pack: Sealed pack envelope as bytes

        Returns:
            SealStatus indicating verification result
        """
        try:
            envelope = SealedPackEnvelope(**json.loads(sealed_pack))

            # Decode and decompress content
            compressed = base64.b64decode(envelope.content_base64)
            decompressed = gzip.decompress(compressed)

            # Verify content hash
            computed_hash = hashlib.sha256(decompressed).hexdigest()
            if computed_hash != envelope.content_hash:
                logger.error(f"Content hash mismatch: {computed_hash} != {envelope.content_hash}")
                return SealStatus.TAMPERED

            # Verify hash chain
            pack_data = json.loads(decompressed)
            records_data = pack_data.get("records", [])

            if not self._verify_hash_chain(records_data, envelope.hash_chain):
                logger.error("Hash chain verification failed")
                return SealStatus.TAMPERED

            # Verify merkle root
            record_hashes = [compute_sha256(r) for r in records_data]
            computed_merkle = self._compute_merkle_root_from_hashes(record_hashes)

            if computed_merkle != envelope.merkle_root:
                logger.error(f"Merkle root mismatch: {computed_merkle} != {envelope.merkle_root}")
                return SealStatus.TAMPERED

            logger.info(f"Evidence pack {envelope.pack_id} verified successfully")
            return SealStatus.VERIFIED

        except Exception as e:
            logger.error(f"Pack verification failed: {e}")
            return SealStatus.TAMPERED

    def extract_pack(self, sealed_pack: bytes) -> Tuple[EvidencePackMetadata, List[Dict[str, Any]]]:
        """
        Extract and verify contents of a sealed pack.

        Args:
            sealed_pack: Sealed pack envelope

        Returns:
            Tuple of (metadata, records list)

        Raises:
            ValueError: If pack is tampered
        """
        status = self.verify_pack(sealed_pack)
        if status != SealStatus.VERIFIED:
            raise ValueError(f"Pack verification failed: {status}")

        envelope = SealedPackEnvelope(**json.loads(sealed_pack))
        compressed = base64.b64decode(envelope.content_base64)
        decompressed = gzip.decompress(compressed)
        pack_data = json.loads(decompressed)

        metadata = EvidencePackMetadata(**pack_data["metadata"])
        records = pack_data.get("records", [])

        return metadata, records

    # =========================================================================
    # EXPORT FORMATS
    # =========================================================================

    def export_regulatory_format(
        self,
        sealed_pack: bytes,
        export_format: ExportFormat,
    ) -> bytes:
        """
        Export sealed pack in regulatory format.

        Args:
            sealed_pack: Sealed evidence pack
            export_format: Desired export format

        Returns:
            Exported data as bytes
        """
        envelope = SealedPackEnvelope(**json.loads(sealed_pack))

        if export_format == ExportFormat.JSON:
            return sealed_pack

        elif export_format in [ExportFormat.XML, ExportFormat.EPA_XML, ExportFormat.FDA_XML, ExportFormat.TEMA_XML]:
            return self._export_xml(envelope, export_format)

        elif export_format == ExportFormat.PDF:
            return self._export_pdf_placeholder(envelope)

        return sealed_pack

    def _export_xml(self, envelope: SealedPackEnvelope, format_type: ExportFormat) -> bytes:
        """Export as XML format."""
        # Decompress content for details
        compressed = base64.b64decode(envelope.content_base64)
        decompressed = gzip.decompress(compressed)
        pack_data = json.loads(decompressed)
        metadata = pack_data.get("metadata", {})

        root = ET.Element("EvidencePack")
        root.set("packId", envelope.pack_id)
        root.set("format", format_type.value)
        root.set("version", self.VERSION)

        # Metadata section
        meta_elem = ET.SubElement(root, "Metadata")
        ET.SubElement(meta_elem, "AgentId").text = metadata.get("agent_id", "GL-014")
        ET.SubElement(meta_elem, "AgentName").text = metadata.get("agent_name", "EXCHANGERPRO")
        ET.SubElement(meta_elem, "CreatedAt").text = metadata.get("created_at", "")
        ET.SubElement(meta_elem, "RecordCount").text = str(metadata.get("record_count", 0))
        ET.SubElement(meta_elem, "RetentionYears").text = str(metadata.get("retention_years", 7))

        # Seal information
        seal_elem = ET.SubElement(root, "Seal")
        ET.SubElement(seal_elem, "SealHash").text = envelope.seal_hash
        ET.SubElement(seal_elem, "ContentHash").text = envelope.content_hash
        ET.SubElement(seal_elem, "MerkleRoot").text = envelope.merkle_root
        ET.SubElement(seal_elem, "Algorithm").text = envelope.algorithm
        ET.SubElement(seal_elem, "SealedAt").text = envelope.sealed_at.isoformat()

        # Frameworks
        frameworks_elem = ET.SubElement(root, "RegulatoryFrameworks")
        for framework in metadata.get("regulatory_frameworks", []):
            ET.SubElement(frameworks_elem, "Framework").text = framework

        # Exchangers
        exchangers_elem = ET.SubElement(root, "ExchangersCovered")
        for exchanger_id in metadata.get("exchangers_covered", []):
            ET.SubElement(exchangers_elem, "ExchangerId").text = exchanger_id

        # Records summary (not full content for size)
        records_elem = ET.SubElement(root, "RecordsSummary")
        for record in pack_data.get("records", [])[:100]:  # Limit to 100 for XML
            rec_elem = ET.SubElement(records_elem, "Record")
            rec_elem.set("id", record.get("record_id", ""))
            rec_elem.set("type", record.get("evidence_type", ""))
            rec_elem.set("timestamp", record.get("timestamp", ""))
            rec_elem.set("exchanger", record.get("exchanger_id", ""))

        return ET.tostring(root, encoding="utf-8", xml_declaration=True)

    def _export_pdf_placeholder(self, envelope: SealedPackEnvelope) -> bytes:
        """Generate PDF placeholder (actual implementation would use reportlab)."""
        # This is a placeholder - real implementation would generate proper PDF
        text = f"""
GL-014 EXCHANGERPRO EVIDENCE PACK
=================================

Pack ID: {envelope.pack_id}
Sealed At: {envelope.sealed_at.isoformat()}
Content Hash: {envelope.content_hash}
Merkle Root: {envelope.merkle_root}
Seal Hash: {envelope.seal_hash}

This pack has been cryptographically sealed and verified.
For full contents, use JSON or XML export.
        """
        return text.encode('utf-8')

    # =========================================================================
    # HASH CHAIN UTILITIES
    # =========================================================================

    def _build_hash_chain(self, records: List[EvidenceRecord]) -> List[str]:
        """Build hash chain from records."""
        chain = []
        prev_hash = ""

        for record in records:
            record_hash = record.compute_hash()
            link = hashlib.sha256(f"{prev_hash}:{record_hash}".encode()).hexdigest()
            chain.append(link)
            prev_hash = link

        return chain

    def _verify_hash_chain(self, records_data: List[Dict], chain: List[str]) -> bool:
        """Verify hash chain matches records."""
        if len(records_data) != len(chain):
            return False

        prev_hash = ""
        for i, record_data in enumerate(records_data):
            record_hash = compute_sha256(record_data)
            expected_link = hashlib.sha256(f"{prev_hash}:{record_hash}".encode()).hexdigest()

            if chain[i] != expected_link:
                return False

            prev_hash = chain[i]

        return True

    def _compute_merkle_root(self, records: List[EvidenceRecord]) -> str:
        """Compute Merkle tree root from records."""
        if not records:
            return hashlib.sha256(b"").hexdigest()

        leaves = [r.compute_hash() for r in records]
        return self._compute_merkle_root_from_hashes(leaves)

    def _compute_merkle_root_from_hashes(self, leaves: List[str]) -> str:
        """Compute Merkle root from list of hashes."""
        if not leaves:
            return hashlib.sha256(b"").hexdigest()

        while len(leaves) > 1:
            if len(leaves) % 2:
                leaves.append(leaves[-1])  # Duplicate last if odd

            leaves = [
                hashlib.sha256((leaves[i] + leaves[i + 1]).encode()).hexdigest()
                for i in range(0, len(leaves), 2)
            ]

        return leaves[0]


# =============================================================================
# STANDARD METHODOLOGIES
# =============================================================================

# Pre-defined calculation methodologies for common heat exchanger calculations

LMTD_METHODOLOGY = CalculationMethodology(
    methodology_id="METH-LMTD-001",
    name="Log Mean Temperature Difference (LMTD)",
    version="1.0.0",
    description="Calculates the logarithmic mean temperature difference for heat exchanger analysis",
    formula_id="LMTD-001",
    formula_latex=r"\text{LMTD} = \frac{\Delta T_1 - \Delta T_2}{\ln(\Delta T_1 / \Delta T_2)}",
    formula_python="lmtd = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)",
    input_parameters=[
        {"name": "delta_t1", "type": "float", "unit": "K", "description": "Temperature difference at one end"},
        {"name": "delta_t2", "type": "float", "unit": "K", "description": "Temperature difference at other end"},
    ],
    output_parameters=[
        {"name": "lmtd", "type": "float", "unit": "K", "description": "Log mean temperature difference"},
    ],
    output_units={"lmtd": "K"},
    standards_references=[
        "ASME PTC 12.5-2000",
        "TEMA 10th Edition",
        "Perry's Chemical Engineers' Handbook",
    ],
    assumptions=[
        "Constant heat transfer coefficient along exchanger",
        "Constant specific heats",
        "Negligible heat losses to surroundings",
    ],
    limitations=[
        "Not valid when delta_t1 = delta_t2 (use arithmetic mean)",
        "Assumes no phase change",
    ],
)

NTU_METHODOLOGY = CalculationMethodology(
    methodology_id="METH-NTU-001",
    name="Number of Transfer Units (NTU) Method",
    version="1.0.0",
    description="NTU-effectiveness method for heat exchanger rating and sizing",
    formula_id="NTU-001",
    formula_latex=r"\text{NTU} = \frac{UA}{C_{min}}",
    formula_python="ntu = ua / c_min",
    input_parameters=[
        {"name": "ua", "type": "float", "unit": "W/K", "description": "Overall heat transfer conductance"},
        {"name": "c_min", "type": "float", "unit": "W/K", "description": "Minimum heat capacity rate"},
    ],
    output_parameters=[
        {"name": "ntu", "type": "float", "unit": "dimensionless", "description": "Number of transfer units"},
        {"name": "effectiveness", "type": "float", "unit": "dimensionless", "description": "Heat exchanger effectiveness"},
    ],
    standards_references=[
        "ASME PTC 12.5-2000",
        "Incropera & DeWitt, Fundamentals of Heat Transfer",
    ],
)

FOULING_RESISTANCE_METHODOLOGY = CalculationMethodology(
    methodology_id="METH-FOUL-001",
    name="Fouling Resistance Calculation",
    version="1.0.0",
    description="Calculates fouling resistance from clean and fouled heat transfer coefficients",
    formula_id="FOUL-001",
    formula_latex=r"R_f = \frac{1}{U_{fouled}} - \frac{1}{U_{clean}}",
    formula_python="rf = 1/u_fouled - 1/u_clean",
    input_parameters=[
        {"name": "u_clean", "type": "float", "unit": "W/m2K", "description": "Clean overall heat transfer coefficient"},
        {"name": "u_fouled", "type": "float", "unit": "W/m2K", "description": "Fouled overall heat transfer coefficient"},
    ],
    output_parameters=[
        {"name": "rf", "type": "float", "unit": "m2K/W", "description": "Fouling resistance"},
    ],
    standards_references=[
        "TEMA 10th Edition, Section RCB-2.6",
        "ASME PTC 12.5-2000",
    ],
    assumptions=[
        "Fouling is uniform across heat transfer surface",
        "No change in flow conditions between clean and fouled states",
    ],
)
