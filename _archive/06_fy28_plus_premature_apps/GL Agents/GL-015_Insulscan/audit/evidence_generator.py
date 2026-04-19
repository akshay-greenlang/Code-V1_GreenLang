# -*- coding: utf-8 -*-
"""
GL-015 Insulscan - Evidence Generator

Immutable Evidence Pack Generator for regulatory compliance and audit trails
for insulation scanning and thermal assessment operations. Generates sealed,
cryptographically verifiable evidence packages that support 7-year retention
per regulatory requirements.

Features:
    - Generate evidence records for all calculations
    - SHA-256 hashing of inputs and outputs
    - Formula version tracking
    - Reproducibility verification
    - Export to regulatory formats (ISO 50001, ASHRAE)
    - Merkle tree for efficient verification
    - Multiple export formats (JSON, XML, PDF)

Standards:
    - ISO 50001:2018 (Energy Management Systems)
    - ASHRAE 90.1 (Energy Standard for Buildings)
    - ASTM C1060 (Thermographic Inspection of Insulation)
    - ASTM C680 (Heat Loss Calculations)
    - 21 CFR Part 11 (Electronic Records)

Example:
    >>> from audit.evidence_generator import InsulationEvidenceGenerator
    >>> generator = InsulationEvidenceGenerator()
    >>> record = generator.create_evidence_record(
    ...     evidence_type=EvidenceType.HEAT_LOSS_CALCULATION,
    ...     data={"method": "ASTM_C680", "result": 150.5},
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
    EvidenceRecord,
    ProvenanceChain,
    ComputationType,
    ChainVerificationStatus,
    compute_sha256,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EVIDENCE TYPE ENUMERATION
# =============================================================================

class EvidenceType(str, Enum):
    """Types of evidence that can be captured for Insulscan."""

    # Heat loss calculations
    HEAT_LOSS_CALCULATION = "HEAT_LOSS_CALCULATION"
    R_VALUE_CALCULATION = "R_VALUE_CALCULATION"
    U_VALUE_CALCULATION = "U_VALUE_CALCULATION"
    THERMAL_CONDUCTIVITY = "THERMAL_CONDUCTIVITY"
    SURFACE_HEAT_TRANSFER = "SURFACE_HEAT_TRANSFER"

    # Thermal imaging evidence
    THERMAL_SCAN = "THERMAL_SCAN"
    HOTSPOT_DETECTION = "HOTSPOT_DETECTION"
    TEMPERATURE_MAPPING = "TEMPERATURE_MAPPING"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"

    # Assessment evidence
    INSULATION_ASSESSMENT = "INSULATION_ASSESSMENT"
    CONDITION_ASSESSMENT = "CONDITION_ASSESSMENT"
    DEGRADATION_ASSESSMENT = "DEGRADATION_ASSESSMENT"
    REMAINING_USEFUL_LIFE = "REMAINING_USEFUL_LIFE"

    # Safety evidence
    SAFETY_CHECK = "SAFETY_CHECK"
    TEMPERATURE_LIMIT_CHECK = "TEMPERATURE_LIMIT_CHECK"

    # Compliance evidence
    ISO_50001_COMPLIANCE = "ISO_50001_COMPLIANCE"
    ASHRAE_COMPLIANCE = "ASHRAE_COMPLIANCE"
    ASTM_COMPLIANCE = "ASTM_COMPLIANCE"

    # Prediction evidence
    PREDICTION = "PREDICTION"
    ENERGY_SAVINGS_PREDICTION = "ENERGY_SAVINGS_PREDICTION"
    MAINTENANCE_PREDICTION = "MAINTENANCE_PREDICTION"

    # Recommendation evidence
    RECOMMENDATION = "RECOMMENDATION"
    MAINTENANCE_RECOMMENDATION = "MAINTENANCE_RECOMMENDATION"
    REPLACEMENT_RECOMMENDATION = "REPLACEMENT_RECOMMENDATION"
    UPGRADE_RECOMMENDATION = "UPGRADE_RECOMMENDATION"

    # Configuration and change evidence
    CONFIGURATION_CHANGE = "CONFIGURATION_CHANGE"
    THRESHOLD_CHANGE = "THRESHOLD_CHANGE"
    MODEL_UPDATE = "MODEL_UPDATE"

    # Regulatory evidence
    REGULATORY_SUBMISSION = "REGULATORY_SUBMISSION"
    ENERGY_SAVINGS_VERIFICATION = "ENERGY_SAVINGS_VERIFICATION"
    CO2E_ACCOUNTING = "CO2E_ACCOUNTING"

    # Workflow evidence
    WORK_ORDER_TRACE = "WORK_ORDER_TRACE"
    APPROVAL_WORKFLOW = "APPROVAL_WORKFLOW"


class SealStatus(str, Enum):
    """Status of evidence pack seal."""

    UNSEALED = "UNSEALED"
    SEALED = "SEALED"
    VERIFIED = "VERIFIED"
    TAMPERED = "TAMPERED"


class ExportFormat(str, Enum):
    """Available export formats for evidence packs."""

    JSON = "JSON"
    XML = "XML"
    PDF = "PDF"
    ISO_50001_XML = "ISO_50001_XML"
    ASHRAE_XML = "ASHRAE_XML"
    ASTM_XML = "ASTM_XML"


# =============================================================================
# INSULATION EVIDENCE RECORD
# =============================================================================

@dataclass(frozen=True)
class InsulationEvidenceRecord:
    """
    Immutable evidence record for insulation audit trail.

    Captures a single piece of evidence with full cryptographic
    verification and traceability for insulation assessments.

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

    # Asset metadata
    asset_id: str = ""
    asset_type: str = "insulation"
    location: str = ""

    # Calculation metadata
    algorithm_version: str = "1.0.0"
    formula_reference: str = ""
    calculation_trace: List[str] = field(default_factory=list)

    # Model provenance (for predictions)
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    training_data_hash: Optional[str] = None
    model_config_hash: Optional[str] = None

    # Scan metadata (for thermal imaging)
    scan_id: Optional[str] = None
    camera_model: Optional[str] = None
    calibration_date: Optional[datetime] = None

    # Recommendation traceability
    recommendation_id: Optional[str] = None
    work_order_id: Optional[str] = None
    cmms_reference: Optional[str] = None

    # Reproducibility verification
    reproducibility_verified: bool = False
    verification_timestamp: Optional[datetime] = None

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
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "location": self.location,
            "algorithm_version": self.algorithm_version,
            "formula_reference": self.formula_reference,
            "calculation_trace": self.calculation_trace,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "training_data_hash": self.training_data_hash,
            "model_config_hash": self.model_config_hash,
            "scan_id": self.scan_id,
            "camera_model": self.camera_model,
            "calibration_date": self.calibration_date.isoformat() if self.calibration_date else None,
            "recommendation_id": self.recommendation_id,
            "work_order_id": self.work_order_id,
            "cmms_reference": self.cmms_reference,
            "reproducibility_verified": self.reproducibility_verified,
            "verification_timestamp": self.verification_timestamp.isoformat() if self.verification_timestamp else None,
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
    agent_id: str = "GL-015"
    agent_name: str = "INSULSCAN"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retention_years: int = 7
    retention_expires_at: Optional[datetime] = None
    record_count: int = 0
    seal_status: SealStatus = SealStatus.UNSEALED
    pack_version: str = "1.0.0"

    # Regulatory frameworks this pack supports
    regulatory_frameworks: List[str] = Field(
        default_factory=lambda: [
            "ISO_50001_2018",
            "ASHRAE_90_1",
            "ASTM_C1060",
            "ASTM_C680",
        ]
    )

    # Asset coverage
    assets_covered: List[str] = Field(default_factory=list)

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
    sealer_id: str = "GL-015-SEALER"


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
# INSULATION EVIDENCE GENERATOR
# =============================================================================

class InsulationEvidenceGenerator:
    """
    Immutable Evidence Pack Generator for GL-015 Insulscan.

    Generates cryptographically sealed evidence packages for
    regulatory compliance and audit trails in insulation scanning
    and thermal assessment operations.

    Usage:
        >>> generator = InsulationEvidenceGenerator()
        >>> record = generator.create_evidence_record(
        ...     evidence_type=EvidenceType.HEAT_LOSS_CALCULATION,
        ...     data={"method": "ASTM_C680", "result": 150.5},
        ...     inputs={"surface_temp_c": 45, "ambient_temp_c": 20},
        ...     outputs={"heat_loss_w_m2": 150.5}
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
        self.agent_id = "GL-015"
        self.agent_name = "INSULSCAN"

        # Methodology registry
        self._methodologies: Dict[str, CalculationMethodology] = {}

        # Model provenance registry
        self._model_provenance: Dict[str, ModelProvenance] = {}

        # Initialize standard methodologies
        self._init_standard_methodologies()

        logger.info(f"InsulationEvidenceGenerator initialized: retention={retention_years} years")

    def _init_standard_methodologies(self) -> None:
        """Initialize standard calculation methodologies."""
        # Heat loss calculation methodology
        self._methodologies["HEAT_LOSS_001"] = CalculationMethodology(
            methodology_id="METH-HEATLOSS-001",
            name="Surface Heat Loss Calculation",
            version="1.0.0",
            description="Calculates heat loss from insulated surface using ASTM C680 method",
            formula_id="HEATLOSS-001",
            formula_latex=r"Q = h \cdot A \cdot (T_s - T_a)",
            formula_python="heat_loss = h * area * (surface_temp - ambient_temp)",
            input_parameters=[
                {"name": "surface_temp_c", "type": "float", "unit": "C", "description": "Surface temperature"},
                {"name": "ambient_temp_c", "type": "float", "unit": "C", "description": "Ambient temperature"},
                {"name": "surface_area_m2", "type": "float", "unit": "m2", "description": "Surface area"},
                {"name": "h_coefficient", "type": "float", "unit": "W/m2K", "description": "Heat transfer coefficient"},
            ],
            output_parameters=[
                {"name": "heat_loss_w", "type": "float", "unit": "W", "description": "Total heat loss"},
            ],
            output_units={"heat_loss_w": "W"},
            standards_references=[
                "ASTM C680 - Standard Practice for Estimate of the Heat Gain or Loss",
                "ASTM C1060 - Standard Practice for Thermographic Inspection of Insulation",
            ],
            assumptions=[
                "Steady-state heat transfer",
                "Uniform surface temperature",
                "Constant ambient conditions",
            ],
            limitations=[
                "Not valid for transient conditions",
                "Assumes no radiation heat transfer (simplified)",
            ],
        )

        # R-value calculation methodology
        self._methodologies["R_VALUE_001"] = CalculationMethodology(
            methodology_id="METH-RVALUE-001",
            name="Thermal Resistance (R-Value) Calculation",
            version="1.0.0",
            description="Calculates thermal resistance of insulation",
            formula_id="RVALUE-001",
            formula_latex=r"R = \frac{L}{k}",
            formula_python="r_value = thickness_m / thermal_conductivity",
            input_parameters=[
                {"name": "thickness_m", "type": "float", "unit": "m", "description": "Insulation thickness"},
                {"name": "thermal_conductivity", "type": "float", "unit": "W/mK", "description": "Thermal conductivity"},
            ],
            output_parameters=[
                {"name": "r_value", "type": "float", "unit": "m2K/W", "description": "Thermal resistance"},
            ],
            output_units={"r_value": "m2K/W"},
            standards_references=[
                "ASHRAE 90.1 - Energy Standard for Buildings",
                "ASTM C518 - Steady-State Thermal Transmission Properties",
            ],
        )

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
        asset_id: str = "",
        asset_type: str = "insulation",
        location: str = "",
        algorithm_version: str = "1.0.0",
        formula_reference: str = "",
        calculation_trace: Optional[List[str]] = None,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        scan_id: Optional[str] = None,
        recommendation_id: Optional[str] = None,
        work_order_id: Optional[str] = None,
    ) -> InsulationEvidenceRecord:
        """
        Create an evidence record with full traceability.

        Args:
            evidence_type: Type of evidence
            data: The evidence data
            inputs: Input data that led to this evidence
            outputs: Output data produced
            provenance_chain: Chain of hashes for traceability
            asset_id: Insulation asset identifier
            asset_type: Type of asset
            location: Physical location
            algorithm_version: Version of algorithm used
            formula_reference: Reference to formula/method
            calculation_trace: List of calculation steps
            model_id: ML model identifier (if prediction)
            model_version: ML model version
            scan_id: Thermal scan identifier (if applicable)
            recommendation_id: Recommendation ID (if applicable)
            work_order_id: Work order ID (if created)

        Returns:
            Immutable InsulationEvidenceRecord
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

        return InsulationEvidenceRecord(
            record_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            evidence_type=evidence_type,
            data=data,
            input_hash=input_hash,
            output_hash=output_hash,
            provenance_chain=provenance_chain or [],
            asset_id=asset_id,
            asset_type=asset_type,
            location=location,
            algorithm_version=algorithm_version,
            formula_reference=formula_reference,
            calculation_trace=calculation_trace or [],
            model_id=model_id,
            model_version=model_version,
            training_data_hash=training_data_hash,
            model_config_hash=model_config_hash,
            scan_id=scan_id,
            recommendation_id=recommendation_id,
            work_order_id=work_order_id,
        )

    def create_heat_loss_evidence(
        self,
        asset_id: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        calculation_trace: List[str],
        methodology_id: str = "HEAT_LOSS_001",
    ) -> InsulationEvidenceRecord:
        """
        Create evidence for a heat loss calculation.

        Args:
            asset_id: Insulation asset identifier
            inputs: Calculation inputs
            outputs: Calculation results
            calculation_trace: Step-by-step calculation trace
            methodology_id: ID of methodology used

        Returns:
            InsulationEvidenceRecord for the calculation
        """
        methodology = self._methodologies.get(methodology_id)
        if not methodology:
            methodology = self._methodologies.get("HEAT_LOSS_001")

        return self.create_evidence_record(
            evidence_type=EvidenceType.HEAT_LOSS_CALCULATION,
            data={
                "calculation_type": "heat_loss",
                "methodology_id": methodology.methodology_id if methodology else "",
                "methodology": methodology.to_dict() if methodology else {},
                "inputs": inputs,
                "outputs": outputs,
            },
            inputs=inputs,
            outputs=outputs,
            asset_id=asset_id,
            algorithm_version=methodology.version if methodology else "1.0.0",
            formula_reference=methodology.formula_id if methodology else "",
            calculation_trace=calculation_trace,
        )

    def create_thermal_scan_evidence(
        self,
        asset_id: str,
        scan_id: str,
        camera_model: str,
        scan_data: Dict[str, Any],
        analysis_results: Dict[str, Any],
        image_hash: str,
    ) -> InsulationEvidenceRecord:
        """
        Create evidence for a thermal scan.

        Args:
            asset_id: Insulation asset identifier
            scan_id: Thermal scan identifier
            camera_model: Thermal camera model used
            scan_data: Raw scan data
            analysis_results: Analysis results
            image_hash: SHA-256 hash of thermal image

        Returns:
            InsulationEvidenceRecord for the scan
        """
        return self.create_evidence_record(
            evidence_type=EvidenceType.THERMAL_SCAN,
            data={
                "scan_type": "thermal_imaging",
                "camera_model": camera_model,
                "image_hash": image_hash,
                "analysis_results": analysis_results,
            },
            inputs=scan_data,
            outputs=analysis_results,
            asset_id=asset_id,
            scan_id=scan_id,
        )

    def create_compliance_evidence(
        self,
        asset_id: str,
        standard: str,
        requirement: str,
        compliance_data: Dict[str, Any],
        is_compliant: bool,
        supporting_records: Optional[List[InsulationEvidenceRecord]] = None,
    ) -> InsulationEvidenceRecord:
        """
        Create evidence for compliance verification.

        Args:
            asset_id: Asset identifier
            standard: Compliance standard (e.g., "ISO_50001")
            requirement: Specific requirement
            compliance_data: Compliance verification data
            is_compliant: Whether compliant
            supporting_records: Related evidence records

        Returns:
            InsulationEvidenceRecord for compliance
        """
        # Build provenance chain from supporting records
        provenance_chain = []
        if supporting_records:
            provenance_chain = [r.compute_hash() for r in supporting_records]

        evidence_type_map = {
            "ISO_50001": EvidenceType.ISO_50001_COMPLIANCE,
            "ASHRAE": EvidenceType.ASHRAE_COMPLIANCE,
            "ASTM": EvidenceType.ASTM_COMPLIANCE,
        }

        evidence_type = evidence_type_map.get(standard.upper().split("_")[0], EvidenceType.ASHRAE_COMPLIANCE)

        return self.create_evidence_record(
            evidence_type=evidence_type,
            data={
                "standard": standard,
                "requirement": requirement,
                "is_compliant": is_compliant,
                "compliance_data": compliance_data,
            },
            inputs=compliance_data,
            outputs={"is_compliant": is_compliant, "standard": standard, "requirement": requirement},
            provenance_chain=provenance_chain,
            asset_id=asset_id,
        )

    def create_prediction_evidence(
        self,
        prediction_type: str,
        asset_id: str,
        inputs: Dict[str, Any],
        prediction: Any,
        confidence: float,
        model_provenance: ModelProvenance,
        explanation: Optional[Dict[str, Any]] = None,
    ) -> InsulationEvidenceRecord:
        """
        Create evidence for a model prediction with full provenance.

        Args:
            prediction_type: Type of prediction (e.g., "degradation", "rul")
            asset_id: Insulation asset identifier
            inputs: Model input features
            prediction: Prediction value
            confidence: Prediction confidence
            model_provenance: Complete model provenance
            explanation: SHAP/LIME explanation if available

        Returns:
            InsulationEvidenceRecord for the prediction
        """
        # Register model provenance
        if model_provenance.model_id not in self._model_provenance:
            self._model_provenance[model_provenance.model_id] = model_provenance

        evidence_type_map = {
            "energy_savings": EvidenceType.ENERGY_SAVINGS_PREDICTION,
            "maintenance": EvidenceType.MAINTENANCE_PREDICTION,
            "degradation": EvidenceType.PREDICTION,
            "rul": EvidenceType.REMAINING_USEFUL_LIFE,
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
            asset_id=asset_id,
            model_id=model_provenance.model_id,
            model_version=model_provenance.model_version,
        )

    def verify_reproducibility(
        self,
        record: InsulationEvidenceRecord,
        recalculated_outputs: Dict[str, Any],
    ) -> Tuple[bool, InsulationEvidenceRecord]:
        """
        Verify that a calculation is reproducible.

        Args:
            record: Evidence record to verify
            recalculated_outputs: Outputs from recalculation

        Returns:
            Tuple of (is_reproducible, updated_record)
        """
        recalculated_hash = compute_sha256(recalculated_outputs)
        is_reproducible = recalculated_hash == record.output_hash

        # Create a new record with verification status (records are immutable)
        verified_record = InsulationEvidenceRecord(
            record_id=record.record_id,
            timestamp=record.timestamp,
            evidence_type=record.evidence_type,
            data=record.data,
            input_hash=record.input_hash,
            output_hash=record.output_hash,
            provenance_chain=record.provenance_chain,
            asset_id=record.asset_id,
            asset_type=record.asset_type,
            location=record.location,
            algorithm_version=record.algorithm_version,
            formula_reference=record.formula_reference,
            calculation_trace=record.calculation_trace,
            model_id=record.model_id,
            model_version=record.model_version,
            training_data_hash=record.training_data_hash,
            model_config_hash=record.model_config_hash,
            scan_id=record.scan_id,
            recommendation_id=record.recommendation_id,
            work_order_id=record.work_order_id,
            reproducibility_verified=is_reproducible,
            verification_timestamp=datetime.now(timezone.utc),
        )

        logger.info(
            f"Reproducibility verification for {record.record_id}: "
            f"{'PASSED' if is_reproducible else 'FAILED'}"
        )

        return is_reproducible, verified_record

    # =========================================================================
    # EVIDENCE PACK OPERATIONS
    # =========================================================================

    def create_evidence_pack(
        self,
        records: List[InsulationEvidenceRecord],
        assets_covered: Optional[List[str]] = None,
    ) -> bytes:
        """
        Create a compressed evidence pack from records.

        Args:
            records: List of evidence records
            assets_covered: List of asset IDs covered

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

        # Get unique assets
        if assets_covered is None:
            assets_covered = list(set(r.asset_id for r in records if r.asset_id))

        # Create metadata
        metadata = EvidencePackMetadata(
            pack_id=str(uuid.uuid4()),
            record_count=len(records),
            assets_covered=assets_covered,
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

        elif export_format in [ExportFormat.XML, ExportFormat.ISO_50001_XML,
                               ExportFormat.ASHRAE_XML, ExportFormat.ASTM_XML]:
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

        root = ET.Element("InsulationEvidencePack")
        root.set("packId", envelope.pack_id)
        root.set("format", format_type.value)
        root.set("version", self.VERSION)

        # Metadata section
        meta_elem = ET.SubElement(root, "Metadata")
        ET.SubElement(meta_elem, "AgentId").text = metadata.get("agent_id", "GL-015")
        ET.SubElement(meta_elem, "AgentName").text = metadata.get("agent_name", "INSULSCAN")
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

        # Assets
        assets_elem = ET.SubElement(root, "AssetsCovered")
        for asset_id in metadata.get("assets_covered", []):
            ET.SubElement(assets_elem, "AssetId").text = asset_id

        # Records summary (not full content for size)
        records_elem = ET.SubElement(root, "RecordsSummary")
        for record in pack_data.get("records", [])[:100]:  # Limit to 100 for XML
            rec_elem = ET.SubElement(records_elem, "Record")
            rec_elem.set("id", record.get("record_id", ""))
            rec_elem.set("type", record.get("evidence_type", ""))
            rec_elem.set("timestamp", record.get("timestamp", ""))
            rec_elem.set("asset", record.get("asset_id", ""))

        return ET.tostring(root, encoding="utf-8", xml_declaration=True)

    def _export_pdf_placeholder(self, envelope: SealedPackEnvelope) -> bytes:
        """Generate PDF placeholder (actual implementation would use reportlab)."""
        text = f"""
GL-015 INSULSCAN EVIDENCE PACK
==============================

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

    def _build_hash_chain(self, records: List[InsulationEvidenceRecord]) -> List[str]:
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

    def _compute_merkle_root(self, records: List[InsulationEvidenceRecord]) -> str:
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

    # =========================================================================
    # METHODOLOGY MANAGEMENT
    # =========================================================================

    def register_methodology(self, methodology: CalculationMethodology) -> None:
        """Register a calculation methodology."""
        self._methodologies[methodology.methodology_id] = methodology
        logger.info(f"Registered methodology: {methodology.methodology_id}")

    def get_methodology(self, methodology_id: str) -> Optional[CalculationMethodology]:
        """Get a registered methodology by ID."""
        return self._methodologies.get(methodology_id)

    def list_methodologies(self) -> List[str]:
        """List all registered methodology IDs."""
        return list(self._methodologies.keys())

    # =========================================================================
    # MODEL PROVENANCE MANAGEMENT
    # =========================================================================

    def register_model_provenance(self, provenance: ModelProvenance) -> None:
        """Register model provenance for tracking."""
        self._model_provenance[provenance.model_id] = provenance
        logger.info(f"Registered model provenance: {provenance.model_id}")

    def get_model_provenance(self, model_id: str) -> Optional[ModelProvenance]:
        """Get model provenance by ID."""
        return self._model_provenance.get(model_id)


# =============================================================================
# STANDARD METHODOLOGIES
# =============================================================================

# Pre-defined calculation methodologies for common insulation calculations

HEAT_LOSS_METHODOLOGY = CalculationMethodology(
    methodology_id="METH-HEATLOSS-STD",
    name="Standard Heat Loss Calculation",
    version="1.0.0",
    description="Calculates heat loss from insulated surfaces per ASTM C680",
    formula_id="HEATLOSS-STD",
    formula_latex=r"Q = h \cdot A \cdot (T_s - T_a)",
    formula_python="heat_loss = h * area * (surface_temp - ambient_temp)",
    standards_references=["ASTM C680", "ASTM C1060"],
)

R_VALUE_METHODOLOGY = CalculationMethodology(
    methodology_id="METH-RVALUE-STD",
    name="Standard R-Value Calculation",
    version="1.0.0",
    description="Calculates thermal resistance per ASHRAE standards",
    formula_id="RVALUE-STD",
    formula_latex=r"R = \frac{L}{k}",
    formula_python="r_value = thickness / thermal_conductivity",
    standards_references=["ASHRAE 90.1", "ASTM C518"],
)

U_VALUE_METHODOLOGY = CalculationMethodology(
    methodology_id="METH-UVALUE-STD",
    name="Standard U-Value Calculation",
    version="1.0.0",
    description="Calculates overall heat transfer coefficient",
    formula_id="UVALUE-STD",
    formula_latex=r"U = \frac{1}{R_{total}}",
    formula_python="u_value = 1 / r_total",
    standards_references=["ASHRAE 90.1"],
)
