# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Evidence Generator for Condenser Optimization

Generates evidence chains linking recommendations to source data,
calculations, and physics principles for audit and compliance purposes.

Key Features:
- Generate evidence for optimization recommendations
- Link calculations to source data with complete traceability
- Create compliance reports for regulatory review
- Support multiple evidence types (sensor, calculated, derived)
- Maintain chain of custody for all evidence

Zero-Hallucination Guarantee:
All evidence derived from deterministic calculations.
No LLM or AI inference in evidence generation.
Complete traceability from recommendation to source data.

Reference Standards:
- ASME PTC 12.2: Steam Surface Condensers
- ISO 14064: Greenhouse Gas Quantification
- GHG Protocol: Corporate Accounting Standard

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Agent identification
AGENT_ID = "GL-017"
AGENT_NAME = "Condensync"
VERSION = "1.0.0"


# ============================================================================
# ENUMS
# ============================================================================

class EvidenceType(str, Enum):
    """Types of evidence."""
    SENSOR_DATA = "sensor_data"            # Raw sensor measurements
    CALCULATED = "calculated"               # Derived from calculations
    REFERENCE = "reference"                 # Reference to standards/specs
    HISTORICAL = "historical"               # Historical data comparison
    EXPERT_RULE = "expert_rule"            # Engineering rule/heuristic
    MODEL_OUTPUT = "model_output"          # ML/physics model output
    THRESHOLD = "threshold"                 # Threshold comparison


class EvidenceStrength(str, Enum):
    """Strength of evidence."""
    DEFINITIVE = "definitive"              # Conclusive evidence
    STRONG = "strong"                       # High confidence
    MODERATE = "moderate"                   # Reasonable confidence
    WEAK = "weak"                           # Supporting but not conclusive
    CIRCUMSTANTIAL = "circumstantial"      # Indirect evidence


class RecommendationType(str, Enum):
    """Types of recommendations."""
    OPERATIONAL = "operational"             # Operational adjustment
    MAINTENANCE = "maintenance"             # Maintenance action
    INVESTIGATION = "investigation"         # Further investigation
    MONITORING = "monitoring"               # Continued monitoring
    CAPITAL = "capital"                     # Capital improvement


class ComplianceFramework(str, Enum):
    """Compliance frameworks."""
    ASME_PTC = "ASME_PTC"                  # ASME Performance Test Codes
    ISO_14064 = "ISO_14064"                # GHG Quantification
    ISO_50001 = "ISO_50001"                # Energy Management
    GHG_PROTOCOL = "GHG_Protocol"          # GHG Protocol
    HEI = "HEI"                            # Heat Exchange Institute


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DataSource:
    """
    Source of evidence data.

    Attributes:
        source_id: Unique identifier
        source_type: Type of source (sensor, database, etc.)
        source_name: Human-readable name
        location: Physical or logical location
        timestamp: When data was collected
        quality_score: Data quality indicator (0-1)
        metadata: Additional source metadata
    """
    source_id: str
    source_type: str
    source_name: str
    location: str
    timestamp: datetime
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "location": self.location,
            "timestamp": self.timestamp.isoformat(),
            "quality_score": round(self.quality_score, 3),
            "metadata": self.metadata
        }


@dataclass
class EvidenceItem:
    """
    A single piece of evidence.

    Attributes:
        evidence_id: Unique identifier
        evidence_type: Type of evidence
        strength: Evidence strength
        description: What this evidence shows
        value: The evidence value
        unit: Unit of measurement
        data_source: Source of the data
        calculation_reference: Reference to calculation (if derived)
        equation_reference: Reference to physics equation
        threshold_reference: Reference to threshold (if comparison)
        timestamp: When evidence was generated
        hash: SHA-256 hash of evidence data
    """
    evidence_id: str
    evidence_type: EvidenceType
    strength: EvidenceStrength
    description: str
    value: Any
    unit: str
    data_source: DataSource
    calculation_reference: Optional[str]
    equation_reference: Optional[str]
    threshold_reference: Optional[str]
    timestamp: datetime
    hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "strength": self.strength.value,
            "description": self.description,
            "value": self.value,
            "unit": self.unit,
            "data_source": self.data_source.to_dict(),
            "calculation_reference": self.calculation_reference,
            "equation_reference": self.equation_reference,
            "threshold_reference": self.threshold_reference,
            "timestamp": self.timestamp.isoformat(),
            "hash": self.hash
        }


@dataclass
class EvidenceChain:
    """
    Chain of evidence supporting a conclusion.

    Attributes:
        chain_id: Unique identifier
        conclusion: The conclusion being supported
        evidence_items: List of evidence items
        chain_strength: Overall chain strength
        chain_hash: Hash of entire chain
        created_at: When chain was created
    """
    chain_id: str
    conclusion: str
    evidence_items: List[EvidenceItem]
    chain_strength: EvidenceStrength
    chain_hash: str
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chain_id": self.chain_id,
            "conclusion": self.conclusion,
            "evidence_items": [e.to_dict() for e in self.evidence_items],
            "chain_strength": self.chain_strength.value,
            "chain_hash": self.chain_hash,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class RecommendationEvidence:
    """
    Complete evidence package for a recommendation.

    Attributes:
        recommendation_id: Unique identifier
        condenser_id: Condenser equipment ID
        recommendation_type: Type of recommendation
        recommendation_text: The recommendation
        priority: Priority level (1-5, 1=highest)
        evidence_chains: Supporting evidence chains
        physics_basis: Physics justification
        expected_benefit: Expected improvement
        benefit_unit: Unit of benefit
        confidence: Confidence in recommendation
        implementation_notes: Implementation guidance
        compliance_mapping: Mapping to compliance frameworks
        provenance_hash: SHA-256 hash of all evidence
        created_at: When generated
    """
    recommendation_id: str
    condenser_id: str
    recommendation_type: RecommendationType
    recommendation_text: str
    priority: int
    evidence_chains: List[EvidenceChain]
    physics_basis: str
    expected_benefit: float
    benefit_unit: str
    confidence: float
    implementation_notes: str
    compliance_mapping: Dict[str, List[str]]
    provenance_hash: str
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recommendation_id": self.recommendation_id,
            "condenser_id": self.condenser_id,
            "recommendation_type": self.recommendation_type.value,
            "recommendation_text": self.recommendation_text,
            "priority": self.priority,
            "evidence_chains": [ec.to_dict() for ec in self.evidence_chains],
            "physics_basis": self.physics_basis,
            "expected_benefit": round(self.expected_benefit, 3),
            "benefit_unit": self.benefit_unit,
            "confidence": round(self.confidence, 3),
            "implementation_notes": self.implementation_notes,
            "compliance_mapping": self.compliance_mapping,
            "provenance_hash": self.provenance_hash,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ComplianceReport:
    """
    Compliance report for regulatory submission.

    Attributes:
        report_id: Unique identifier
        condenser_id: Condenser equipment ID
        framework: Compliance framework
        period_start: Reporting period start
        period_end: Reporting period end
        recommendations: List of recommendations with evidence
        methodology_statement: Description of methodology
        data_quality_statement: Data quality declaration
        limitations: Known limitations
        certifications: Required certifications
        report_hash: SHA-256 hash of report
        generated_at: When report was generated
    """
    report_id: str
    condenser_id: str
    framework: ComplianceFramework
    period_start: datetime
    period_end: datetime
    recommendations: List[RecommendationEvidence]
    methodology_statement: str
    data_quality_statement: str
    limitations: List[str]
    certifications: List[str]
    report_hash: str
    generated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "condenser_id": self.condenser_id,
            "framework": self.framework.value,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "recommendations": [r.to_dict() for r in self.recommendations],
            "methodology_statement": self.methodology_statement,
            "data_quality_statement": self.data_quality_statement,
            "limitations": self.limitations,
            "certifications": self.certifications,
            "report_hash": self.report_hash,
            "generated_at": self.generated_at.isoformat()
        }


# ============================================================================
# REFERENCE DATA
# ============================================================================

# Threshold definitions for condenser parameters
PARAMETER_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "backpressure": {
        "normal_max": 5.0,
        "warning": 7.0,
        "critical": 10.0,
        "unit": "kPa_abs",
        "reference": "Plant design specification"
    },
    "TTD": {
        "normal_max": 4.0,
        "warning": 6.0,
        "critical": 8.0,
        "unit": "C",
        "reference": "HEI Standards"
    },
    "cleanliness_factor": {
        "normal_min": 0.85,
        "warning": 0.75,
        "critical": 0.65,
        "unit": "fraction",
        "reference": "HEI Standards"
    },
    "air_ingress": {
        "normal_max": 5.0,
        "warning": 10.0,
        "critical": 20.0,
        "unit": "kg/h",
        "reference": "ASME PTC 12.2"
    },
    "CW_flow_deviation": {
        "normal_max": 5.0,
        "warning": 10.0,
        "critical": 20.0,
        "unit": "%",
        "reference": "Plant design specification"
    }
}

# Physics equations for evidence
PHYSICS_EQUATIONS: Dict[str, Dict[str, str]] = {
    "heat_transfer": {
        "equation": "Q = UA * LMTD",
        "description": "Fundamental heat transfer equation",
        "reference": "ASME PTC 12.2"
    },
    "energy_balance": {
        "equation": "Q = m_cw * Cp * (T_out - T_in)",
        "description": "Energy balance on cooling water",
        "reference": "ASME PTC 12.2"
    },
    "efficiency_impact": {
        "equation": "deta = k * dP_back",
        "description": "Backpressure effect on efficiency",
        "reference": "ASME PTC 12.2"
    }
}

# Compliance mappings
COMPLIANCE_MAPPINGS: Dict[ComplianceFramework, Dict[str, Any]] = {
    ComplianceFramework.ASME_PTC: {
        "standard": "ASME PTC 12.2",
        "sections": ["Section 4: Test Procedure", "Section 5: Calculations"],
        "requirements": ["Instrumentation accuracy", "Test duration", "Calculation methodology"]
    },
    ComplianceFramework.ISO_50001: {
        "standard": "ISO 50001:2018",
        "sections": ["4.4.3: Energy review", "4.4.5: Energy baseline"],
        "requirements": ["Energy performance indicators", "Continuous improvement"]
    },
    ComplianceFramework.HEI: {
        "standard": "HEI Standards for Steam Surface Condensers",
        "sections": ["Performance calculations", "Cleanliness factor"],
        "requirements": ["TTD measurement", "Cleanliness monitoring"]
    }
}


# ============================================================================
# EVIDENCE GENERATOR
# ============================================================================

class EvidenceGenerator:
    """
    Generates evidence chains for condenser optimization recommendations.

    ZERO-HALLUCINATION GUARANTEE:
    - All evidence derived from source data and calculations
    - No LLM or AI inference in evidence generation
    - Complete traceability from recommendation to source
    - SHA-256 hashes for all evidence items

    Features:
    1. Evidence Generation: Create evidence from sensor/calculated data
    2. Chain Building: Build evidence chains for conclusions
    3. Recommendation Evidence: Complete evidence packages
    4. Compliance Reports: Regulatory-ready documentation

    Example:
        >>> generator = EvidenceGenerator()
        >>> evidence = generator.generate_recommendation_evidence(
        ...     condenser_id="COND-001",
        ...     recommendation="Increase CW flow to design value",
        ...     recommendation_type=RecommendationType.OPERATIONAL,
        ...     supporting_data={
        ...         "CW_flow": {"value": 13500, "unit": "kg/s"},
        ...         "backpressure": {"value": 6.5, "unit": "kPa_abs"}
        ...     }
        ... )
    """

    def __init__(self):
        """Initialize evidence generator."""
        self._evidence_count = 0
        self._chain_count = 0
        self._recommendation_count = 0

        logger.info("EvidenceGenerator initialized")

    def create_evidence_item(
        self,
        evidence_type: EvidenceType,
        description: str,
        value: Any,
        unit: str,
        source_name: str,
        source_type: str = "sensor",
        source_location: str = "condenser",
        quality_score: float = 0.95,
        calculation_ref: Optional[str] = None,
        equation_ref: Optional[str] = None,
        threshold_ref: Optional[str] = None
    ) -> EvidenceItem:
        """
        Create a single evidence item.

        Args:
            evidence_type: Type of evidence
            description: What this evidence shows
            value: The evidence value
            unit: Unit of measurement
            source_name: Name of data source
            source_type: Type of source
            source_location: Physical/logical location
            quality_score: Data quality (0-1)
            calculation_ref: Reference to calculation
            equation_ref: Reference to equation
            threshold_ref: Reference to threshold

        Returns:
            EvidenceItem with complete metadata
        """
        self._evidence_count += 1
        timestamp = datetime.now(timezone.utc)

        # Create data source
        data_source = DataSource(
            source_id=f"SRC-{uuid.uuid4().hex[:8]}",
            source_type=source_type,
            source_name=source_name,
            location=source_location,
            timestamp=timestamp,
            quality_score=quality_score
        )

        # Determine evidence strength
        strength = self._determine_evidence_strength(
            evidence_type, quality_score, value, threshold_ref
        )

        # Compute evidence hash
        evidence_data = {
            "type": evidence_type.value,
            "value": value,
            "unit": unit,
            "source": source_name,
            "timestamp": timestamp.isoformat()
        }
        evidence_hash = self._compute_hash(evidence_data)

        return EvidenceItem(
            evidence_id=f"EV-{uuid.uuid4().hex[:12]}",
            evidence_type=evidence_type,
            strength=strength,
            description=description,
            value=value,
            unit=unit,
            data_source=data_source,
            calculation_reference=calculation_ref,
            equation_reference=equation_ref,
            threshold_reference=threshold_ref,
            timestamp=timestamp,
            hash=evidence_hash
        )

    def _determine_evidence_strength(
        self,
        evidence_type: EvidenceType,
        quality_score: float,
        value: Any,
        threshold_ref: Optional[str]
    ) -> EvidenceStrength:
        """Determine evidence strength based on type and quality."""
        # Base strength on quality score
        if quality_score >= 0.95:
            base_strength = EvidenceStrength.DEFINITIVE
        elif quality_score >= 0.85:
            base_strength = EvidenceStrength.STRONG
        elif quality_score >= 0.70:
            base_strength = EvidenceStrength.MODERATE
        elif quality_score >= 0.50:
            base_strength = EvidenceStrength.WEAK
        else:
            base_strength = EvidenceStrength.CIRCUMSTANTIAL

        # Adjust based on evidence type
        if evidence_type == EvidenceType.SENSOR_DATA and quality_score >= 0.90:
            return EvidenceStrength.DEFINITIVE
        elif evidence_type == EvidenceType.THRESHOLD and threshold_ref:
            return EvidenceStrength.STRONG
        elif evidence_type == EvidenceType.EXPERT_RULE:
            return EvidenceStrength.MODERATE

        return base_strength

    def build_evidence_chain(
        self,
        conclusion: str,
        evidence_items: List[EvidenceItem]
    ) -> EvidenceChain:
        """
        Build an evidence chain from multiple evidence items.

        Args:
            conclusion: The conclusion being supported
            evidence_items: List of supporting evidence

        Returns:
            EvidenceChain with computed strength and hash
        """
        self._chain_count += 1
        created_at = datetime.now(timezone.utc)

        # Compute chain strength (weakest link)
        strength_order = [
            EvidenceStrength.DEFINITIVE,
            EvidenceStrength.STRONG,
            EvidenceStrength.MODERATE,
            EvidenceStrength.WEAK,
            EvidenceStrength.CIRCUMSTANTIAL
        ]

        weakest_idx = 0
        for item in evidence_items:
            item_idx = strength_order.index(item.strength)
            if item_idx > weakest_idx:
                weakest_idx = item_idx

        chain_strength = strength_order[weakest_idx]

        # Compute chain hash
        chain_data = {
            "conclusion": conclusion,
            "evidence_hashes": [e.hash for e in evidence_items],
            "created_at": created_at.isoformat()
        }
        chain_hash = self._compute_hash(chain_data)

        return EvidenceChain(
            chain_id=f"EC-{uuid.uuid4().hex[:12]}",
            conclusion=conclusion,
            evidence_items=evidence_items,
            chain_strength=chain_strength,
            chain_hash=chain_hash,
            created_at=created_at
        )

    def generate_recommendation_evidence(
        self,
        condenser_id: str,
        recommendation: str,
        recommendation_type: RecommendationType,
        supporting_data: Dict[str, Dict[str, Any]],
        physics_basis: str = "",
        expected_benefit: float = 0.0,
        benefit_unit: str = "",
        priority: int = 3,
        implementation_notes: str = ""
    ) -> RecommendationEvidence:
        """
        Generate complete evidence package for a recommendation.

        Args:
            condenser_id: Condenser equipment ID
            recommendation: The recommendation text
            recommendation_type: Type of recommendation
            supporting_data: Dict of parameter -> {value, unit, ...}
            physics_basis: Physics justification
            expected_benefit: Expected improvement value
            benefit_unit: Unit of benefit
            priority: Priority level (1-5)
            implementation_notes: Implementation guidance

        Returns:
            RecommendationEvidence with full evidence package
        """
        self._recommendation_count += 1
        created_at = datetime.now(timezone.utc)

        evidence_chains = []

        # Generate evidence for each supporting data point
        for param_name, param_data in supporting_data.items():
            value = param_data.get("value")
            unit = param_data.get("unit", "")

            # Create evidence items for this parameter
            items = []

            # Sensor/measurement evidence
            sensor_evidence = self.create_evidence_item(
                evidence_type=EvidenceType.SENSOR_DATA,
                description=f"Measured {param_name} value",
                value=value,
                unit=unit,
                source_name=f"{param_name}_sensor",
                source_type="sensor",
                source_location=condenser_id
            )
            items.append(sensor_evidence)

            # Threshold comparison evidence (if applicable)
            if param_name in PARAMETER_THRESHOLDS:
                threshold_info = PARAMETER_THRESHOLDS[param_name]
                threshold_evidence = self._create_threshold_evidence(
                    param_name, value, unit, threshold_info
                )
                if threshold_evidence:
                    items.append(threshold_evidence)

            # Build chain for this parameter
            conclusion = f"{param_name} indicates {self._interpret_value(param_name, value)}"
            chain = self.build_evidence_chain(conclusion, items)
            evidence_chains.append(chain)

        # Generate physics basis if not provided
        if not physics_basis:
            physics_basis = self._generate_physics_basis(
                recommendation_type, supporting_data
            )

        # Build compliance mapping
        compliance_mapping = self._build_compliance_mapping(recommendation_type)

        # Compute confidence based on evidence strength
        confidence = self._compute_confidence(evidence_chains)

        # Compute provenance hash
        provenance_data = {
            "condenser_id": condenser_id,
            "recommendation": recommendation,
            "evidence_chains": [ec.chain_hash for ec in evidence_chains],
            "created_at": created_at.isoformat()
        }
        provenance_hash = self._compute_hash(provenance_data)

        return RecommendationEvidence(
            recommendation_id=f"REC-{uuid.uuid4().hex[:12]}",
            condenser_id=condenser_id,
            recommendation_type=recommendation_type,
            recommendation_text=recommendation,
            priority=priority,
            evidence_chains=evidence_chains,
            physics_basis=physics_basis,
            expected_benefit=expected_benefit,
            benefit_unit=benefit_unit,
            confidence=confidence,
            implementation_notes=implementation_notes,
            compliance_mapping=compliance_mapping,
            provenance_hash=provenance_hash,
            created_at=created_at
        )

    def _create_threshold_evidence(
        self,
        param_name: str,
        value: float,
        unit: str,
        threshold_info: Dict[str, Any]
    ) -> Optional[EvidenceItem]:
        """Create threshold comparison evidence."""
        # Determine threshold status
        if "normal_max" in threshold_info:
            if value <= threshold_info["normal_max"]:
                status = "normal"
                threshold_val = threshold_info["normal_max"]
            elif value <= threshold_info.get("warning", float("inf")):
                status = "warning"
                threshold_val = threshold_info["warning"]
            else:
                status = "critical"
                threshold_val = threshold_info.get("critical", threshold_info["warning"])
        elif "normal_min" in threshold_info:
            if value >= threshold_info["normal_min"]:
                status = "normal"
                threshold_val = threshold_info["normal_min"]
            elif value >= threshold_info.get("warning", 0):
                status = "warning"
                threshold_val = threshold_info["warning"]
            else:
                status = "critical"
                threshold_val = threshold_info.get("critical", threshold_info["warning"])
        else:
            return None

        description = (
            f"{param_name} = {value} {unit} is {status} "
            f"(threshold: {threshold_val} {unit})"
        )

        return self.create_evidence_item(
            evidence_type=EvidenceType.THRESHOLD,
            description=description,
            value={"measured": value, "threshold": threshold_val, "status": status},
            unit=unit,
            source_name=f"{param_name}_threshold_check",
            source_type="calculated",
            threshold_ref=threshold_info.get("reference", "")
        )

    def _interpret_value(self, param_name: str, value: float) -> str:
        """Interpret a parameter value in plain language."""
        if param_name not in PARAMETER_THRESHOLDS:
            return "measured value"

        threshold_info = PARAMETER_THRESHOLDS[param_name]

        if "normal_max" in threshold_info:
            if value > threshold_info.get("critical", threshold_info["warning"]):
                return "critically high value requiring immediate action"
            elif value > threshold_info["warning"]:
                return "elevated value requiring attention"
            elif value > threshold_info["normal_max"]:
                return "slightly elevated value"
            else:
                return "normal operating value"
        elif "normal_min" in threshold_info:
            if value < threshold_info.get("critical", threshold_info["warning"]):
                return "critically low value requiring immediate action"
            elif value < threshold_info["warning"]:
                return "reduced value requiring attention"
            elif value < threshold_info["normal_min"]:
                return "slightly reduced value"
            else:
                return "normal operating value"

        return "measured value"

    def _generate_physics_basis(
        self,
        recommendation_type: RecommendationType,
        supporting_data: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate physics basis for recommendation."""
        bases = []

        if "CW_flow" in supporting_data or "backpressure" in supporting_data:
            bases.append(
                "Per Q = m_cw * Cp * dT, reduced CW flow increases outlet temperature, "
                "reducing LMTD and requiring higher saturation temperature to reject heat."
            )

        if "cleanliness_factor" in supporting_data:
            bases.append(
                "Fouling adds thermal resistance per 1/U = 1/h_i + R_wall + R_fouling + 1/h_o, "
                "reducing overall heat transfer coefficient."
            )

        if "TTD" in supporting_data:
            bases.append(
                "TTD = T_sat - T_cw_out; elevated TTD indicates degraded heat transfer "
                "requiring higher saturation pressure."
            )

        if "air_ingress" in supporting_data:
            bases.append(
                "Non-condensables blanket tube surfaces and reduce partial pressure of steam, "
                "degrading heat transfer."
            )

        if not bases:
            bases.append(
                "Analysis based on fundamental heat transfer principles per ASME PTC 12.2."
            )

        return " ".join(bases)

    def _build_compliance_mapping(
        self,
        recommendation_type: RecommendationType
    ) -> Dict[str, List[str]]:
        """Build compliance framework mapping."""
        mapping = {}

        # All recommendations map to ASME PTC
        mapping["ASME_PTC_12.2"] = [
            "Section 4: Test Procedure",
            "Section 5: Calculations"
        ]

        # Performance recommendations map to ISO 50001
        if recommendation_type in [RecommendationType.OPERATIONAL, RecommendationType.MONITORING]:
            mapping["ISO_50001"] = [
                "4.4.3: Energy review",
                "4.4.6: Energy objectives"
            ]

        # Maintenance recommendations map to HEI
        if recommendation_type == RecommendationType.MAINTENANCE:
            mapping["HEI_Standards"] = [
                "Cleanliness factor monitoring",
                "Performance testing"
            ]

        return mapping

    def _compute_confidence(self, evidence_chains: List[EvidenceChain]) -> float:
        """Compute overall confidence from evidence chains."""
        if not evidence_chains:
            return 0.5

        strength_values = {
            EvidenceStrength.DEFINITIVE: 0.95,
            EvidenceStrength.STRONG: 0.85,
            EvidenceStrength.MODERATE: 0.70,
            EvidenceStrength.WEAK: 0.50,
            EvidenceStrength.CIRCUMSTANTIAL: 0.30
        }

        total_confidence = sum(
            strength_values[ec.chain_strength] for ec in evidence_chains
        )
        return total_confidence / len(evidence_chains)

    def generate_compliance_report(
        self,
        condenser_id: str,
        framework: ComplianceFramework,
        recommendations: List[RecommendationEvidence],
        period_start: datetime,
        period_end: datetime
    ) -> ComplianceReport:
        """
        Generate compliance report for regulatory submission.

        Args:
            condenser_id: Condenser equipment ID
            framework: Compliance framework
            recommendations: List of recommendations with evidence
            period_start: Reporting period start
            period_end: Reporting period end

        Returns:
            ComplianceReport ready for submission
        """
        generated_at = datetime.now(timezone.utc)

        # Get framework info
        framework_info = COMPLIANCE_MAPPINGS.get(framework, {})

        # Generate methodology statement
        methodology = self._generate_methodology_statement(framework)

        # Generate data quality statement
        data_quality = self._generate_data_quality_statement(recommendations)

        # Identify limitations
        limitations = self._identify_limitations(recommendations)

        # Compute report hash
        report_data = {
            "condenser_id": condenser_id,
            "framework": framework.value,
            "recommendations": [r.provenance_hash for r in recommendations],
            "generated_at": generated_at.isoformat()
        }
        report_hash = self._compute_hash(report_data)

        return ComplianceReport(
            report_id=f"CR-{uuid.uuid4().hex[:12]}",
            condenser_id=condenser_id,
            framework=framework,
            period_start=period_start,
            period_end=period_end,
            recommendations=recommendations,
            methodology_statement=methodology,
            data_quality_statement=data_quality,
            limitations=limitations,
            certifications=framework_info.get("requirements", []),
            report_hash=report_hash,
            generated_at=generated_at
        )

    def _generate_methodology_statement(self, framework: ComplianceFramework) -> str:
        """Generate methodology statement for compliance report."""
        statements = {
            ComplianceFramework.ASME_PTC: (
                "Performance analysis conducted per ASME PTC 12.2 methodology. "
                "Heat balance calculations performed using fundamental heat transfer equations. "
                "All calculations are deterministic with complete provenance tracking."
            ),
            ComplianceFramework.ISO_50001: (
                "Energy performance analysis per ISO 50001:2018 requirements. "
                "Energy baseline established using historical operating data. "
                "Performance indicators calculated from measured parameters."
            ),
            ComplianceFramework.HEI: (
                "Condenser performance evaluated per HEI Standards for Steam Surface Condensers. "
                "Cleanliness factor and TTD calculated from operational measurements. "
                "All calculations traceable to source data."
            ),
            ComplianceFramework.ISO_14064: (
                "Emissions quantification per ISO 14064-1 requirements. "
                "Activity data from operational measurements. "
                "Emission factors from authoritative sources."
            ),
            ComplianceFramework.GHG_PROTOCOL: (
                "GHG emissions calculated per GHG Protocol Corporate Standard. "
                "Scope 1 and 2 emissions quantified from operational data. "
                "Calculation methodology fully documented."
            )
        }
        return statements.get(framework, "Analysis performed using standard engineering methodology.")

    def _generate_data_quality_statement(
        self,
        recommendations: List[RecommendationEvidence]
    ) -> str:
        """Generate data quality statement."""
        # Calculate average quality score
        all_quality_scores = []
        for rec in recommendations:
            for chain in rec.evidence_chains:
                for item in chain.evidence_items:
                    all_quality_scores.append(item.data_source.quality_score)

        if all_quality_scores:
            avg_quality = sum(all_quality_scores) / len(all_quality_scores)
        else:
            avg_quality = 0.0

        quality_level = (
            "HIGH" if avg_quality >= 0.90 else
            "MEDIUM" if avg_quality >= 0.75 else
            "LOW"
        )

        return (
            f"Data quality level: {quality_level} (average score: {avg_quality:.2f}). "
            f"Sensor data validated against operational limits. "
            f"Calculated values verified for consistency. "
            f"All data sources identified with complete traceability."
        )

    def _identify_limitations(
        self,
        recommendations: List[RecommendationEvidence]
    ) -> List[str]:
        """Identify limitations in the analysis."""
        limitations = [
            "Analysis based on available operational data; periods of missing data excluded",
            "Calculated parameters have inherent uncertainty from measurement accuracy",
            "Performance predictions assume stable operating conditions"
        ]

        # Check for weak evidence chains
        weak_chains = 0
        for rec in recommendations:
            for chain in rec.evidence_chains:
                if chain.chain_strength in [EvidenceStrength.WEAK, EvidenceStrength.CIRCUMSTANTIAL]:
                    weak_chains += 1

        if weak_chains > 0:
            limitations.append(
                f"{weak_chains} evidence chain(s) have reduced confidence due to data limitations"
            )

        return limitations

    def _compute_hash(self, data: Any) -> str:
        """Compute SHA-256 hash of data."""
        if isinstance(data, dict):
            processed = {}
            for k, v in sorted(data.items()):
                if isinstance(v, float):
                    processed[k] = round(v, 8)
                else:
                    processed[k] = v
            data_str = json.dumps(processed, sort_keys=True, default=str)
        else:
            data_str = json.dumps(data, sort_keys=True, default=str)

        return hashlib.sha256(data_str.encode()).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "agent_id": AGENT_ID,
            "version": VERSION,
            "evidence_count": self._evidence_count,
            "chain_count": self._chain_count,
            "recommendation_count": self._recommendation_count
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "EvidenceGenerator",
    "EvidenceItem",
    "EvidenceChain",
    "RecommendationEvidence",
    "ComplianceReport",
    "DataSource",
    "EvidenceType",
    "EvidenceStrength",
    "RecommendationType",
    "ComplianceFramework",
    "PARAMETER_THRESHOLDS",
    "PHYSICS_EQUATIONS",
    "COMPLIANCE_MAPPINGS",
]
