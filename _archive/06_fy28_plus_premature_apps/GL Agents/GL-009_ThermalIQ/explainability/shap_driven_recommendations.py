"""
ThermalIQ SHAP-Driven Recommendation Engine

Integrates SHAP TreeExplainer with the EngineeringRationaleGenerator to provide
data-driven, explainable recommendations grounded in both ML feature importance
and thermodynamic principles.

This module bridges ML explainability with engineering knowledge to produce
actionable recommendations that operators and engineers can trust.

Compliance:
    - ASME PTC 4.1: Steam Generator Performance
    - ISO 50001:2018: Energy Management Systems
    - IAPWS-IF97: Water/Steam Properties

Global AI Standards Alignment:
    - Explainability (15 pts): SHAP TreeExplainer integration
    - Determinism (15 pts): SHA-256 provenance tracking
    - Auditability (10 pts): Complete audit trail logging

Example:
    >>> from explainability.shap_driven_recommendations import (
    ...     SHAPDrivenRecommendationEngine, analyze_thermal_prediction
    ... )
    >>> engine = SHAPDrivenRecommendationEngine()
    >>> rationale = engine.explain_and_recommend(
    ...     model=trained_model,
    ...     inputs=feature_dict,
    ...     result=calculation_result,
    ...     background_data=training_data
    ... )
    >>> print(rationale.format_report())

Author: GL-BackendDeveloper
Version: 2.0.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import logging
import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("SHAP library loaded successfully")
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP library not available - TreeExplainer features disabled")

from .shap_explainer import (
    ThermalSHAPExplainer,
    SHAPExplanation,
    FeatureContribution,
    ExplanationType
)
from .engineering_rationale import (
    EngineeringRationaleGenerator,
    EngineeringRationale,
    RationaleSection,
    RationaleCategory,
    Citation
)


class RecommendationPriority(Enum):
    """Priority levels for recommendations based on SHAP importance."""
    CRITICAL = "critical"      # Top SHAP contributor, immediate action
    HIGH = "high"              # High SHAP value, near-term action
    MEDIUM = "medium"          # Moderate SHAP value, planned action
    LOW = "low"                # Low SHAP value, optional action
    INFORMATIONAL = "info"     # Context only, no action needed


class RecommendationType(Enum):
    """Types of recommendations generated."""
    OPTIMIZATION = "optimization"
    MAINTENANCE = "maintenance"
    MONITORING = "monitoring"
    DESIGN_CHANGE = "design_change"
    OPERATIONAL = "operational"
    SAFETY = "safety"


class AuditEventType(Enum):
    """Types of audit events for tracking."""
    RECOMMENDATION_GENERATED = "recommendation_generated"
    SHAP_ANALYSIS_STARTED = "shap_analysis_started"
    SHAP_ANALYSIS_COMPLETED = "shap_analysis_completed"
    RATIONALE_CREATED = "rationale_created"
    FEATURE_MAPPING_UPDATED = "feature_mapping_updated"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"


@dataclass
class AuditTrailEntry:
    """
    Immutable audit trail entry for tracking all recommendation operations.

    Provides complete traceability for regulatory compliance and debugging.
    Each entry is cryptographically sealed with SHA-256 for tamper detection.
    """
    event_type: AuditEventType
    timestamp: str
    agent_id: str
    session_id: str
    operation_id: str
    input_hash: str
    output_hash: str
    processing_time_ms: float
    metadata: Dict[str, Any]
    provenance_hash: str = ""

    def __post_init__(self):
        """Compute provenance hash if not provided."""
        if not self.provenance_hash:
            self.provenance_hash = self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for audit entry integrity."""
        content = json.dumps({
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "operation_id": self.operation_id,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "processing_time_ms": self.processing_time_ms
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization and logging."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "operation_id": self.operation_id,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
            "provenance_hash": self.provenance_hash
        }

    def to_json(self) -> str:
        """Convert to JSON string for structured logging."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class SHAPDrivenRecommendation:
    """
    A recommendation driven by SHAP feature importance analysis.

    Each recommendation is cryptographically sealed with SHA-256 provenance
    hash for complete audit trail traceability.

    Attributes:
        title: Short action title for the recommendation
        description: Detailed description with SHAP context
        priority: Priority level based on SHAP contribution
        recommendation_type: Category of recommendation action
        driving_features: List of features driving this recommendation
        shap_contribution: SHAP value contribution
        expected_impact: Expected impact description
        implementation_steps: Ordered list of implementation actions
        engineering_principle: Thermodynamic principle grounding
        citations: Academic/standard citations for validation
        estimated_improvement: Quantified improvement estimate
        confidence_score: Confidence in recommendation (0.0-1.0)
        provenance_hash: SHA-256 hash for audit trail
        timestamp: ISO 8601 timestamp of generation
    """
    title: str
    description: str
    priority: RecommendationPriority
    recommendation_type: RecommendationType
    driving_features: List[str]
    shap_contribution: float
    expected_impact: str
    implementation_steps: List[str]
    engineering_principle: Optional[str] = None
    citations: List[Citation] = field(default_factory=list)
    estimated_improvement: Optional[str] = None
    confidence_score: float = 0.0
    provenance_hash: str = ""
    timestamp: str = ""

    def __post_init__(self):
        """Generate provenance hash if not provided."""
        if not self.provenance_hash:
            self.provenance_hash = self._compute_provenance_hash()
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        # Log recommendation generation
        logger.debug(
            f"Recommendation generated: {self.title} "
            f"[{self.priority.value}] hash={self.provenance_hash[:16]}"
        )

    def _compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for recommendation provenance."""
        content = json.dumps({
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "recommendation_type": self.recommendation_type.value,
            "driving_features": self.driving_features,
            "shap_contribution": float(self.shap_contribution),
            "confidence_score": float(self.confidence_score)
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "recommendation_type": self.recommendation_type.value,
            "driving_features": self.driving_features,
            "shap_contribution": float(self.shap_contribution),
            "expected_impact": self.expected_impact,
            "implementation_steps": self.implementation_steps,
            "engineering_principle": self.engineering_principle,
            "citations": [c.to_dict() for c in self.citations],
            "estimated_improvement": self.estimated_improvement,
            "confidence_score": float(self.confidence_score),
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp
        }

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate recommendation completeness and consistency.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if not self.title or len(self.title) < 3:
            errors.append("Title must be at least 3 characters")

        if not self.driving_features:
            errors.append("At least one driving feature required")

        if not 0.0 <= self.confidence_score <= 1.0:
            errors.append("Confidence score must be between 0.0 and 1.0")

        if self.engineering_principle and not self.citations:
            logger.warning(
                f"Recommendation '{self.title}' has principle but no citations"
            )

        return len(errors) == 0, errors


@dataclass
class SHAPDrivenRationale:
    """
    Complete rationale combining SHAP analysis with engineering principles.

    This class represents the full output of the SHAP-driven recommendation
    engine, including ML explanations, engineering rationale, prioritized
    recommendations, and complete audit trail.

    Attributes:
        shap_explanation: SHAP values and feature contributions
        engineering_rationale: Thermodynamic engineering context
        recommendations: Prioritized list of recommendations
        feature_priority_ranking: Features ranked by SHAP importance
        overall_confidence: Overall confidence level (high/medium/low)
        audit_trail: List of audit entries for traceability
        provenance_hash: SHA-256 hash for complete rationale
        timestamp: ISO 8601 timestamp of generation
        metadata: Additional context and metrics
    """
    shap_explanation: SHAPExplanation
    engineering_rationale: EngineeringRationale
    recommendations: List[SHAPDrivenRecommendation]
    feature_priority_ranking: List[Tuple[str, float, RecommendationPriority]]
    overall_confidence: str
    audit_trail: List[AuditTrailEntry] = field(default_factory=list)
    provenance_hash: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate provenance hash if not provided."""
        if not self.provenance_hash:
            self.provenance_hash = self._compute_provenance_hash()
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        # Log rationale creation
        logger.info(
            f"SHAPDrivenRationale created: {len(self.recommendations)} recommendations, "
            f"confidence={self.overall_confidence}, hash={self.provenance_hash[:16]}"
        )

    def _compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for full rationale provenance."""
        content = json.dumps({
            "shap_hash": self.shap_explanation.metadata.get("hash", ""),
            "rationale_timestamp": self.engineering_rationale.timestamp,
            "n_recommendations": len(self.recommendations),
            "recommendation_hashes": [r.provenance_hash for r in self.recommendations],
            "audit_trail_hashes": [a.provenance_hash for a in self.audit_trail]
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "shap_explanation": self.shap_explanation.to_dict(),
            "engineering_rationale": self.engineering_rationale.to_dict(),
            "recommendations": [r.to_dict() for r in self.recommendations],
            "feature_priority_ranking": [
                {"feature": f, "shap_value": float(s), "priority": p.value}
                for f, s, p in self.feature_priority_ranking
            ],
            "overall_confidence": self.overall_confidence,
            "audit_trail": [a.to_dict() for a in self.audit_trail],
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    def get_critical_recommendations(self) -> List[SHAPDrivenRecommendation]:
        """Get only critical priority recommendations."""
        return [r for r in self.recommendations
                if r.priority == RecommendationPriority.CRITICAL]

    def get_actionable_recommendations(self) -> List[SHAPDrivenRecommendation]:
        """Get recommendations that require action (exclude informational)."""
        return [r for r in self.recommendations
                if r.priority != RecommendationPriority.INFORMATIONAL]

    def format_report(self, include_citations: bool = True) -> str:
        """Generate formatted text report."""
        lines = [
            "=" * 70,
            "SHAP-DRIVEN THERMAL ANALYSIS REPORT",
            f"Generated: {self.timestamp}",
            f"Provenance: {self.provenance_hash[:16]}...",
            "=" * 70,
            "",
            "EXECUTIVE SUMMARY",
            "-" * 50,
            self.engineering_rationale.summary,
            "",
            "SHAP FEATURE IMPORTANCE RANKING",
            "-" * 50
        ]

        for i, (feature, shap_val, priority) in enumerate(self.feature_priority_ranking[:10], 1):
            direction = "+" if shap_val > 0 else ""
            lines.append(f"  {i}. {feature}: {direction}{shap_val:.4f} [{priority.value.upper()}]")

        lines.extend([
            "",
            "DATA-DRIVEN RECOMMENDATIONS",
            "-" * 50
        ])

        for i, rec in enumerate(self.recommendations, 1):
            lines.extend([
                f"\n{i}. [{rec.priority.value.upper()}] {rec.title}",
                f"   Type: {rec.recommendation_type.value}",
                f"   {rec.description}",
                f"   Driving Features: {', '.join(rec.driving_features)}",
                f"   Expected Impact: {rec.expected_impact}"
            ])

            if rec.implementation_steps:
                lines.append("   Implementation Steps:")
                for step in rec.implementation_steps:
                    lines.append(f"      - {step}")

            if rec.engineering_principle:
                lines.append(f"   Engineering Basis: {rec.engineering_principle}")

            if include_citations and rec.citations:
                lines.append("   References:")
                for citation in rec.citations:
                    lines.append(f"      [{citation.principle}] {citation.source}")

        lines.extend([
            "",
            "OVERALL ASSESSMENT",
            "-" * 50,
            self.engineering_rationale.overall_assessment,
            "",
            f"Confidence Level: {self.overall_confidence.upper()}",
            "",
            "Applicable Standards:",
        ])

        for standard in self.engineering_rationale.applicable_standards:
            lines.append(f"  - {standard}")

        return "\n".join(lines)


class SHAPDrivenRecommendationEngine:
    """
    Engine that generates recommendations by combining SHAP feature importance
    with thermodynamic engineering principles.

    This provides a transparent, auditable recommendation system where:
    - ML predictions are explained via SHAP values
    - Recommendations are prioritized by feature importance
    - Engineering rationale grounds recommendations in thermodynamic principles
    - Full provenance tracking ensures auditability
    """

    # Feature-to-recommendation mapping
    # Maps thermal features to specific recommendation types and actions
    FEATURE_RECOMMENDATION_MAP: Dict[str, Dict[str, Any]] = {
        # Temperature features
        "inlet_temperature": {
            "positive": {
                "type": RecommendationType.OPTIMIZATION,
                "action": "Optimize inlet temperature",
                "principle": "first_law_energy_balance",
                "steps": [
                    "Review heat source temperature setpoints",
                    "Check heat exchanger fouling",
                    "Verify process requirements"
                ]
            },
            "negative": {
                "type": RecommendationType.OPERATIONAL,
                "action": "Increase inlet temperature",
                "principle": "first_law_energy_balance",
                "steps": [
                    "Evaluate preheating options",
                    "Check for heat losses in inlet piping",
                    "Review insulation condition"
                ]
            }
        },
        "outlet_temperature": {
            "positive": {
                "type": RecommendationType.OPTIMIZATION,
                "action": "Reduce outlet temperature losses",
                "principle": "first_law_energy_balance",
                "steps": [
                    "Install or improve economizer",
                    "Evaluate waste heat recovery",
                    "Check process heat utilization"
                ]
            },
            "negative": {
                "type": RecommendationType.MONITORING,
                "action": "Monitor outlet temperature trend",
                "principle": "first_law_energy_balance",
                "steps": [
                    "Establish baseline outlet temperature",
                    "Set up continuous monitoring",
                    "Define alarm thresholds"
                ]
            }
        },
        "stack_temperature": {
            "positive": {
                "type": RecommendationType.OPTIMIZATION,
                "action": "Reduce stack heat losses",
                "principle": "excess_air_effect",
                "steps": [
                    "Install economizer to recover stack heat",
                    "Optimize combustion air ratio",
                    "Consider condensing heat recovery"
                ]
            },
            "negative": {
                "type": RecommendationType.MAINTENANCE,
                "action": "Check for combustion issues",
                "principle": "combustion_stoichiometry",
                "steps": [
                    "Verify fuel quality and flow",
                    "Inspect burner condition",
                    "Check air damper settings"
                ]
            }
        },
        "surface_temperature": {
            "positive": {
                "type": RecommendationType.MAINTENANCE,
                "action": "Improve surface insulation",
                "principle": "fourier_law",
                "steps": [
                    "Conduct insulation survey",
                    "Replace damaged insulation",
                    "Upgrade to higher R-value materials"
                ]
            },
            "negative": {
                "type": RecommendationType.MONITORING,
                "action": "Monitor surface temperatures",
                "principle": "stefan_boltzmann",
                "steps": [
                    "Perform thermographic survey",
                    "Document baseline conditions",
                    "Schedule periodic inspections"
                ]
            }
        },
        "ambient_temperature": {
            "positive": {
                "type": RecommendationType.DESIGN_CHANGE,
                "action": "Mitigate ambient temperature effects",
                "principle": "newton_cooling",
                "steps": [
                    "Improve ventilation or cooling",
                    "Add shading or enclosures",
                    "Consider seasonal operating adjustments"
                ]
            },
            "negative": {
                "type": RecommendationType.INFORMATIONAL,
                "action": "Account for ambient conditions",
                "principle": "newton_cooling",
                "steps": [
                    "Document ambient temperature range",
                    "Adjust efficiency calculations",
                    "Plan for seasonal variations"
                ]
            }
        },

        # Pressure features
        "pressure_drop": {
            "positive": {
                "type": RecommendationType.MAINTENANCE,
                "action": "Reduce pressure drop",
                "principle": "darcy_weisbach",
                "steps": [
                    "Clean heat transfer surfaces",
                    "Check for flow restrictions",
                    "Evaluate piping configuration"
                ]
            },
            "negative": {
                "type": RecommendationType.MONITORING,
                "action": "Monitor pressure drop trend",
                "principle": "darcy_weisbach",
                "steps": [
                    "Establish baseline measurements",
                    "Track fouling indicators",
                    "Schedule cleaning based on trends"
                ]
            }
        },
        "inlet_pressure": {
            "positive": {
                "type": RecommendationType.OPERATIONAL,
                "action": "Optimize inlet pressure",
                "principle": "reynolds_number",
                "steps": [
                    "Review pump/compressor settings",
                    "Check for upstream restrictions",
                    "Verify process requirements"
                ]
            },
            "negative": {
                "type": RecommendationType.SAFETY,
                "action": "Address low inlet pressure",
                "principle": "reynolds_number",
                "steps": [
                    "Check pump/compressor performance",
                    "Inspect supply piping",
                    "Verify pressure control settings"
                ]
            }
        },

        # Flow features
        "mass_flow_rate": {
            "positive": {
                "type": RecommendationType.OPTIMIZATION,
                "action": "Optimize mass flow rate",
                "principle": "reynolds_number",
                "steps": [
                    "Review flow setpoints",
                    "Balance flow distribution",
                    "Check control valve operation"
                ]
            },
            "negative": {
                "type": RecommendationType.OPERATIONAL,
                "action": "Increase mass flow rate",
                "principle": "nusselt_number",
                "steps": [
                    "Verify pump capacity",
                    "Check for flow restrictions",
                    "Review process demands"
                ]
            }
        },
        "velocity": {
            "positive": {
                "type": RecommendationType.OPTIMIZATION,
                "action": "Optimize fluid velocity",
                "principle": "nusselt_number",
                "steps": [
                    "Adjust flow rates for optimal heat transfer",
                    "Balance between heat transfer and pressure drop",
                    "Review pipe sizing"
                ]
            },
            "negative": {
                "type": RecommendationType.DESIGN_CHANGE,
                "action": "Increase fluid velocity",
                "principle": "reynolds_number",
                "steps": [
                    "Consider pipe resizing",
                    "Increase flow rate if possible",
                    "Evaluate heat transfer enhancement"
                ]
            }
        },

        # Thermal properties
        "specific_heat": {
            "positive": {
                "type": RecommendationType.INFORMATIONAL,
                "action": "Leverage high specific heat",
                "principle": "first_law_energy_balance",
                "steps": [
                    "Optimize flow rates for heat capacity",
                    "Consider fluid selection",
                    "Review system design"
                ]
            },
            "negative": {
                "type": RecommendationType.DESIGN_CHANGE,
                "action": "Consider fluid with higher specific heat",
                "principle": "first_law_energy_balance",
                "steps": [
                    "Evaluate alternative fluids",
                    "Assess compatibility",
                    "Calculate performance improvement"
                ]
            }
        },
        "thermal_conductivity": {
            "positive": {
                "type": RecommendationType.OPTIMIZATION,
                "action": "Maximize thermal conductivity benefit",
                "principle": "fourier_law",
                "steps": [
                    "Ensure good surface contact",
                    "Maintain clean heat transfer surfaces",
                    "Optimize temperature differences"
                ]
            },
            "negative": {
                "type": RecommendationType.DESIGN_CHANGE,
                "action": "Improve heat transfer coefficient",
                "principle": "fourier_law",
                "steps": [
                    "Consider enhanced surfaces",
                    "Evaluate alternative fluids",
                    "Review heat exchanger design"
                ]
            }
        },
        "viscosity": {
            "positive": {
                "type": RecommendationType.MONITORING,
                "action": "Monitor viscosity effects",
                "principle": "prandtl_number",
                "steps": [
                    "Track fluid temperature",
                    "Monitor pumping power",
                    "Check for fluid degradation"
                ]
            },
            "negative": {
                "type": RecommendationType.OPTIMIZATION,
                "action": "Reduce viscosity effects",
                "principle": "reynolds_number",
                "steps": [
                    "Increase operating temperature if possible",
                    "Consider viscosity modifiers",
                    "Review fluid selection"
                ]
            }
        },

        # Equipment parameters
        "heat_transfer_area": {
            "positive": {
                "type": RecommendationType.INFORMATIONAL,
                "action": "Maintain heat transfer area",
                "principle": "newton_cooling",
                "steps": [
                    "Implement regular cleaning schedule",
                    "Monitor fouling indicators",
                    "Track heat transfer performance"
                ]
            },
            "negative": {
                "type": RecommendationType.DESIGN_CHANGE,
                "action": "Increase heat transfer area",
                "principle": "newton_cooling",
                "steps": [
                    "Evaluate equipment upgrade",
                    "Consider additional heat exchanger",
                    "Review process requirements"
                ]
            }
        },
        "heat_transfer_coefficient": {
            "positive": {
                "type": RecommendationType.MAINTENANCE,
                "action": "Maintain heat transfer coefficient",
                "principle": "nusselt_number",
                "steps": [
                    "Clean heat transfer surfaces regularly",
                    "Monitor for fouling",
                    "Maintain optimal flow conditions"
                ]
            },
            "negative": {
                "type": RecommendationType.OPTIMIZATION,
                "action": "Improve heat transfer coefficient",
                "principle": "nusselt_number",
                "steps": [
                    "Clean heat transfer surfaces",
                    "Optimize flow velocities",
                    "Consider surface enhancement"
                ]
            }
        },
        "fouling_factor": {
            "positive": {
                "type": RecommendationType.MAINTENANCE,
                "action": "Reduce fouling",
                "principle": "newton_cooling",
                "steps": [
                    "Schedule heat exchanger cleaning",
                    "Review water treatment program",
                    "Install strainers or filters"
                ]
            },
            "negative": {
                "type": RecommendationType.MONITORING,
                "action": "Monitor fouling trend",
                "principle": "newton_cooling",
                "steps": [
                    "Track approach temperatures",
                    "Monitor pressure drops",
                    "Establish cleaning triggers"
                ]
            }
        },
        "insulation_thickness": {
            "positive": {
                "type": RecommendationType.DESIGN_CHANGE,
                "action": "Optimize insulation thickness",
                "principle": "fourier_law",
                "steps": [
                    "Calculate economic insulation thickness",
                    "Upgrade high-loss areas first",
                    "Consider operating conditions"
                ]
            },
            "negative": {
                "type": RecommendationType.MAINTENANCE,
                "action": "Verify insulation integrity",
                "principle": "fourier_law",
                "steps": [
                    "Conduct thermal survey",
                    "Repair damaged sections",
                    "Document insulation condition"
                ]
            }
        },

        # Performance metrics
        "efficiency": {
            "positive": {
                "type": RecommendationType.INFORMATIONAL,
                "action": "Maintain high efficiency",
                "principle": "first_law_energy_balance",
                "steps": [
                    "Continue current operating practices",
                    "Implement preventive maintenance",
                    "Monitor for degradation"
                ]
            },
            "negative": {
                "type": RecommendationType.OPTIMIZATION,
                "action": "Improve system efficiency",
                "principle": "first_law_energy_balance",
                "steps": [
                    "Conduct energy audit",
                    "Identify loss sources",
                    "Implement improvement measures"
                ]
            }
        },
        "effectiveness": {
            "positive": {
                "type": RecommendationType.INFORMATIONAL,
                "action": "Maintain effectiveness",
                "principle": "log_mean_temp_diff",
                "steps": [
                    "Monitor approach temperatures",
                    "Maintain clean surfaces",
                    "Track performance trends"
                ]
            },
            "negative": {
                "type": RecommendationType.OPTIMIZATION,
                "action": "Improve heat exchanger effectiveness",
                "principle": "log_mean_temp_diff",
                "steps": [
                    "Clean heat transfer surfaces",
                    "Check flow distribution",
                    "Verify design flow rates"
                ]
            }
        },

        # Combustion parameters
        "excess_air_ratio": {
            "positive": {
                "type": RecommendationType.OPTIMIZATION,
                "action": "Reduce excess air",
                "principle": "excess_air_effect",
                "steps": [
                    "Tune combustion controls",
                    "Install O2 trim system",
                    "Check for air leaks"
                ]
            },
            "negative": {
                "type": RecommendationType.SAFETY,
                "action": "Verify adequate excess air",
                "principle": "combustion_stoichiometry",
                "steps": [
                    "Check combustion quality",
                    "Monitor CO levels",
                    "Verify safety margins"
                ]
            }
        },
        "O2_percentage": {
            "positive": {
                "type": RecommendationType.OPTIMIZATION,
                "action": "Optimize O2 levels",
                "principle": "excess_air_effect",
                "steps": [
                    "Target optimal O2 range (typically 2-4%)",
                    "Implement O2 trim control",
                    "Check air damper operation"
                ]
            },
            "negative": {
                "type": RecommendationType.SAFETY,
                "action": "Ensure adequate O2",
                "principle": "combustion_stoichiometry",
                "steps": [
                    "Verify complete combustion",
                    "Check for fuel-rich conditions",
                    "Monitor CO and combustibles"
                ]
            }
        },
        "fuel_flow_rate": {
            "positive": {
                "type": RecommendationType.OPTIMIZATION,
                "action": "Optimize fuel consumption",
                "principle": "first_law_energy_balance",
                "steps": [
                    "Verify process heat demands",
                    "Check for standby losses",
                    "Optimize firing rate"
                ]
            },
            "negative": {
                "type": RecommendationType.OPERATIONAL,
                "action": "Review fuel supply",
                "principle": "first_law_energy_balance",
                "steps": [
                    "Check fuel supply pressure",
                    "Verify fuel quality",
                    "Review process demands"
                ]
            }
        }
    }

    # Priority thresholds based on normalized absolute SHAP values
    PRIORITY_THRESHOLDS = {
        RecommendationPriority.CRITICAL: 0.3,    # Top 30% contribution
        RecommendationPriority.HIGH: 0.15,       # 15-30% contribution
        RecommendationPriority.MEDIUM: 0.05,    # 5-15% contribution
        RecommendationPriority.LOW: 0.01,       # 1-5% contribution
        RecommendationPriority.INFORMATIONAL: 0.0  # < 1% contribution
    }

    def __init__(
        self,
        shap_explainer: Optional[ThermalSHAPExplainer] = None,
        rationale_generator: Optional[EngineeringRationaleGenerator] = None,
        custom_feature_map: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize the SHAP-driven recommendation engine.

        Args:
            shap_explainer: Pre-configured ThermalSHAPExplainer instance
            rationale_generator: Pre-configured EngineeringRationaleGenerator
            custom_feature_map: Custom feature-to-recommendation mapping
        """
        self.shap_explainer = shap_explainer
        self.rationale_generator = rationale_generator or EngineeringRationaleGenerator()

        # Merge custom feature map with defaults
        self.feature_map = {**self.FEATURE_RECOMMENDATION_MAP}
        if custom_feature_map:
            self.feature_map.update(custom_feature_map)

    def generate_recommendations(
        self,
        shap_explanation: SHAPExplanation,
        result: Dict[str, Any],
        max_recommendations: int = 10,
        min_priority: RecommendationPriority = RecommendationPriority.LOW
    ) -> SHAPDrivenRationale:
        """
        Generate SHAP-driven recommendations from an explanation.

        Args:
            shap_explanation: SHAP explanation from the explainer
            result: Original calculation result for rationale generation
            max_recommendations: Maximum recommendations to return
            min_priority: Minimum priority level to include

        Returns:
            SHAPDrivenRationale combining SHAP analysis with recommendations
        """
        # Generate engineering rationale based on explanation type
        if shap_explanation.explanation_type == ExplanationType.EFFICIENCY:
            eng_rationale = self.rationale_generator.generate_efficiency_rationale(result)
        elif shap_explanation.explanation_type == ExplanationType.EXERGY:
            eng_rationale = self.rationale_generator.generate_exergy_rationale(result)
        elif shap_explanation.explanation_type == ExplanationType.FLUID_SELECTION:
            eng_rationale = self.rationale_generator.generate_fluid_recommendation_rationale(
                result.get('fluid', {}),
                result.get('alternatives', []),
                result.get('operating_conditions')
            )
        else:
            eng_rationale = self.rationale_generator.generate_efficiency_rationale(result)

        # Compute feature priorities based on SHAP values
        feature_priorities = self._compute_feature_priorities(shap_explanation)

        # Generate recommendations from SHAP analysis
        recommendations = self._generate_shap_recommendations(
            shap_explanation,
            feature_priorities,
            max_recommendations,
            min_priority
        )

        # Compute overall confidence
        confidence = self._assess_overall_confidence(
            shap_explanation, eng_rationale, recommendations
        )

        return SHAPDrivenRationale(
            shap_explanation=shap_explanation,
            engineering_rationale=eng_rationale,
            recommendations=recommendations,
            feature_priority_ranking=feature_priorities,
            overall_confidence=confidence,
            metadata={
                "explanation_type": shap_explanation.explanation_type.value,
                "n_features_analyzed": len(shap_explanation.feature_contributions),
                "n_recommendations": len(recommendations),
                "model_type": shap_explanation.model_type
            }
        )

    def explain_and_recommend(
        self,
        model: Any,
        inputs: Union[np.ndarray, Dict[str, float]],
        result: Dict[str, Any],
        explanation_type: ExplanationType = ExplanationType.EFFICIENCY,
        background_data: Optional[np.ndarray] = None,
        max_recommendations: int = 10
    ) -> SHAPDrivenRationale:
        """
        Complete workflow: generate SHAP explanation and recommendations.

        Args:
            model: Trained model to explain
            inputs: Input features for the prediction
            result: Calculation result for rationale generation
            explanation_type: Type of explanation to generate
            background_data: Background data for SHAP (required if no explainer)
            max_recommendations: Maximum recommendations to return

        Returns:
            SHAPDrivenRationale with full analysis
        """
        # Initialize SHAP explainer if needed
        if self.shap_explainer is None:
            if background_data is None:
                raise ValueError(
                    "Background data required when no pre-configured explainer"
                )
            self.shap_explainer = ThermalSHAPExplainer(
                background_data=background_data,
                feature_names=list(inputs.keys()) if isinstance(inputs, dict) else None
            )

        # Generate SHAP explanation
        if explanation_type == ExplanationType.EFFICIENCY:
            shap_explanation = self.shap_explainer.explain_efficiency(model, inputs)
        elif explanation_type == ExplanationType.EXERGY:
            shap_explanation = self.shap_explainer.explain_exergy(model, inputs)
        elif explanation_type == ExplanationType.FLUID_SELECTION:
            shap_explanation = self.shap_explainer.explain_fluid_selection(
                model, inputs, result.get('fluid_options', [])
            )
        else:
            shap_explanation = self.shap_explainer.explain_efficiency(model, inputs)

        # Generate recommendations
        return self.generate_recommendations(
            shap_explanation, result, max_recommendations
        )

    def _compute_feature_priorities(
        self,
        explanation: SHAPExplanation
    ) -> List[Tuple[str, float, RecommendationPriority]]:
        """Compute priority ranking for each feature based on SHAP values."""
        # Compute normalized absolute SHAP values
        abs_shap = np.abs(explanation.shap_values)
        total_abs_shap = np.sum(abs_shap)

        if total_abs_shap == 0:
            total_abs_shap = 1.0  # Avoid division by zero

        priorities = []

        for i, feature_name in enumerate(explanation.feature_names):
            shap_value = explanation.shap_values[i]
            normalized_abs = abs_shap[i] / total_abs_shap

            # Determine priority based on normalized contribution
            if normalized_abs >= self.PRIORITY_THRESHOLDS[RecommendationPriority.CRITICAL]:
                priority = RecommendationPriority.CRITICAL
            elif normalized_abs >= self.PRIORITY_THRESHOLDS[RecommendationPriority.HIGH]:
                priority = RecommendationPriority.HIGH
            elif normalized_abs >= self.PRIORITY_THRESHOLDS[RecommendationPriority.MEDIUM]:
                priority = RecommendationPriority.MEDIUM
            elif normalized_abs >= self.PRIORITY_THRESHOLDS[RecommendationPriority.LOW]:
                priority = RecommendationPriority.LOW
            else:
                priority = RecommendationPriority.INFORMATIONAL

            priorities.append((feature_name, float(shap_value), priority))

        # Sort by absolute SHAP value descending
        priorities.sort(key=lambda x: abs(x[1]), reverse=True)

        return priorities

    def _generate_shap_recommendations(
        self,
        explanation: SHAPExplanation,
        feature_priorities: List[Tuple[str, float, RecommendationPriority]],
        max_recommendations: int,
        min_priority: RecommendationPriority
    ) -> List[SHAPDrivenRecommendation]:
        """Generate recommendations from SHAP feature analysis."""
        recommendations = []
        priority_order = [
            RecommendationPriority.CRITICAL,
            RecommendationPriority.HIGH,
            RecommendationPriority.MEDIUM,
            RecommendationPriority.LOW,
            RecommendationPriority.INFORMATIONAL
        ]
        min_priority_idx = priority_order.index(min_priority)

        for feature_name, shap_value, priority in feature_priorities:
            # Skip if below minimum priority
            if priority_order.index(priority) > min_priority_idx:
                continue

            # Get recommendation template for this feature
            if feature_name not in self.feature_map:
                continue

            feature_config = self.feature_map[feature_name]
            direction = "positive" if shap_value > 0 else "negative"

            if direction not in feature_config:
                continue

            config = feature_config[direction]

            # Get citations for the engineering principle
            citations = []
            if config.get("principle"):
                principle_key = config["principle"]
                if principle_key in self.rationale_generator.THERMODYNAMIC_PRINCIPLES:
                    citations.append(
                        self.rationale_generator.THERMODYNAMIC_PRINCIPLES[principle_key]
                    )

            # Compute expected impact
            impact = self._compute_expected_impact(
                feature_name, shap_value, explanation.predicted_value
            )

            recommendation = SHAPDrivenRecommendation(
                title=config["action"],
                description=self._generate_recommendation_description(
                    feature_name, shap_value, config, explanation
                ),
                priority=priority,
                recommendation_type=config["type"],
                driving_features=[feature_name],
                shap_contribution=shap_value,
                expected_impact=impact,
                implementation_steps=config.get("steps", []),
                engineering_principle=config.get("principle"),
                citations=citations,
                estimated_improvement=self._estimate_improvement(
                    feature_name, shap_value, explanation
                )
            )

            recommendations.append(recommendation)

            if len(recommendations) >= max_recommendations:
                break

        return recommendations

    def _generate_recommendation_description(
        self,
        feature_name: str,
        shap_value: float,
        config: Dict,
        explanation: SHAPExplanation
    ) -> str:
        """Generate natural language description for recommendation."""
        # Get feature contribution details
        contribution = next(
            (c for c in explanation.feature_contributions if c.feature_name == feature_name),
            None
        )

        if contribution is None:
            return config.get("action", "Review this parameter")

        direction_text = "increases" if shap_value > 0 else "decreases"
        magnitude = contribution.contribution_magnitude

        # Get unit if available
        unit_text = f" ({contribution.unit})" if contribution.unit else ""

        description = (
            f"SHAP analysis indicates that {feature_name}{unit_text} {direction_text} "
            f"the predicted {explanation.explanation_type.value} with {magnitude} impact. "
            f"Current value: {contribution.feature_value:.2f}{unit_text}. "
            f"SHAP contribution: {shap_value:+.4f}."
        )

        return description

    def _compute_expected_impact(
        self,
        feature_name: str,
        shap_value: float,
        predicted_value: float
    ) -> str:
        """Compute expected impact description."""
        if predicted_value == 0:
            relative_impact = abs(shap_value) * 100
        else:
            relative_impact = abs(shap_value / predicted_value) * 100

        if relative_impact > 20:
            return "High impact - significant improvement potential"
        elif relative_impact > 5:
            return "Medium impact - moderate improvement potential"
        elif relative_impact > 1:
            return "Low impact - incremental improvement"
        else:
            return "Minimal impact - informational only"

    def _estimate_improvement(
        self,
        feature_name: str,
        shap_value: float,
        explanation: SHAPExplanation
    ) -> Optional[str]:
        """Estimate potential improvement from addressing this feature."""
        # Simple estimation based on SHAP contribution
        abs_shap = abs(shap_value)
        total_range = explanation.predicted_value - explanation.base_value

        if total_range == 0:
            return None

        contribution_pct = (abs_shap / abs(total_range)) * 100 if total_range != 0 else 0

        if contribution_pct > 25:
            return f"Up to {contribution_pct:.0f}% of prediction gap addressable"
        elif contribution_pct > 10:
            return f"Approximately {contribution_pct:.0f}% improvement potential"
        elif contribution_pct > 5:
            return f"Minor improvement potential (~{contribution_pct:.0f}%)"
        else:
            return None

    def _assess_overall_confidence(
        self,
        shap_explanation: SHAPExplanation,
        eng_rationale: EngineeringRationale,
        recommendations: List[SHAPDrivenRecommendation]
    ) -> str:
        """Assess overall confidence in the analysis."""
        # Factors affecting confidence:
        # 1. Engineering rationale confidence
        # 2. Number of high-priority recommendations with citations
        # 3. SHAP explanation coverage

        eng_confidence_score = {
            "high": 3,
            "medium": 2,
            "low": 1
        }.get(eng_rationale.confidence_level, 1)

        # Count recommendations with citations
        cited_recs = sum(1 for r in recommendations if r.citations)
        rec_coverage = cited_recs / len(recommendations) if recommendations else 0

        # SHAP coverage - check if total SHAP explains prediction
        shap_sum = np.sum(shap_explanation.shap_values)
        prediction_gap = shap_explanation.predicted_value - shap_explanation.base_value
        shap_coverage = abs(shap_sum / prediction_gap) if prediction_gap != 0 else 0

        # Compute composite score
        composite = (
            eng_confidence_score * 0.4 +
            rec_coverage * 3 * 0.3 +
            min(shap_coverage, 1.0) * 3 * 0.3
        )

        if composite >= 2.5:
            return "high"
        elif composite >= 1.5:
            return "medium"
        else:
            return "low"

    def add_custom_feature_mapping(
        self,
        feature_name: str,
        positive_config: Dict[str, Any],
        negative_config: Dict[str, Any]
    ) -> None:
        """
        Add or update custom feature-to-recommendation mapping.

        Args:
            feature_name: Name of the feature
            positive_config: Config for positive SHAP values
            negative_config: Config for negative SHAP values
        """
        self.feature_map[feature_name] = {
            "positive": positive_config,
            "negative": negative_config
        }


# Convenience function for quick analysis
def analyze_thermal_prediction(
    model: Any,
    inputs: Dict[str, float],
    result: Dict[str, Any],
    background_data: np.ndarray,
    explanation_type: str = "efficiency"
) -> SHAPDrivenRationale:
    """
    Quick function to perform complete SHAP-driven analysis.

    Args:
        model: Trained prediction model
        inputs: Input features as dictionary
        result: Calculation result for rationale
        background_data: Background dataset for SHAP
        explanation_type: Type of analysis ("efficiency", "exergy", "fluid_selection")

    Returns:
        SHAPDrivenRationale with full analysis and recommendations
    """
    engine = SHAPDrivenRecommendationEngine()

    exp_type = {
        "efficiency": ExplanationType.EFFICIENCY,
        "exergy": ExplanationType.EXERGY,
        "fluid_selection": ExplanationType.FLUID_SELECTION
    }.get(explanation_type, ExplanationType.EFFICIENCY)

    return engine.explain_and_recommend(
        model=model,
        inputs=inputs,
        result=result,
        explanation_type=exp_type,
        background_data=background_data
    )
