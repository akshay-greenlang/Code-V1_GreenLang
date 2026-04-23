"""
GreenLang Framework - Explainability Data Schemas

Pydantic models for SHAP, LIME, and engineering rationale explanations.
Provides type-safe, validated data structures for all explainability outputs.

Author: GreenLang AI Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import hashlib
import json


class PredictionType(Enum):
    """Types of predictions that can be explained."""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    ANOMALY_DETECTION = "anomaly_detection"
    OPTIMIZATION = "optimization"
    CALCULATION = "calculation"


class ExplainerType(Enum):
    """Types of explainers available."""
    SHAP_TREE = "shap_tree"
    SHAP_KERNEL = "shap_kernel"
    SHAP_DEEP = "shap_deep"
    LIME_TABULAR = "lime_tabular"
    ENGINEERING_RATIONALE = "engineering_rationale"
    CAUSAL_ANALYSIS = "causal_analysis"


class StandardSource(Enum):
    """Standard sources for engineering citations."""
    ASME = "ASME"  # American Society of Mechanical Engineers
    EPA = "EPA"  # US Environmental Protection Agency
    NIST = "NIST"  # National Institute of Standards and Technology
    IAPWS = "IAPWS"  # International Association for Properties of Water and Steam
    ISO = "ISO"  # International Organization for Standardization
    ASHRAE = "ASHRAE"  # American Society of Heating, Refrigerating and Air-Conditioning Engineers
    API = "API"  # American Petroleum Institute
    IPCC = "IPCC"  # Intergovernmental Panel on Climate Change
    IEC = "IEC"  # International Electrotechnical Commission
    IEEE = "IEEE"  # Institute of Electrical and Electronics Engineers
    DEFRA = "DEFRA"  # UK Department for Environment, Food & Rural Affairs
    GHG_PROTOCOL = "GHG_PROTOCOL"  # Greenhouse Gas Protocol


@dataclass
class ConfidenceBounds:
    """Confidence interval bounds for explanations."""
    lower_bound: float
    upper_bound: float
    confidence_level: float = 0.95
    method: str = "bootstrap"

    def contains(self, value: float) -> bool:
        """Check if value falls within bounds."""
        return self.lower_bound <= value <= self.upper_bound

    def width(self) -> float:
        """Get interval width."""
        return self.upper_bound - self.lower_bound


@dataclass
class UncertaintyRange:
    """Uncertainty quantification for explanations."""
    mean: float
    std: float
    min_value: float
    max_value: float
    percentile_5: float
    percentile_95: float
    num_samples: int

    @property
    def coefficient_of_variation(self) -> float:
        """Get coefficient of variation (CV)."""
        if self.mean == 0:
            return float('inf')
        return abs(self.std / self.mean)


@dataclass
class FeatureContribution:
    """Contribution of a single feature to the prediction."""
    feature_name: str
    feature_value: float
    contribution: float
    contribution_percentage: float
    direction: str  # "positive" or "negative"
    baseline_value: float = 0.0
    confidence_bounds: Optional[ConfidenceBounds] = None
    description: Optional[str] = None
    unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "feature_name": self.feature_name,
            "feature_value": self.feature_value,
            "contribution": self.contribution,
            "contribution_percentage": self.contribution_percentage,
            "direction": self.direction,
            "baseline_value": self.baseline_value,
        }
        if self.confidence_bounds:
            result["confidence_bounds"] = {
                "lower": self.confidence_bounds.lower_bound,
                "upper": self.confidence_bounds.upper_bound,
                "level": self.confidence_bounds.confidence_level,
            }
        if self.description:
            result["description"] = self.description
        if self.unit:
            result["unit"] = self.unit
        return result


@dataclass
class InteractionEffect:
    """Interaction effect between two features."""
    feature_1: str
    feature_2: str
    interaction_value: float
    interaction_type: str = "multiplicative"  # or "additive"
    significance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_1": self.feature_1,
            "feature_2": self.feature_2,
            "interaction_value": self.interaction_value,
            "interaction_type": self.interaction_type,
            "significance": self.significance,
        }


@dataclass
class SHAPExplanation:
    """Complete SHAP explanation for a prediction."""
    explanation_id: str
    prediction_type: PredictionType
    base_value: float
    prediction_value: float
    feature_contributions: List[FeatureContribution]
    interaction_effects: Optional[Dict[str, Dict[str, float]]] = None
    consistency_check: float = 0.0
    explainer_type: str = "tree"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    computation_time_ms: float = 0.0
    random_seed: int = 42
    provenance_hash: Optional[str] = None

    def __post_init__(self):
        """Compute provenance hash after initialization."""
        if self.provenance_hash is None:
            self.provenance_hash = self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "explanation_id": self.explanation_id,
            "base_value": round(self.base_value, 10),
            "prediction_value": round(self.prediction_value, 10),
            "contributions": [
                {
                    "name": c.feature_name,
                    "value": round(c.contribution, 10)
                }
                for c in self.feature_contributions
            ],
            "random_seed": self.random_seed,
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_top_contributors(self, n: int = 5) -> List[FeatureContribution]:
        """Get top n contributing features."""
        sorted_contribs = sorted(
            self.feature_contributions,
            key=lambda x: abs(x.contribution),
            reverse=True
        )
        return sorted_contribs[:n]

    def get_positive_contributors(self) -> List[FeatureContribution]:
        """Get features with positive contributions."""
        return [c for c in self.feature_contributions if c.direction == "positive"]

    def get_negative_contributors(self) -> List[FeatureContribution]:
        """Get features with negative contributions."""
        return [c for c in self.feature_contributions if c.direction == "negative"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "explanation_id": self.explanation_id,
            "prediction_type": self.prediction_type.value,
            "base_value": self.base_value,
            "prediction_value": self.prediction_value,
            "feature_contributions": [c.to_dict() for c in self.feature_contributions],
            "interaction_effects": self.interaction_effects,
            "consistency_check": self.consistency_check,
            "explainer_type": self.explainer_type,
            "timestamp": self.timestamp.isoformat(),
            "computation_time_ms": self.computation_time_ms,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class LIMEExplanation:
    """Complete LIME explanation for a prediction."""
    explanation_id: str
    prediction_type: PredictionType
    prediction_value: float
    feature_contributions: List[FeatureContribution]
    local_model_r2: float
    local_model_intercept: float
    neighborhood_size: int
    kernel_width: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    computation_time_ms: float = 0.0
    random_seed: int = 42
    provenance_hash: Optional[str] = None

    def __post_init__(self):
        """Compute provenance hash after initialization."""
        if self.provenance_hash is None:
            self.provenance_hash = self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "explanation_id": self.explanation_id,
            "prediction_value": round(self.prediction_value, 10),
            "local_model_r2": round(self.local_model_r2, 10),
            "contributions": [
                {
                    "name": c.feature_name,
                    "value": round(c.contribution, 10)
                }
                for c in self.feature_contributions
            ],
            "random_seed": self.random_seed,
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def is_reliable(self, min_r2: float = 0.7) -> bool:
        """Check if explanation has acceptable local fidelity."""
        return self.local_model_r2 >= min_r2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "explanation_id": self.explanation_id,
            "prediction_type": self.prediction_type.value,
            "prediction_value": self.prediction_value,
            "feature_contributions": [c.to_dict() for c in self.feature_contributions],
            "local_model_r2": self.local_model_r2,
            "local_model_intercept": self.local_model_intercept,
            "neighborhood_size": self.neighborhood_size,
            "kernel_width": self.kernel_width,
            "timestamp": self.timestamp.isoformat(),
            "computation_time_ms": self.computation_time_ms,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class StandardCitation:
    """Citation to an engineering standard."""
    source: StandardSource
    standard_id: str
    section: Optional[str] = None
    year: Optional[int] = None
    title: Optional[str] = None
    url: Optional[str] = None

    def format_citation(self) -> str:
        """Format as citation string."""
        parts = [f"{self.source.value} {self.standard_id}"]
        if self.section:
            parts.append(f"Section {self.section}")
        if self.year:
            parts.append(f"({self.year})")
        if self.title:
            parts.append(f'"{self.title}"')
        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source.value,
            "standard_id": self.standard_id,
            "section": self.section,
            "year": self.year,
            "title": self.title,
            "url": self.url,
        }


@dataclass
class ThermodynamicPrinciple:
    """Reference to a thermodynamic principle."""
    name: str
    formula: str
    description: str
    variables: Dict[str, str]  # variable_name -> description
    citations: List[StandardCitation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "formula": self.formula,
            "description": self.description,
            "variables": self.variables,
            "citations": [c.to_dict() for c in self.citations],
        }


@dataclass
class EngineeringRationale:
    """Complete engineering rationale for a calculation."""
    rationale_id: str
    calculation_type: str
    summary: str
    methodology: List[str]
    principles: List[ThermodynamicPrinciple]
    citations: List[StandardCitation]
    assumptions: List[str]
    limitations: List[str]
    input_parameters: Dict[str, Any]
    output_results: Dict[str, Any]
    validation_status: str = "PASS"
    confidence_level: float = 0.95
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: Optional[str] = None

    def __post_init__(self):
        """Compute provenance hash after initialization."""
        if self.provenance_hash is None:
            self.provenance_hash = self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "rationale_id": self.rationale_id,
            "calculation_type": self.calculation_type,
            "input_parameters": str(self.input_parameters),
            "output_results": str(self.output_results),
            "citations": [c.format_citation() for c in self.citations],
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rationale_id": self.rationale_id,
            "calculation_type": self.calculation_type,
            "summary": self.summary,
            "methodology": self.methodology,
            "principles": [p.to_dict() for p in self.principles],
            "citations": [c.to_dict() for c in self.citations],
            "assumptions": self.assumptions,
            "limitations": self.limitations,
            "input_parameters": self.input_parameters,
            "output_results": self.output_results,
            "validation_status": self.validation_status,
            "confidence_level": self.confidence_level,
            "timestamp": self.timestamp.isoformat(),
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class CausalNode:
    """Node in a causal graph."""
    node_id: str
    name: str
    node_type: str  # "variable", "intervention", "outcome"
    value: Optional[float] = None
    unit: Optional[str] = None
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type,
            "value": self.value,
            "unit": self.unit,
            "parents": self.parents,
            "children": self.children,
            "metadata": self.metadata,
        }


@dataclass
class CausalEdge:
    """Edge in a causal graph."""
    source: str
    target: str
    effect_size: float
    confidence: float = 0.95
    edge_type: str = "direct"  # "direct", "indirect", "confounded"
    mechanism: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "effect_size": self.effect_size,
            "confidence": self.confidence,
            "edge_type": self.edge_type,
            "mechanism": self.mechanism,
        }


@dataclass
class CounterfactualExplanation:
    """Counterfactual explanation for causal analysis."""
    counterfactual_id: str
    original_outcome: float
    counterfactual_outcome: float
    interventions: Dict[str, float]  # variable -> new value
    effect_size: float
    confidence: float = 0.95
    feasibility_score: float = 1.0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "counterfactual_id": self.counterfactual_id,
            "original_outcome": self.original_outcome,
            "counterfactual_outcome": self.counterfactual_outcome,
            "interventions": self.interventions,
            "effect_size": self.effect_size,
            "confidence": self.confidence,
            "feasibility_score": self.feasibility_score,
            "description": self.description,
        }


@dataclass
class RootCauseAnalysis:
    """Root cause analysis result."""
    analysis_id: str
    outcome_variable: str
    outcome_deviation: float
    root_causes: List[Dict[str, Any]]  # Ranked list of causes
    causal_paths: List[List[str]]  # Paths from causes to outcome
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: Optional[str] = None

    def __post_init__(self):
        """Compute provenance hash after initialization."""
        if self.provenance_hash is None:
            self.provenance_hash = self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "analysis_id": self.analysis_id,
            "outcome_variable": self.outcome_variable,
            "root_causes": str(self.root_causes),
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analysis_id": self.analysis_id,
            "outcome_variable": self.outcome_variable,
            "outcome_deviation": self.outcome_deviation,
            "root_causes": self.root_causes,
            "causal_paths": self.causal_paths,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class CausalAnalysisResult:
    """Complete causal analysis result."""
    analysis_id: str
    nodes: List[CausalNode]
    edges: List[CausalEdge]
    root_cause_analysis: Optional[RootCauseAnalysis] = None
    counterfactuals: List[CounterfactualExplanation] = field(default_factory=list)
    intervention_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: Optional[str] = None

    def __post_init__(self):
        """Compute provenance hash after initialization."""
        if self.provenance_hash is None:
            self.provenance_hash = self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "analysis_id": self.analysis_id,
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analysis_id": self.analysis_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "root_cause_analysis": self.root_cause_analysis.to_dict() if self.root_cause_analysis else None,
            "counterfactuals": [c.to_dict() for c in self.counterfactuals],
            "intervention_recommendations": self.intervention_recommendations,
            "timestamp": self.timestamp.isoformat(),
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class DashboardExplanationData:
    """Aggregated explanation data for dashboard display."""
    shap_explanation: Optional[SHAPExplanation] = None
    lime_explanation: Optional[LIMEExplanation] = None
    engineering_rationale: Optional[EngineeringRationale] = None
    causal_analysis: Optional[CausalAnalysisResult] = None
    visualization_data: Dict[str, Any] = field(default_factory=dict)
    summary_text: str = ""
    confidence_level: float = 0.95
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "shap_explanation": self.shap_explanation.to_dict() if self.shap_explanation else None,
            "lime_explanation": self.lime_explanation.to_dict() if self.lime_explanation else None,
            "engineering_rationale": self.engineering_rationale.to_dict() if self.engineering_rationale else None,
            "causal_analysis": self.causal_analysis.to_dict() if self.causal_analysis else None,
            "visualization_data": self.visualization_data,
            "summary_text": self.summary_text,
            "confidence_level": self.confidence_level,
            "timestamp": self.timestamp.isoformat(),
        }
